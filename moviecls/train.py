import os
import sys
import json
import pickle
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from moviecls.config import load_config, print_config, save_config
from moviecls.data.dataset import MovieFramesDataset, create_subset, create_data_loaders
from moviecls.data.transforms import get_train_transform, get_test_transform
from moviecls.models.visual_model import create_model
from moviecls.utils.embeddings import load_embedding_model, create_text_embeddings
from moviecls.utils.metrics import train_epoch, validate
from moviecls.utils.logger import TrainingLogger


def main(args):
    config = load_config(args.config)
    print_config(config)

    save_config(
        config,
        os.path.join(config.data.output_dir, "config.yaml")
    )

    device = torch.device(config.training.device)
    print(f"Используем устройство: {device}")

    print(f"Загрузка метаданных из {config.data.metadata_file}")
    with open(config.data.metadata_file, "r", encoding="utf-8") as f:
        movies_metadata = json.load(f)

    movies_with_frames = {movie_id: data for movie_id, data in movies_metadata.items()
                          if data.get("frame_count", 0) > 0}
    print(f"Найдено {len(movies_with_frames)} фильмов с кадрами")

    image_size = config.model.image_size
    train_transform = get_train_transform(image_size)
    test_transform = get_test_transform(image_size)

    print(f"Используем размер изображения: {image_size}x{image_size}")

    full_dataset = MovieFramesDataset(
        movies_with_frames, config.data.frames_dir, transform=None)

    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=config.data.test_size,
        random_state=42
    )

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=config.data.val_size,
        random_state=42
    )

    train_dataset = create_subset(full_dataset, train_indices, train_transform)
    val_dataset = create_subset(full_dataset, val_indices, test_transform)
    test_dataset = create_subset(full_dataset, test_indices, test_transform)

    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config.training.batch_size
    )

    embeddings_path = config.embeddings.embeddings_path
    embedding_type = config.embeddings.type
    embedding_dim = config.model.embedding_dim

    if os.path.exists(embeddings_path) and getattr(config.embeddings, 'use_cached_embeddings', True):
        print(
            f"Загружаем предварительно созданные текстовые эмбеддинги из {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            text_embeddings = pickle.load(f)
    else:
        print(
            f"Создаем текстовые эмбеддинги типа {embedding_type} для фильмов...")

        embedding_model, actual_embedding_type = load_embedding_model(config)

        if embedding_model is None:
            print(
                f"Не удалось загрузить модель для {embedding_type}, используем случайные эмбеддинги")

        text_embeddings = create_text_embeddings(
            movies_with_frames,
            embedding_model,
            embedding_type=actual_embedding_type,
            embedding_dim=embedding_dim
        )

        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(text_embeddings, f)

        print(f"Текстовые эмбеддинги сохранены в {embeddings_path}")

    model = create_model(config, device)
    print(f"Модель {config.model.backbone} создана и перемещена на {device}")

    freeze_mode = getattr(config.model, 'freeze_layers', 'none')

    backbone_params = [p for p in model.backbone.parameters()
                       if p.requires_grad]
    projection_params = list(model.projection.parameters())

    if freeze_mode == "all":
        optimizer = torch.optim.Adam([
            {'params': projection_params, 'lr': config.training.learning_rate}
        ], weight_decay=config.training.weight_decay)
        print("Оптимизатор: обучается только проекционный слой")
    else:
        base_lr = config.training.learning_rate
        backbone_lr = base_lr * 0.1

        projection_lr = base_lr * \
            2.0 if freeze_mode in ["all_except_last", "partial"] else base_lr

        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': projection_params, 'lr': projection_lr}
        ], weight_decay=config.training.weight_decay)
        print(
            f"Оптимизатор: backbone LR={backbone_lr}, projection LR={projection_lr}")

    criterion = nn.CosineEmbeddingLoss()

    logger = TrainingLogger(config, config.data.output_dir)

    best_top5_acc = 0
    best_model_path = os.path.join(config.data.output_dir, 'best_model.pth')

    precision_k = getattr(config.evaluation, 'precision_k', 10) if hasattr(
        config, 'evaluation') else 10

    print(f"Начало обучения на {device}...")

    for epoch in range(config.training.num_epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            text_embeddings,
            device
        )

        val_loss, top1_acc, top5_acc, mrr, precision_at_k = validate(
            model,
            val_loader,
            criterion,
            text_embeddings,
            device,
            precision_k=precision_k
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)
        scheduler.step(val_loss)

        logger.log_epoch(epoch + 1, train_loss, val_loss,
                         top1_acc, top5_acc, mrr, precision_at_k)

        if top5_acc > best_top5_acc:
            best_top5_acc = top5_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Модель сохранена в {best_model_path}")

    logger.plot_training_history()

    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_top1_acc, test_top5_acc, test_mrr, test_precision_at_k = validate(
        model,
        test_loader,
        criterion,
        text_embeddings,
        device,
        precision_k=precision_k
    )

    logger.log_test_results(test_loss, test_top1_acc,
                            test_top5_acc, test_mrr, test_precision_at_k)

    logger.log_prediction_samples(model, test_loader, text_embeddings, device)

    print(f"Обучение завершено. Лучшая модель сохранена в {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение модели классификации кадров")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Путь к файлу конфигурации"
    )

    args = parser.parse_args()
    main(args)
