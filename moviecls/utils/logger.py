import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from omegaconf import OmegaConf

try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML не установлен. Для логирования в ClearML установите: pip install clearml")


class TrainingLogger:
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.config = config
        self.use_clearml = False
        self.task = None
        self.logger = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'top1_acc': [],
            'top5_acc': [],
            'mrr': [],
            'precision_at_k': []
        }

        os.makedirs(output_dir, exist_ok=True)

        if config.logging.use_clearml and CLEARML_AVAILABLE:
            self.use_clearml = True
            self.task = Task.init(
                project_name=config.logging.project_name,
                task_name=config.logging.task_name,
            )
            self.logger = self.task.get_logger()

            config_dict = OmegaConf.to_container(config, resolve=True)
            self.task.connect_configuration(config_dict)

            print("ClearML логирование активировано")

    def log_epoch(self, epoch, train_loss, val_loss, top1_acc, top5_acc, mrr=None, precision_at_k=None):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['top1_acc'].append(top1_acc)
        self.history['top5_acc'].append(top5_acc)

        if mrr is not None:
            self.history['mrr'].append(mrr)

        if precision_at_k is not None:
            self.history['precision_at_k'].append(precision_at_k)

        print(f"Эпоха {epoch}:")
        print(
            f"Потери при обучении: {train_loss:.4f}, Потери при валидации: {val_loss:.4f}")
        print(
            f"Top-1 точность: {top1_acc:.4f}, Top-5 точность: {top5_acc:.4f}")

        if mrr is not None:
            print(f"MRR: {mrr:.4f}")

        if precision_at_k is not None:
            k_value = getattr(self.config.evaluation, 'precision_k', 10)
            print(f"Precision@{k_value}: {precision_at_k:.4f}")

        if self.use_clearml:
            self.logger.report_scalar("Loss", "Обучение", train_loss, epoch)
            self.logger.report_scalar("Loss", "Валидация", val_loss, epoch)
            self.logger.report_scalar("Accuracy", "Top-1", top1_acc, epoch)
            self.logger.report_scalar("Accuracy", "Top-5", top5_acc, epoch)

            if mrr is not None:
                self.logger.report_scalar("Metrics", "MRR", mrr, epoch)

            if precision_at_k is not None:
                k_value = getattr(self.config.evaluation, 'precision_k', 10)
                self.logger.report_scalar(
                    "Metrics", f"Precision@{k_value}", precision_at_k, epoch)

    def log_test_results(self, test_loss, test_top1_acc, test_top5_acc, test_mrr=None, test_precision_at_k=None):
        print(f"Финальная оценка на тестовой выборке:")
        print(f"Потери: {test_loss:.4f}")
        print(f"Top-1 accuracy: {test_top1_acc:.4f}")
        print(f"Top-5 accuracy: {test_top5_acc:.4f}")

        if test_mrr is not None:
            print(f"MRR: {test_mrr:.4f}")

        if test_precision_at_k is not None:
            k_value = getattr(self.config.evaluation, 'precision_k', 10)
            print(f"Precision@{k_value}: {test_precision_at_k:.4f}")

        if self.use_clearml:
            self.logger.report_scalar("Тестирование", "Loss", test_loss, 0)
            self.logger.report_scalar(
                "Тестирование", "Top-1 accuracy", test_top1_acc, 0)
            self.logger.report_scalar(
                "Тестирование", "Top-5 accuracy", test_top5_acc, 0)

            if test_mrr is not None:
                self.logger.report_scalar("Тестирование", "MRR", test_mrr, 0)

            if test_precision_at_k is not None:
                k_value = getattr(self.config.evaluation, 'precision_k', 10)
                self.logger.report_scalar(
                    "Тестирование", f"Precision@{k_value}", test_precision_at_k, 0)

    def plot_training_history(self):
        num_plots = 2

        has_mrr = len(self.history.get('mrr', [])) > 0
        has_precision = len(self.history.get('precision_at_k', [])) > 0

        if has_mrr or has_precision:
            num_plots += 1

        plt.figure(figsize=(6 * num_plots, 5))

        plt.subplot(1, num_plots, 1)
        plt.plot(self.history['train_loss'], label='Обучение')
        plt.plot(self.history['val_loss'], label='Валидация')
        plt.title('Динамика потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()

        plt.subplot(1, num_plots, 2)
        plt.plot(self.history['top1_acc'], label='Top-1')
        plt.plot(self.history['top5_acc'], label='Top-5')
        plt.title('Динамика точности')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()

        if has_mrr or has_precision:
            plt.subplot(1, num_plots, 3)

            if has_mrr:
                plt.plot(self.history['mrr'], label='MRR')

            if has_precision:
                k_value = getattr(self.config.evaluation, 'precision_k', 10)
                plt.plot(self.history['precision_at_k'],
                         label=f'Precision@{k_value}')

            plt.title('Дополнительные метрики')
            plt.xlabel('Эпоха')
            plt.ylabel('Значение')
            plt.legend()

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path)

        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        if self.use_clearml:
            self.logger.report_image("Графики", "История обучения",
                                     iteration=0, local_path=plot_path)

        return plot_path

    def log_prediction_samples(self, model, dataloader, text_embeddings, device, num_samples=5):
        model.eval()

        all_movie_ids = list(text_embeddings.keys())
        all_text_emb = np.array([text_embeddings[mid]
                                for mid in all_movie_ids])

        movie_titles = {}
        for movie_id in all_movie_ids:
            metadata = dataloader.dataset.samples[0]['metadata']
            movie_titles[movie_id] = metadata.get('title', 'Unknown')

        samples = []
        data_iterator = iter(dataloader)
        batch = next(data_iterator)

        for i in range(min(num_samples, len(batch['image']))):
            samples.append((
                batch['image'][i],
                batch['movie_id'][i],
                batch['path'][i]
            ))

        fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))

        for i, (image, movie_id, path) in enumerate(samples):
            image_tensor = image.unsqueeze(0).to(device)
            with torch.no_grad():
                visual_emb = model(image_tensor).cpu().numpy()

            similarities = cosine_similarity(visual_emb, all_text_emb)[0]
            top5_indices = np.argsort(-similarities)[:5]

            ax = axes[i, 0]

            img_np = image.permute(1, 2, 0).numpy()
            img_np = img_np * \
                np.array([0.229, 0.224, 0.225]) + \
                np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            ax.imshow(img_np)
            ax.set_title(f"Кадр из: {movie_titles.get(movie_id, 'Unknown')}")
            ax.axis('off')

            ax = axes[i, 1]
            ax.barh([movie_titles.get(all_movie_ids[idx], 'Unknown') for idx in top5_indices],
                    [similarities[idx] for idx in top5_indices])
            ax.set_title("Top-5 совпадений")
            ax.set_xlim(0, 1)

        plt.tight_layout()

        samples_path = os.path.join(self.output_dir, 'prediction_samples.png')
        plt.savefig(samples_path)

        if self.use_clearml:
            self.logger.report_image("Примеры", "Предсказания модели",
                                     iteration=0, local_path=samples_path)

        return samples_path
