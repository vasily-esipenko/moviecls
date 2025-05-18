import os
import json
import pickle
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from moviecls.config import load_config
from moviecls.models.visual_model import create_model
from moviecls.data.transforms import get_test_transform
from moviecls.utils.metrics import find_movie_by_frame


def find_movie(config, model_path, frame_path, device, top_k=5):
    print(
        f"Загружаем текстовые эмбеддинги из {config.embeddings.embeddings_path}")
    with open(config.embeddings.embeddings_path, 'rb') as f:
        text_embeddings = pickle.load(f)

    print("Инициализация модели...")
    model = create_model(config, device)

    print(f"Загружаем веса модели из {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_test_transform(config.model.image_size)

    results = find_movie_by_frame(
        frame_path, model, text_embeddings, transform, device, top_k)

    with open(config.data.metadata_file, "r", encoding="utf-8") as f:
        movies_metadata = json.load(f)

    full_results = []
    for result in results:
        movie_id = result['movie_id']
        if movie_id in movies_metadata:
            metadata = movies_metadata[movie_id].copy()
            metadata['similarity'] = result['similarity']
            full_results.append(metadata)

    return full_results


def main(args):
    config = load_config(args.config)

    device = torch.device(
        args.device if args.device else config.training.device)
    print(f"Используем устройство: {device}")

    results = find_movie(
        config,
        args.model,
        args.frame,
        device,
        args.top_k
    )

    print(f"\nРезультаты поиска для кадра {args.frame}:")
    for i, result in enumerate(results):
        print(
            f"{i+1}. {result.get('title', 'Неизвестно')} ({result.get('year', 'Н/Д')})")
        print(f"   Сходство: {result['similarity']:.4f}")
        print(f"   Жанры: {', '.join(result.get('genres', []))}")
        if 'director' in result:
            print(f"   Режиссер: {result['director']}")
        print()

    if args.output:
        visualize_results(args.frame, results, args.output)
        print(f"Визуализация сохранена в {args.output}")


def visualize_results(frame_path, results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(Image.open(frame_path))
    axes[0].set_title("Исходный кадр")
    axes[0].axis('off')

    titles = [
        f"{r.get('title', 'Неизвестно')} ({r.get('year', 'Н/Д')})" for r in results]
    similarities = [r['similarity'] for r in results]

    y_pos = np.arange(len(titles))
    axes[1].barh(y_pos, similarities)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(titles)
    axes[1].invert_yaxis()
    axes[1].set_title("Результаты поиска")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск фильма по кадру")
    parser.add_argument("--config", type=str, default=None,
                        help="Путь к файлу конфигурации")
    parser.add_argument("--model", type=str, required=True,
                        help="Путь к весам модели")
    parser.add_argument("--frame", type=str, required=True,
                        help="Путь к изображению кадра")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Количество результатов")
    parser.add_argument("--output", type=str,
                        help="Путь для сохранения визуализации")
    parser.add_argument("--device", type=str, help="Устройство (cuda/cpu)")

    args = parser.parse_args()
    main(args)
