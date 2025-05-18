import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class MovieFramesDataset(Dataset):
    def __init__(self, movie_data, frames_dir, transform=None):
        self.movie_data = movie_data
        self.frames_dir = frames_dir
        self.transform = transform
        self.samples = []

        for movie_id, metadata in movie_data.items():
            movie_dir = os.path.join(frames_dir, str(movie_id))
            if os.path.exists(movie_dir):
                frame_files = [f for f in os.listdir(movie_dir)
                               if f.endswith(('.jpg', '.jpeg', '.png'))]

                for frame_file in frame_files:
                    self.samples.append({
                        'movie_id': movie_id,
                        'frame_path': os.path.join(movie_dir, frame_file),
                        'metadata': metadata
                    })

        print(
            f"Всего доступно {len(self.samples)} кадров из {len(movie_data)} фильмов")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        movie_id = sample['movie_id']
        frame_path = sample['frame_path']
        metadata = sample['metadata']

        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Ошибка при загрузке {frame_path}: {e}")
            random_idx = random.randint(0, len(self) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)

        return {
            'image': image,
            'movie_id': movie_id,
            'title': metadata.get('title', ''),
            'genres': metadata.get('genres', []),
            'year': metadata.get('year', ''),
            'director': metadata.get('director', ''),
            'cast': metadata.get('cast', []),
            'path': frame_path
        }


class MovieSubset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        movie_id = sample['movie_id']
        frame_path = sample['frame_path']
        metadata = sample['metadata']

        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Ошибка при загрузке {frame_path}: {e}")
            random_idx = random.randint(0, len(self) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)

        return {
            'image': image,
            'movie_id': movie_id,
            'title': metadata.get('title', ''),
            'genres': metadata.get('genres', []),
            'year': metadata.get('year', ''),
            'director': metadata.get('director', ''),
            'cast': metadata.get('cast', []),
            'path': frame_path
        }


def create_subset(dataset, indices, transform):
    subset = [dataset.samples[i] for i in indices]
    return MovieSubset(subset, transform)


def movie_collate_fn(batch):
    result = {
        'image': [],
        'movie_id': [],
        'title': [],
        'genres': [],
        'year': [],
        'director': [],
        'cast': [],
        'path': []
    }

    for item in batch:
        for key in result.keys():
            result[key].append(item[key])

    if torch.is_tensor(batch[0]['image']):
        result['image'] = torch.stack(result['image'])

    return result


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=4):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=movie_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader
