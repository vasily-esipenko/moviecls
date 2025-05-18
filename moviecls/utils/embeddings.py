import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import gensim.downloader
import gensim

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("SBERT не установлен. Для использования SBERT установите: pip install sentence-transformers")

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("FastText не установлен. Для использования FastText установите: pip install fasttext")


def load_embedding_model(config):
    embedding_type = config.embeddings.type
    cache_path = config.embeddings.cache_path

    if getattr(config.embeddings, 'use_cached_embeddings', True) and os.path.exists(cache_path):
        print(f"Загружаем модель эмбеддингов из кэша: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f), embedding_type

    if embedding_type == "word2vec":
        model = load_word2vec_model(config.embeddings.model_name, cache_path)
    elif embedding_type == "fasttext":
        if not FASTTEXT_AVAILABLE:
            raise ImportError(
                "FastText не установлен. Установите с помощью: pip install fasttext")
        model = load_fasttext_model(
            config.embeddings.fasttext_model_path, cache_path)
    elif embedding_type == "glove":
        model = load_glove_model(
            config.embeddings.glove_model_path, cache_path)
    elif embedding_type == "sbert":
        if not SBERT_AVAILABLE:
            raise ImportError(
                "SBERT не установлен. Установите с помощью: pip install sentence-transformers")
        model = load_sbert_model(
            config.embeddings.sbert_model_name, cache_path)
    else:
        raise ValueError(f"Неизвестный тип эмбеддингов: {embedding_type}")

    return model, embedding_type


def load_word2vec_model(model_name, cache_path):
    try:
        if os.path.exists(cache_path):
            print(f"Загружаем word2vec модель из кэша: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Загрузка предобученной word2vec модели {model_name}...")
        word2vec_model = gensim.downloader.load(model_name)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(word2vec_model, f)

        print(f"Модель word2vec успешно загружена и сохранена в {cache_path}")
        return word2vec_model

    except Exception as e:
        print(f"Ошибка при загрузке word2vec модели: {e}")
        return None


def load_fasttext_model(model_path, cache_path):
    try:
        print(f"Загрузка предобученной FastText модели из {model_path}...")
        fasttext_model = fasttext.load_model(model_path)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(fasttext_model, f)

        print(f"Модель FastText успешно загружена и сохранена в {cache_path}")
        return fasttext_model
    except Exception as e:
        print(f"Ошибка при загрузке FastText модели: {e}")
        return None


def load_glove_model(model_path, cache_path):
    try:
        print(f"Загрузка предобученной GloVe модели из {model_path}...")
        glove_model = {}
        with open(model_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Загрузка GloVe"):
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                glove_model[word] = vector

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(glove_model, f)

        print(f"Модель GloVe успешно загружена и сохранена в {cache_path}")
        return glove_model
    except Exception as e:
        print(f"Ошибка при загрузке GloVe модели: {e}")
        return None


def load_sbert_model(model_name, cache_path):
    try:
        print(f"Загрузка предобученной SBERT модели {model_name}...")
        sbert_model = SentenceTransformer(model_name)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(sbert_model, f)

        print(f"Модель SBERT успешно загружена и сохранена в {cache_path}")
        return sbert_model
    except Exception as e:
        print(f"Ошибка при загрузке SBERT модели: {e}")
        return None


def create_text_embeddings(movie_data, embedding_model, embedding_type="word2vec", embedding_dim=300):
    text_embeddings = {}

    for movie_id, metadata in tqdm(movie_data.items(), desc="Создание текстовых эмбеддингов"):
        title = metadata.get('title', '')
        original_title = metadata.get('original_title', '')
        overview = metadata.get('overview', '')
        genres = metadata.get('genres', [])
        director = metadata.get('director', '')
        cast = metadata.get('cast', [])
        keywords = metadata.get('keywords', [])

        if embedding_type == "sbert":
            movie_text = f"{title} {original_title} "

            if genres:
                movie_text += f"Жанры: {', '.join(genres)}. "

            if overview:
                movie_text += f"Описание: {overview} "

            if director:
                movie_text += f"Режиссер: {director}. "

            if cast:
                top_cast = cast[:5] if len(cast) > 5 else cast
                movie_text += f"В ролях: {', '.join(top_cast)}. "

            if keywords:
                movie_text += f"Ключевые слова: {', '.join(keywords)}."

            if embedding_model:
                embedding = embedding_model.encode(movie_text)
            else:
                embedding = np.random.randn(embedding_dim)
        else:
            all_words = []

            title_words = title.lower().split()
            original_title_words = original_title.lower().split() if original_title else []
            all_words.extend(title_words * 3)
            all_words.extend(original_title_words * 2)

            if overview:
                overview_words = overview.lower().split()
                all_words.extend(overview_words)

            all_words.extend([genre.lower() for genre in genres] * 5)

            if director:
                all_words.extend(director.lower().split() * 2)

            all_words.extend([actor.lower() for actor in cast])

            all_words.extend([keyword.lower() for keyword in keywords] * 3)

            if embedding_model:
                if embedding_type == "word2vec" or embedding_type == "glove":
                    word_vectors = []
                    for word in all_words:
                        if word in embedding_model:
                            word_vectors.append(embedding_model[word])

                    if word_vectors:
                        embedding = np.mean(word_vectors, axis=0)
                    else:
                        embedding = np.zeros(embedding_dim)

                elif embedding_type == "fasttext":
                    word_vectors = [embedding_model.get_word_vector(
                        word) for word in all_words]
                    if word_vectors:
                        embedding = np.mean(word_vectors, axis=0)
                    else:
                        embedding = np.zeros(embedding_dim)
                else:
                    embedding = np.zeros(embedding_dim)
            else:
                embedding = np.random.randn(embedding_dim)

        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        text_embeddings[movie_id] = embedding

    return text_embeddings


def get_text_embeddings_batch(batch, text_embeddings, device):
    movie_ids = batch['movie_id']
    embedding_dim = next(iter(text_embeddings.values())).shape[0]
    batch_embeddings = []

    for movie_id in movie_ids:
        embedding = text_embeddings.get(movie_id, np.zeros(embedding_dim))
        batch_embeddings.append(embedding)

    return torch.tensor(np.array(batch_embeddings), dtype=torch.float32).to(device)
