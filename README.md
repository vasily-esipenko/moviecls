# MovieCLS - Movie Classification by Frames

MovieCLS - система для распознавания фильмов по кадрам с использованием различных архитектур нейронных сетей и методов текстовых эмбеддингов.

## Возможности

- Распознавание фильмов по кадрам
- Поддержка различных визуальных архитектур:
  - ResNet (resnet50, resnet18)
  - EfficientNet (efficientnet_b0, efficientnet_b3)
  - Vision Transformer (vit_b_16, vit_l_16)
- Поддержка различных текстовых эмбеддингов:
  - Word2Vec
  - SBERT (Sentence-BERT)
  - FastText
  - GloVe
- Возможность выбора разных размеров входных изображений (224px, 456px, 600px и др.)
- Контроль заморозки слоев для трансферного обучения
- Логирование метрик с использованием ClearML
- Сохранение и визуализация результатов

## Установка

1. Клонировать репозиторий:
```bash
git clone https://github.com/username/moviecls.git
cd moviecls
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Установить дополнительные пакеты для работы с выбранными эмбеддингами:
```bash
# Для SBERT
pip install sentence-transformers

# Для FastText
pip install fasttext

# Для ClearML
pip install clearml
```

## Использование

### Выбор конфигурации

Cистема позволяет гибко настраивать модели и эмбеддинги через конфигурационный файл:

```yaml
model:
  backbone: "resnet50"  # Возможные варианты: "resnet50", "resnet18", "efficientnet_b0", "efficientnet_b3", "vit_b_16", "vit_l_16"
  embedding_dim: 300    # Размерность текстовых эмбеддингов
  image_size: 224       # Размер изображения (224, 456, 600 и т.д.)
  freeze_layers: "none" # Режим заморозки слоев: "none", "partial", "all_except_last", "all"

embeddings:
  type: "word2vec"      # Тип эмбеддингов: "word2vec", "sbert", "fasttext", "glove"
  # Параметры для конкретного типа эмбеддингов:
  model_name: "word2vec-google-news-300"  # Для word2vec
  sbert_model_name: "paraphrase-MiniLM-L6-v2"  # Для SBERT
  fasttext_model_path: "cc.ru.300.bin"  # Для FastText
  glove_model_path: "glove.6B.300d.txt"  # Для GloVe
```

### Режимы заморозки слоев

Система поддерживает следующие режимы заморозки слоев для трансферного обучения:

- `"none"` - нет заморозки, все слои обучаются
- `"partial"` - замораживаются ранние слои:
  - Для ResNet: первые 2 блока (layer1, layer2)
  - Для EfficientNet: первая половина блоков
  - Для ViT: только слой эмбеддингов
- `"all_except_last"` - замораживается вся модель кроме последних слоев:
  - Для ResNet: только последний блок (layer4) обучается
  - Для EfficientNet: только 2 последних блока обучаются
  - Для ViT: обучаются все слои кроме эмбеддингов и первых трансформеров
- `"all"` - замораживается вся базовая модель, обучается только проекционный слой

### Запуск обучения

```bash
python -m moviecls.train --config path/to/config.yaml
```

### Примеры конфигураций

#### ResNet50 с Word2Vec (224px)
```yaml
model:
  backbone: "resnet50"
  embedding_dim: 300
  image_size: 224
  freeze_layers: "none"

embeddings:
  type: "word2vec"
  model_name: "word2vec-google-news-300"
```

#### EfficientNet-B3 с SBERT (456px) и частичной заморозкой
```yaml
model:
  backbone: "efficientnet_b3"
  embedding_dim: 768
  image_size: 456
  freeze_layers: "partial"

embeddings:
  type: "sbert"
  sbert_model_name: "paraphrase-MiniLM-L6-v2"
```

#### Vision Transformer с FastText (224px) и заморозкой всех слоев кроме последнего
```yaml
model:
  backbone: "vit_b_16"
  embedding_dim: 300
  image_size: 224
  freeze_layers: "all_except_last"

embeddings:
  type: "fasttext"
  fasttext_model_path: "cc.ru.300.bin"
```

## Оценка и метрики

Система рассчитывает и логирует следующие метрики:
- Top-1 Accuracy (точность определения фильма на первой позиции)
- Top-5 Accuracy (точность определения фильма в топ-5 результатах)
- MRR (Mean Reciprocal Rank)
- Precision@k (точность в первых k результатах)

## Логирование с ClearML

Для настройки логирования с ClearML:

1. Установите ClearML:
```bash
pip install clearml
```

2. Настройте ClearML:
```bash
clearml-init
```

3. Активируйте логирование в конфигурации:
```yaml
logging:
  use_clearml: true
  project_name: "MovieCLS"
  task_name: "experment_1"
``` 