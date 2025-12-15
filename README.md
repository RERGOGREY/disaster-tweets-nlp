# Disaster Tweets NLP (Kaggle)

Проект решает задачу бинарной классификации твитов: определить, относится ли твит к **реальной катастрофе** (`target=1`) или нет (`target=0`).

Датасет соответствует соревнованию Kaggle **“Natural Language Processing with Disaster Tweets”** и содержит поля:

- `id` — идентификатор
- `text` — текст твита (основной признак)
- `keyword` — ключевое слово (может быть пустым)
- `location` — локация (может быть пустой / шум)
- `target` — целевая метка (только в `train.csv`)

Метрика соревнования — **F1-score**.
В проекте логируются метрики в MLflow (минимум 3 графика).

---

## Что реализовано

### Baseline

Классический бейзлайн без нейросетей:

- векторизация текста: **BoW / TF-IDF / LSA (TruncatedSVD)**
- классификатор: **Logistic Regression**
- логирование метрик: `val/f1`, `val/accuracy`, `val/roc_auc`

### Основная модель

Нейросетевая модель на **PyTorch Lightning**:

- токенизация (простая whitespace) + словарь из train
- модель: **BiLSTM** (bidirectional) для классификации
- логирование метрик: `train/loss`, `val/loss`, `val/accuracy`, `val/f1` + learning rate

---
