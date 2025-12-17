# Полносвязная нейронная сеть для анализа эмоциональной окраски текстов
# Разработка для Google Colab | TensorFlow 2.x | Русские отзывы студентов
# Тема: "Разработка полносвязной нейронной сети на базе фреймворка TensorFlow Python
# для анализа эмоциональной окраски студенческих эссе и отзывов о лекциях"

# ===============================================
# 1. УСТАНОВКА И ИМПОРТ БИБЛИОТЕК
# ===============================================
!pip install tensorflow transformers datasets scikit-learn matplotlib seaborn plotly kaleido

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow версия: {tf.__version__}")
print(f"GPU доступен: {tf.config.list_physical_devices('GPU')}")

# ===============================================
# 2. ПОДКЛЮЧЕНИЕ РУССКОЯЗЫЧНОГО ДАТАСЕТА
# ===============================================
# Загружаем датасет RuSentiment с HuggingFace (русские отзывы и тексты)
dataset = load_dataset("k1tub/sentiment_dataset", split='train')

# Преобразуем в pandas для удобства
df = pd.DataFrame(dataset)
print("Размер датасета:", df.shape)
print("\nРаспределение классов:")
print(df['label'].value_counts())

# Маппинг лейблов: 0=negative, 1=neutral, 2=positive
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['label_name'] = df['label'].map(label_map)

# Берем подвыборку для быстрого обучения в Colab (10k примеров)
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)

# ===============================================
# 3. ПРЕДОБРАБОТКА ТЕКСТОВ
# ===============================================
def preprocess_text(text):
    """Очистка и нормализация русского текста"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Удаляем специальные символы, оставляем кириллицу и базовую пунктуацию
    text = re.sub(r'[^а-яёa-z0-9\s.,!?()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Применяем предобработку
df_sample['clean_text'] = df_sample['text'].apply(preprocess_text)

# Удаляем слишком короткие тексты
df_sample = df_sample[df_sample['clean_text'].str.len() > 10]
print(f"После фильтрации: {len(df_sample)} примеров")

# ===============================================
# 4. СОЗДАНИЕ СЛОВАРЯ И ВЕКТОРИЗАЦИЯ
# ===============================================
# Параметры словаря (полностью параметризированные)
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128

# Строим словарь
all_words = []
for text in df_sample['clean_text']:
    all_words.extend(text.split())

word_counts = Counter(all_words)
most_common = word_counts.most_common(VOCAB_SIZE - 2)  # -2 для padding и unk
vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

print(f"Размер словаря: {len(vocab)} слов")

def text_to_sequence(text, vocab, max_len):
    """Преобразование текста в последовательность индексов"""
    words = text.split()
    seq = [vocab.get(word, vocab['<UNK>']) for word in words]
    # Обрезаем или дополняем до max_len
    if len(seq) >= max_len:
        seq = seq[:max_len]
    else:
        seq.extend([vocab['<PAD>']] * (max_len - len(seq)))
    return seq

# Векторизуем тексты
X = np.array([text_to_sequence(text, vocab, MAX_LEN) for text in df_sample['clean_text']])
y = tf.keras.utils.to_categorical(df_sample['label'], num_classes=3)

# Разделяем на train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df_sample['label']  # здесь всё ок
)

# ВАЖНО: использовать те же метки df_sample, а не df_temp
labels_temp = np.argmax(y_temp, axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,          # 0.25 от 0.8 = 0.2, итого 60/20/20
    random_state=42,
    stratify=labels_temp     # вместо df_temp['label']
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ===============================================
# 5. ПОСТРОЕНИЕ ПОЛНОСВЯЗНОЙ НЕЙРОННОЙ СЕТИ
# ===============================================
# Полностью параметризированная архитектура
def create_model(vocab_size, embedding_dim, max_len, num_classes,
                hidden_units=[256, 128, 64], dropout_rate=0.3,
                embedding_trainable=True, l2_reg=1e-4):
    """
    Создание полносвязной НС с полным параметрическим контролем
    """
    model = keras.Sequential([
        # Слой встраивания слов
        layers.Embedding(vocab_size, embedding_dim,
                        input_length=max_len,
                        trainable=embedding_trainable,
                        name='embedding'),

        # Global Average Pooling для агрегации последовательностей
        layers.GlobalAveragePooling1D(name='global_avg_pool'),

        # Полносвязные слои
        layers.Dense(hidden_units[0], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),

        layers.Dense(hidden_units[1], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),

        layers.Dense(hidden_units[2], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name='dense_3'),
        layers.Dropout(dropout_rate/2, name='dropout_3'),

        # Выходной слой
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    return model

# Создаем модель с оптимальными параметрами
model = create_model(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    max_len=MAX_LEN,
    num_classes=3,
    hidden_units=[256, 128, 64],
    dropout_rate=0.3,
    embedding_trainable=True,
    l2_reg=1e-4
)

# Компиляция с оптимизатором AdamW и focal loss
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================================
# 6. CALLBACKS ДЛЯ ОПТИМИЗАЦИИ ОБУЧЕНИЯ
# ===============================================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_sentiment_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ===============================================
# 7. ОБУЧЕНИЕ МОДЕЛИ
# ===============================================
print("\n🚀 НАЧИНАЕМ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# ===============================================
# 8. ОЦЕНКА МОДЕЛИ
# ===============================================
# Тестирование
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Тестовая точность: {test_accuracy:.4f}")

# Предсказания
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Детальный отчет
print("\n📊 КЛАССИФИКАЦИОННЫЙ ОТЧЕТ:")
print(classification_report(y_true, y_pred,
                          target_names=['negative', 'neutral', 'positive']))

# ===============================================
# ===============================================
# 9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (ГЛАДКИЕ ГРАФИКИ)
# ===============================================

train_acc  = np.array(history.history['accuracy'])
val_acc    = np.array(history.history['val_accuracy'])
train_loss = np.array(history.history['loss'])
val_loss   = np.array(history.history['val_loss'])

# реальное число обученных эпох
n_epochs = len(train_acc)
epochs_range = np.arange(1, n_epochs + 1)

# --- сглаживание скользящим средним ---
def smooth_curve(x, window=3):
    if window <= 1:
        return x
    x_pad = np.pad(x, (window-1, 0), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode='valid')

smooth_train_acc  = smooth_curve(train_acc,  window=3)
smooth_val_acc    = smooth_curve(val_acc,    window=3)
smooth_train_loss = smooth_curve(train_loss, window=3)
smooth_val_loss   = smooth_curve(val_loss,   window=3)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# --------- ЛИНЕЙНЫЕ ГРАФИКИ ----------
axes[0].plot(epochs_range, smooth_train_acc,  label='Обучающая точность')
axes[0].plot(epochs_range, smooth_val_acc,    label='Валидирующая точность')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Точность')
axes[0].set_title('Динамика точности (сглаженная)')
axes[0].grid(alpha=0.3)
axes[0].legend(loc='lower right')

# --------- ЛОГАРИФМИЧЕСКИЕ/«ГИПЕРБОЛИЧЕСКИЕ» ГРАФИКИ ПОТЕРЬ ----------
axes[1].plot(epochs_range, smooth_train_loss, label='Обучающие потери')
axes[1].plot(epochs_range, smooth_val_loss,   label='Валидационные потери')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Потери')
axes[1].set_title('Динамика потерь (log-scale)')
axes[1].set_yscale('log')   # логарифмическая шкала – график получается «гиперболический»
axes[1].grid(alpha=0.3)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

# ===============================================
# 10. ТЕСТИРОВАНИЕ НА ПРИМЕРАХ СТУДЕНЧЕСКИХ ОТЗЫВОВ
# ===============================================
student_reviews = [
    "Лекция была очень интересной, преподаватель отлично объясняет материал!",
    "Сложно понять тему, нужно больше примеров и практики",
    "Обычная лекция, ничего особенного, средние знания",
    "Отличный курс! Очень полезная информация для экзамена",
    "Преподаватель запаздывает, материал подается скучно"
]

print("\n🎓 АНАЛИЗ СТУДЕНЧЕСКИХ ОТЗЫВОВ О ЛЕКЦИЯХ:")
for i, review in enumerate(student_reviews, 1):
    seq = text_to_sequence(preprocess_text(review), vocab, MAX_LEN)
    seq = np.array([seq])
    pred = model.predict(seq, verbose=0)
    emotion = label_map[np.argmax(pred)]
    confidence = np.max(pred) * 100

    print(f"{i}. '{review}' → {emotion.upper()} ({confidence:.1f}%)")

# ===============================================
# 11. HTML ОТЧЕТ (СОХРАНЕНИЕ РЕЗУЛЬТАТОВ)
# ===============================================
html_report = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ эмоциональной окраски студенческих эссе | TensorFlow НС</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card {{ transition: transform 0.3s; }}
        .metric-card:hover {{ transform: translateY(-5px); }}
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen p-8">
    <div class="max-w-6xl mx-auto">
        <header class="text-center mb-12">
            <h1 class="text-5xl font-bold bg-gradient-to-r from-purple-600 to-blue-600
                       bg-clip-text text-transparent mb-4">
                🎓 Анализ эмоциональной окраски
            </h1>
            <h2 class="text-2xl text-gray-700 mb-2">
                Полносвязная нейронная сеть TensorFlow
            </h2>
            <p class="text-xl text-gray-600">Студенческие эссе и отзывы о лекциях</p>
            <div class="flex justify-center gap-4 mt-6 text-sm text-gray-500">
                <span>Точность: <strong>{test_accuracy:.3f}</strong></span>
                <span>Размер датасета: {len(df_sample):,}</span>
                <span>Словарь: {VOCAB_SIZE:,}</span>
            </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            <div class="metric-card bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
                <div class="text-3xl mb-2">🎯</div>
                <h3 class="text-xl font-semibold text-gray-800 mb-2">Тестовая точность</h3>
                <div class="text-4xl font-bold text-green-600">{test_accuracy:.3f}</div>
            </div>
            <div class="metric-card bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
                <div class="text-3xl mb-2">📚</div>
                <h3 class="text-xl font-semibold text-gray-800 mb-2">Обучающих примеров</h3>
                <div class="text-4xl font-bold text-blue-600">{len(X_train):,}</div>
            </div>
            <div class="metric-card bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
                <div class="text-3xl mb-2">🧠</div>
                <h3 class="text-xl font-semibold text-gray-800 mb-2">Параметров модели</h3>
                <div class="text-4xl font-bold text-purple-600">{model.count_params():,}</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            <div class="bg-white p-8 rounded-2xl shadow-xl">
                <h3 class="text-2xl font-bold mb-6">📊 Результаты на студенческих отзывах</h3>
"""

for i, review in enumerate(student_reviews, 1):
    seq = text_to_sequence(preprocess_text(review), vocab, MAX_LEN)
    seq = np.array([seq])
    pred = model.predict(seq, verbose=0)
    emotion = label_map[np.argmax(pred)]
    confidence = np.max(pred) * 100

    color = "bg-red-100 text-red-800" if emotion == "negative" else \
            "bg-yellow-100 text-yellow-800" if emotion == "neutral" else \
            "bg-green-100 text-green-800"

    html_report += f"""
                <div class="mb-6 p-4 border-l-4 {color} border border-gray-200 rounded-r-lg">
                    <div class="font-semibold text-lg mb-1">Отзыв {i}</div>
                    <div class="text-gray-700 mb-3">"{review}"</div>
                    <div class="flex items-center gap-4">
                        <span class="px-4 py-1 rounded-full text-sm font-bold {color.split()[0]}">
                            {emotion.upper()}
                        </span>
                        <span class="text-sm text-gray-600">Уверенность: {confidence:.1f}%</span>
                    </div>
                </div>
    """

html_report += """
            </div>

            <div class="bg-white p-8 rounded-2xl shadow-xl">
                <h3 class="text-2xl font-bold mb-6">⚙️ Архитектура нейронной сети</h3>
                <div class="space-y-3 text-sm">
"""

html_report += """
            <div class="bg-white p-8 rounded-2xl shadow-xl">
                <h3 class="text-2xl font-bold mb-6">⚙️ Архитектура нейронной сети</h3>
                <div class="space-y-3 text-sm">
"""

for layer in model.layers:
    try:
        out_shape = tuple(layer.output.shape)
    except Exception:
        out_shape = "unknown"

    html_report += f"""
                    <div class="flex justify-between p-3 bg-gray-50 rounded-lg">
                        <span>{layer.name}</span>
                        <span class="font-mono bg-blue-100 px-2 py-1 rounded text-xs">{out_shape}</span>
                    </div>
    """

html_report += """
                </div>
            </div>
        </div>
"""


html_report += """
                </div>
            </div>
        </div>

        <div class="bg-white p-8 rounded-2xl shadow-xl">
            <h3 class="text-2xl font-bold mb-6">📈 Классификационный отчет</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="p-3 text-left font-semibold">Класс</th>
                            <th class="p-3 text-left font-semibold">Precision</th>
                            <th class="p-3 text-left font-semibold">Recall</th>
                            <th class="p-3 text-left font-semibold">F1-score</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-200">
"""

report = classification_report(y_true, y_pred,
                              target_names=['negative', 'neutral', 'positive'],
                              output_dict=True)

for label in ['negative', 'neutral', 'positive']:
    html_report += f"""
                        <tr>
                            <td class="p-3 font-medium">{label}</td>
                            <td class="p-3">{report[label]['precision']:.3f}</td>
                            <td class="p-3">{report[label]['recall']:.3f}</td>
                            <td class="p-3 font-semibold">{report[label]['f1-score']:.3f}</td>
                        </tr>
"""

html_report += """
                    </tbody>
                </table>
            </div>
        </div>

        <footer class="text-center mt-16 text-gray-500 text-sm">
            <p>Разработано для анализа студенческих эссе и отзывов о лекциях</p>
            <p>TensorFlow Dense Neural Network | Google Colab | 2025</p>
        </footer>
    </div>
</body>
</html>
"""

# Сохраняем HTML отчет
with open('sentiment_analysis_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("\n✅ HTML отчет сохранен: sentiment_analysis_report.html")
print("📁 Скачайте файл для просмотра интерактивного отчета!")

# ФУНКЦИЯ ДЛЯ ИНТЕРАКТИВНОГО ТЕСТИРОВАНИЯ
def predict_emotion(text):
    """Предсказание эмоции для произвольного текста"""
    clean_text = preprocess_text(text)
    seq = text_to_sequence(clean_text, vocab, MAX_LEN)
    seq = np.array([seq])
    pred = model.predict(seq, verbose=0)
    emotion = label_map[np.argmax(pred)]
    confidence = np.max(pred) * 100
    return emotion, confidence

print("\n🎯 ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ:")
test_text = input("Введите отзыв о лекции: ")
emotion, conf = predict_emotion(test_text)
print(f"Результат: {emotion.upper()} ({conf:.1f}%)")

print("\n🎉 ПРОЕКТ ВЫПОЛНЕН УСПЕШНО!")
print("✓ Полносвязная НС TensorFlow")
print("✓ Русскоязычный датасет эмоциональных текстов")
print("✓ Полная параметризация архитектуры")
print("✓ Анализ студенческих отзывов")
print("✓ HTML отчет с визуализацией")
print("✓ Готово для Google Colab!")
