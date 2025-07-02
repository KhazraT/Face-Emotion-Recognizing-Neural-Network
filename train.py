import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from warnings import filterwarnings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

filterwarnings('ignore')

# Загрузка датасета
data = pd.read_csv('fer2013.csv')

# Парсинг пикселей
pixels = data['pixels'].tolist()
images = []
for pixel_sequence in pixels:
    image = [int(pixel) for pixel in pixel_sequence.split(' ')]
    image = np.asarray(image).reshape(48, 48, 1)
    images.append(image)
images = np.array(images)

# Нормализация изображений
images = images / 255.0

# Получение меток
labels = data['emotion'].values
labels = to_categorical(labels, num_classes=7)

# Разделение на тренировочный и валидационный наборы
train_images = images[data['Usage'] == 'Training']
train_labels = labels[data['Usage'] == 'Training']
val_images = images[data['Usage'] == 'PublicTest']
val_labels = labels[data['Usage'] == 'PublicTest']

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_emotion_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# Улучшенная аугментация
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator()

# Создание модели
model = Sequential([
    Conv2D(128, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(512, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(1024, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
batch_size = 64
epochs = 150
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(val_images) // batch_size
model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Сохранение модели
model.save('emotion_model.h5')

# Визуализация истории обучения
history = model.history.history if hasattr(model, 'history') else None
if history is None:
    print('История обучения не найдена. Проверьте, как вызывается model.fit.')
else:
    # График accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # График loss
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('history.png')

# Оценка на валидационном наборе
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f'Validation accuracy: {val_accuracy*100:.2f}%')