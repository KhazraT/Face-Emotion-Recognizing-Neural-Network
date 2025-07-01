import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Аугментация данных
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator()

# Создание модели
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
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
epochs = 50
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(val_images) // batch_size
model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps)

# Сохранение модели
model.save('emotion_model.h5')

# Оценка на валидационном наборе
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print(f'Validation accuracy: {val_accuracy*100:.2f}%')