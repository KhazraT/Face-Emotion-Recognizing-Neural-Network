import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse

# Загрузка модели
model = load_model('emotion_model.h5')

# Метки эмоций
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Путь к входному изображению')
args = parser.parse_args()

# Загрузка изображения
image_path = args.image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

for (x, y, w, h) in faces:
    # Извлечение области лица
    face_roi = gray[y:y+h, x:x+w]
    # Изменение размера до 48x48
    face_roi = cv2.resize(face_roi, (48,48))
    # Нормализация
    face_roi = face_roi / 255.0
    # Изменение формы для ввода в модель
    face_roi = face_roi.reshape(1, 48, 48, 1)
    # Предсказание эмоции
    prediction = model.predict(face_roi)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = prediction[0][emotion_index] * 100
    # Отрисовка прямоугольника и метки
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f'{emotion} ({confidence:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# Отображение изображения
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()