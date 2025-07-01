from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from django.conf import settings

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        # Сохраняем загруженное изображение
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        image_path = fs.path(filename)

        # Загрузка модели
        model_path = os.path.join(settings.BASE_DIR, '..', 'emotion_model.h5')
        model = load_model(model_path)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Обработка изображения
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotions = []
        # Рисуем прямоугольники и подписи
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = face_roi.reshape(1, 48, 48, 1)

            prediction = model.predict(face_roi)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            confidence = prediction[0][emotion_index] * 100
            emotions.append({'emotion': emotion, 'confidence': confidence})

            # Рисуем прямоугольник
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Добавляем текст с эмоцией
            cv2.putText(image, f'{emotion} ({confidence:.2f}%)',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Сохраняем оригинальное и обработанное изображения
        original_image_url = fs.url(filename)
        processed_filename = 'processed_' + filename
        processed_image_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
        cv2.imwrite(processed_image_path, image)
        processed_image_url = fs.url(processed_filename)

        return render(request, 'result.html', {
            'original_image': original_image_url,
            'processed_image': processed_image_url,
            'emotions': emotions
        })

    return render(request, 'home.html')