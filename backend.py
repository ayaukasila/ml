# backend.py - FastAPI для Fake News Detection

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import pickle
import re
from googlesearch import search
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib

# === Отключаем отображение графиков в GUI ===
matplotlib.use('Agg')

# === Загрузка моделей ===
try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    print("Модели успешно загружены!")
except FileNotFoundError as e:
    print(f"Ошибка загрузки моделей: {e}")
    model, tfidf, label_encoder = None, None, None


# === Инициализация FastAPI ===
app = FastAPI()

# === Вспомогательная функция для BarChart ===
positive_count = 0
negative_count = 0

def generate_chart():
    labels = ['Real', 'Fake']
    values = [positive_count, negative_count]
    plt.figure(figsize=(5, 3))
    plt.bar(labels, values, color=['green', 'red'])
    plt.title('News Classification Statistics')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return data


# === Маршрут для предсказания ===
@app.post('/predict', response_class=JSONResponse)
def predict(news_text: str = Form(...)):
    global positive_count, negative_count
    if tfidf is None or model is None or label_encoder is None:
        return JSONResponse(status_code=500, content={"error": "Модели не загружены"})

    vector = tfidf.transform([news_text])
    prediction = model.predict(vector)
    label = label_encoder.inverse_transform(prediction)[0]
    
    # Обновляем статистику
    if label == 'real':
        positive_count += 1
    else:
        negative_count += 1

    # Поиск правды в Google если новость фейковая
    search_results = []
    if label == 'fake':
        for url in search(news_text, num_results=3):
            search_results.append(url)

    return JSONResponse(content={
        "label": label,
        "search_results": search_results,
        "positive_count": positive_count,
        "negative_count": negative_count
    })

# === Маршрут для получения графика ===
@app.get('/chart')
def get_chart():
    chart_data = generate_chart()
    return JSONResponse(content={"chart": chart_data})
