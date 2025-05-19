
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

print('Загрузка данных')
data = pd.read_csv('train.csv')
print('Данные успешно загружены:')
print(data.head())


print('Предобработка данных')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

data['text'] = data['title'] + ' ' + data['author'].fillna('') + ' ' + data['text'].fillna('')
data['text'] = data['text'].apply(clean_text)


data = data.dropna()

print('Векторизация текста')
vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=5, max_df=0.85)
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

print('Преобразование меток')
le = LabelEncoder()
y = le.fit_transform(y)

print("Количество фейковых новостей:", sum(y == 0))
print("Количество настоящих новостей:", sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Обучение модели (LogisticRegression)...')
model = LogisticRegression(max_iter=1000, solver='lbfgs')

print('Кросс-валидация модели')
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print('Результаты кросс-валидации:', scores)
print('Средняя точность на кросс-валидации:', scores.mean())

model.fit(X_train, y_train)

print('Оценка модели')
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


print('Построение WordCloud')
fake_data = data[data['label'] == 0]
real_data = data[data['label'] == 1]

wordcloud_fake = WordCloud(width=800, height=400, max_words=30).generate(' '.join(fake_data['text']))
wordcloud_real = WordCloud(width=800, height=400, max_words=30).generate(' '.join(real_data['text']))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title('WordCloud - Fake News')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title('WordCloud - Real News')
plt.axis('off')

plt.show()

print('Сохранение модели')
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Модель, векторизатор и энкодер успешно сохранены!')
