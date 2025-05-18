from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import urllib.parse
import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загрузка модели и TF-IDF векторизатора
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

df = pd.read_csv("train.csv")
total_articles = len(df)
# Берём 3 примера для списка
examples = df['text'].dropna().sample(n=3, random_state=42).tolist()

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "examples": examples, "total": total_articles}
    )

@app.post("/predict")
async def predict(text: str = Form(...)):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    real_pct = round(probs[0] * 100)
    fake_pct = 100 - real_pct
    label = "REAL" if real_pct >= fake_pct else "FAKE"

    response = {"label": label, "probs": {"REAL": real_pct, "FAKE": fake_pct}}
    if label == "FAKE":
        query = urllib.parse.quote_plus(text)
        response["links"] = [
            f"https://www.google.com/search?q={query}",
            f"https://www.google.com/search?q=fact+check+{query}",
            f"https://www.google.com/search?q=news+{query}"
        ]
    return JSONResponse(response)