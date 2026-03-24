from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# ── Load Saved Model ───────────────────────────────────────
print("Loading model...")
model   = joblib.load('model.pkl')
tfidf   = joblib.load('tfidf.pkl')
encoder = joblib.load('encoder.pkl')
print("✅ Model loaded!")

# ── Same Cleaning Function from preprocess.py ─────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
        and len(word) > 2
    ]
    return ' '.join(tokens)

# ── Home Page ──────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── Predict Page ───────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        # Get resume text from webpage
        resume_text = request.form['resume']
        
        # ── Check if empty ─────────────────────────
        if resume_text.strip() == '':
            return render_template(
                'index.html',
                error="⚠️ Please paste your resume text first!"
            )
        
        # ── Check minimum length ───────────────────
        if len(resume_text.split()) < 10:
            return render_template(
                'index.html',
                error="⚠️ Resume is too short! Please paste complete resume."
            )

        # Clean it
        cleaned = clean_resume(resume_text)

        # Convert to numbers
        vectorized = tfidf.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)
        category   = encoder.inverse_transform(prediction)[0]

        # Get confidence score
        proba      = model.predict_proba(vectorized)
        confidence = round(max(proba[0]) * 100, 2)

        # ── Check low confidence ───────────────────
        if confidence < 40:
            return render_template(
                'index.html',
                error="⚠️ Could not predict! Please paste a proper resume."
            )

        return render_template(
            'index.html',
            prediction=category,
            confidence=confidence
        )

# ── Run App ────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)