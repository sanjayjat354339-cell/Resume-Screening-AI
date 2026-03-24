import pandas as pd
import re
import nltk


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ── Cleaning Function 
def clean_resume(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords + lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(tokens)

# ── Load Dataset 
print("Loading dataset...")
df = pd.read_csv("D:/vs code/csvfiles/UpdatedResumeDataSet.csv")
print(f"Original shape: {df.shape}")

# ── Apply Cleaning 
print("\nCleaning resumes... (takes 1-2 mins)")
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# ── Check Results 
print("\n✅ Cleaning Done!")
print(f"\nOriginal resume sample:\n{df['Resume'][0][:200]}")
print(f"\nCleaned resume sample:\n{df['cleaned_resume'][0][:200]}")

# ── Save Cleaned Data 
df.to_csv("cleaned_resume_data.csv", index=False)
print(f"\n✅ Saved as cleaned_resume_data.csv")
print(f"Shape: {df.shape}")
print("\n✅ Step 2 Complete! Ready for Step 3.")