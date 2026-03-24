import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Step 1: Load Cleaned Data 
print("Loading cleaned data...")
df = pd.read_csv("cleaned_resume_data.csv")
print(f"Shape: {df.shape}")

# ── Step 2: Encode Categories to Numbers
# ML doesn't understand "Data Science", "HR" etc
# We convert them to numbers
# "Data Science" → 0
# "HR"           → 1
# "Java"         → 2  etc.

le = LabelEncoder()
df['Category_Num'] = le.fit_transform(df['Category'])
print(f"\nCategories encoded: {list(le.classes_)}")

# ── Step 3: TF-IDF Vectorization ──────────────────────────
# Convert words → numbers


print("\nConverting text to numbers using TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1,2),    # looks at 2 words together
    sublinear_tf=True,    # reduces very common words
    min_df=2              # ignore very rare words
)
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category_Num']

print(f"Features created: {X.shape}")

# ── Step 4: Split Data 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── Step 5: Train Random Forest Model 
print("\nTraining Random Forest model...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,          # limits tree depth
    min_samples_split=5,   # needs 5 samples to split
    max_features='sqrt',   # uses random features
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── Step 6: Test the Model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=le.classes_))

# ── Step 7: Save Model 
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le,    'encoder.pkl')

print("\n✅ Model saved as model.pkl")
print("✅ TF-IDF saved as tfidf.pkl")
print("✅ Encoder saved as encoder.pkl")
print("\n✅ Step 3 Complete! Ready for Step 4.")