import pandas as pd

df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
df.to_csv("spam.csv", index=False)
print("CSV file saved as spam.csv")

#Data Preprocessing + Visualization (train_model.py)
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # ham=0, spam=1

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Vectorization
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc}")

# Save best model (Naive Bayes)
joblib.dump(models["Naive Bayes"], "spam_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

# Visualization
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()









