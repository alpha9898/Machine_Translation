from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Read and preprocess data
with open("data.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

X = []
y = []
for line in lines:
    source, target = line.strip().split('\t')
    X.append(source)
    y.append(target)

# Step 2: Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Training the SVM Model
svm_clf = Pipeline([
    ('clf', SVC(kernel='linear'))
])
svm_clf.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", 100*accuracy)

# Example of prediction
new_sentences = ["Hello!", "Run faster!","Goodbye!"]
X_new_tfidf = tfidf_vectorizer.transform(new_sentences)
predictions = svm_clf.predict(X_new_tfidf)
print("Predictions:")
for sentence, prediction in zip(new_sentences, predictions):
    print(f"{sentence} -> {prediction}")
