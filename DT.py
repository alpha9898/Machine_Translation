from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Step 4: Training the Decision Tree Model
decision_tree_clf = Pipeline([
    ('clf', DecisionTreeClassifier())
])
decision_tree_clf.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = decision_tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example of prediction
new_sentences = ["Hello!", "Run faster!"]
X_new_tfidf = tfidf_vectorizer.transform(new_sentences)
predictions = decision_tree_clf.predict(X_new_tfidf)
print("Predictions:")
for sentence, prediction in zip(new_sentences, predictions):
    print(f"{sentence} -> {prediction}")
