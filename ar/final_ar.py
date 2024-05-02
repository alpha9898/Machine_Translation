import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# English to Arabic dictionary
translation_dict_en_to_ar = {
    "I": "انا",
    "you": "انت",
    "Ali": "علي",
    "loves": "يحب",
    "football": "كرة القدم",
    "Football": "كرة القدم",
    "is": "تكون",
    "fun": "ممتعة",
    "plays": "يلعب",
    "every": "كل",
    "day": "يوم",
    # Add more translations as needed
}

# Arabic to English dictionary
translation_dict_ar_to_en = {
    "انا": "I",
    "انت": "you",
    "علي": "Ali",
    "يحب": "loves",
    "الكرة": "football",
    "هي": "is",
    "ممتعة": "fun",
    "يلعب": "plays",
    "كل": "every",
    "يوم": "day",
    # Add more translations as needed
}


def translate_text():
    input_text = input_entry.get().strip()
    translated_text = ""

    # Detect input language
    is_arabic = any(char in 'ابتح' for char in input_text)

    if is_arabic:
        translation_dict = translation_dict_ar_to_en
    else:
        translation_dict = translation_dict_en_to_ar

    for word in input_text.split():
        translated_word = translation_dict.get(word, word)
        translated_text += translated_word + " "

    translated_text_var.set(translated_text.strip())


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

# Step 4b: Training the Decision Tree Model
dt_clf = Pipeline([
    ('clf', DecisionTreeClassifier())
])
dt_clf.fit(X_train, y_train)


# Step 5: Evaluation
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


svm_accuracy = evaluate_model(svm_clf, X_test, y_test)
dt_accuracy = evaluate_model(dt_clf, X_test, y_test)

# Create GUI
root = tk.Tk()
root.title("Machine Translation System")
root.geometry("400x250")

# Input field
input_label = tk.Label(root, text="Enter Text:")
input_label.grid(row=0, column=0, padx=5, pady=5)
input_entry = tk.Entry(root, width=30)
input_entry.grid(row=0, column=1, padx=5, pady=5)

# Translate button
translate_button = tk.Button(root, text="Translate", command=translate_text)
translate_button.grid(row=0, column=2, padx=5, pady=5)

# Output field
translated_text_var = tk.StringVar()
output_label = tk.Label(root, textvariable=translated_text_var, wraplength=380)
output_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

# SVM Accuracy
svm_accuracy_label = tk.Label(root, text=f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
svm_accuracy_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Decision Tree Accuracy
dt_accuracy_label = tk.Label(root, text=f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
dt_accuracy_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

root.mainloop()
