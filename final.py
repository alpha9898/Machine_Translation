import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Translation App")

        # Model selection
        self.model_label = ttk.Label(root, text="Choose Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(root, textvariable=self.model_var, values=["SVM", "Decision Tree"])
        self.model_combobox.grid(row=0, column=1, padx=10, pady=5)

        # Input text
        self.text_label = ttk.Label(root, text="Input Text:")
        self.text_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.text_entry = ttk.Entry(root, width=50)
        self.text_entry.grid(row=1, column=1, padx=10, pady=5)

        # Accuracy label
        self.accuracy_label = ttk.Label(root, text="")
        self.accuracy_label.grid(row=3, column=0, columnspan=2, pady=5)

        # Translate button
        self.translate_button = ttk.Button(root, text="Translate", command=self.translate_text)
        self.translate_button.grid(row=2, column=0, columnspan=2, pady=10)

    def train_model(self, X, y, model_type):
        if model_type == "SVM":
            clf = Pipeline([
                ('clf', SVC(kernel='linear'))
            ])
        elif model_type == "Decision Tree":
            clf = Pipeline([
                ('clf', DecisionTreeClassifier())
            ])
        else:
            return None

        clf.fit(X, y)
        return clf

    def translate_text(self):
        input_text = self.text_entry.get()
        chosen_model = self.model_var.get()

        if not chosen_model:
            messagebox.showerror("Error", "Please choose a model.")
            return

        if not input_text:
            messagebox.showerror("Error", "Please enter some text.")
            return

        # Load and preprocess data
        with open("data_eng.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()

        X = []
        y = []
        for line in lines:
            source, target = line.strip().split('\t')
            X.append(source)
            y.append(target)

        # Feature extraction
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Train the selected model
        clf = self.train_model(X_train, y_train, chosen_model)
        if clf is None:
            messagebox.showerror("Error", "Invalid model selection.")
            return

        # Evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy_label.config(text=f"Accuracy: {100*accuracy:.2f}")

        # Translate the input text
        X_input_tfidf = tfidf_vectorizer.transform([input_text])
        translation = clf.predict(X_input_tfidf)[0]

        messagebox.showinfo("Translation", f"Translated text: {translation}")

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
# all code