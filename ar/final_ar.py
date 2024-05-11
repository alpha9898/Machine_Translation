import tkinter as tk
from transformers import MarianMTModel, MarianTokenizer


def translate_arabic_to_english():
    arabic_text = arabic_entry.get()

    # Load pre-trained model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.encode(arabic_text, return_tensors="pt")

    # Translate input text to English
    translation = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)

    # Decode the translated output
    english_translation = tokenizer.decode(translation[0], skip_special_tokens=True)

    # Display the translation
    english_label.config(text=english_translation)


# Create the main application window
root = tk.Tk()
root.title("Arabic to English Translation")

# Create text entry field for Arabic input
arabic_entry = tk.Entry(root, width=50)
arabic_entry.pack(pady=10)

# Create button to trigger translation
translate_button = tk.Button(root, text="Translate", command=translate_arabic_to_english)
translate_button.pack()

# Create label to display English translation
english_label = tk.Label(root, text="")
english_label.pack(pady=10)

# Run the application
root.mainloop()
