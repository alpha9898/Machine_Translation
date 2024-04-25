import tkinter as tk
from tkinter import ttk
from transformers import MarianMTModel, MarianTokenizer


def translate(text, model_name, src_lang='en', tgt_lang='ar'):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    if src_lang == 'ar' and tgt_lang == 'en':
        translated_text = translate_ar_to_en(text, tokenizer, model)
    elif src_lang == 'en' and tgt_lang == 'ar':
        translated_text = translate_en_to_ar(text, tokenizer, model)
    else:
        translated_text = "Translation direction not supported."

    return translated_text


def translate_ar_to_en(text, tokenizer, model):
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokenized_text)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def translate_en_to_ar(text, tokenizer, model):
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, src_lang="en")
    translated = model.generate(**tokenized_text)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def translate_text():
    src_lang = src_lang_var.get()
    tgt_lang = tgt_lang_var.get()
    text_to_translate = text_entry.get()

    if src_lang == 'ar' and tgt_lang == 'en':
        translation = translate(text_to_translate, model_name_ar_to_en, src_lang, tgt_lang)
    elif src_lang == 'en' and tgt_lang == 'ar':
        translation = translate(text_to_translate, model_name_en_to_ar, src_lang, tgt_lang)
    else:
        translation = "Translation direction not supported."

    translated_label.config(text=f'Translated Text: {translation}')


# Model names
model_name_ar_to_en = 'Helsinki-NLP/opus-mt-ar-en'
model_name_en_to_ar = 'Helsinki-NLP/opus-mt-en-ar'

# Create Tkinter window
window = tk.Tk()
window.title("Text Translator")

# Source language selection
src_lang_label = ttk.Label(window, text="Source Language:")
src_lang_label.grid(row=0, column=0, padx=5, pady=5)
src_lang_var = tk.StringVar(value="en")
src_lang_combobox = ttk.Combobox(window, values=["en", "ar"], textvariable=src_lang_var, state="readonly")
src_lang_combobox.grid(row=0, column=1, padx=5, pady=5)

# Target language selection
tgt_lang_label = ttk.Label(window, text="Target Language:")
tgt_lang_label.grid(row=1, column=0, padx=5, pady=5)
tgt_lang_var = tk.StringVar(value="ar")
tgt_lang_combobox = ttk.Combobox(window, values=["en", "ar"], textvariable=tgt_lang_var, state="readonly")
tgt_lang_combobox.grid(row=1, column=1, padx=5, pady=5)

# Text entry
text_label = ttk.Label(window, text="Enter Text:")
text_label.grid(row=2, column=0, padx=5, pady=5)
text_entry = ttk.Entry(window, width=40)
text_entry.grid(row=2, column=1, padx=5, pady=5)

# Translate button
translate_button = ttk.Button(window, text="Translate", command=translate_text)
translate_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Translated text label
translated_label = ttk.Label(window, text="")
translated_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

window.mainloop()
