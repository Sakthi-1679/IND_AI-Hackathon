import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from keras import layers
from keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras import backend as K  # Import Keras backend

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Text Preprocessing Class
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

# Custom Attention Layer
class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.exp(u)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        output = inputs * a
        return K.sum(output, axis=1)

# Main Application Class
class CrimePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Crime Prediction App")
        
        self.load_model()

        # Create UI elements
        self.text_input = tk.Text(master, height=5, width=50)
        self.text_input.pack(pady=10)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_text)
        self.predict_button.pack(pady=5)

        self.upload_button = tk.Button(master, text="Upload CSV", command=self.upload_csv)
        self.upload_button.pack(pady=5)

        self.output_label = tk.Label(master, text="")
        self.output_label.pack(pady=10)

    def load_model(self):
        # Load the trained model and tokenizer
        with custom_object_scope({'AttentionLayer': AttentionLayer}):  # Include your custom layer
            self.model = load_model("crime_classifier_multi_output_model.h5")
        
        self.tokenizer = joblib.load("tokenizer.pkl")
        self.category_map = joblib.load("category_map.pkl")
        self.subcategory_map = joblib.load("subcategory_map.pkl")
        self.reverse_category_map = {v: k for k, v in self.category_map.items()}
        self.reverse_subcategory_map = {v: k for k, v in self.subcategory_map.items()}

    def preprocess_text(self, text):
        preprocessor = TextPreprocessor()
        return preprocessor.preprocess_text(text)

    def predict_text(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        if input_text:
            preprocessed_text = self.preprocess_text(input_text)
            sequence = pad_sequences(self.tokenizer.texts_to_sequences([preprocessed_text]), maxlen=200)
            predictions = self.model.predict(sequence)

            main_category_idx = np.argmax(predictions[0])
            sub_category_idx = np.argmax(predictions[1])

            main_category = self.reverse_category_map[main_category_idx]
            sub_category = self.reverse_subcategory_map[sub_category_idx]

            self.output_label.config(text=f"Main Category: {main_category}, Sub Category: {sub_category}")

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            if 'crimeaditionalinfo' in data.columns:
                results = []
                for text in data['crimeaditionalinfo']:
                    preprocessed_text = self.preprocess_text(text)
                    sequence = pad_sequences(self.tokenizer.texts_to_sequences([preprocessed_text]), maxlen=200)
                    predictions = self.model.predict(sequence)

                    main_category_idx = np.argmax(predictions[0])
                    sub_category_idx = np.argmax(predictions[1])

                    main_category = self.reverse_category_map[main_category_idx]
                    sub_category = self.reverse_subcategory_map[sub_category_idx]

                    results.append({'crimeaditionalinfo': text, 'main_category': main_category, 'sub_category': sub_category})

                output_df = pd.DataFrame(results)
                output_df.to_csv('predictions_output.csv', index=False)
                self.output_label.config(text="Predictions saved to 'predictions_output.csv'")
            else:
                self.output_label.config(text="CSV file must contain 'crimeaditionalinfo' column.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CrimePredictionApp(root)
    root.mainloop()
