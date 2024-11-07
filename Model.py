Enable desktop notifications for Gmail.
   OK  No, thanks
2 of 2,653
vels
Inbox

Team Vss
Attachments
14:04 (8 hours ago)
to me


One attachment
  • Scanned by Gmail
import re
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, LSTM, Dense, Bidirectional, Layer
from tensorflow.keras import backend as K
from googletrans import Translator, LANGUAGES

# Download NLTK stopwords
nltk.download('stopwords')

# Text Preprocessing Class with Translation
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.translator = Translator()

    def translate_to_english(self, text):
        if not text:
            return ""
        try:
            detected_lang = self.translator.detect(text).lang
            if detected_lang != 'en':
                translated_text = self.translator.translate(text, src=detected_lang, dest='en').text
                return translated_text if translated_text else ""
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return the original text if translation fails
        return text

    def preprocess_text(self, text):
        text = self.translate_to_english(text)
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

# Custom Attention Layer
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.exp(u)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        output = inputs * a
        return K.sum(output, axis=1)

# Crime Classification Model with Multi-Output
class CrimeClassifierMultiOutputModel:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.model = None
        self.categories = None
        self.subcategories = None
        self.category_map = None
        self.subcategory_map = None

    def prepare_data(self, data, text_column, main_category_column, sub_category_column):
        # Check for necessary columns in the dataset
        required_columns = [text_column, main_category_column, sub_category_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

        # Drop any rows with missing values in required columns
        data = data.dropna(subset=[text_column, main_category_column, sub_category_column])

        # Map categories and subcategories to indices
        self.categories = data[main_category_column].unique()
        self.subcategories = data[sub_category_column].unique()
        self.category_map = {category: idx for idx, category in enumerate(self.categories)}
        self.subcategory_map = {subcategory: idx for idx, subcategory in enumerate(self.subcategories)}

        # Create target columns for model training
        data = data.copy()
        data['main_category_target'] = data[main_category_column].map(self.category_map)
        data['sub_category_target'] = data[sub_category_column].map(self.subcategory_map)

        # Preprocess text data
        preprocessor = TextPreprocessor()
        data[text_column] = data[text_column].apply(preprocessor.preprocess_text)

        # Tokenize and pad text data
        self.tokenizer.fit_on_texts(data[text_column])
        sequences = pad_sequences(self.tokenizer.texts_to_sequences(data[text_column]), maxlen=self.max_len, padding='post')

        # Split into training and testing sets
        return train_test_split(sequences, data[['main_category_target', 'sub_category_target']], test_size=0.2, random_state=42)

    def build_model(self):
        # Define model structure
        input_layer = Input(shape=(self.max_len,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len)(input_layer)
        gru_layer = Bidirectional(GRU(64, return_sequences=True))(embedding_layer)
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(gru_layer)
        attention_layer = AttentionLayer()(lstm_layer)

        # Output layers for main and subcategory classification
        main_category_output = Dense(len(self.categories), activation='softmax', name='main_category')(attention_layer)
        sub_category_output = Dense(len(self.subcategories), activation='softmax', name='sub_category')(attention_layer)

        # Compile the model
        model = Model(inputs=input_layer, outputs=[main_category_output, sub_category_output])
        model.compile(optimizer='adam', 
                      loss={'main_category': 'sparse_categorical_crossentropy', 'sub_category': 'sparse_categorical_crossentropy'}, 
                      metrics={'main_category': 'accuracy', 'sub_category': 'accuracy'})
        
        self.model = model
        return model

    def train_model(self, X_train_seq, y_train, validation_split=0.1, epochs=10, batch_size=64):
        # Train the model with provided data
        history = self.model.fit(X_train_seq, 
                                 {'main_category': y_train['main_category_target'], 'sub_category': y_train['sub_category_target']},
                                 validation_split=validation_split, 
                                 epochs=epochs, 
                                 batch_size=batch_size)

        # Make predictions on the test set
        X_test_seq = pad_sequences(self.tokenizer.texts_to_sequences(X_test), maxlen=self.max_len, padding='post')
        y_pred = self.model.predict(X_test_seq)
        
        # Convert predictions from probabilities to class labels
        y_pred_main = np.argmax(y_pred[0], axis=1)
        y_pred_sub = np.argmax(y_pred[1], axis=1)

        # Calculate metrics for main category
        accuracy_main = accuracy_score(y_test['main_category_target'], y_pred_main)
        precision_main = precision_score(y_test['main_category_target'], y_pred_main, average='weighted')
        recall_main = recall_score(y_test['main_category_target'], y_pred_main, average='weighted')
        f1_main = f1_score(y_test['main_category_target'], y_pred_main, average='weighted')

        # Calculate metrics for sub category
        accuracy_sub = accuracy_score(y_test['sub_category_target'], y_pred_sub)
        precision_sub = precision_score(y_test['sub_category_target'], y_pred_sub, average='weighted')
        recall_sub = recall_score(y_test['sub_category_target'], y_pred_sub, average='weighted')
        f1_sub = f1_score(y_test['sub_category_target'], y_pred_sub, average='weighted')

        # Print the classification report for detailed metrics
        print("Main Category Classification Report:")
        print(classification_report(y_test['main_category_target'], y_pred_main, target_names=self.categories))
        
        print("Sub Category Classification Report:")
        print(classification_report(y_test['sub_category_target'], y_pred_sub, target_names=self.subcategories))

        # Print overall accuracy, precision, recall, and F1 scores
        print("Overall Metrics for Main Category:")
        print(f"Accuracy: {accuracy_main:.4f}, Precision: {precision_main:.4f}, Recall: {recall_main:.4f}, F1 Score: {f1_main:.4f}")

        print("Overall Metrics for Sub Category:")
        print(f"Accuracy: {accuracy_sub:.4f}, Precision: {precision_sub:.4f}, Recall: {recall_sub:.4f}, F1 Score: {f1_sub:.4f}")

        return history

    def save_model(self, model_path, tokenizer_path, category_map_path, subcategory_map_path):
        # Save model and supporting files
        self.model.save(model_path)
        joblib.dump(self.tokenizer, tokenizer_path)
        joblib.dump(self.category_map, category_map_path)
        joblib.dump(self.subcategory_map, subcategory_map_path)

# Load your dataset and specify the file path
# Replace 'your_dataset.csv' with the path to your actual file
data = pd.read_csv("C:/Users/511522104045/Downloads/train_sorted.csv")

# Initialize and prepare the model
crime_model = CrimeClassifierMultiOutputModel()
X_train, X_test, y_train, y_test = crime_model.prepare_data(data, 'crimeaditionalinfo', 'category', 'sub_category')

# Build and train the model
crime_model.build_model()
history = crime_model.train_model(X_train, y_train)

# Save the trained model and tokenizer
crime_model.save_model("crime_classifier_multi_output_model.h5", "tokenizer.pkl", "category_map.pkl", "subcategory_map.pkl")
