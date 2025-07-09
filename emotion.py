import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("emotion")

# Extracting features and labels
texts = dataset['train']['text']
labels = dataset['train']['label']

# Tokenize the text
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_sequence_length = 100  # Adjust according to your data and requirements
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to one-hot encoding
label_dict = {label: i for i, label in enumerate(set(labels))}
num_classes = len(label_dict)
labels = [label_dict[label] for label in labels]
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16),  # Removed input_length
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Define a function for making predictions
def predict_emotion(input_text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    
    # Decode the prediction
    emotion_labels = ['joy', 'anger', 'love', 'sadness', 'fear', 'surprise']
    predicted_label = emotion_labels[np.argmax(prediction)]
    prediction_value = np.max(prediction)
    
    return predicted_label, prediction_value

# Streamlit app
st.title("Emotion Prediction")

input_text = st.text_input("Enter a text:", "")

if input_text:
    predicted_emotion, confidence = predict_emotion(input_text)
    st.write("Predicted Emotion:", predicted_emotion)
    st.write("Confidence:", confidence)