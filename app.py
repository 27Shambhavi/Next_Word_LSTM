<<<<<<< HEAD
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure the token list has the correct length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title("Next Word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words:", "To be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = 20  # Must match the length used during training
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.success(f"The next word is: **{next_word}**")
    else:
        st.error("Could not predict the next word.")
=======
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure the token list has the correct length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title("Next Word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words:", "To be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = 20  # Must match the length used during training
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.success(f"The next word is: **{next_word}**")
    else:
        st.error("Could not predict the next word.")
>>>>>>> 5d56099ebe44508ff3ddb0302c489040d06a6135
