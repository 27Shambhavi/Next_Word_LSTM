🧠 Next Word Prediction with LSTM

This is a Streamlit-based web application that predicts the next word in a sentence using a trained LSTM (Long Short-Term Memory) model. The app is designed to demonstrate basic Natural Language Processing (NLP) with deep learning, showing how sequence modeling can be applied to generate language predictions.

🔍 Overview

- Uses a pre-trained LSTM model (`next_word_lstm.h5`) for word prediction.
- Tokenizer (`tokenizer.pickle`) is used to convert text to sequences.
- Streamlit is used to create a simple, interactive web interface.
- Input: A partial sentence (e.g., “To be or not to be”)
- Output: The predicted next word based on learned sequences.

📁 Files

- `app.py`: Main Streamlit app script.
- `next_word_lstm.h5`: Trained Keras LSTM model file.
- `tokenizer.pickle`: Fitted tokenizer used during model training.

▶️ Run the App
streamlit run app.py
