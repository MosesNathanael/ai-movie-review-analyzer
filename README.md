# AI Movie Review Analyzer 

An AI-powered application that analyze movie reviews using Machine Learning (ML) and Natural Language Processing (NLP).
The app predicts whether a movie review is a positive or negative review based on the user input. Users can choose between
2 different models for prediction:
 - TF-IDF + Logistic Regression = A traditional ML model that is built by text vectorization and classification method.
 - DistilBERT (transformers model) = A transformer-based deep learning model for more advanced language understanding.


## Features
- **Analyze movie reviews instantly**
- **Predict positive or negative sentiment with prediction score**
- **Choose between traditional ML and transformer models**
- **NLP-based text preprocessing**
- **Interactive UI**


## Tech Stack 
- **Python**
- **Numpy**
- **Pandas**
- **Hugging Face**
- **Scikit-learn**
- **NLP**
- **Streamlit**
- **Transformers (Hugging Face)**


## Models Used 
1. **TF-IDF + Logistic Regression**
  A traditional Machine learning model:
    - Convert text into vectors using TF-IDF vectorizer
    - Use logistic regression for sentiment classfication
    - Fast but less accuracy (not bad is recognize bad)

2. **DistilBERT**
   A transformer NLP model:
     - Use pretrained model from hugginface
     - Higher accuracy
     - Understand contextual meaning in text
  


## Machine Learning Workflow 
1. **Data Collection**
  Labeled movie review datasets from imdb were collected 
2. **Text Preprocessing**
   The raw datasets were cleaned before traning:
    - Lowercasing
    - Removing html tags and urls
    - Removing punctuation and special characters
    - Contractions
    - Tokenization
    - Stop_words
    - Lemmatization
3. **Feature Engineering**
   For **logistic regression model**, the cleaned text was converted into vectors using TF-IDF vectorizer.
   For **distilBERT model**, DistilBERT tokenizer and embeddings were used to capture contextual meaning from text.
4. **Model Training**
   **TF-IDF + Logistic Regression Model**:
     The vectors were trained using Logistic Regression for sentiment classification
   **DistilBERT**:
     The model was already trained, we just need to use pipeline() function to pass our input and the model.
5. **Evaluation**
   The models were evaluated using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
6. **Deployment**
   The trained models were integrated into real web-based application.
   

## How it works
1. User paste a movie review
2. User select the model
3. The text is preprocessed using NLP techniques (tokenization, cleaning, stop_word)
4. The model analyze the preprocessed text
5. The app displyas the positive or negative output and also the prediction score.


## Installation
1. git clone https://github.com/MosesNathanael/ai-movie-review-analyzer.git
2. cd ai-movie-review-analyzer
3. pip install -r requirements.txt


## Run the app
streamlit run app.py


## How to Deploy (Streamlit Cloud)
1. Push this repo to github
2. Connect your github account to your streamlit account (https://share.streamlit.io)
4. Select this repo → Deploy
5. Your app is live


## Author
Built by Moses Nathanael as part of an AI portfolio demonstrating NLP techniques.


