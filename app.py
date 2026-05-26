import streamlit as st
import pickle
import time
from transformers import pipeline
# import train_model


tf_idf_model = pickle.load(open("Models/tf_idf_model.pkl", "rb"))
vectorizer = pickle.load(open("Models/vectorizer.pkl", "rb"))
distilBert_model = pipeline("sentiment-analysis", model="omidroshani/imdb-sentiment-analysis")

if "response" not in st.session_state:
    st.session_state.response = ""

if "input" not in st.session_state:
    st.session_state.input = ""


def run_tf_idf(input):
    lst = []
    lst.append(input)
    text = vectorizer.transform(lst)
    response = tf_idf_model.predict(text)
    return response[0]


def run_distilBert(input):
    response = distilBert_model(input)[0]["label"]
    if response == "LABEL_0":
        response = "negative"
    else:
        response = "positive"
    return response

def is_submitted():
    if st.session_state.input == "":
        st.error("Please input a text")
        st.session_state.response = ""
        return 
    if st.session_state.selected_model == "TF-IDF + Logistic Regression":
        with st.spinner("", show_time=True):
            response = run_tf_idf(st.session_state.input)
    else:
        with st.spinner("", show_time=True):
            response = run_distilBert(st.session_state.input)
        
    st.session_state.response = response    

st.title(':blue[🎬 AI Movie Review Analyzer]', help=None, width="stretch", text_alignment="center")



with st.form("form", enter_to_submit=False, border=True, ):
    st.write("Paste any text to check if it is a positive/negative review")
    left, right = st.columns([1,1] , vertical_alignment="center")    
    left2, mid2, right2 = st.columns([2,1,2], gap="large",vertical_alignment="bottom")

    with left:
        input = st.text_area("Paste Movie Review", height="content", key="input")
        
    with right:
        selected_model = st.selectbox("Choose Model",
                    ("TF-IDF + Logistic Regression", "DistilBERT"), key="selected_model")
        
    with mid2:
        submit = st.form_submit_button("Submit", on_click=is_submitted)
    
if submit and st.session_state.response != "":
    if st.session_state.response == "positive":
        st.success("Positive Review 😄")
    else:
        st.error("Negative Review 🙁") 
