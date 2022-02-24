import streamlit as st
import pandas as pd
import tensorflow as tf
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


st.title("Twitter data sentiment analysis")

def get_prediction(text):
  tokenizer = Tokenizer()
  model = load_model()
  encoded = tokenizer.texts_to_sequences([text])[0]
  encoded = pad_sequences([encoded],maxlen=25,padding='post')
  prediction=model.predict(encoded)
  return prediction

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Model')
    return model

tweet = str(st.text_input('Enter your tweet'))

if tweet:
  model = load_model()
  text = re.sub('\W+',' ', tweet)
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  encoded = tokenizer.texts_to_sequences([text])[0]
  encoded = pad_sequences([encoded],maxlen=25,padding='post')
  prediction=model.predict(encoded)
  if prediction>0.5:
    st.header("Positive sentiment")
  else:
    st.header("Negative sentiment")
