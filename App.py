import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import load_model

import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model = load_model('Embedding_RNN.h5')
dic = imdb.get_word_index()
# Input

st.title('IMDB review')
st.write("Enter Review so that we can classify: ")

review_input = st.text_input("Enter Your Review: ")

def preprocessing(sample):
    example_review =sample
    l_word = example_review.lower().split()
    val_word = [dic.get(word, 2)+ 3 for word in l_word]
    padded_seq = sequence.pad_sequences([val_word], maxlen= 500)
    return padded_seq

def prediction(seq):
    result= model.predict(seq)[0][0]

    if result > 0.5:
        rev = "Review is positive with {:.2f}".format(result*100) + "%"
    else:
        rev = "Review is negative with {:.2f}".format(result*100) + "%"
    return rev
    


if st.button("Classify"):
    padded_seq = preprocessing(review_input)
    rev = prediction(padded_seq)

    st.write(rev)
else:
    st.write("Please share your review")