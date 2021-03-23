import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from swachhdata.text import *
from keras.utils.data_utils import get_file
model= get_file('best_model2.hdf5','https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Model/best_model2.hdf5?raw=true')
model = load_model(model)
tokenizer =Tokenizer()
df=pd.read_csv("https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Data/cleaned_tweet.csv?raw=true")
sentiment = ['Hate Speech','Offensive Language','No Issues']
PAGE_CONFIG = {"page_title":"Hate Speech Detector","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

def process(tweet):
    return TextRecast(tweet, urlRecast = {'process': 'remove'},
                      htmlRecast = True,
                      EscapeSequenceRecast = True,
                      MentionRecast = {'process': 'extract_remove'},
                      ContractionsRecast = True,
                      CaseRecast = {'process': 'lower'},
                      EmojiRecast = {'process': 'remove', 'space_out': False},
                      HashtagRecast = {'process': 'remove'},
                      StopWordsRecast = {'package': 'nltk', 'stopwords': None},
                      NumberRecast = {'process': 'remove', 'seperator': None},
                      PunctuationRecast = True,
                      LemmatizationRecast = {'package':'nltk'})

def main():
  menu = ["Tool"]
  tokenizer =Tokenizer()
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Tool':
    st.header("Hate Speech Detector")
    st.write("A machine learning tool which can detect if a particular input is hateful or not")
    opt=st.selectbox('Select input format',['Text','Twitter Link','Audio File'])
    if opt == 'Text':
      user_input = st.text_input('Enter text')
    elif opt == 'Twitter Link':
      user_input = st.text_input('Enter link')
    elif opt == 'Audio File':
      user_input = st.file_uploader('Upload File')
    if st.button('Generate Text'):
      input = process(user_input)
      clean=df['tweet'].astype('str')
      tokenizer.fit_on_texts(clean.values)
      input = tokenizer.texts_to_sequences([input])
      test = pad_sequences(input, maxlen=30)
      generated_text = sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
      st.write(generated_text)

if __name__ == '__main__':
	main()
