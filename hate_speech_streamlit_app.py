%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from swachhdata.text import *
from keras.utils.data_utils import get_file
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence


model= get_file('best_model2.hdf5','https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Model/best_model2.hdf5?raw=true')
model = load_model(model)
tokenizer =Tokenizer()
r = sr.Recognizer()
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

def get_large_audio_transcription(path):
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                #print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

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
      sound = AudioSegment.from_mp3(path)
      dst="test.wav"
      sound.export(dst, format="wav")
      user_input= get_large_audio_transcription(dst)
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
