import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from swachhdata.text import *
from keras.utils.data_utils import get_file
import speech_recognition as sr 
import matplotlib.pyplot as plt
import os
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence
import plotly.express as px
from wordcloud import WordCloud
import tweepy as tw

model= get_file('best_model2.hdf5','https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Model/best_model2.hdf5?raw=true')
model = load_model(model)
tokenizer =Tokenizer()
r = sr.Recognizer()

df=pd.read_csv("https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Data/cleaned_tweet.csv?raw=true")
df2=pd.read_csv("https://github.com/TazeemKhan9/Hate-Speech-Detector/blob/main/Data/HateSpeechData.csv?raw=true")
sentiment = ['Hate Speech','Offensive Language','No Issues']

Access_Token='2276034169-WiASclcqTMhnaZeskrA1CYSULuGS3X3nZQztHQo'
Access_token_secret= 'aCvI9D3H9J22YI5AvbiUNWlusLiqMiqSP7kRs2u181MiM'
API_Key='YEHbdylfRU9f7m7IIwGrMl597'
API_key_secret ='rbNIqQ0vwMWdaGVK8qXzld0I4zXPK5Kama5FiYQKrkfAoAS3q0'
auth = tw.OAuthHandler(API_Key, API_key_secret)
auth.set_access_token(Access_Token, Access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

data={
  "type": "service_account",
  "project_id": "acoustic-arch-308907",
  "private_key_id": "a25c1c779c6e690f84d5555a7040c250dc8384bd",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDF5inqmMM7riI3\nQrt0ZNpGNEeVyBzoGn1OHt24TmbHyY2t7cXZp03EI86RYu+8GcMlSC2glxMBS/GT\nGPOrIfMo9wPa1yWEiONy702TDhOo+hpaknePR/nycDbpQnS7ddZeLqKV8bdwBwnO\nr2PTR36NT79gd0xBn3eU+U6gVRZ8L3RJ6B5eyCsQbr+4JJLI+jA7Bdr+FuNXvGiG\nvDiFkayJyQCmImhEe5Rf2r9etxRBHa+e3FpcrZMQVK9kMQlrsZAa1q9w+bOkXIL2\nQrSayDH1KSqLNkvJq/ivp4alQjflDHjD6K1S8xSRHoza23Ip4h8PiMiiyGpmDzJG\nc9fspNaTAgMBAAECggEAK1lA2kHvFf4NQaSM8jB7PYCLI6yLZG4U/400TQBjdCTa\ns3wPaB9FIg8j2uXclIviGqMGV5RwFcsgVfPTwCs2G8WL4x5YegEQLWdsyyb8uHlc\nV3WY2dpOzkO1S4ACmON1evjlJUomv9PS2qLBj5CsEGElDwjiu8cmBwxbJDSvMbFC\nd28glOeWpDKtMwaJu9ZaPpcC8h6w9H2a9aKTRjZJMHUkpMFNaEEf/inCyLylef6J\n+7xvXz0UdOioe8+rD1ZrIGSSs9QqthhUgqVQTAfxSAXZ53jobgglARcStFbi5CEV\nDYAODR5AWaZPWS88JZc8aQ3igmocH3hOl4YT/5HkFQKBgQDy5ogPZi9sAnInRiKA\nFfJOIDB9X8a3SAlgyCUQu0jUzH3QDBofEofDECnA69Lor66sTepPRx/OwsXLt7o8\nbyN6/FWZKUaTN9oBeW9blA6psa3exRaXGCJSeCpCiF8bB3Z0CQGX881Q/1x0z4cF\nls475TkZjOU5P6/38B/VrDSe5QKBgQDQklhqs+/ggQenWZvFIy4msO2lAEjb+sDn\ntlwl5Q8+iGLzS0Y+TqCK5Jv6lD+6tc20uw135boRsqjfsEcQe+HYHcBvyPtHHdba\nffF6OZa50Bm5gwEUYH6sKOshFOoLru0+quwn8QhwNhgjPAeoE7npWR4JObPjvGRz\nqf89YgdQFwKBgQC54uZ5MnBULkMB/1BjyWfXlhbFu8gtdzmGEWUcOtdv0tbtonVT\nFjFDfFkXxOFxJRF911rbNMkIyFHqpz4lBcCXXAh93/Kcs39o5W/tG49lGg6/jwDM\nvLF3f3KH3Ck8XCewgTvw96lGtUYiNrdT9ab6e1+JSCQb/btC+UbDlLfoaQKBgQDH\n/Dqo/SU5N58WGHainKvoz2bd+hriSlnjE1jhwPNP+0gdjgSpQ4zuAGuK5dEBfsbh\nzyUH3I7/3zXLXeOV66LOLSDSTnyZYQQc9fuvPT7HpcC0vucvGaL8AjQJwVr0nuK+\nXvcXCScVKNkWF74jq95r31ZMdDaHW6FZwhuJSNBIOwKBgQDwh1KvfKG1uEADeuAc\nrdVb18rB/hHECuLq9pcobFR5fZTSaQOoffwmjbOxhdwgwSmjiI0yfrNQe0YmaFNo\nr4JExebdrqSR1z+Rx+T1skEh6Nz9NBkOXoL3KMpT4MWrmGemwlw+BRzXhFMd3KIf\nuigRkaoEoGsswIZfaCq2cZtdJA==\n-----END PRIVATE KEY-----\n",
  "client_email": "speech-to-text@acoustic-arch-308907.iam.gserviceaccount.com",
  "client_id": "104798248999236730114",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/speech-to-text%40acoustic-arch-308907.iam.gserviceaccount.com"
}
y=json.dumps(data)




PAGE_CONFIG = {"page_title":"Hate Speech Detector","page_icon":":H:","layout":"centered"}
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
                text = r.recognize_google_cloud(audio_listened,credentials_json=y)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                #print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

def main():
  menu = ["Tool","More"]
  tokenizer =Tokenizer()
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Tool':
    st.header("Hate Speech Detector")
    st.write("A machine learning tool which can detect if a particular input is hateful or not")
    opt=st.selectbox('Select input format',['Text','Twitter Link','Audio File'])
    if opt == 'Text':
      user_input = st.text_input('Enter text')
    elif opt == 'Twitter Link':
      x= st.text_input('Enter link')
      try:
        id=x.rsplit('/', 1)[1]
        status = api.get_status(id,tweet_mode="extended")
        user_input=status.full_text
        st.subheader("Text of the tweet")
        st.write(user_input)
      except ValueError:
        st.error('Please enter a valid input')
    elif opt == 'Audio File':
      uploaded_file = st.file_uploader('Upload File',type='wav')
      if uploaded_file is not None:
        user_input= get_large_audio_transcription(uploaded_file)
        st.subheader("Speech to Text Result")
        st.write(user_input)
    if st.button('Generate Result'):
      input = process(user_input)
      clean=df['tweet'].astype('str')
      tokenizer.fit_on_texts(clean.values)
      input = tokenizer.texts_to_sequences([input])
      test = pad_sequences(input, maxlen=30)
      generated_text = sentiment[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
      st.write(generated_text)
  elif choice == "More":
    st.header('About the project')
    st.subheader('Data')
    st.write('The dataset used for this project consists of Tweets labeled as hate_speech, offensive_language, or neither. We have added multiple steps in our preprocessing like removal of stop words, lemitizing, removal of emojis etc. on both our training dataset and the input which we take. Below you can select which dataset you want to see.')
    pos_df= st.selectbox("Select a Dataset",['None','Original Data','Cleaned Data'])
    if pos_df =='Original Data':
      st.dataframe(df2)
    elif pos_df =='Cleaned Data':
      st.dataframe(df)
    st.write('One of the major issues faced by us in the dataset was the class imbalance. The class imbalance can bee seen in the histogram below')
    fig = px.histogram(df2, x="class",color='class',labels=['Hate Speech','Offensive Language','No issues'])
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Modelling')
    st.write('Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They work tremendously well on a large variety of problems, and are now widely used especially in NLP applications. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!. LSTMs also have this chain like neural network structure, but the repeating module has a different structure from that of RNN. Instead of having a single neural network layer, there are four, interacting in a very special way. This helps in learning the context of statements and dealing with the class imabalance in the dataset. Below you can select wordclouds of the dataset.')
    wc= st.selectbox("Select an option",['All','Hate Speech','Offensive Language'])
    x = st.select_slider(label='Number of words',options=[25,50,100,150])
    if wc == "All":
      all_words = ' '.join([text for text in df['tweet'].astype(str)])
      wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,max_words=x).generate(all_words)
      fig, ax = plt.subplots()
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      st.pyplot(fig)
    elif wc== 'Hate Speech':
      hatred_words = ' '.join([text for text in df['tweet'][df2['class'] == 0].astype(str)])
      wordcloud = WordCloud(width=800, height=500,max_font_size=110,max_words=x).generate(hatred_words)
      fig, ax = plt.subplots()
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      st.pyplot(fig)
    elif wc== 'Offensive Language':
      offensive_words = ' '.join([text for text in df['tweet'][df2['class'] == 1].astype(str)])
      wordcloud = WordCloud(width=800, height=500,max_font_size=110,max_words=x).generate(offensive_words)
      fig, ax= plt.subplots()
      plt.imshow(wordcloud, interpolation='bilinear')
      plt.axis("off")
      st.pyplot(fig)

if __name__ == '__main__':
  main()
