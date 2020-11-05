import os
import shutil

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pafy
import speech_recognition as sr 

from pydub import AudioSegment
from pydub.silence import split_on_silence

import moviepy.editor as mp

import re
import string

import nltk

from rake_nltk import Rake,Metric

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

r = Rake(max_length=3, ranking_metric= Metric.WORD_DEGREE)

def get_clean_text(text):
  text = text.lower()
  text = re.sub(r'\d+', '', text) 
  text = text.translate(text.maketrans('', '', string.punctuation))
  text = text.strip()
  text = word_tokenize(text)
  text = [word for word in text if word.isalpha()] 
  text = [w for w in text if not w in stop_words] 
  text = ' '.join(text)
  return text

def get_youtube_video(video_file_url):
  if not os.path.isdir('video'):
    os.makedirs('video')
  
  video=pafy.new(video_file_url)
  Yt_video = video.getbest(preftype="mp4")
  filename = get_clean_text(Yt_video.title) + ".mp4"
  base = os.getcwd()
  path = base + "\\video" +"\\"+ filename
  Yt_video.download(filepath=path)
  return path

def get_text_from_video(filepath):
  recognized_text = ""
  if os.path.isdir('chunk'):
    shutil.rmtree('chunk')
    os.makedirs('chunk')
  else:
    os.makedirs('chunk')
  
  if not os.path.isdir('audio'):
    os.makedirs('audio')
  

  speech_recognizer  = sr.Recognizer()
  video = mp.VideoFileClip(filepath)
  name = video.filename
  name = name.split("\\")[-1]
  name = name.strip(".mp4")
  base = os.getcwd()
  video.audio.write_audiofile(base + "\\audio" + "\\" + name + ".wav",verbose=False)
  audio = AudioSegment.from_wav(base + "\\audio" + "\\" + name + ".wav")

  chunks = split_on_silence(audio,
        min_silence_len = 500,
        silence_thresh = audio.dBFS-13,
        keep_silence=500)

  for i, chunk in enumerate(chunks):
    filename = os.path.join(base, "chunk")
    filename = os.path.join(filename,f"chunk{i}.wav")
    chunk.export(filename, format="wav")
    with sr.AudioFile(filename) as source:
      audio_listened = speech_recognizer.record(source)
      ### Try converting it to text
      try:
        text = speech_recognizer.recognize_google(audio_listened)
        print("text -> ",text)
      except sr.UnknownValueError as e:
          pass
      else:
        text = f"{text.capitalize()}. "
        recognized_text += text
  
  audio_filepath = base + "\\audio" + "\\" + name + ".wav"
  return recognized_text,audio_filepath

def get_keywords(text):
    r = Rake(max_length=2, ranking_metric= Metric.WORD_DEGREE)
    r.extract_keywords_from_text(text)
    data = r.get_ranked_phrases_with_scores()
    keywords = []
    for keyword in data:
        keywords.append(keyword[1])

    return keywords
