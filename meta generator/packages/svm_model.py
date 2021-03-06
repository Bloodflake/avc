import os
from  pathlib import Path
import  sys
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

save_filepath =  str(sys.argv[0])
save_filepath = Path(save_filepath)

save_filepath = str(save_filepath.parent)
if(save_filepath == "."):
  save_filepath = os.getcwd()


if not os.path.isdir(str(save_filepath) + str(Path('/output'))):
    os.makedirs(str(save_filepath) + str(Path('/output')))

save_filepath = str(save_filepath) + str(Path('/output'))

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
  if not os.path.isdir(str(save_filepath) + str(Path('/video'))):
    os.makedirs(str(save_filepath) + str(Path('/video')))
  
  video = pafy.new(video_file_url)
  Yt_video = video.getbest(preftype="mp4")
  filename = get_clean_text(Yt_video.title) + ".mp4"
  #base = os.getcwd()
  path = str(save_filepath) + str(Path("/video" + "/" + filename)) 
  Yt_video.download(filepath = path)
  return path

def get_text_from_video(filepath):
  recognized_text = ""
  if os.path.isdir(str(save_filepath) + str(Path('/chunk'))):
    shutil.rmtree(str(save_filepath) + str(Path('/chunk')))
    os.makedirs(str(save_filepath) + str(Path('/chunk')))
  else:
    os.makedirs(str(save_filepath) + str(Path('/chunk')))
  
  if not os.path.isdir(str(save_filepath) + str(Path('/audio'))):
    os.makedirs(str(save_filepath) + str(Path('/audio')))
  

  speech_recognizer  = sr.Recognizer()
  video = mp.VideoFileClip(filepath)
  name = video.filename
  name = name.split(str(Path("/")))[-1]
  name = name.strip(".mp4")
  base = os.getcwd()
  video.audio.write_audiofile(str(save_filepath) + str(Path("/audio" + "/" + name)) + ".wav",verbose=False)
  audio = AudioSegment.from_wav(str(save_filepath) + str(Path("/audio" + "/" + name)) + ".wav")

  chunks = split_on_silence(audio,
        min_silence_len = 500,
        silence_thresh = audio.dBFS-13,
        keep_silence=500)

  for i, chunk in enumerate(chunks):
    filename = os.path.join(save_filepath, "chunk")
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
  
  audio_filepath = str(save_filepath) + str(Path("/audio" + "/" + name)) + ".wav"
  return recognized_text,audio_filepath

def get_keywords(text):
    r = Rake(max_length=2, ranking_metric= Metric.WORD_DEGREE)
    r.extract_keywords_from_text(text)
    data = r.get_ranked_phrases_with_scores()
    keywords = []
    for keyword in data:
        keywords.append(keyword[1])

    return keywords
