import os
import shutil
import pafy
import speech_recognition as sr 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mp

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from rake_nltk import Rake,Metric

stop_words = set(stopwords.words('english'))

from nltk.corpus import stopwords

import pke
import spacy
import pytextrank

filee = open(".//packages//custom_stopword.txt", "r")
try:
    content = filee.read()
    custom_stopwords = content.split(",")
finally:
    filee.close()

from pyate.term_extraction_pipeline import TermExtractionPipeline


stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stopwords.words('english')
stoplist = stoplist + custom_stopwords
nlp = spacy.load("en_core_web_sm")
for word in custom_stopwords:
  nlp.vocab[word].is_stop = True

tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

nlp.add_pipe(TermExtractionPipeline())

def keywords_generators(text, stoplist = stoplist, n = 30):
    keywords = []
    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input=text, language='en')
    extractor.candidate_selection(n=3, stoplist=stoplist)
    extractor.candidate_weighting()
    keyphrases1 = extractor.get_n_best(n)

    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input=text, language='en')
    extractor.candidate_selection(n=3, stoplist=stoplist)
    extractor.candidate_weighting()
    keyphrases2 = extractor.get_n_best(n)

    for w1, w2 in zip(keyphrases1, keyphrases2):
        if (w1[0] not in keywords):
            keywords.append(w1[0])
        if (w2[0] not in keywords):
            keywords.append(w2[0])
  
    return keywords

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

    print("GETTING TEXT FROM VIDEO")
    for i, chunk in enumerate(chunks):
        filename = os.path.join(base, "chunk")
        filename = os.path.join(filename,f"chunk{i}.wav")
        chunk.export(filename, format="wav")
        with sr.AudioFile(filename) as source:
            audio_listened = speech_recognizer.record(source)
            ### Try converting it to text
            try:
                text = speech_recognizer.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                  pass
            else:
                text = f"{text.capitalize()}. "
                recognized_text += text
    return recognized_text

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
  
