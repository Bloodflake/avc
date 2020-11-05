import sys
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import svm_model

import pprint

import json

import sub_classes


youtube_video_url = str(sys.argv[1])
print(youtube_video_url)
le_filename = "le"
le = pickle.load(open(le_filename, "rb"))

model_filename = "model"
model = pickle.load(open(model_filename, "rb"))

tfidf_filename = "tfidf"
tfidf_text = pickle.load(open(tfidf_filename, "rb"))


video_filepath = svm_model.get_youtube_video(youtube_video_url)
text, audio_filepath = svm_model.get_text_from_video(video_filepath)
clean_text = svm_model.get_clean_text(text)
keywords = svm_model.get_keywords(text)
test = tfidf_text.transform([clean_text])
y_pred = model.predict(test)
sub_class = sub_classes.get_sub_class(le.classes_[y_pred[0]], clean_text)

meta = {
    "category": le.classes_[y_pred[0]],
    "sub class": sub_class,
    "target": [],
    "video location": video_filepath,
    "audio location": audio_filepath,
    "keyword": keywords,
    "text": text

}

pprint.pprint(meta)

out_file = open("Data.json", "a")

json.dump(meta, out_file, indent=2)
out_file.write(",")
out_file.write("\n")
out_file.close()