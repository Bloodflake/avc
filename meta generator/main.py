import sys
import os
import pickle
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from packages import svm_model

import pprint

import json

from packages import sub_classes

from packages.keyword_generator import keywords_generators

#base = os.path.dirname(os.path.abspath(__file__))
base = str(Path(__file__).absolute().parent)
print(base)

youtube_video_url = str(sys.argv[1])
le_filename = str(Path(base +"/packages/main_classes/le"))
le = pickle.load(open(le_filename, "rb"))

model_filename = str(Path(base + "/packages/main_classes/model"))
model = pickle.load(open(model_filename, "rb"))

tfidf_filename = str(Path(base + "/packages/main_classes/tfidf"))
tfidf_text = pickle.load(open(tfidf_filename, "rb"))


video_filepath = svm_model.get_youtube_video(youtube_video_url)
text, audio_filepath = svm_model.get_text_from_video(video_filepath)
clean_text = svm_model.get_clean_text(text)
#keywords = svm_model.get_keywords(text)
keywords = keywords_generators(text)
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