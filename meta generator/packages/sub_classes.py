import pickle
import os

base = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def get_sub_class(predicted_class, text):
    if(predicted_class == "Technology"):
        le_filename = base + "\\tech\\tech_le.pickle"
        le = pickle.load(open(le_filename, "rb"))

        model_filename = base + "\\tech\\tech_model.pickle"
        model = pickle.load(open(model_filename, "rb"))

        tfidf_filename = base + "\\tech\\tech_tfidf.pickle"
        tfidf_text = pickle.load(open(tfidf_filename, "rb"))

        tranform_text = tfidf_text.transform([text])
        y_pred = model.predict(tranform_text)

        return le.classes_[y_pred[0]]

    elif (predicted_class == "health"):
        le_filename = base + "\\health\\health_le.pickle"
        le = pickle.load(open(le_filename, "rb"))

        model_filename = base + "\\health\\health_model.pickle"
        model = pickle.load(open(model_filename, "rb"))

        tfidf_filename = base + "\\health\\health_tfidf.pickle"
        tfidf_text = pickle.load(open(tfidf_filename, "rb"))

        tranform_text = tfidf_text.transform([text])
        y_pred = model.predict(tranform_text)

        return le.classes_[y_pred[0]]
