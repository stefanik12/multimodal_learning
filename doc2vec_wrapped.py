"""
Usage:

import doc2vec_wrapped
vectors_series = doc2vec_wrapped.vectorize_content(vector_len=400)

# output is Series format, all keys are in vectors_series.index,
# you can retrieve particular vector with:
print(vectors_series["000000154087.jpg"])

"""

import pandas as pd
from collections import namedtuple
from gensim.utils import simple_preprocess
from gensim.models import doc2vec
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

content_dir = "data/COCO/coco-easier.txt"
df = pd.read_json(content_dir,
                  lines=True)
df["captions"] = df["captions"].apply(lambda sents: " ".join(sents))

CategorizedDocument = namedtuple('CategorizedDocument', 'words tags category')
df["vocab_docs"] = df.apply(lambda row: CategorizedDocument(simple_preprocess(row["captions"]),
                                                            [row["category"]],
                                                            row["category"]), axis=1)

def linear_svc_predictions(vector_len=400):
    vectorize_content(vector_len)
    classifier = LinearSVC()
    df["predictions"] = cross_val_predict(classifier, 
                                      pd.DataFrame(df["vectors"].tolist()), 
                                      df["category"].values, 
                                      cv=5)
    out_series = pd.Series(df["predictions"].values, index=df["file_name"].values)
    return out_series

def logistic_regression_predictions(vector_len=400):
    vectorize_content(vector_len)
    classifier = LogisticRegression()
    df["predictions_lr"] = cross_val_predict(classifier, 
                                      pd.DataFrame(df["vectors"].tolist()), 
                                      df["category"].values, 
                                      cv=5)
    out_series = pd.Series(df["predictions_lr"].values, index=df["file_name"].values)
    return out_series

def vectorize_content(vector_len=400):
    doc2vec_model = doc2vec.Doc2Vec(dm=0, vector_size=vector_len, negative=12,
                                    hs=0, min_count=5, workers=1,
                                    alpha=0.1, window=8)

    doc2vec_model.build_vocab(df["vocab_docs"])
    # doc2vec_model.docvec1 = copy.copy(doc2vec_model.docvecs[0])
    doc2vec_model.train(df["vocab_docs"], total_examples=doc2vec_model.corpus_count, epochs=10)

    df["vectors"] = list(map(lambda doc: doc2vec_model.infer_vector(doc.words), df["vocab_docs"]))

    out_series = pd.Series(df["vectors"].values, index=df["file_name"].values)

    return out_series
