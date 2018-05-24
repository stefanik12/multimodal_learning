"""
Usage:

import word2vec_wrapped
vectors_series = word2vec_wrapped.vectorize_content()

# output is Series format, all keys are in vectors_series.index,
# you can retrieve particular vector with:
print(vectors_series["000000154087.jpg"])

"""

import pandas as pd
import numpy as np
from collections import namedtuple
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

content_dir = "data/COCO/coco-easier.txt"


word2vec = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)


def get_vectors(words):
    return list(map(lambda word: word2vec[word], filter(lambda word: word in word2vec, words)))


def vectorize_content():
    df = pd.read_json(content_dir, lines=True)

    df["captions"] = df["captions"].apply(lambda sents: " ".join(sents))

    df["words"] = df.apply(lambda row: simple_preprocess(row["captions"]), axis=1)

    df["vectors"] = df.apply(lambda row: get_vectors(row["words"]), axis=1)
    df["vector_avg"] = df.apply(lambda row: list(np.average(np.array(row["vectors"]), axis=0)), axis=1)

    out_series = pd.Series(df["vector_avg"].values, index=df["file_name"].values)

    return out_series
