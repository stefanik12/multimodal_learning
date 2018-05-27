import word2vec_wrapped
import doc2vec_wrapped
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict

def concatenate(vector1, vector2):
    a = np.array(vector1)
    b = np.array(vector2)
    return np.concatenate((a, b))

word2vec_vectors_series = word2vec_wrapped.vectorize_content()
doc2vec_vectors_series = doc2vec_wrapped.vectorize_content()

inception_df = pd.read_json("data/inception_output.txt", lines=True)
inception_df.set_index('image',inplace=True)

df = pd.read_json("data/COCO/coco-easier.txt", lines=True)
df = df[['file_name', 'category']]
df = df.sample(frac=1).reset_index(drop=True)

def w2v_d2v_svc():
    df["concat_vectors"] = df.apply(lambda row: concatenate(doc2vec_vectors_series[row['file_name']], 
                                                       word2vec_vectors_series[row['file_name']]), axis=1)

    classifier = LinearSVC()
    df["predictions"] = cross_val_predict(classifier, 
                                      pd.DataFrame(df["concat_vectors"].tolist()), 
                                      df["category"].values, 
                                      cv=5)
    out_series = pd.Series(df["predictions"].values, index=df["file_name"].values)
    return out_series

def d2v_inception_svc():
    df["concat_vectors"] = df.apply(lambda row: concatenate(doc2vec_vectors_series[row['file_name']],
                                                        inception_df['vector'][row['file_name']]), axis=1)

    classifier = LinearSVC()
    df["predictions"] = cross_val_predict(classifier, 
                                      pd.DataFrame(df["concat_vectors"].tolist()), 
                                      df["category"].values, 
                                      cv=5)
    out_series = pd.Series(df["predictions"].values, index=df["file_name"].values)
    
    return out_series

