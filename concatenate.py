
# coding: utf-8

# In[1]:


import numpy as np

def concatenate(vector1, vector2):
    a = np.array(vector1)
    b = np.array(vector2)
    return np.concatenate((a, b))


# In[2]:


import doc2vec_wrapped
import word2vec_wrapped
import pandas as pd
import for_each_file
import for_each_file

def concat_ser_dic(series1, series2):
    if isinstance(series1, dict):
        series1 = pd.Series(series1)
    if isinstance(series2, dict):
        series1 = pd.Series(series2)
    concatenate_series = pd.Series()
    for i in series1.index:
        concatenate_series[i] = concatenate(series1[i], series2[i])
    return concatenate_series

## doc2vec+word2vec ##
def doc2vec_word2vec():
    return concat_ser_dic(doc2vec_wrapped.vectorize_content(vector_len=400), word2vec_wrapped.vectorize_content())

def doc2vec_inception():
    return concat_ser_dic(doc2vec_wrapped.vectorize_content(vector_len=400), for_each_file("./data/Pascal_VOC_images"))
    
def word2vec_inception():
    return concat_ser_dic(word2vec_wrapped.vectorize_content(), for_each_file("./data/Pascal_VOC_images"))

def doc2vec_word2vec_inception():
    tmp = concat_ser_dic(doc2vec_wrapped.vectorize_content(vector_len=400), word2vec_wrapped.vectorize_content())
    return concat_ser_dic(tmp, for_each_file("./data/Pascal_VOC_images"))


# In[3]:


print(len(doc2vec_word2vec()))

