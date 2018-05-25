
# coding: utf-8

# In[1]:


# Do not use !
def save_vec_result(series, outname):
    output = open(outname, "w")
    for word in series.index:
        output.write(word + ":" + str(series[word]) + ", ")
    return outname
    


# In[ ]:


import doc2vec_wrapped
import word2vec_wrapped
import pandas as pd
import inception_for_each_file

def run_to_pickle():
    doc2vec_wrapped.vectorize_content(vector_len=400).to_pickle("./doc2vec")
    print("doc2vec done")
    word2vec_wrapped.vectorize_content().to_pickle("./word2vec")
    print("word2vec done")
    for_each_file("./data/Pascal_VOC_images").to_pickle("./inception")
    print("inception done")
    return


# In[ ]:


run_to_pickle()

