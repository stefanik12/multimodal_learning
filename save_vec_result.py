
# coding: utf-8

# In[1]:


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

save_vec_result(doc2vec_wrapped.vectorize_content(vector_len=400), "doc2vec")
print("doc2vec done")
save_vec_result(word2vec_wrapped.vectorize_content(), "word2vec")
print("word2vec done")
save_vec_result(for_each_file("./data/Pascal_VOC_images"), "inception")
print("inception done")

