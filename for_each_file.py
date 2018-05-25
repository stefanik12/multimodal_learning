
# coding: utf-8

# In[ ]:


import os
import InceptionV3_conv
def for_each_file(dirname):
    newdic = {}
    for filename in os.listdir(dirname):
        
        newdic[filename] = description.flatten()
    return newdic



# In[ ]:


print(for_each_file("./data/Pascal_VOC_images"))

