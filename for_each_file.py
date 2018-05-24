
# coding: utf-8

# In[6]:


import os
def for_each_file(dirname):
    newdic = {}
    for filename in os.listdir(dirname):
        print(filename)
        ##newdic[filename] = callthecommandhere(blablahbla, filename, foo)   Get value from your funcion that returns vector for that image.
    return ##newdic

for_each_file("./data/Pascal_VOC_images")

## Remove print and uncomment newdic lines, change callthecommandhere for your function

