
# coding: utf-8

# In[1]:


import word2vec_wrapped
import doc2vec_wrapped
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score


# In[2]:


inception_df = pd.read_json("data/inception_output.txt", lines=True)
inception_df.set_index('image',inplace=True)
inception_df[:5]


# In[3]:


def concatenate(vector1, vector2, vector3 = 0):
    a = np.array(vector1)
    b = np.array(vector2)
    if (vector3 == 0):
        return np.concatenate((a, b))
    c = np.array(vector3)
    return np.concatenate((np.concatenate((a, b)), c))


# In[4]:


def cross_val_accuracy(classifier, df):
    # Returns the mean accuracy on the given test data and labels, in 5 cross validation splits
    scores = cross_val_score(classifier, 
                             pd.DataFrame(df["concat_vectors"].tolist()), 
                             df["category"].values, 
                             cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # doc2vec_inception_class

# In[5]:


def doc2vec_inception_class():
    df = pd.read_json("data/COCO/coco-easier.txt", 
                  lines=True)
    df = df.sample(frac=1).reset_index(drop=True)
    doc2vec_vectors_series = doc2vec_wrapped.vectorize_content()
    df["concat_vectors"] = df.apply(lambda row: concatenate(doc2vec_vectors_series[row['file_name']],
                                                           inception_df['vector'][row['file_name']]), axis=1)
    X = df["concat_vectors"]
    X

    from sklearn.neural_network import MLPClassifier
    X = df["concat_vectors"].tolist()
    y = df["category"].tolist()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(2500, 2500), random_state=1)
    clf.fit(X, y)                         
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    print("doc2vec_inception")
    cross_val_accuracy(clf, df)


# # word2vec_inception_class

# In[6]:


def word2vec_inception_class():
    df = pd.read_json("data/COCO/coco-easier.txt", 
                  lines=True)
    df = df.sample(frac=1).reset_index(drop=True)
    word2vec_vectors_series = word2vec_wrapped.vectorize_content()
    df["concat_vectors"] = df.apply(lambda row: concatenate(word2vec_vectors_series[row['file_name']],
                                                           inception_df['vector'][row['file_name']]), axis=1)
    X = df["concat_vectors"]
    X

    from sklearn.neural_network import MLPClassifier
    X = df["concat_vectors"].tolist()
    y = df["category"].tolist()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(2500, 2500), random_state=1)
    clf.fit(X, y)                         
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    print("word2vec_inception")
    cross_val_accuracy(clf, df)


# # word2vec_doc2vec_class

# In[7]:


def word2vec_doc2vec_class():
    df = pd.read_json("data/COCO/coco-easier.txt", 
                  lines=True)
    df = df.sample(frac=1).reset_index(drop=True)
    word2vec_vectors_series = word2vec_wrapped.vectorize_content()
    doc2vec_vectors_series = doc2vec_wrapped.vectorize_content()
    df["concat_vectors"] = df.apply(lambda row: concatenate(word2vec_vectors_series[row['file_name']],
                                                           doc2vec_vectors_series[row['file_name']]), axis=1)
    X = df["concat_vectors"]
    X

    from sklearn.neural_network import MLPClassifier
    X = df["concat_vectors"].tolist()
    y = df["category"].tolist()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(2500, 2500), random_state=1)
    clf.fit(X, y)                         
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    print("word2vec_doc2vec")
    cross_val_accuracy(clf, df)


# # word2vec_doc2vec_inception_class

# In[8]:


def word2vec_doc2vec_inception_class():
    df = pd.read_json("data/COCO/coco-easier.txt", 
                  lines=True)
    df = df.sample(frac=1).reset_index(drop=True)
    word2vec_vectors_series = word2vec_wrapped.vectorize_content()
    doc2vec_vectors_series = doc2vec_wrapped.vectorize_content()

    df["concat_vectors"] = df.apply(lambda row: concatenate(word2vec_vectors_series[row['file_name']],
                                                            doc2vec_vectors_series[row['file_name']],
                                                            inception_df['vector'][row['file_name']]), axis=1)
  
    X = df["concat_vectors"]
    X

    from sklearn.neural_network import MLPClassifier
    X = df["concat_vectors"].tolist()
    y = df["category"].tolist()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(2500, 2500), random_state=1)
    clf.fit(X, y)                         
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    print("word2vec_doc2vec_inception")
    cross_val_accuracy(clf, df)


# In[ ]:


doc2vec_inception_class()
word2vec_inception_class()
word2vec_doc2vec_class()
word2vec_doc2vec_inception_class()

