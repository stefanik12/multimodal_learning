
# coding: utf-8

# In[1]:


import glob, os
import tarfile
import sys
<<<<<<< HEAD
import json
=======

>>>>>>> 73cfe2021a01978690e2732661fd92fa8f85acd5
import numpy as np
import tensorflow as tf

from six.moves import urllib


# In[2]:


<<<<<<< HEAD
model_dir = './model'
=======
model_dir = '/model'
>>>>>>> 73cfe2021a01978690e2732661fd92fa8f85acd5
model_name = 'classify_image_graph_def.pb'
data_dir = './data/COCO/easier'

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# In[3]:


def maybe_download_and_extract():
    """Download and extract model tar file."""
    
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# In[4]:


def load_data():
<<<<<<< HEAD
    image_files = []
    
    for file in glob.glob(data_dir + "/*.jpg"):
=======
    os.chdir(data_dir)
    image_files = []
    
    for file in glob.glob("*.jpg"):
>>>>>>> 73cfe2021a01978690e2732661fd92fa8f85acd5
        image_files.append(file)
        
    return image_files


# In[5]:


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    
    with tf.gfile.FastGFile(os.path.join(model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, name='')


# In[6]:


def run_inference_on_image(image):
    """Runs inference on an image.
    Args:
        image: Image file name.
    Returns:
        Description of the image (2048 floats)
    """
    
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()
    
    with tf.Session() as sess:
        pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        image_description = sess.run(pool_tensor, {'DecodeJpeg/contents:0': image_data})
        
        return image_description


# In[7]:

<<<<<<< HEAD
def run_inference_on_images(images):
    """Runs inference on an image.
    Args:
        image: Image file name.
    Returns:
        Description of the image (2048 floats)
    """

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        with open('./data/inception_output.txt', 'a') as outfile:
            for image in image_files:
                image_data = tf.gfile.FastGFile(image, 'rb').read()

                description = sess.run(pool_tensor, {'DecodeJpeg/contents:0': image_data})
                json.dump({image.rsplit('/', 1)[-1]: description.flatten().tolist()}, outfile)
                outfile.write('\n')


        return

# if the model is already downloaded, comment this
# maybe_download_and_extract()
=======

# if the model is already downloaded, comment this
maybe_download_and_extract()

>>>>>>> 73cfe2021a01978690e2732661fd92fa8f85acd5

# In[ ]:


image_files = load_data()
<<<<<<< HEAD

run_inference_on_images(image_files)



=======
result_dic = {}

for image in image_files:
    description = run_inference_on_image(image)
    result_dic[image] = description.flatten()
    
    print(result_dic[image])
>>>>>>> 73cfe2021a01978690e2732661fd92fa8f85acd5

