from tensorflow.keras.utils import load_img,img_to_array
import tensorflow as tf
import random, os
import numpy as np
from matplotlib import pyplot as plt

test_dir = 'flower-data/test/'
img_sz = 224

class_name = list(os.listdir('flower-data/train/'))

img_name = random.choice(os.listdir(test_dir))
img_path = test_dir + img_name

img = load_img(img_path,target_size=(256,256))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array,0)

model = tf.keras.models.load_model('model10epochs.h5')

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"{img_name} most likely belongs to {class_name[np.argmax(score)]} with a {100*np.max(score):.2f}")

plt.figure(figsize=(2,2))
plt.imshow(img_array[0].numpy(),vmin=0,vmax=255)
plt.title("{}:{:.2f}".format(class_name[np.argmax(score)], 100*np.max(score)))
plt.axis('off')