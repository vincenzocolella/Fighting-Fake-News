from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from extract_exif import extract_exif_test,generate_label,cropping_list,get_np_arrays,remove_elements,create_batch_samples
from matplotlib import image
from lib.utils import benchmark_utils, util,io
import cv2
import os
import numpy as np
import keras
import pickle
from keras.engine import keras_tensor

EPOCHS = 100 
def datagenerator(images,images2,labels, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            x = images[start:end] 
            y = labels[start:end]
            x2 = images2[start:end]
            yield (x,x2),y

            start += batchsize
            end += batchsize

print("[INFO] starting test")
"""
with open("dict.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	training_dict = pickle.load(fp)
fp.close()
training_dict = remove_elements(training_dict)

#--------------------------------------------------------------- EXTRACT 
dict,image_list,dict_keys = extract_exif_test(list(training_dict.keys()))
#--------------------------------------------------------------- REMOVE ELEMENTS
for key in dict:
    print("KEY: ",key)
print("[INFO] number of keys: ", len(dict_keys))

#--------------------------------------------------------------- CREATE SAMPLES
list1_img,list2_img = create_batch_samples(dict,image_list)
#--------------------------------------------------------------- GENERATE LABELS
exif_lbl = generate_label(dict_keys,list1_img,list2_img)


"""
with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()
missing_labels = np.zeros(11)

for i in range(len(exif_lbl)):
    exif_lbl[i] = np.concatenate((np.array(exif_lbl[i]),missing_labels),axis=0)
exif_lbl = np.array(exif_lbl)
print(len(exif_lbl[0]))
#--------------------------------------------------------------- CROP IMAGES
#list1,list2 = cropping_list(list1_img,list2_img)

list1,list2 = get_np_arrays('test_cropped_arrays.npy')
#--------------------------------------------------------------- GET ELEMENTS
#list1,list2 = get_np_arrays('test_cropped_arrays.npy')
#with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
#	exif_lbl = pickle.load(fp)
#fp.close()
#--------------------------------------------------------------- RUN MODEL
x_train = datagenerator(list1,list2,exif_lbl,32)

model = tf.keras.models.load_model('siameseMLP.h5')

print("[INFO] Starting Evaluation")

print(model.evaluate(x_train,steps=len(list1)))

print(model.metrics_names)
