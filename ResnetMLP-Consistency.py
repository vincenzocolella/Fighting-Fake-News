from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model,Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from extract_exif import extract_exif, random_list,generate_label,cropping_list,get_np_arrays
from matplotlib import image
from lib.utils import benchmark_utils, util,io
import cv2
import numpy as np
import keras
import pickle
from keras.engine import keras_tensor


EPOCHS = 100

def SiameseMLP2():
    siameseMLP = tf.keras.models.load_model('final_model.h5')
    
    for layer in siameseMLP.layers:
        layer.trainable = False
    last_layer  = siameseMLP.output
    x = Flatten(name="flat21")(last_layer)
    x = Dense(512, activation='relu',name="pre_final_out")(x)
    x = Dense(1, activation='sigmoid',name="final_out")(x)

    model = Model(inputs=siameseMLP.input,outputs=x)
    return model

def datagenerator(images,images2, labels, batchsize, mode="train"):
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


with open("exif_lbl.txt", "rb") as fp:  
	exif_lbl = pickle.load(fp)
fp.close()

for i in range(len(exif_lbl)):
    exif_lbl[i] = np.array(exif_lbl[i])
exif_lbl = np.array(exif_lbl)

list1,list2 = get_np_arrays('cropped_arrays.npy')

train_set = int(len(list1)*(2/3))

list1_train = list1[:train_set]
list2_train = list2[:train_set]
exif_lbl1 = exif_lbl[:train_set]

list1_test = list1[train_set:]
list2_test = list2[train_set:]
exif_lbl2 = exif_lbl[train_set:]

x_train = datagenerator(list1_train,list2_train,exif_lbl1,32)
x_test = datagenerator(list1_test,list2_test,exif_lbl2,32)

steps = int(train_set/EPOCHS)
image_shape = (128,128,3)

model = SiameseMLP2()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x = x_train,epochs=EPOCHS,steps_per_epoch=steps,validation_data = x_test,validation_steps=steps,validation_batch_size=32)
model.save("siameseMLP.h5")

