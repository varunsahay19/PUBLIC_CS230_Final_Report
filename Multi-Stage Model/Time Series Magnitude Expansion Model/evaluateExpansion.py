# (c) 2024 Patrick Nieman and Varun Sahay
# Plots intermediate magnitude time series predicted by chaotic interpolation model

import numpy as np
from matplotlib import pyplot as p
import tensorflow as tf
from random import randrange
from utilLoss import *

length=4096
tf.keras.config.enable_unsafe_deserialization()

g=386.09
casesEvaluated=256

#Load linear model and test data
inputs=np.load("/Applications/CS230 Data/Export/condensedOutput%s.npy"%length)
outputs=np.load("/Applications/CS230 Data/Export/condensedOutputIntermediate%s.npy"%length)

model=tf.keras.models.load_model("/Applications/CS230 Data/expansionModel.keras",custom_objects={"expansion":expansion})
model.trainable=False

#Shuffle data and predict time histories
shuffle=np.random.shuffle(np.arange(outputs.shape[0]))
outputs=outputs[shuffle,:]
outputs=outputs[:,:casesEvaluated,:]
inputs=inputs[shuffle,:]
inputs=inputs[:,:casesEvaluated,:]
inputs=np.squeeze(inputs)
outputs=np.squeeze(outputs)
predictions=model.predict(inputs)

#Plot example time series
p.figure(1)
for i in range(16):
    index=randrange(0,casesEvaluated)
    p.subplot(8,6,3*i+1)
    p.plot(inputs[index,:])

    p.subplot(8,6,3*i+2)
    p.plot(outputs[index,:])

    p.subplot(8,6,3*i+3)
    p.plot(predictions[index,:])
p.show()