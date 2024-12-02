# (c) 2024 Patrick Nieman and Varun Sahay
# Plots time series predicted by coarse temporal dependency model

import numpy as np
from matplotlib import pyplot as p
import tensorflow as tf
from random import randrange
from utilLoss import *

normalize=True
length=4096
tf.keras.config.enable_unsafe_deserialization()

g=386.09
casesEvaluated=256

#Load linear model and test data
inputs=np.loadtxt("/Applications/CS230 Data/Export/inputExpanded.csv",dtype=float,delimiter=",")
inputs=inputs[:,74:]
yTest=np.load("/Applications/CS230 Data/Export/condensedOutput%s.npy"%length)

model=tf.keras.models.load_model("/Applications/CS230 Data/condensedModel.keras",custom_objects={"elementAverageRelativeAbsolute":elementAverageRelativeAbsolute})
model.trainable=False

#Shuffle data and predict time histories
if normalize:
    yTest/=np.max(np.abs(yTest),axis=1,keepdims=True)
yTest=yTest[-casesEvaluated:,:]
inputs=inputs[-casesEvaluated:,:]
predictions=model.predict(inputs)

#Plot example time series
p.figure(1)
for i in range(16):
    p.subplot(8,4,2*i+1)
    index=randrange(0,casesEvaluated)
    p.plot(yTest[index,:])
    p.subplot(8,4,2*i+2)
    p.plot(predictions[index,:])
p.show()