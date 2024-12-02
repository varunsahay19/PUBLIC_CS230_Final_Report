# (c) 2024 Patrick Nieman and Varun Sahay
# Plots spectra predicted by response spectrum model

import numpy as np
from matplotlib import pyplot as p
import tensorflow as tf
from random import randrange
from utilLoss import *

final=False
length=1024
tf.keras.config.enable_unsafe_deserialization()
#Redefine losses to allow easy loading of keras model
responsePath="/Applications/CS230 Data/responseModel%s.keras"%(length if not final else "Final")
model=tf.keras.models.load_model(responsePath,custom_objects={"elementAverageRelativeAbsolute":elementAverageRelativeAbsolute})
model.trainable=False

g=386.09
casesEvaluated=256

#Load linear model and test data
Tr=np.load("/Applications/CS230 Data/Export/spectraPeriods.npy")
inputs=np.load("/Applications/CS230 Data/Export/spectraInput.npy")
inputs=inputs[:,0:length]
yTest=np.load("/Applications/CS230 Data/Export/spectraOutput.npy")

#Shuffle data and predict time histories
shuffle=np.random.shuffle(np.arange(yTest.shape[0]))
yTest=yTest[shuffle,:]
yTest=yTest[:,:casesEvaluated,:]
inputs=inputs[shuffle,:]
inputs=inputs[:,:casesEvaluated,:]
inputs=np.squeeze(inputs)
yTest=np.squeeze(yTest)
predictions=model.predict(inputs)

#Normalize for plotting 
yTest*=g
predictions*=g

#Take average of response spectra
SaTotal=np.sum(yTest,axis=0)
SaHatTotal=np.sum(predictions,axis=0)
SaTotal/=(casesEvaluated)
SaHatTotal/=(casesEvaluated)

#Plot response spectrum
p.figure(1)
p.plot(Tr,SaTotal,label="Ground truth")
p.plot(Tr,SaHatTotal,label="Predicted")
p.legend()
p.xlabel("Fundamental period (s)")
p.ylabel("Pseudospectral acceleration (in/s^2)")

#Plot example time series
p.figure(2)
for i in range(8):
    p.subplot(4,4,2*i+1)
    index=randrange(0,casesEvaluated)
    p.plot(yTest[index,:],label="Ground truth")
    p.legend()
    p.subplot(4,4,2*i+2)
    p.plot(predictions[index,:],label="Predicted")
p.figure(3)
p.plot(yTest[12,:])
p.show()