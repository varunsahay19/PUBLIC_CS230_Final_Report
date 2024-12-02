# (c) 2024 Patrick Nieman and Varun Sahay
# Plots records predicted by high- or low-frequency oscillation model

import numpy as np
from matplotlib import pyplot as p
import tensorflow as tf
from random import randrange
from loss import *

HF=False
length=4096
tf.keras.config.enable_unsafe_deserialization()

g=386.09
casesEvaluated=256

#Load linear model and test data
inputs=np.load("/Applications/CS230 Data/Export/condensedOutputIntermediate%s.npy"%length)
outputs=np.load("/Applications/CS230 Data/Export/output.npy")
outputs/=np.max(np.abs(outputs),axis=1,keepdims=True)

model=tf.keras.models.load_model("/Applications/CS230 Data/oscillationModel%s.keras"%("HF" if HF else "LF"),custom_objects={"convLoss":convLossHF if HF else convLossLF})
model.trainable=False

#Predict time histories
inputs=inputs[-casesEvaluated:,:]
outputs=outputs[-casesEvaluated:,:]
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