# (c) 2024 Patrick Nieman and Varun Sahay
# Plots example predictions and computes combined metrics for the baseline model

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from random import randrange
from loss import *

input=np.loadtxt("/Applications/CS230 Data/Export/inputExpanded.csv",delimiter=",")
output=np.load("/Applications/CS230 Data/Export/output.npy")

#Normalize except for modified one-hot rock path classification
metaStart=74
input[:,metaStart:]=input[:,metaStart:]-np.mean(input[:,metaStart:],axis=0)
input[:,metaStart:]=input[:,metaStart:]/np.std(input[:,metaStart:],axis=0)

#Isolate test data (data is pre-shuffled)
m=53960
testSplit=0.05
index=int(np.floor(m*(1-testSplit)))
input=input[index:,:]
output=output[index:,:]

model=tf.keras.models.load_model("/Applications/CS230 Data/model.keras",custom_objects={"responseLoss":responseLoss})
final=model.predict(input)

#Compute metrics
print(np.mean(ariasNP(output,final)))
print(np.mean(spectrumNP(output,final)))
smTotal=0
i=0
while i<output.shape[0]:
    smTotal+=smearedNP(output[i,:],final[i,:])
    i+=1
smTotal/=i
print(smTotal)

#Plot example time series
p.figure(1)
for i in range(16):
    nr=2
    index=randrange(0,m*testSplit-1)
    p.subplot(8,2*nr,nr*i+1)
    p.plot(output[index,:])
    p.subplot(8,2*nr,nr*i+2)
    p.plot(final[index,:])

#Plot response spectra
p.figure(2)
Tr=np.load("/Applications/CS230 Data/Export/spectraPeriods.npy")
p.plot(Tr,np.mean(responseModel.predict(final),axis=0),label="Predicted")
p.plot(Tr,np.mean(responseModel.predict(output),axis=0),label="Truth")
p.legend()
p.show()