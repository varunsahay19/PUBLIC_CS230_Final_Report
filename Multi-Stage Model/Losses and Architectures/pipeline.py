# (c) 2024 Patrick Nieman and Varun Sahay
# Combines all elements of the multi-stage model into one system
# Generates predictions and evaluates on several metrics

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from random import randrange
from loss import *

def predict(input):
    inputReduced=input[:,74:]
    pgaModel=tf.keras.models.load_model('/Applications/CS230 Data/pgaModel.keras',custom_objects={"relativeBiabsolute":relativeBiabsolute})
    pgaModel.trainable=False
    pgas=pgaModel.predict(inputReduced)

    condensedModel=tf.keras.models.load_model('/Applications/CS230 Data/condensedModel.keras',custom_objects={"relativeBiabsolute":relativeBiabsolute})
    condensedModel.trainable=False
    condensed=condensedModel.predict(input)
    condensed/=np.max(np.abs(condensed),axis=1,keepdims=True)

    expansionModel=tf.keras.models.load_model('/Applications/CS230 Data/expansionModel.keras',custom_objects={"expansion":expansion})
    expansionModel.trainable=False
    expanded=expansionModel.predict(condensed)
    expanded/=np.max(np.abs(expanded),axis=1,keepdims=True)

    oscillationModelHF=tf.keras.models.load_model('/Applications/CS230 Data/oscillationModelHF.keras',custom_objects={"convLoss":convLossHF})
    oscillationModelHF.trainable=False
    oscillationModelLF=tf.keras.models.load_model('/Applications/CS230 Data/oscillationModelLF.keras',custom_objects={"convLoss":convLossLF})
    oscillationModelLF.trainable=False

    predictionHF=oscillationModelHF.predict(expanded)
    predictionHF/=np.max(np.abs(predictionHF),axis=1,keepdims=True)
    predictionLF=oscillationModelLF.predict(expanded)
    predictionLF/=np.max(np.abs(predictionLF),axis=1,keepdims=True)

    prediction=predictionHF+predictionLF
    prediction/=np.max(np.abs(prediction),axis=1,keepdims=True)
    final=prediction*pgas
    return condensed,expanded,predictionHF,predictionLF,final

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

#Predict time series and record intermediate steps
condensed,expanded,predictionHF,predictionLF,final=predict(input)

#Calculate evaluation metrics
ariasAvg=np.mean(ariasNP(output,final))
spectrumAvg=np.mean(spectrumNP(output,final))
smAvg=0
i=0
while i<output.shape[0]:
    smAvg+=smearedNP(output[i,:],final[i,:])
    i+=1
smAvg/=i
print("Arias error ",ariasAvg)
print("Spectrum error ",spectrumAvg)
print("Smeared motion error ",smAvg)
print("\nTotal error ",ariasAvg+spectrumAvg+smAvg)

#Plot example time series, including intermediate steps
p.figure(1)
for i in range(16):
    nr=6
    index=randrange(0,m*testSplit-1)

    p.subplot(8,2*nr,nr*i+1)
    p.plot(output[index,:])
    p.subplot(8,2*nr,nr*i+2)
    p.plot(condensed[index,:])
    p.subplot(8,2*nr,nr*i+3)
    p.plot(expanded[index,:])
    p.subplot(8,2*nr,nr*i+4)
    p.plot(predictionHF[index,:])
    p.subplot(8,2*nr,nr*i+5)
    p.plot(predictionLF[index,:])
    p.subplot(8,2*nr,nr*i+6)
    p.plot(final[index,:])

#Plot response spectra
p.figure(2)
Tr=np.load("/Applications/CS230 Data/Export/spectraPeriods.npy")
p.plot(Tr,np.mean(responseModel.predict(final),axis=0),label="Predicted")
p.plot(Tr,np.mean(responseModel.predict(output),axis=0),label="Truth")
p.legend()
p.show()