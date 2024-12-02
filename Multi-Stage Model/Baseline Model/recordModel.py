# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a linear model to predict acceleration time histories from earthquake metadata
# Baseline end-to-end model

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from loss import *
import architectures

length=4096
examples=2560

tf.compat.v1.enable_eager_execution()
xpath="/Applications/CS230 Data/Export/inputExpanded.csv"
ypath="/Applications/CS230 Data/Export/output.npy"

testSplit=0.05

normalize=True

#Load X and Y
x=np.loadtxt(xpath,delimiter=",")
y=np.load(ypath)
x=x[:examples,:]
y=y[:examples,:]
y=y[:,0:length]

#Normalize except for modified one-hot rock path classification
metaStart=74
x[:,metaStart:]=x[:,metaStart:]-np.mean(x[:,metaStart:],axis=0)
x[:,metaStart:]=x[:,metaStart:]/np.std(x[:,metaStart:],axis=0)

if normalize:
    y/=np.max(np.abs(y),axis=1,keepdims=True)

#Shuffle and split data
m=x.shape[0]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[1:index,:]
yTrain=y[1:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

#Instantiate the model
model=architectures.linear()

#Complile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=responseLoss,metrics=[spectrum,arias,smearedMotion])

class BatchLossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchLossHistory, self).__init__()
        self.lBatch=[]
        self.lSmeared=[]
        self.lSpectrum=[]
        self.lArias=[]

    def on_train_batch_end(self,_,logs=None):
        self.lBatch.append(logs.get('loss'))
        self.lSmeared.append(logs.get('smeared_motion'))
        self.lSpectrum.append(logs.get('spectrum'))
        self.lArias.append(logs.get('arias'))
batchLoss=BatchLossHistory()

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/Checkpoints/modelConv-{epoch:02d}-{val_loss:.5f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=16, batch_size=32, validation_split=testSplit,callbacks=[checkpoint,batchLoss])
model.evaluate(xTest,yTest)
print(model.summary())

#Save model
model.save("/Applications/CS230 Data/model.keras")

#Plot training history metrics
p.figure(1)
p.plot(history.history['val_loss'])
p.figure(2)
p.plot(batchLoss.lBatch)
p.xlabel("Mini-batch")
p.ylabel("Training loss")
p.figure(3)
p.plot(batchLoss.lSmeared,label="Smeared")
p.plot(batchLoss.lSpectrum,label="Spectrum")
p.plot(batchLoss.lArias,label="Arias")
p.xlabel("Mini-batch")
p.ylabel("Training loss")
p.legend()
p.show()

