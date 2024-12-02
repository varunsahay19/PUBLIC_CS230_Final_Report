# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a linear model to predict full time series from a relative magnitude time history

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from loss import *
import architectures

tf.compat.v1.enable_eager_execution()

testSplit=0.05
HF=False
examples=53960
length=4096
#Load X and Y
x=np.load("/Applications/CS230 Data/Export/condensedOutputIntermediate%s.npy"%length)
y=np.load("/Applications/CS230 Data/Export/output.npy")
x=x[:examples,:]
y=y[:examples,:]
y/=np.max(np.abs(y),axis=1,keepdims=True)

#Split data (pre-shuffled)
m=x.shape[0]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[1:index,:]
yTrain=y[1:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

#Instantiate the model
model=architectures.oscillation()

#Complile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=convLossHF if HF else convLossLF,metrics=['mse'])

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/Checkpoints/modelConv-{epoch:02d}-{val_loss:.5f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=2, batch_size=64, validation_split=testSplit,callbacks=[checkpoint])
model.evaluate(xTest,yTest)
print(model.summary())
#Save model
model.save("/Applications/CS230 Data/oscillationModel%s.keras"%("HF" if HF else "LF"))
p.figure(1)
p.plot(history.history['val_loss'])
p.show()

