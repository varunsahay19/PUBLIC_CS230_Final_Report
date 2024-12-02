# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a 1D convolutional linear network to predict the response spectrum of a record

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from utilLoss import *
import architectures

tf.compat.v1.enable_eager_execution()
xpath="/Applications/CS230 Data/Export/spectraInput4096.npy"
ypath="/Applications/CS230 Data/Export/spectraOutput4096.npy"

#Load X and Y and expand dimensions
x=np.load(xpath)
y=np.load(ypath)

x=np.expand_dims(x,-1)
y=np.expand_dims(y,-1)

#Shuffle and split data
m=x.shape[0]
testSplit=0.05
shuffle=np.arange(m)
np.random.shuffle(shuffle)
x=x[shuffle,:,:]
y=y[shuffle,:,:]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[0:index,:,:]
yTrain=y[0:index,:,:]
xTest=x[index:,:,:]
yTest=y[index:,:,:]

#model=architectures.responseLinear30000(yTrain.shape[1])
model=architectures.responseLinearFC1024(yTrain.shape[1])

#Complile model
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss=elementAverageRelativeAbsolute,metrics=['mse'])

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/epoch-{epoch:02d}-val_loss-{val_loss:.2f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=32, batch_size=64, validation_split=0.05,callbacks=[checkpoint])
print(model.evaluate(xTest,yTest))

#Save model
model.save('/Applications/CS230 Data/responseModel.keras')
p.figure(1)
p.plot(history.history['val_loss'])
p.show()