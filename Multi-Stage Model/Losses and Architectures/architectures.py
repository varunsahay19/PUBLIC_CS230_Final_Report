# (c) 2024 Patrick Nieman and Varun Sahay
# Architectures for record model


import tensorflow as tf

#Baseline models
def linear():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[128,128,128,192,256,384,512,1024,2048]
    #layers=[128,128,512]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(4096,activation="linear"))
    return model

def linear2():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[128,256,384,512,768]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(1024,activation="linear"))
    return model

#PGA models
def pga1():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[84,76,64,52,42,30,22,16,10,6,3]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(1,activation="linear",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    return model

def pga2():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[16,32,48,48,48,32,16,8,4]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.17))
    model.add(tf.keras.layers.Dense(1,activation="linear",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    return model

def pga3():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[16,96,128,128,32]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.17))
    model.add(tf.keras.layers.Dense(1,activation="linear",kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    return model

#Deconvolution models
def linearConv():
    model=tf.keras.models.Sequential()
    layers=[128,128,128,192,256,384,512,768,1200]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))

    #Add deconvolution layers
    model.add(tf.keras.layers.Reshape((-1, 1)))
    model.add(tf.keras.layers.Conv1DTranspose(3,5,strides=5,activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(3,5,strides=5,activation="linear"))
    model.add(tf.keras.layers.Conv1DTranspose(1,1,strides=1,activation="linear"))
    model.add(tf.keras.layers.Flatten())
    return model

#Early recurrent strategies
def recurrent():
    model=tf.keras.models.Sequential()
    denseLayers=[64,64,64,32]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(1024))
    model.add(tf.keras.layers.SimpleRNN(32,activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16,activation="relu")))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="linear")))
    model.add(tf.keras.layers.Flatten())
    return model

def recurrent2():
    model=tf.keras.models.Sequential()
    denseLayers=[64,64,96,128]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(30000))
    model.add(tf.keras.layers.SimpleRNN(128,activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    model.add(tf.keras.layers.Flatten())
    return model

def recurrentLSTM():
    model=tf.keras.models.Sequential()
    denseLayers=[96,96,64,64,128]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(1024))
    model.add(tf.keras.layers.LSTM(128,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.LSTM(64,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation='linear')))
    model.add(tf.keras.layers.Flatten())
    return model

def recurrentLSTMDecode():
    model=tf.keras.models.Sequential()
    denseLayers=[96,96,128,256]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(64))
    model.add(tf.keras.layers.LSTM(512,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.LSTM(256,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.Conv1DTranspose(256,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(128,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(128,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(64,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(32,7,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(16,9,1,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(1,11,2,padding="same",activation="linear"))
    model.add(tf.keras.layers.Flatten())
    return model

#Multi-stage oscillation deconv
def oscillation():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((-1, 1)))
    model.add(tf.keras.layers.Conv1DTranspose(16,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(32,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(64,5,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(32,7,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(16,9,2,padding="same",activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(1,11,1,padding="same",activation="linear"))
    model.add(tf.keras.layers.Flatten())
    return model

#Recurrent forms of first-stage temporal trend model
def recurrentCondensed():
    model=tf.keras.models.Sequential()
    denseLayers=[96,96,128]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(16))
    model.add(tf.keras.layers.LSTM(128,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="linear")))

    model.add(tf.keras.layers.Flatten())
    return model

def recurrentCondensedReducedInput():
    model=tf.keras.models.Sequential()
    denseLayers=[96,96,128]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.RepeatVector(16))
    model.add(tf.keras.layers.LSTM(128,return_sequences=True,go_backwards=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation="linear")))

    model.add(tf.keras.layers.Flatten())
    return model

def recurrentCondensedAlternative():
    model=tf.keras.models.Sequential()
    denseLayers=[96,128,256,256,256,128,92,64,48,32]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(16,activation="linear"))
    return model

#Chaotic interpolation
def expansion():
    model=tf.keras.models.Sequential()
    denseLayers=[32,64,96,128,128]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(128,activation="linear"))
    return model

#Reduced metadata linear form
def recurrentCondensedAlternativeReducedInput():
    model=tf.keras.models.Sequential()
    denseLayers=[16,48,84,128,256,256,256,128,92,64,48,32]
    for i in denseLayers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(16,activation="linear"))

    model.add(tf.keras.layers.Flatten())
    return model

#Response spectrum models
def responseLinear30000(output):
    #Build model
    #Add convolutional layers
    model=tf.keras.models.Sequential()
    for i in range(6):
        model.add(tf.keras.layers.Conv1D(6,7,strides=1,padding="same",activation='relu'))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))

    #Add fully-connected layers
    layers=[256,128,64,32,32,32]
    model.add(tf.keras.layers.Flatten())
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(output,activation="linear"))
    return model

def responseLinear1024(output):
    #Build model
    #Add convolutional layers
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(4,7,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=1))
    model.add(tf.keras.layers.Conv1D(7,11,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=1))
    model.add(tf.keras.layers.Conv1D(9,7,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv1D(13,7,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv1D(1,1,strides=1,padding="same",activation='relu'))
    #Add fully-connected layers
    layers=[128,128,128,64,64,32,32,32,32]
    model.add(tf.keras.layers.Flatten())
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(output,activation="linear"))
    return model

def responseLinearFC1024(output):
    model=tf.keras.models.Sequential()

    #Add fully-connected layers
    #model.add(tf.keras.layers.Lambda(lambda i: tf.keras.backend.squeeze(i,axis=-1)))
    model.add(tf.keras.layers.Flatten())
    layers=[992,768,512,384,256,128,92,64]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(output,activation="linear"))
    return model

def responseLinearFC4096(output):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(4,5,strides=2,padding="same",activation='relu'))
    model.add(tf.keras.layers.Conv1D(1,5,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.Flatten())
    #Add fully-connected layers
    layers=[992,768,512,384,256,128,92,64]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(output,activation="linear"))
    return model