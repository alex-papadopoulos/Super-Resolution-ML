# hDSC_MS.py
# 2018 K. Fukami

## Hybrid Down-sampled skip-connection (DSC) multi-scale (MS) model.
## Author: Kai Fukami (Keio University, Florida State University, University of California, Los Angeles)

## Kai Fukami provides no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citations, please use the reference below:
# Ref: K. Fukami, K. Fukagata, & K. Taira,
#     "Super-resolution reconstruction of turbulent flows with machine learning,"
#     Journal of Fluid Mechanics, 2019
#
# The code is written for educational clarity and not for speed.
# -- version 1: Nov 21, 2018

# For making hDSC/MS model, the user has to install 'keras', 'numpy', 'pandas' and 'sklearn'.
from keras.layers import Input, Dense, Conv2D, merge, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
####

# Setting for Training datasets
datasetPrefix_omega = './DATABANK_2D_turbulence/2Dturbulence_omega/2Dturbulence_omega_'
datasetPrefix_omega_LR = './DATABANK_2D_turbulence/2Dturbulence_omega_downed32/2Dturbulence_omega_32_'
datasetSuffix = '.csv'
datasetSerial = np.arange(1,10001,1)
gridSetting = (128,128)
dim = 1  #Vorticity model
#dim = 2  #Velocity model


X_1 = np.zeros((len(datasetSerial),128,128,dim))
y = np.zeros((len(datasetSerial),128,128,dim))
for i in range(len(datasetSerial)):
    name = datasetPrefix_omega + '{0:06d}'.format(datasetSerial[i]) + datasetSuffix
    da = pd.read_csv(name, header=None, delim_whitespace=False)
    dataset = da.values
    u = dataset[:,:]
    y[i,:,:,0] = u[:,:]

    name = datasetPrefix_omega_LR + '{0:06d}'.format(datasetSerial[i]) + datasetSuffix
    da = pd.read_csv(name, header=None, delim_whitespace=False)
    dataset = da.values
    u = dataset[:,:]
    X_1[i,:,:,0] = u[:,:]
    print(i)


#Model
input_img = Input(shape=(128,128,dim))

#Down sampled skip-connection model
down_1 = MaxPooling2D((8,8),padding='same')(input_img)
x1 = Conv2D(32, (3,3),activation='relu', padding='same')(down_1)
x1 = Conv2D(32, (3,3),activation='relu', padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)

down_2 = MaxPooling2D((4,4),padding='same')(input_img)
x2 = merge([x1,down_2],mode='concat')
x2 = Conv2D(32, (3,3),activation='relu', padding='same')(x2)
x2 = Conv2D(32, (3,3),activation='relu', padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)

down_3 = MaxPooling2D((2,2),padding='same')(input_img)
x3 = merge([x2,down_3],mode='concat')
x3 = Conv2D(32, (3,3),activation='relu', padding='same')(x3)
x3 = Conv2D(32, (3,3),activation='relu', padding='same')(x3)
x3 = UpSampling2D((2,2))(x3)

x4 = merge([x3,input_img],mode='concat')
x4 = Conv2D(32, (3,3),activation='relu', padding='same')(x4)
x4 = Conv2D(32, (3,3),activation='relu', padding='same')(x4)

#Multi-scale model (Du et al., 2018)
layer_1 = Conv2D(16, (5,5),activation='relu', padding='same')(input_img)
x1m = Conv2D(8, (5,5),activation='relu', padding='same')(layer_1)
x1m = Conv2D(8, (5,5),activation='relu', padding='same')(x1m)

layer_2 = Conv2D(16, (9,9),activation='relu', padding='same')(input_img)
x2m = Conv2D(8, (9,9),activation='relu', padding='same')(layer_2)
x2m = Conv2D(8, (9,9),activation='relu', padding='same')(x2m)

layer_3 = Conv2D(16, (13,13),activation='relu', padding='same')(input_img)
x3m = Conv2D(8, (13,13),activation='relu', padding='same')(layer_3)
x3m = Conv2D(8, (13,13),activation='relu', padding='same')(x3m)

x_add = merge([x1m,x2m,x3m,input_img],mode='concat')
x4m = Conv2D(8, (7,7),activation='relu',padding='same')(x_add)
x4m = Conv2D(3, (5,5),activation='relu',padding='same')(x4m)

x_final = merge([x4,x4m],mode='concat')
x_final = Conv2D(dim, (3,3),padding='same')(x_final)
autoencoder = Model(input_img, x_final)
autoencoder.compile(optimizer='adam', loss='mse')


#Learning parameters
from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=None)
model_cb=ModelCheckpoint('./Model.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=20,verbose=1)
cb = [model_cb, early_cb]
history = autoencoder.fit(X_train,y_train,epochs=5000,batch_size=100,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History.csv',index=False)