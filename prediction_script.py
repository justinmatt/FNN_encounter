import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
#from models.model import RegressionModel
from preprocess.data_preprocessor import DataPreprocessor
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model


def create_neural_network(inputs, num_layers, num_nodes, activation_function, num_outputs):
    # Define the model
    model = tf.keras.Sequential()

    # Add the input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(inputs,)))

    # Add the hidden layers
    for _ in range(num_layers):
        model.add(Dense(num_nodes, activation=activation_function, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=10)))

    # Add the output layer
    model.add(Dense(num_outputs, activation='linear'))

    # Compile the model
    #model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    return model

#concatenated model
def make_model(input_shape, node, activation='relu', initializer = 'he_normal'):
    inputs = tf.keras.Input(shape=(input_shape,),name='input_layer')
    X = Dense(node, activation=activation, kernel_initializer=initializer, name='hidden1')(inputs)
    #X = Dropout(drp_rate)(X)
    X = Dense(node, activation=activation, kernel_initializer=initializer, name='hidden2')(X)
    #X = Dropout(drp_rate)(X)
    X = Dense(node, activation=activation, kernel_initializer=initializer, name='hidden3')(X)
    #X = Dense(64, activation='relu', kernel_initializer='he_normal', name='hidden3')(X)

    #outputs = Dense(3,activation='linear',name='outputs')(X)
    output_tph = Dense(1,activation='linear', name='tph')(X)
    output_dph = Dense(1,activation='linear', name='dph')(X)
    output_vph = Dense(1,activation='linear', name='vph')(X)

    tph_std = Dense(1,activation='linear', name='tph_std')(X)
    dph_std = Dense(1,activation='linear', name='dph_std')(X)
    vph_std = Dense(1,activation='linear', name='vph_std')(X)
    
    outputs = tf.concat([output_dph, dph_std, output_vph, vph_std, output_tph, tph_std],axis=1)
    # Functional Model
    Model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return Model


#load training data
X = pd.read_pickle('./data/corr_data/train_set/train_inputs_corr.pkl')
Y = pd.read_pickle('./data/corr_data/train_set/train_labels_corr.pkl')

X = X[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
       'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']]

Y = Y[['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std','tph_dph_corr','tph_vph_corr','dph_vph_corr']]

#convert to log
Y[['dph_med','dph_std','vph_med','vph_std','tph_std']] = Y[['dph_med','dph_std','vph_med','vph_std','tph_std']].apply(np.log)

#columns = ['ra','dec','parallax','pmra','pmdec','radial_velocity']

#return difference in cartesian coordinates of star and sun
#X = cart(X[columns], DX=True)

xval = X.sample(frac=0.20, random_state=10)
yval = Y.loc[xval.index]

#make training data
xtrain = X.drop(index=xval.index)
ytrain = Y.drop(index=yval.index)
#print(xtrain['dx0'])
#print(ytrain)


#rescale the inputs and outputs to mean=0 and std=1
rescaler = DataPreprocessor()

rescaler.fit(xtrain, ytrain)

#load test data
xtest = pd.read_pickle('./data/corr_data/test_set/test_inputs_corr.pkl')
ytest = pd.read_pickle('./data/corr_data/test_set/test_labels_corr.pkl')


xtest = xtest[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
       'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']]

ytest = ytest[['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std','tph_dph_corr','tph_vph_corr','dph_vph_corr']]

#convert to log
#ytest[['dph_med','vph_med']] = ytest[['dph_med','vph_med']].apply(np.log)



xtest_scl = rescaler.transform_input(xtest)


epchs = 10000
btch_size = 256
drp_rate = 0.0
lr = 1e-5
layers = 3
nodes = 300
activation_func = 'relu'
num_inputs = 13
num_outputs = 9

EncModel = create_neural_network(num_inputs, layers, nodes, activation_func, num_outputs)  
#EncModel = make_model(num_inputs, node = nodes, activation=activation_func)

EncModel.load_weights('./save_weights_corr/corr_X_enc_params_full_relu_split_ratio_0.2_rnd_state_10_layers_3_node_300_btch_size_256_lr_1e-05_epchs_10000.keras')

ypred = rescaler.inverse_transform_output(EncModel.predict(xtest_scl))

ypred[:,0] = np.exp(ypred[:,0])
ypred[:,1] = np.exp(ypred[:,1])
ypred[:,2] = np.exp(ypred[:,2])
ypred[:,3] = np.exp(ypred[:,3])
ypred[:,5] = np.exp(ypred[:,5])


print('Parameters prediction bias and scatter\n')
print("Bias (tph_med):",np.median((ypred[:,4] - ytest['tph_med'])),'yr')
print('scatter (tph_med):',np.median(np.abs(ypred[:,4] - ytest['tph_med'])),'yr\n')

print("Bias (dph_med):",np.median((ypred[:,0] - ytest['dph_med']))*1000, "pc")
print('scatter (dph_med):',np.median(np.abs(ypred[:,0] - ytest['dph_med']))*1000, "pc\n")

print("Bias (vph_med):",np.median((ypred[:,2] - ytest['vph_med'])), "km/s")
print('scatter (vph_med):',np.median(np.abs(ypred[:,2] - ytest['vph_med'])), "km/s\n")

print("Parameters uncertainty prediction bias and scatter\n")
print("Bias (tph_std):",np.median(ypred[:,5] - ytest['tph_std']), "yr")
print('scatter (tph_std):',np.median(np.abs(ypred[:,5] - ytest['tph_std'])), "yr\n")

print("Bias (dph_std):",np.median(ypred[:,1] - ytest['dph_std'])*1000, "pc")
print('scatter (dph_std):',np.median(np.abs(ypred[:,1] - ytest['dph_std']))*1000, "pc\n")

print("Bias (vph_std):",np.median(ypred[:,3] - ytest['vph_std']), "km/s")
print('scatter (vph_std):',np.median(np.abs(ypred[:,3] - ytest['vph_std'])), "km/s\n")

print("Correlation prediction bias and scatter\n")

print("Bias (tph_dph_corr):",np.median(ypred[:,6] - ytest['tph_dph_corr']))
print('scatter (tph_dph_corr):',np.median(np.abs(ypred[:,6] - ytest['tph_dph_corr'])),'\n')

print("Bias (tph_vph_corr):",np.median(ypred[:,7] - ytest['tph_vph_corr']))
print('scatter (tph_vph_corr):',np.median(np.abs(ypred[:,7] - ytest['tph_vph_corr'])),'\n')

print("Bias (dph_vph_corr):",np.median(ypred[:,8] - ytest['dph_vph_corr']))
print('scatter (dph_vph_corr):',np.median(np.abs(ypred[:,8] - ytest['dph_vph_corr'])),'\n')


print('Scatter of true value (tph_dph_corr):', np.median(np.abs(ytest['tph_dph_corr'] - np.median(ytest['tph_dph_corr']))))
print('Scatter of true value (tph_vph_corr):', np.median(np.abs(ytest['tph_vph_corr'] - np.median(ytest['tph_vph_corr']))))
print('Scatter of true value (dph_vph_corr):', np.median(np.abs(ytest['dph_vph_corr'] - np.median(ytest['dph_vph_corr']))))