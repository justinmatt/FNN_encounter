
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Input, Concatenate
from tensorflow.keras.models import Model
#from models.model import RegressionModel
from preprocess.data_preprocessor import DataPreprocessor


import numpy as np
from tensorflow.keras.models import load_model

import argparse 


def create_neural_network(inputs, num_layers, num_nodes, activation_function, num_outputs):
    # Define the input
    input_layer = Input(shape=(inputs,))
    
    # Add the hidden layers
    x = input_layer
    for _ in range(num_layers):
        x = Dense(num_nodes, activation=activation_function, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=10))(x)
    
    # Add the output layer with linear activation
    output_linear = Dense(num_outputs-3, activation='linear')(x)
    
    # Add the additional output layer with tanh activation
    output_tanh = Dense(3, activation='tanh')(x)
    
    # Concatenate the two output layers
    concatenated_outputs = Concatenate()([output_linear, output_tanh])
    
    # Define the model
    model = Model(inputs=input_layer, outputs=concatenated_outputs)
    
    return model


# Create an argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')


# Add an argument for the saved weights
parser.add_argument('--inputs_X', type=str, default=None, help='Path to file containing input data (X)')
parser.add_argument('--scaling_factor', type=str, default='./saved_models/rescale_factor/encounter_params_30-06_MAE_kpc_tanh_relu_split_ratio_0.1_rnd_state_10_layers_3_node_300_btch_size_512_lr_0.0001_epchs_10000.pkl', help='Path to rescaling factor')
parser.add_argument('--saved_weight', type=str, default='./saved_weights/encounter_params_full_5Mdata__30-06_MAE_kpc_tanh_relu_split_ratio_0.1_rnd_state_10_layers_3_node_300_btch_size_512_lr_0.0001_epchs_10000.keras', help='Path to saved weights')
parser.add_argument('--save', type=bool, default=True, help='Save the result to csv file if True')
parser.add_argument('--ouput_file', type=str, default=None, help='Path to save predictions')
# Parse the command line arguments
args = parser.parse_args()
#load data
xtest = args.inputs_X

xtest = xtest[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
       'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']]

rescaler = DataPreprocessor()
rescaler.load_scalers(args.scaling_factor)

xtest_scl = rescaler.transform_input(xtest)

# print(xtest_scl[0].reshape(1,13).shape)
layers = 3
nodes = 300
activation_func = 'relu'
num_inputs = 13
num_outputs = 9

EncModel = create_neural_network(num_inputs, layers, nodes, activation_func, num_outputs)  
#EncModel = make_model(num_inputs, node = nodes, activation=activation_func)

EncModel.load_weights(args.saved_weight)
# import time
# start = time.time()
ypred = rescaler.inverse_transform_output(EncModel.predict(xtest_scl))
# end = time.time()
# print('Time taken:',end-start)

ypred[:,0] = np.exp(ypred[:,0])
ypred[:,1] = np.exp(ypred[:,1])
ypred[:,2] = np.exp(ypred[:,2])
ypred[:,3] = np.exp(ypred[:,3])
ypred[:,5] = np.exp(ypred[:,5])

ypred_df = pd.DataFrame({
    'dph_med': ypred[:, 0]*1000,
    'dph_std': ypred[:, 1]*1000,
    'vph_med': ypred[:, 2],
    'vph_std': ypred[:, 3],
    'tph_med': ypred[:, 4]/1000,
    'tph_std': ypred[:, 5]/1000,
    'tph_dph_corr': ypred[:, 6],
    'tph_vph_corr': ypred[:, 7],
    'dph_vph_corr': ypred[:, 8]
})

if args.save:
    print('Saving results')
    ypred_df.to_csv(args.output_file)
else:
    print(ypred_df)

    # print('Parameters prediction bias and scatter\n')
    # print('scatter of true value (tph_med):', np.median(np.abs(ytest['tph_med'] - np.median(ytest['tph_med'])))/1000,'kyr')
    # print("Bias (tph_med):",np.median((ypred[:,4] - ytest['tph_med']))/1000,'kyr')
    # print('scatter (tph_med):',np.median(np.abs(ypred[:,4] - ytest['tph_med']))/1000,'kyr\n')

    # print('scatter of true value (dph_med):', np.median(np.abs(ytest['dph_med'] - np.median(ytest['dph_med'])))*1000,'pc')
    # print("Bias (dph_med):",np.median((ypred[:,0] - ytest['dph_med']))*1000, "pc")
    # print('scatter (dph_med):',np.median(np.abs(ypred[:,0] - ytest['dph_med']))*1000, "pc\n")

    # print('scatter of true value (vph_med):', np.median(np.abs(ytest['vph_med'] - np.median(ytest['vph_med']))))
    # print("Bias (vph_med):",np.median((ypred[:,2] - ytest['vph_med'])), "km/s")
    # print('scatter (vph_med):',np.median(np.abs(ypred[:,2] - ytest['vph_med'])), "km/s\n")

    # print("Parameters uncertainty prediction bias and scatter\n")
    # print('scatter of true value (tph_std):', np.median(np.abs(ytest['tph_std'] - np.median(ytest['tph_std'])))/1000,'kyr')
    # print("Bias (tph_std):",np.median(ypred[:,5] - ytest['tph_std'])/1000, "kyr")
    # print('scatter (tph_std):',np.median(np.abs(ypred[:,5] - ytest['tph_std']))/1000, "kyr\n")

    # print('scatter of true value (dph_std):', np.median(np.abs(ytest['dph_std'] - np.median(ytest['dph_std'])))*1000,'pc')
    # print("Bias (dph_std):",np.median(ypred[:,1] - ytest['dph_std'])*1000, "pc")
    # print('scatter (dph_std):',np.median(np.abs(ypred[:,1] - ytest['dph_std']))*1000, "pc\n")

    # print('scatter of true value (vph_std):', np.median(np.abs(ytest['vph_std'] - np.median(ytest['vph_std']))))
    # print("Bias (vph_std):",np.median(ypred[:,3] - ytest['vph_std']), "km/s")
    # print('scatter (vph_std):',np.median(np.abs(ypred[:,3] - ytest['vph_std'])), "km/s\n")

    # print("Correlation prediction bias and scatter\n")

    # print('Scatter of true value (tph_dph_corr):', np.median(np.abs(ytest['tph_dph_corr'] - np.median(ytest['tph_dph_corr']))))
    # print("Bias (tph_dph_corr):",np.median(ypred[:,6] - ytest['tph_dph_corr']))
    # print('scatter (tph_dph_corr):',np.median(np.abs(ypred[:,6] - ytest['tph_dph_corr'])),'\n')

    # print('Scatter of true value (tph_vph_corr):', np.median(np.abs(ytest['tph_vph_corr'] - np.median(ytest['tph_vph_corr']))))
    # print("Bias (tph_vph_corr):",np.median(ypred[:,7] - ytest['tph_vph_corr']))
    # print('scatter (tph_vph_corr):',np.median(np.abs(ypred[:,7] - ytest['tph_vph_corr'])),'\n')

    # print('Scatter of true value (dph_vph_corr):', np.median(np.abs(ytest['dph_vph_corr'] - np.median(ytest['dph_vph_corr']))))
    # print("Bias (dph_vph_corr):",np.median(ypred[:,8] - ytest['dph_vph_corr']))
    # print('scatter (dph_vph_corr):',np.median(np.abs(ypred[:,8] - ytest['dph_vph_corr'])),'\n')




