
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Input, Concatenate
from tensorflow.keras.models import Model
#from models.model import RegressionModel
from preprocess.data_preprocessor import DataPreprocessor
import os

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
parser.add_argument('--saved_weight', type=str, default='./saved_models/encounter_params_full_5Mdata__30-06_MAE_kpc_tanh_relu_split_ratio_0.1_rnd_state_10_layers_3_node_300_btch_size_512_lr_0.0001_epchs_10000.h5', help='Path to saved weights')
parser.add_argument('--output_file', type=str, default=None, help='Path to save predictions')
# Parse the command line arguments
args = parser.parse_args()
#load data
print('Loading data...\n')
xtest = pd.read_csv(args.inputs_X)
#convert all columns to lower case
xtest.columns = xtest.columns.str.lower()
#store source_id in seperate variable

source_id = xtest['source_id'].values

#input columns
xtest = xtest[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
       'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']]

# Load the saved rescaling factor
rescaler = DataPreprocessor()
rescaler.load_scalers(args.scaling_factor)

# Rescale the input data
xtest_scl = rescaler.transform_input(xtest)

# Load the saved model
layers = 3
nodes = 300
activation_func = 'relu'
num_inputs = 13
num_outputs = 9

#create model frame
EncModel = create_neural_network(num_inputs, layers, nodes, activation_func, num_outputs)  

#load trained weights
EncModel.load_weights(args.saved_weight)

# Predict the output
ypred = rescaler.inverse_transform_output(EncModel.predict(xtest_scl))

#take antilog of the output (during training we took log of the output)
ypred[:,0] = np.exp(ypred[:,0])
ypred[:,1] = np.exp(ypred[:,1])
ypred[:,2] = np.exp(ypred[:,2])
ypred[:,3] = np.exp(ypred[:,3])
ypred[:,5] = np.exp(ypred[:,5])

#store the results in a dataframe
ypred_df = pd.DataFrame({
    'source_id': source_id,
    'dph_med': ypred[:, 0]*1000, #in pc
    'dph_std': ypred[:, 1]*1000, #in pc
    'vph_med': ypred[:, 2], #in km/s
    'vph_std': ypred[:, 3], #in km/s
    'tph_med': ypred[:, 4]/1000, #in Kyr
    'tph_std': ypred[:, 5]/1000, #in Kyr
    'tph_dph_corr': ypred[:, 6], 
    'tph_vph_corr': ypred[:, 7],
    'dph_vph_corr': ypred[:, 8]
})

# Save the results to a file if an output file is provided
if args.output_file is not None:
    print('Saving results..')
    ypred_df.to_csv(args.output_file, index=False)
else:
    print(ypred_df) #print the results





