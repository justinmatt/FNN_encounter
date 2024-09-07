import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Concatenate, Input
from tensorflow.keras.models import Model
#from models.model import RegressionModel
from preprocess.data_preprocessor import DataPreprocessor
from keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
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


def dN_dr(r):
    bin_edges = np.linspace(0, 1000, 101)
    f_ri, _ = np.histogram(r, bins=bin_edges, density=False)

    #F_ri = (f_ri - np.mean(f_ri))/np.std(f_ri) # normalise the histogram

    bin_index = np.digitize(r, bin_edges)

    return f_ri[bin_index - 1]


def custom_mae_loss_with_weights(y_true, y_pred, sample_weights):
    absolute_error = tf.abs(y_true - y_pred)  # Calculate absolute error
    weighted_error = absolute_error * sample_weights
    sum_weighted_error = tf.reduce_sum(weighted_error)  # Sum of weighted errors
    sum_sample_weights = tf.reduce_sum(sample_weights)  # Sum of sample weights
    
    weighted_average_error = sum_weighted_error / sum_sample_weights  # Weighted average of errors
    
    return weighted_average_error

def custom_metric(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

# Define the R-squared metric
def r_squared(y_true, y_pred):
    residual_sum_of_squares = K.sum(K.square(y_true - y_pred),axis=0)
    total_sum_of_squares = K.sum(K.square(y_true - K.mean(y_true)),axis=0)
    r2 = 1 - residual_sum_of_squares / (total_sum_of_squares + K.epsilon())
    return K.mean(r2)


def scatter_metric_dph(ypred, ytrue):
    dph_pred = ypred[:,0]
    dph_true = ytrue[:,0]

    scatter = K.mean(K.abs(dph_pred - dph_true))*1000

    return scatter

def scatter_metric_dph_std(ypred, ytrue):
    dph_std_pred = ypred[:,1]
    dph_std_true = ytrue[:,1]

    scatter = K.mean(K.abs(dph_std_pred - dph_std_true))

    return scatter

def scatter_metric_vph_std(ypred, ytrue):
    vph_std_pred = ypred[:,3]
    vph_std_true = ytrue[:,3]

    scatter = K.mean(K.abs(vph_std_pred - vph_std_true))

    return scatter

def scatter_metric_tph_std(ypred, ytrue):
    tph_std_pred = ypred[:,5]
    tph_std_true = ytrue[:,5]

    scatter = K.mean(K.abs(tph_std_pred - tph_std_true))

    return scatter



def scatter_metric_vph(ypred, ytrue):
    vph_pred = ypred[:,2]
    vph_true = ytrue[:,2]

    scatter = K.mean(K.abs(vph_pred - vph_true))

    return scatter

def scatter_metric_tph(ypred, ytrue):
    tph_pred = ypred[:,4]
    tph_true = ytrue[:,4]

    scatter = K.mean(K.abs(tph_pred - tph_true))

    return scatter


def scatter_metric_tph_dph_corr(ypred, ytrue):
    tph_dph_corr_pred = ypred[:,6]
    tph_dph_corr_true = ytrue[:,6]

    scatter = K.mean(K.abs(tph_dph_corr_pred - tph_dph_corr_true))

    return scatter

def scatter_metric_tph_vph_corr(ypred, ytrue):
    tph_vph_corr_pred = ypred[:,7]
    tph_vph_corr_true = ytrue[:,7]

    scatter = K.mean(K.abs(tph_vph_corr_pred - tph_vph_corr_true))

    return scatter

def scatter_metric_dph_vph_corr(ypred, ytrue):
    dph_vph_corr_pred = ypred[:,8]
    dph_vph_corr_true = ytrue[:,8]

    scatter = K.mean(K.abs(dph_vph_corr_pred - dph_vph_corr_true))

    return scatter






# Create an argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')


# Add an argument for the saved weights
parser.add_argument('--load_weight', type=str, default=None, help='Path to weights of the pre-trained model')

# Add argument for saving checkpoints
parser.add_argument('--save_checkpoints', type=bool, default=True, help='Save checkpoints during training')

# Add an argument for frac
parser.add_argument('--frac', type=float, default=0.20, help='Fraction of data to use for validation')

# Add an argument for random_state
parser.add_argument('--random_state', type=int, default=10, help='Random state for sampling')

# Add an argument for number of epochs
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model')

# Add an argument for number of layers
parser.add_argument('--layers', type=int, default=3, help='Number of layers in the neural network')

# Add an argument for number of nodes
parser.add_argument('--nodes', type=int, default=300, help='Number of nodes in each layer')

# Add an argument for activation function
parser.add_argument('--activation', type=str, default='relu', help='Activation function for hidden layers') 

# Add an argument for learning rate
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer')

# Add an argument for batch size
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')

# Add an argument for loading the model
parser.add_argument('--load_model', type=str, default=None, help='Path to the model to load')

# Add an argument for loading training data
parser.add_argument('--train_data', type=str, default='./data/train_data/train_inputs.csv', help='Path to training data (X)')
parser.add_argument('--train_labels', type=str, default='./data/train_data/train_labels.csv', help='Path to training labels (Y)')

# Parse the command line arguments
args = parser.parse_args()

# Access the value of frac
split_ratio = args.frac
rnd_state = args.random_state

#load training data
X_path = args.train_data
Y_path = args.train_labels

X = pd.read_csv(X_path, usecols=['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
       'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr'])


Y = pd.read_csv(Y_path, usecols = ['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std','tph_dph_corr','tph_vph_corr','dph_vph_corr'])


#convert to log
Y[['dph_med','dph_std','vph_med','vph_std','tph_std']] = Y[['dph_med','dph_std','vph_med','vph_std','tph_std']].apply(np.log)



xtrain, xval, ytrain, yval = train_test_split(X, Y, test_size=split_ratio, random_state=rnd_state)

epchs = args.epochs
btch_size = args.batch_size
drp_rate = 0.0
lr = args.lr
layers = args.layers
nodes = args.nodes
activation_func = args.activation
num_inputs = 13
num_outputs = 9

#rescale the inputs and outputs to mean=0 and std=1
rescaler = DataPreprocessor()

# rescaler.fit(xtrain, ytrain)

rescaler.fit(xtrain, ytrain[['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std']])

rescaler.save_scalers(f'./save_rescaler/encounter_params_full_5Mdata_MAE_kpc_{activation_func}_split_ratio_{split_ratio}_rnd_state_{rnd_state}_layers_{layers}_node_{nodes}_btch_size_{btch_size}_lr_{lr}_epchs_{epchs}.pkl')

#rescaled values stored for input to model
Xtrn_scl = rescaler.transform_input(xtrain)
ytrn_scl = rescaler.transform_output(ytrain[['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std']])
ytrn_scl = np.hstack((ytrn_scl, ytrain[['tph_dph_corr','tph_vph_corr','dph_vph_corr']].values))

#rescaled validation data
Xval_scl = rescaler.transform_input(xval)
Yval_scl = rescaler.transform_output(yval[['dph_med','dph_std','vph_med','vph_std','tph_med','tph_std']])
Yval_scl = np.hstack((Yval_scl, yval[['tph_dph_corr','tph_vph_corr','dph_vph_corr']].values))


EncModel = create_neural_network(num_inputs, layers, nodes, activation_func, num_outputs)    

if args.load_weight is not None:
    print('Loading saved weights for initialising')
    EncModel.load_weights(args.load_weight)

#train model
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./training_logs/encounter_params_full_5Mdata_MAE_tanh_{activation_func}_split_ratio_{split_ratio}_rnd_state_{rnd_state}_layers_{layers}_node_{nodes}_btch_size_{btch_size}_lr_{lr}_epchs_{epchs}', histogram_freq=1)
#save checkpoints
if args.save_checkpoints:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./saved_models/encounter_params_full_5Mdata_MAE_tanh_{activation_func}_split_ratio_{split_ratio}_rnd_state_{rnd_state}_layers_{layers}_node_{nodes}_btch_size_{btch_size}_lr_{lr}_epchs_{epchs}.keras",  # Filepath to save the checkpoint
        save_best_only=True,  # Save only the best model
        save_weights_only=True,  # Save only the model weights
        monitor='val_loss',  # Monitor validation loss (you can choose a different metric)
        mode='min',  # 'min' if you want to minimize the monitored metric, 'max' if you want to maximize it
        verbose=1  # Display a message when a checkpoint is saved
    )


EncModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='MAE', metrics=[r_squared, scatter_metric_dph, scatter_metric_vph, scatter_metric_tph, scatter_metric_dph_std, scatter_metric_vph_std, scatter_metric_tph_std, scatter_metric_tph_dph_corr, scatter_metric_tph_vph_corr, scatter_metric_dph_vph_corr])

print(f'training {nodes} node model with adam optimizer, lr={lr}, batch size={btch_size}, dropout rate={drp_rate}, epochs={epchs}')
#print(f'Model:{std_model.name}')

EncModel.fit(Xtrn_scl, ytrn_scl, batch_size=btch_size, epochs=epchs, validation_data=(Xval_scl, Yval_scl), verbose=0, callbacks=[tb_callback,checkpoint_callback])
