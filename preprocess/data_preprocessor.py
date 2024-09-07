# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:05:08 2023

@author: ACER
"""

import pickle
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def fit(self, input_data, output_data):
        if output_data.ndim==1:
            output_data = output_data.reshape(-1,1)
        # Fit the scalers on the input and output data
        self.input_scaler.fit(input_data)
        self.output_scaler.fit(output_data)

    def transform_input(self, input_data):
        # Standardize the input data
        standardized_input = self.input_scaler.transform(input_data)
        return standardized_input

    def transform_output(self, output_data):
        if output_data.ndim==1:
            output_data = output_data.reshape(-1,1)
        # Standardize the output data
        standardized_output = self.output_scaler.transform(output_data)
        return standardized_output

    def inverse_transform_output(self, standardized_output):
        if standardized_output.ndim==1:
            standardized_output = standardized_output.reshape(-1,1)
        # Inverse transform the standardized output to get the original scale
        original_output = self.output_scaler.inverse_transform(standardized_output)
        return original_output

    def save_scalers(self, filename):
        # Save the scalers to a file using pickle
        with open(filename, 'wb') as file:
            pickle.dump((self.input_scaler, self.output_scaler), file)

    def load_scalers(self, filename):
        # Load the scalers from a file using pickle
        with open(filename, 'rb') as file:
            self.input_scaler, self.output_scaler = pickle.load(file)
            
            
            

