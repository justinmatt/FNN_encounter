# FNN_encounter
This is a neural network used for finding close encounter to the Sun using Gaia data.

# Usage
## For inference

1. Open terminal or cmd.
2. Navigate to folder containing run_prediction.py
3. python ./run_prediction.py --X 'path to input features of the network' --scaler_file 'name of the rescaling file used for training' --save_results 'path to csv file that stores the predicted results'