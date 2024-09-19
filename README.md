# FNN_encounter
This is a neural network used for finding close encounter to the Sun using Gaia data.

# Usage
## For inference

1. Open terminal or cmd.
2. Navigate to folder containing run_prediction.py
3. `python ./run_predict.py --inputs_X 'path to input features of the network' --scaling_factor 'path to the rescaling file used for training' --save_results 'path to csv file that stores the predicted results'`

The saved weights and corresponding scaling factor is stored in the folder `saved_model`. The default values (path towards the scaling factor and model weights) are given for `--scaling_factor` and `saved_weight`.
