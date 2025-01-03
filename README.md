# Machine Learning Augmented Data Assimilation
This repository contains the code for a ML-augmented data assimilation method, demonstrated on the Lorenz96 system. The code is implemented in four classes, each contained in a single .py file in the root directory.

## Processed data
Processed data for generating the figures is contained in the "/Processed Data" directory. Data is in netcdf, text, and numpy array format. Figures in the paper can be generated using code in the "tables_and_figs.ipynb" notebook. Within this directory are several subdirectories:

* /AlphaTuning
  * Runs using the augmented and CNN-only methods with alpha weighting factors from 0 to 1 in steps of 0.1. Files contain time series of RMSE.
* /Sensitivity
  * RMSE time series for the base AllObs run and all 9 sensitivity runs.
* /Training
  * Results from NN hyperparameter tuning
* /Other
  * Rank histogram data, forecast data, and raw error data

## Classes and Run Scripts
All custom classes are defined in the /src directory.

"EnKF.py" contains methods for implementing the Ensemble Kalman Filter on the Lorenz 96 system. Options for covariance localization and covariance inflation are included. "L96.py" contains methods that specify the Lorenz system and allow for it to be integrated forward in time. "Experiment.py" contains the bulk of the codebase. It has as an attribute an Xarray DataSet object containing time series data of the true evolution of the system, synthetic observations, ensemble forecasts, and ensemble analyses. Its attributes specify the run settings, including localization, inflation, observation error standard deviation, ensemble size, and observation time step. It also contains methods for generating true trajectories and synthetic observations and for performing assimilation of those observations. Lastly, "NeuralNet.py" contains methods for training and using a simple CNN for assimilation. It implements a cyclic padding procedure and has methods for transforming provided input into the size and shape needed by the neural network.

Run scripts are included for running EnKF sensitivity experiments, NN training, and running validation experiments, i.e. experiments that begin halfway into the time series. These are in the "/RunScripts" directory. Yaml files containing settings for the experiments are also included as examples. Sensitivity runs can be initiated by calling "sensitivity.py settings.yml" from the command line, and validation runs can be initiated by calling "RunValidationAssim.py settings.yml", with "settings.yml" being the path of the settings file in each case.

The RawResults directory is intended for raw results generated by the experiment run scripts above in ".pickle" format. Subdirectories for Alpha tuning experiments, sensitivity runs, and validation runs are included. Due to file sizes, only the base and s9 sensitivity runs are included. Any others can be generated using the run scripts described above with appropriate settings. The included files allow for NN training, or for validation runs with any combination of settings to be performed.

Lastly, scripts for processing raw results are included in the "/Processing Scripts" directory. "postprocess_shap.py" computes SHAP values from 1000 samples. "postprocess_forecast.py" computes forecast accuracy using analyses from the three compared methods as initial conditions. Both save their results to .npy files.


## Dependencies
* Numpy
* Scipy
* Keras
* Tensorflow
* Xarray
* Pickle
* Tqdm
* Matplotlib
* Netcdf4