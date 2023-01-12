# machine_learning_DA_part_1
Code repository for Luke Howard's project on using Machine Learning for data assimilation.

There are 3 classes.

L96 contains methods for differentiating and integrating the Lorenz96 system for arbitrary N and F.

EnKF contains methods for assimilating data using the Ensemble Kalman Filter

Experiment objects each contain run settings, methods for generating truth and observations, and methods for
assimilating a set of observations. When a run is complete the object contains all input and results.
