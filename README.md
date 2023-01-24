# Bio-Inspired Artificial Intelligence project
Solving SEIR ODE models for COVID-19 parameter prediction with evolutionary algorithms.

## Run SEIRD model
Create a new conda environment with `environment.yml`. Activate the environment.  
Enter the SEIRD_model folder and run `runner.py`. The folder is self-contained.  
`inspyred_utils.py` contains functions from the Inspyred Python library.  
`multiobjective.py` contains functions for running single- and multi-objective optimization.  
`plot_utils.py` contains the plotting function used to visualize optimization results.  
`read_data.py` contains the functions to retrieve, read and preprocess data.  
`seird_problem.py` contains the definition of the SEIRD model problem.  
`seird_plot_utils.py` contains the function for visualizing specific results.

## Run modified SEIRD model
In order to run the experiments for the SEIRD model use `python exp_seird_base.py`
This file contains the model for predicting the Infected, Recovered and Deceased for a given period of time.
We use the first 30 days as data for modelling the curve, and other 60 for comparing the real data with the predicted one.
Inside this file and in `seird_class.py` there are some comments blocks in order to run the experiments with NSGA-II or EA.
