# Bio-Inspired Artificial Intelligence project
Solving SEIR ODE models for COVID-19 prediction with evolutionary algorithms

## How to run
In order to run the experiments for the SEIRD model use `python exp_seird_base.py`
This file contains the model for predicting the Infected, Recovered and Deceased for a given period of time.
We use the first 30 days as data for modelling the curve, and other 60 for comparing the real data with the predicted one.
Inside this file and in `seird_class.py` there are some comments blocks in order to run the experiments with NSGA-II or EA.