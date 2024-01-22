# Bank Customer Retention Predictor

## The Problem

Banking is a valuable business. At least one study has found that bankers contribute $1 of value for every $7 they extract from an economy. You can see why people want to be bankers!  But to be a banker, you have to have bank customers.  You have to attract new customers and retain the ones you have.  This repository contains a service that predicts the probability that a bank customer will be retained.

This project was inspired by and its dataset drawn from the [bank churn Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e1).

## The Repository

The code for this project is split across several different directories which are largely self-contained:

`data` - This directory contains data used for training.  It also contains data (`test.csv`) which can be used to generate submissions for the associated Kaggle competition.

`notebooks` - This directory contains a Jupyter notebook for exploring and preprocessing data. The notebook also contains code exploring several different models (Logistic Regression, a Random Forest Classifier) and tuning their hyperparameters. This folder contains its own Pipenv files, isolating data and model development work from the deployed prediction service.

`scripts` - This directory contains `preprocess.py`, which contains functions for preprocessing data, which are importedd by the prediction `service `and `train.py`.

`service` - Dockerized Flask app that exposes the predicition service via a `/predict` endpoint.

`kaggle` - Contains a Python script that utilizes the `/predict` endpoint to generate a Kaggle submission file.
