# Disaster Response Pipeline Project

## Installation
There are neccessary libraries to run the code here as following; json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle and warnings. The code should run with no issues using Python versions 3.7.1.

## Project Overview
This repository contains code for web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions
process_data.py: This code contains the data cleaning pipeline that loads the messages and categories datasets, merges the two datasets, cleans the data, stores it in a SQLite database.
train_classifier.py: This code contains a machine learning pipeline that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file.
run.py: This code contains the function to run the web app using Frask.
ETL Pipeline Preparation.ipynb: The code is for data cleaning pipeline preparation.
ML Pipeline Preparation.ipynb: The code is for ML Pipeline preparation.
disaster_messages.csv: Processing data sample
disaster_categories.csv: Processing data sample

## Running Instructions

1. Create 'data' folder and save process_data.py and csv files in the folder and create 'models' folder and save train_classifier. Create 'app' folder and save run.py in the folder.
2. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the following command in the app's directory to run your web app.
    `python run.py`
4. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree.
