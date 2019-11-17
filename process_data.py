import sys
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates()
    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates()
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = []
    for i in row:
        category_colnames.append(i.split('-',1)[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.split('-',expand=True)[1]
    categories[column] = categories[column].astype(int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
