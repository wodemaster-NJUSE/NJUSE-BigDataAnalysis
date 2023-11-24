# src/preprocess.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def preprocess_data(data):
    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
