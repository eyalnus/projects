import pandas as pd
import os

import IDA
import EDA
import Classification
PATH = os.path.join('data', 'mushrooms1.csv')

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    df = df.drop(columns='Unnamed: 0')

    # rename columns
    df = df.rename(
        columns={"0": "classes", "1": "cap-shape", "2": "cap-surface", "3": "cap-color", "5": "odor",
                 "6": "gill-attachment", "7": "gill-spacing", "10": "stalk-shape", "17": "veil-color",
                 "18": "ring-number", "21": "population", "22": "latitude", "23": "longitude"})

    ### IDA
    clean_df = IDA.performIDA(df)
    print(clean_df.head())

    ### EDA
    clean_df = EDA.performEDA(clean_df)

    ### Classification using decision tree
    print("Perform classification for the original dataset. ")
    Classification.performDT(df, True)

    print("Perform classification for the manipulated dataset. ")
    Classification.performDT(clean_df, False)
