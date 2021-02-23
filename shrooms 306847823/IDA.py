import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

CRED = '\033[91m'
CEND = '\033[0m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'


def remove_outlier(data):
    Q1 = data.quantile(0.10)
    Q3 = data.quantile(0.90)
    IQR = Q3 - Q1
    data_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    print('Shape of data before handling outlier values: ', data.shape)
    print('Shape of data after handling outlier values: ', data_out.shape)
    return data_out


def getStatistics(df):
    # Data statistics
    print("*" * 60)
    print(CGREEN + 'Data statistics:' + CEND)
    print(CYELLOW + f'  Number of samples: {df.shape[0]}' + CEND)
    print(CYELLOW + f'  Number of attributes: {df.shape[1]}' + CEND)

    value_counts = df['classes'].value_counts()
    print(CYELLOW + f'  Edible mushrooms:    {value_counts["e"]}' + CEND)
    print(CYELLOW + f'  Poisonous mushrooms: {value_counts["p"]}' + CEND)
    print(
        CYELLOW + f'  Number of samples with missing values: {sum(df.apply(lambda x: sum(x.isnull().values), axis=1) > 0)}' + CEND)
    print(CYELLOW + '  Missing values per columns:' + CEND)

    null_col = dict(df.isnull().sum())
    for col in null_col.keys():
        print(f'   {col}, Number of missing values: {null_col[col]}')

    print(df.describe())

    str_df = df.select_dtypes(include=object)
    print(str_df.describe())


def performIDA(df):
    print(df.head())

    df.loc[(df['gill-spacing'] == -1), 'gill-spacing'] = 'c'
    df.loc[(df['gill-spacing'] == 0), 'gill-spacing'] = 'r'
    df.loc[(df['gill-spacing'] == 1), 'gill-spacing'] = 'd'

    getStatistics(df)

    #################################################
    # replace missing values with most frequent value if its bigger then 90% from the counts,
    # else, we will create new unique category - 'Missing'
    print("*" * 60)
    print(CGREEN + "Handle missing values:" + CEND)
    missing_values = dict(df.isna().any())
    for col in missing_values.keys():
        if missing_values[col]:
            count = df[col].value_counts()  # frequency of each category in the given column
            print(f"Values count for {col}:\n{count}")
            if count[count.idxmax()] / sum(count) > 0.9:
                df[col].fillna(count.idxmax(), inplace=True)
            else:
                df[col].fillna('Missing', inplace=True)

    print("Data after handle missing values:\n", df.head())

    # Encoding
    import Utils

    print("*" * 60)
    print(CGREEN + "Encoding features:" + CEND)
    data_cols = df.applymap(lambda x: isinstance(x, str)).all(0)
    data_cols = df.columns[data_cols]
    df2_encoder = OneHotEncoder(sparse=False, drop="first").fit(df[data_cols].drop("classes", axis=1))
    df2 = Utils.ohe_to_df(df2_encoder.transform(df[data_cols].drop("classes", axis=1)), df[data_cols])
    encoded_df = pd.concat([df.select_dtypes(include=[np.number]), df2], axis=1)
    df = encoded_df

    print("Data after encoding:\n", df.head())
    print(df.head())

    # Handle outliers
    print("*" * 60)
    print(CGREEN + "Handle outliers:" + CEND)
    # Remove outliers
    data_cleaned = remove_outlier(df)

    plt.savefig("Boxplot_after_remove_outliers.jpg")
    data_cleaned.to_csv('data/cleaned_mushrooms.csv')

    return data_cleaned
