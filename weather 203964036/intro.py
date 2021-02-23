import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cutting_columns_by_values(df):
    # Cutting numeric columns according to values of average, quarter of maximum and minimum values,
    # and for columns categories quantity, unique value and more...
    print((df.describe(include='object')))
    print((df.describe()))


def percent_missing_data_by_feature(df):
    # By percentages and numeric value some values are missing in each attribute
    data_na = (df.isnull().sum() / len(df)) * 100
    data_miss = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=data_miss.index, y=data_miss)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()


def data_missing_by_average(df):
    # Checking how data is null of average
    # We see there are some columns with null values.
    # Before we start pre-processing, let's find out which of the columns have maximum null values
    number_of_rows = df.shape[0]
    number_of_nan_in_column = df.isnull().sum(axis=0)
    print(
        pd.concat([number_of_nan_in_column, (number_of_nan_in_column / number_of_rows * 100).round(1)], axis=1).rename(
            columns={0: 'Number of NaN', 1: 'Number of NaN in %'}))


def data_types(df):
    #What types of values exist in the data
    print('Data types of this dataset :')
    print(list(df.dtypes.unique()))


def summary_categorical_numerical_values(df):
    # Summary of values by numerical and category division in the data
    categorical_type_columns = []
    numerical_type_columns = []
    for one_column_name in df:
        if 'object' in str(df[one_column_name].dtype):
            categorical_type_columns.append(one_column_name)
        elif 'float' in str(df[one_column_name].dtype):
            numerical_type_columns.append(one_column_name)

    print(categorical_type_columns)
    print(numerical_type_columns)
    print('Categorical type columns : {} / {}'.format(len(categorical_type_columns), len(df.columns)))
    print('Numerical type columns : {} / {}'.format(len(numerical_type_columns), len(df.columns)))

    # cardinality of categorical columns
    print()
    print("Categorical column cardinality :")
    for var in categorical_type_columns:
        print('{} : {} labels'.format(var, len(df[var].unique())))


if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")

    print('Size of weather data frame is :', weather.shape)
    print(weather.head())
    print(weather.info())

    summary_categorical_numerical_values(weather)
    data_types(weather)
    data_missing_by_average(weather)
    cutting_columns_by_values(weather)
    percent_missing_data_by_feature(weather)

