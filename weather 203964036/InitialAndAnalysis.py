import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def balance_target_class(df):
    # Plotting balance between positive and negative classes in target class
    weather['RainTomorrow'].value_counts().plot(kind='bar')
    plt.show()


def rows_duplicate(df):
    # Removing rows with all duplicate values
    print(df.drop_duplicates(keep=False, inplace=True))


def row_with_more_8_missing_values(df):
    # number of missing values for each row having more than 8 missing values
    tmp = len(df)
    print(df.shape)
    df.dropna(thresh=15, inplace=True)
    print(tmp - len(df))
    return df


def statistics_in_numerical(df):

    # find numerical variables
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    print('There are {} numerical variables\n'.format(len(numerical)))
    print('The numerical variables are :', numerical)

    # view summary statistics in numerical variables
    weather_statistics_numerical = (round(df[numerical].describe()))
    pd.set_option("display.max_columns", None)
    print(weather_statistics_numerical)


def dismantle_column_date(df):
    # parse the dates, currently coded as strings, into datetime format
    # i have divided the date column into 3 different columns as 'day', 'month' and 'year'
    # to know which day of month in a particular year has more rainfall.
    tmp = df
    tmp['Date'] = pd.to_datetime(df['Date'])
    tmp['Year'] = tmp['Date'].dt.year
    tmp['Month'] = tmp['Date'].dt.month
    tmp['Day'] = tmp['Date'].dt.day
    tmp.drop(["Date"], inplace=True, axis=1)

    return tmp


def target_without_nan(df):
    # remove all target with Nan
    df = df[df['RainTomorrow'].notna()]
    return df


def categorical_to_numerical(df):
    #Switching categorical values into numerical values

    WindGustDir_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindGustDir_bin'] = pd.Categorical(df.WindGustDir, ordered=False, categories=WindGustDir_list).codes+1

    WindDir9am_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir9am_bin'] = pd.Categorical(df.WindDir9am, ordered=False, categories=WindDir9am_list).codes+1

    WindDir3pm_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir3pm_bin'] = pd.Categorical(df.WindDir3pm, ordered=False, categories=WindDir3pm_list).codes+1

    RainToday_list = ['No', 'Yes']
    df['RainToday_bin'] = pd.Categorical(df.RainToday, ordered=False, categories=RainToday_list).codes+1

    RainTomorrow_list = ['No', 'Yes']
    df['RainTomorrow_bin'] = pd.Categorical(df.RainTomorrow, ordered=False, categories=RainTomorrow_list).codes

    df['Location_bin'] = pd.Categorical(df.Location, ordered=False).codes

    return df.drop(['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am', 'Location'], axis=1)


def fields_average(df):
    # numerical - Replace the missing values nan with the mean if the column is numerical .
    df['MinTemp'].fillna(round((df['MinTemp'].mean()), 2), inplace=True)
    df['MaxTemp'].fillna(round((df['MaxTemp'].mean()), 2), inplace=True)
    df['WindSpeed9am'].fillna(round((df['WindSpeed9am'].mean()), 2), inplace=True)
    df['Temp9am'].fillna(round((df['Temp9am'].mean()), 2), inplace=True)
    df['Humidity9am'].fillna(round((df['Humidity9am'].mean()), 2), inplace=True)
    df['WindSpeed3pm'].fillna(round((df['WindSpeed3pm'].mean()), 2), inplace=True)
    df['Rainfall'].fillna(round((df['Rainfall'].mean()), 2), inplace=True)
    df['Temp3pm'].fillna(round((df['Temp3pm'].mean()), 2), inplace=True)
    df['Humidity3pm'].fillna(round((df['Humidity3pm'].mean()), 2), inplace=True)
    df['WindGustSpeed'].fillna(round((df['WindGustSpeed'].mean()), 2), inplace=True)
    df['Pressure9am'].fillna(round((df['Pressure9am'].mean()), 2), inplace=True)
    df['Pressure3pm'].fillna(round((df['Pressure3pm'].mean()), 2), inplace=True)
    df['Evaporation'].fillna(round((df['Evaporation'].mean()), 2), inplace=True)


    # categorical - Replace the missing values nan with the mode if the column is categorical
    df['WindDir3pm_bin'].fillna((df['WindDir3pm_bin'].mode()[0]), inplace=True)
    df['WindGustDir_bin'].fillna((df['WindGustDir_bin'].mode()[0]), inplace=True)
    df['WindDir9am_bin'].fillna((df['WindDir9am_bin'].mode()[0]), inplace=True)
    df['RainToday_bin'].fillna((df['RainToday_bin'].mode()[0]), inplace=True)

    return df


def histogram_plot(df):
    #histogram_plot is method to see a scattering of values
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.Rainfall.hist(bins=10)
    fig.set_xlabel('Rainfall')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 2)
    fig = df.Evaporation.hist(bins=10)
    fig.set_xlabel('Evaporation')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 3)
    fig = df.WindSpeed9am.hist(bins=10)
    fig.set_xlabel('WindSpeed9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 4)
    fig = df.WindSpeed3pm.hist(bins=10)
    fig.set_xlabel('WindSpeed3pm')
    fig.set_ylabel('RainTomorrow')

    plt.show()


    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.MinTemp.hist(bins=10)
    fig.set_xlabel('MinTemp')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 2)
    fig = df.MaxTemp.hist(bins=10)
    fig.set_xlabel('MaxTemp')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 3)
    fig = df.Humidity9am.hist(bins=10)
    fig.set_xlabel('Humidity9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 4)
    fig = df.Humidity3pm.hist(bins=10)
    fig.set_xlabel('Humidity3pm')
    fig.set_ylabel('RainTomorrow')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.Pressure9am.hist(bins=10)
    fig.set_xlabel('Pressure9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 2)
    fig = df.Pressure3pm.hist(bins=10)
    fig.set_xlabel('Pressure3pm')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 3)
    fig = df.Cloud9am.hist(bins=10)
    fig.set_xlabel('Cloud9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 4)
    fig = df.Cloud3pm.hist(bins=10)
    fig.set_xlabel('Cloud3pm')
    fig.set_ylabel('RainTomorrow')

    plt.show()


def draw_boxplots(df):

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.boxplot(column='Rainfall')
    fig.set_title('')
    fig.set_ylabel('Rainfall')

    plt.subplot(2, 2, 2)
    fig = df.boxplot(column='Evaporation')
    fig.set_title('')
    fig.set_ylabel('Evaporation')

    plt.subplot(2, 2, 3)
    fig = df.boxplot(column='WindSpeed9am')
    fig.set_title('')
    fig.set_ylabel('WindSpeed9am')

    plt.subplot(2, 2, 4)
    fig = df.boxplot(column='WindSpeed3pm')
    fig.set_title('')
    fig.set_ylabel('WindSpeed3pm')

    plt.show()


def remove_row_outliers(df):
    df.drop(df[(df["Rainfall"] > 250)].index, inplace=True)
    df.drop(df[(df["Evaporation"] > 85)].index, inplace=True)
    df.drop(df[(df["WindSpeed9am"] > 80)].index, inplace=True)
    df.drop(df[(df["WindSpeed9am"] > 80)].index, inplace=True)

    return df


def check_outliers_with_zero(df):
    # Remove the rows that contain 4 zeros from the dataset
    df.drop(df[(df["Humidity9am"] == 0) | (df["Humidity3pm"] == 0)].index, inplace=True)


def modified_heatmap(df):
    # Shows correlations between columns
    data = df.select_dtypes(np.number)
    corr_matrix = round(data.corr(), 3)
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1)
    plt.show()


def scatter_plot(df):
    sns.scatterplot(data=df, x="Humidity9am", y="Humidity3pm", hue='Cloud9am', alpha=.6)
    plt.show()
    sns.scatterplot(data=df, x="Humidity3pm", y="Cloud9am", hue='Cloud3pm', alpha=.6)
    plt.show()
    sns.scatterplot(data=df, x="Cloud9am", y="Cloud3pm", hue='Sunshine', alpha=.6)
    plt.show()


def fill_nan_with_other_columns(df):
    # For these three values there is almost 50 percent data missing so,
    # we will supplement them with a connection to other columns
    df['Cloud9am'] = df.apply(
        lambda row: calculate_cloud9am(row['Humidity9am'], row['Humidity3pm'], row['Cloud9am']), axis=1)

    df['Cloud3pm'] = df.apply(
        lambda row: calculate_cloud3pm(row['Cloud9am'], row['Humidity3pm'], row['Cloud3pm']), axis=1)

    df['Sunshine'] = df.apply(
        lambda row: calculate_sunshine(row['Cloud9am'], row['Cloud3pm'], row['Sunshine']), axis=1)

    return df


def calculate_cloud9am(Humidity9am, Humidity3pm, Cloud9am):
    # method for calculate the fill nan with other columns
    if(Cloud9am == Cloud9am):
        return Cloud9am

    else:
        if(Humidity9am>=80 and Humidity3pm>=40):
               return 7.5
        if (Humidity3pm < 20):
                return 1.5
        else:
                return 4.5


def calculate_cloud3pm(Cloud9am, Humidity3pm, Cloud3pm):
    # method for calculate the fill nan with other columns

    if (Cloud3pm == Cloud3pm):
        return Cloud3pm

    else:
        if (Cloud9am >= 4 and Humidity3pm >= 20):
            return 6
        if (Cloud9am == 0):
            return 0
        else:
            return 3


def calculate_sunshine(Cloud9am, Cloud3pm, Sunshine):
    # method for calculate the fill nan with other columns

    if (Sunshine == Sunshine):
        return Sunshine

    else:
        if (0<=Cloud3pm<6 and 0<=Cloud9am<2):
            return 10
        if (Cloud3pm<5 and Cloud9am>=2):
            return 7.5
        if (Cloud3pm>=6 and Cloud9am<4):
            return 7.5
        if (Cloud3pm>=5 and 4<=Cloud9am<8):
            return 5
        if (Cloud3pm>=5 and Cloud9am>=8):
            return 2.5
        else: return 7.5



if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    balance_target_class(weather)
    rows_duplicate(weather)
    weather = row_with_more_8_missing_values(weather)
    statistics_in_numerical(weather)                                   #columns may contain outliers.
    weather = dismantle_column_date(weather)                           # 'day', 'month' and 'year'
    weather = target_without_nan(weather)
    weather = categorical_to_numerical(weather)                        # swipe categorical value to numerical value
    weather = fields_average(weather)                                  # filling values in columns by the average of the values in the column
    draw_boxplots(weather)                                             # plot the histograms to check distributions to find out if they are normal or skewed.
    histogram_plot(weather)
    weather = remove_row_outliers(weather)
    check_outliers_with_zero(weather)                                  # chek Exceeding the value range
    scatter_plot(weather)
    weather = fill_nan_with_other_columns(weather)
    modified_heatmap(weather)                                        # chek correlation

    weather.to_csv("weather_clean.csv", index=False)






