import numpy as np
import matplotlib.pyplot as plt
import pydotplus as pydotplus
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.tree import export_graphviz
from io import StringIO
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import InitialAndAnalysis


def check_correlation_gnb(df):
    # Checking which columns are most relevant to sending for the algorithm gnb
    df['Cloud'] = ((df['Cloud9am'] * df['Cloud3pm']))        #0.31
    df['Humidity9_Rainfall'] = ((df['Humidity9am'] + df['Rainfall'])/2)                            # 0.28
    df['RainHum3'] = ((df['RainToday_bin'] * df['Humidity3pm']))                      # plus 0.45
    df['differenceTemp'] = ((df['MaxTemp'] - df['MinTemp']))                                     # 0.31
    df['RainHum3_Pressure3pm'] = ((df['RainHum3'] * df['Pressure3pm']))
    df['RainHum3+Pressure3pm'] = ((df['RainHum3'] - df['Pressure3pm']))                #  0.47
    df['Cloud_RainHum3_Pressure3pm'] = ((df['Cloud'] + df['RainHum3_Pressure3pm']))    #0.48
    df['WindGustSpeed_Humidity9_Rainfall'] = ((df['WindGustSpeed'] + df['Humidity9_Rainfall']))  #0.36

    return df.drop(['Location_bin', 'Year', 'Month', 'Day', 'MaxTemp', 'MinTemp', 'Cloud9am', 'Cloud3pm' , 'RainToday_bin', 'Humidity3pm', 'Humidity9am', 'Rainfall', 'Pressure3pm', 'WindGustSpeed'], axis=1)

    return df


def check_correlation_decision_tree(df):

    # Checking which columns are most relevant to sending for the algorithm decision_tree
    df['Cloud'] = ((df['Cloud9am'] * df['Cloud3pm']))
    df['Sunshine_Humidity3pm'] = ((df['Sunshine'] + df['Humidity3pm'])/ 2 )
    df['differenceHum3Temp3'] = ((df['Humidity3pm'] + df['Temp3pm']) / 2)
    df['Sunshine_differenceCloud'] = ((df['Sunshine'] - df['Cloud']))
    df['Humidity9_Rainfall'] = ((df['Humidity9am'] + df['Rainfall'])/2)
    df['Cloud_Humidity9_Rainfall'] = ((df['Cloud'] * df['Humidity9_Rainfall']))
    df['RainHum3'] = ((df['RainToday_bin'] * df['Humidity3pm']))
    df['RainHum3_Pressure9am'] = ((df['RainHum3'] - df['Pressure9am']))
    df['WindGustSpeed_RainHum3_Pressure9am'] = ((df['WindGustSpeed'] + df['RainHum3_Pressure9am']))
    df['differenceTemp'] = ((df['MaxTemp'] - df['MinTemp']))
    df['RainHum3_Pressure3pm'] = ((df['RainHum3'] * df['Pressure3pm']))
    df['RainHum3+Pressure3pm'] = ((df['RainHum3'] - df['Pressure3pm']))
    df['Cloud_RainHum3_Pressure3pm'] = ((df['Cloud'] + df['RainHum3_Pressure3pm']))
    df['WindGustSpeed_Humidity9_Rainfall'] = ((df['WindGustSpeed'] + df['Humidity9_Rainfall']))
    df['Cloud_Temp'] = ((df['Cloud'] - df['differenceTemp']))
    df['WindGustSpeed_Humidity9_Rainfall_differenceTemp'] = ((df['WindGustSpeed_Humidity9_Rainfall'] - df['differenceTemp']))


    return df


def pairplot(df):
    # Tests see correlations between columns and target class
    data = df[['Sunshine', 'Year', 'Month', 'RainTomorrow_bin']]
    sns.pairplot(data=data, hue='RainTomorrow_bin')
    plt.show()

    # data = df[['Cloud9am', 'Cloud3pm', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['MaxTemp', 'MinTemp', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Pressure3pm', 'Pressure9am', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Humidity9am', 'Sunshine', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Humidity3pm', 'Sunshine', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Humidity3pm', 'Cloud3pm', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Humidity3pm', 'WindGustSpeed','RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['Humidity3pm', 'RainToday_bin', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()

    # data = df[['differenceRainHum3', 'Pressure9am', 'RainTomorrow_bin']]
    # sns.pairplot(data=data, hue='RainTomorrow_bin')
    # plt.show()


def modified_heatmap(df):
    data = df.select_dtypes(np.number)
    corr_matrix = round(data.corr(), 3)
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1)
    plt.show()


def accuracy_score_comparison(df):
    #Checking which columns will bring the highest success for this algorithm
    target = df[['RainTomorrow_bin']]

    print("'Cloud_RainHum3_Pressure3pm' and 'WindGustSpeed_Humidity9_Rainfall'")     # 0.80
    data = df[['Cloud_RainHum3_Pressure3pm', 'WindGustSpeed_Humidity9_Rainfall']]
    print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')


    # print("'Cloud_RainHum3_Pressure3pm' and 'differenceTemp'")
    # data = df[['Cloud_RainHum3_Pressure3pm', 'differenceTemp']]
    # print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')
    #
    # print("'Cloud_RainHum3_Pressure3pm' and 'Sunshine'")
    # data = df[['Cloud_RainHum3_Pressure3pm', 'Sunshine']]
    # print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')
    #
    # print("'Cloud_RainHum3_Pressure3pm' and 'Evaporation'")
    # data = df[['Cloud_RainHum3_Pressure3pm', 'Evaporation']]
    # print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')
    #
    # print("'Cloud_RainHum3_Pressure3pm' and 'Pressure9am'")
    # data = df[['Cloud_RainHum3_Pressure3pm', 'Pressure9am']]
    # print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')
    #
    #
    # print("'WindGustSpeed_Humidity9_Rainfall' and 'Cloud'")
    # data = df[['WindGustSpeed_Humidity9_Rainfall', 'Cloud']]
    # print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')
    #


def gnb_accuracy_score(data, target):
    #calculate the accuracy score of the gnb model
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train.values.ravel())

    y_model = gnb_model.predict(x_test)

    return metrics.accuracy_score(y_test, y_model)


def baseline_tree():
    # Initial data for the basic tree
    weather_baseline = pd.read_csv("weather1.csv")
    weather_baseline.dropna(inplace=True)
    weather_baseline = InitialAndAnalysis.dismantle_column_date(weather_baseline)
    weather_baseline = InitialAndAnalysis.categorical_to_numerical(weather_baseline)
    target = weather_baseline[['RainTomorrow_bin']]
    weather_baseline.drop(['RainTomorrow_bin'],  inplace=True,  axis=1)
    tree_plot(weather_baseline, target)


def manipulated_data_tree(df):
    df = df[['WindGustSpeed_Humidity9_Rainfall', 'Cloud_RainHum3_Pressure3pm', 'RainHum3_Pressure3pm', 'Pressure3pm', 'Sunshine', 'differenceHum3Temp3']]
    return df


def tree_plot(data, target):

    clf = report_decision_tree(data, target)
    result = permutation_importance(clf, data, target, n_repeats=10, random_state=0)
    plt.bar(range(len(data.columns)), result['importances_mean'])
    plt.xticks(ticks=range(len(data.columns)), labels=data.columns, rotation=90)
    plt.tight_layout()
    plt.show()


def report_decision_tree(df, target):
    # Create a decision tree classifier model
    # train size is 80%, and testing is 20%
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=1)
    clf = DecisionTreeClassifier(max_depth=8)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(metrics.classification_report(y_test, y_pred))
    return clf


def tree_visual_png(df, target ,img_name):
    # Print the decision tree as a stylish PNG image
    clf = report_decision_tree(df, target)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                    feature_names=df.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(img_name)
    Image(graph.create_png())


if __name__ == '__main__':

    weather = pd.read_csv("weather_clean.csv")
    # weather = check_correlation_gnb(weather)
    # accuracy_score_comparison(weather)

    # pairplot(weather)
    # modified_heatmap(weather)

    weather = check_correlation_decision_tree(weather)
    # baseline_tree()
    tmp = manipulated_data_tree(weather)
    target = weather[['RainTomorrow_bin']]
    tree_visual_png(tmp, target, 'tree.png')
