import matplotlib.pyplot as plt
import pydotplus as pydotplus
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


def load_data(data_path):
    """load the dataset"""
    data = pd.read_csv(data_path)
    return data


def category_to_num(data):
    """change category to ordinal numerical"""
    category_numb = {"density": {"low": 1, "medium": 2, "high": 3, "very high": 4},
                     "kind": {"white": 0, "red": 1}}
    data = data.replace(category_numb)
    return data


def mean_to_val(data):
    """enter an average value to empty rows"""
    data['volatile acidity'] = data['volatile acidity'].fillna(data['volatile acidity'].mean())
    data['total sulfur dioxide'] = data['total sulfur dioxide'].fillna(data['total sulfur dioxide'].mean())
    data['pH'] = data['pH'].fillna(data['pH'].mean())
    return data


def remove_outliers(data, category, val1, val2=None):
    """for each outlier in the data take the val and drop them"""
    data.drop(data[(data[category] > val1)].index, inplace=True)
    if val2 != None:
        wine.drop(data[(data[category] < val2)].index, inplace=True)
    return data


def fill_kind(kind, volatile_acidity, total_sulfur_dioxide):
    """fill the row with the correct kind according to total sulfur dioxide and volatile acidity"""
    if kind == 0 or kind == 1:
        return kind
    else:
        if total_sulfur_dioxide:
            return 0
        if total_sulfur_dioxide < 50:
            return 1
        if volatile_acidity > 0.4:
            return 1
        else:
            return 0


def gnb(df, x_name, y_name, target_name):
    """create GNB and prediction"""
    data = df[[x_name, y_name]]
    target = df[[target_name]]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train.values.ravel())

    y_model = gnb_model.predict(x_test)
    print(x_name, '&', y_name)
    print(metrics.accuracy_score(y_test, y_model))
    print(metrics.classification_report(y_test, y_model.round(), digits=3))

def decision_Tree(data, target_name, depth=None):
    """create a decision tree from dataset"""
    target = data[[target_name]]
    data.drop(['quality'], inplace=True, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(x_train, y_train)
    # Evaluate the model with out-of-sample test set
    y_pred = tree.predict(x_test)
    # Classification report
    print(metrics.classification_report(y_test, y_pred.round(), digits=3))
    return tree


def tree_png(file_path, model_tree):
    """create a decision tree png from dataset"""
    dot_data = StringIO()
    export_graphviz(model_tree, out_file=dot_data, filled=True, rounded=True, feature_names=tmp.columns,
                    class_names=['4', '5', '6', '7'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(file_path)
    Image(graph.create_png())


if __name__ == '__main__':
    wine = load_data('wine2.csv')

    # print the firs 5 rows to check it
    print(wine.head().to_string())

    # see data statistics
    print(wine.describe().to_string())

    # missing value in the data
    print(wine.isnull().sum())
    print(wine.shape)

    # drop the sample column
    wine.drop(['sample'], inplace=True, axis=1)
    print(wine.head().to_string())

    # categorical to numerical
    wine = category_to_num(wine)

    # corollitions by heatmap
    X = wine.iloc[:, 0:20]
    y = wine.iloc[:, -1]
    corrmat = wine.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10, 10))
    map1 = sns.heatmap(wine[top_corr_features].corr(), annot=True, cmap="YlGnBu")
    plt.show()

    # decied what to put in empty values in kind column
    sns.scatterplot(data=wine, x='total sulfur dioxide', y='volatile acidity', hue='kind')
    plt.show()
    wine['kind'] = wine.apply(lambda row: fill_kind(row['kind'], row['volatile acidity'],
                                                    row['total sulfur dioxide']), axis=1)
    print(wine.isnull().sum())

    # decied what to put in empty values in density column
    sns.scatterplot(data=wine, x='alcohol', y='residual sugar', hue='density')
    plt.show()
    sns.pairplot(data=wine, hue='density', kind='kde', y_vars=['residual sugar'], x_vars=['alcohol'])
    plt.show()
    wine = wine[wine.density.notnull()]
    print(wine.isnull().sum())
    print(len(wine))

    # enter an average value instaed of deleting it
    wine = mean_to_val(wine)
    print(wine.isnull().sum())
    print(wine.shape)

    # find outliers in the data and drop them
    # sns.boxplot(x=wine['fixed acidity'])
    # plt.show()
    # wine = remove_outliers(wine, 'fixed acidity', 13, 4.5)
    # sns.boxplot(x=wine['fixed acidity'])
    # plt.show()
    #
    # sns.boxplot(x=wine['volatile acidity'])
    # plt.show()
    # wine = remove_outliers(wine, 'volatile acidity', 0.9)
    # sns.boxplot(x=wine['volatile acidity'])
    # plt.show()
    #
    # sns.boxplot(y=wine['citric acid'])
    # plt.show()
    # wine = remove_outliers(wine, 'citric acid', 4.5)
    # sns.boxplot(x=wine['citric acid'])
    # plt.show()
    #
    # sns.boxplot(x=wine['residual sugar'])
    # plt.show()
    # wine = remove_outliers(wine, 'residual sugar', 21)
    # sns.boxplot(x=wine['residual sugar'])
    # plt.show()
    #
    # sns.boxplot(x=wine['chlorides'])
    # plt.show()
    # wine = remove_outliers(wine, 'chlorides', 0.2)
    # sns.boxplot(x=wine['chlorides'])
    # plt.show()
    #
    # sns.boxplot(x=wine['free sulfur dioxide'])
    # plt.show()
    # wine = remove_outliers(wine, 'free sulfur dioxide', 220, 80)
    # sns.boxplot(x=wine['free sulfur dioxide'])
    # plt.show()
    #
    # sns.boxplot(x=wine['total sulfur dioxide'])
    # plt.show()
    # wine = remove_outliers(wine, 'total sulfur dioxide', 260)
    # sns.boxplot(x=wine['total sulfur dioxide'])
    # plt.show()
    #
    # sns.boxplot(x=wine['pH'])
    # plt.show()
    # wine = remove_outliers(wine, 'pH', 3.8, 2.8)
    # sns.boxplot(x=wine['pH'])
    # plt.show()
    #
    # sns.boxplot(x=wine['sulphates'])
    # plt.show()
    # wine = remove_outliers(wine, 'sulphates', 1.2)
    # sns.boxplot(x=wine['sulphates'])
    # plt.show()
    #
    # sns.boxplot(x=wine['alcohol'])
    # plt.show()
    # wine = remove_outliers(wine, 'alcohol', 14)
    # sns.boxplot(x=wine['alcohol'])
    # plt.show()
    #
    # sns.boxplot(x=wine['quality'])
    # plt.show()
    # wine = remove_outliers(wine, 'quality', 7, 4)
    # sns.boxplot(x=wine['quality'])
    # plt.show()
    # print(wine.shape)
    # wine.to_csv('clean_wine.csv', index=False)

    # question 3
    # load clean data
    wine2 = load_data('clean_wine.csv')

    # check corroletione of target class
    X = wine2.iloc[:, 0:20]
    y = wine2.iloc[:, -1]
    corrmat = wine2.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10, 10))
    map2 = sns.heatmap(wine2[top_corr_features].corr(), annot=True)
    plt.show()

    # quality column corroletione\influence
    # sns.scatterplot(data=wine2, x='free sulfur dioxide', y='alcohol', hue='quality')
    # plt.show()
    # sns.pairplot(data=wine2, hue='quality', kind='kde', y_vars=['alcohol'], x_vars=['free sulfur dioxide'])
    # plt.show()
    #
    # sns.scatterplot(data=wine2, x='free sulfur dioxide', y='citric acid', hue='quality')
    # plt.show()
    # sns.pairplot(data=wine2, hue='quality', kind='kde', y_vars=['citric acid'], x_vars=['free sulfur dioxide'])
    # plt.show()
    #
    # sns.scatterplot(data=wine2, x='alcohol', y='citric acid', hue='quality')
    # plt.show()
    # sns.pairplot(data=wine2, hue='quality', kind='kde', y_vars=['citric acid'], x_vars=['alcohol'])
    # plt.show()

    # join column for better corroletione
    # sns.scatterplot(data=wine2, x='citric acid', y='volatile acidity', hue='quality')
    # plt.show()
    #
    # sns.scatterplot(data=wine2, x='citric acid', y='total sulfur dioxide', hue='free sulfur dioxide')
    # plt.show()

    a = 'free sulfur dioxide'   # corroletione to quality 0.69
    b = 'alcohol'               # corroletione to quality 0.44
    c = 'citric acid'           # corroletione to quality 0.36
    d = 'density'               # corroletione to quality 0.29
    e = 'volatile acidity'      # corroletione to quality 0.25
    f = 'chlorides'             # corroletione to quality 0.22
    # testing influence by scatterplot
    # sns.scatterplot(data=wine2, x=a, y=b, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=a, y=c, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=a, y=d, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=a, y=e , hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=a, y=f, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=b, y=c, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=b, y=d, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=b, y=e, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=b, y=f, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=c, y=d, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=c, y=e, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=c, y=f, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=d, y=e, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=d, y=f, hue='quality')
    # plt.show()
    # sns.scatterplot(data=wine2, x=e, y=f, hue='quality')
    # plt.show()

    # test joining columns by deffrence/join/multiply
    tmp = wine2

    # tmp['af'] = tmp[a] - tmp[f] #no improve 0.69
    # tmp['ae'] = tmp[a] - tmp[e] #no improve 0.69
    # tmp['ad'] = tmp[a] - tmp[d] #improve to 0.70
    # tmp['ac'] = tmp[a] + tmp[c] #no improve 0.69
    # tmp['ab'] = tmp[a] * tmp[b]  #improve 0.77

    # tmp['bc'] = tmp[b] + tmp[c]  #improve 0.54
    # tmp['bd'] = tmp[b] - tmp[d]  #worst 0.4
    # tmp['be'] = tmp[b] - tmp[e]  #improve 0.47
    # tmp['bf'] = tmp[b] + tmp[f]  #no improve 0.44

    # tmp['cd'] = tmp[c] - tmp[d]  #improve 0.44
    # tmp['ce'] = tmp[c] - tmp[e]  #improve 0.38
    # tmp['cf'] = tmp[c] - tmp[f]  #improve 0.37

    # tmp['de'] = tmp[d] * tmp[e]  #improve -0.33
    # tmp['df'] = tmp[d] - tmp[f]  #no improve -0.29
    #
    # tmp['ef'] = tmp[e] * tmp[f]  #improve -0.26
    #
    # tmp['cdef'] = tmp['cd'] + tmp['ef']  #improve 0.44
    # tmp['cedf'] = tmp['ce'] - tmp['df']  #improve 0.44
    # tmp['bcde'] = tmp['bc'] + tmp['de']  #no improve 0.54
    # tmp['bcdf'] = tmp['bc'] - tmp['df']  #worst 0.49
    # tmp['bcef'] = tmp['bc'] + tmp['ef']  # no improve 0.54
    # tmp['becd'] = tmp['be'] + tmp['cd']  #improv 0.5

    # best results after checking
    tmp['alcohol * free sulfur dioxide'] = tmp['alcohol'] * tmp['free sulfur dioxide']
    tmp['volatile acidity - citric acid'] = tmp['volatile acidity'] - tmp['citric acid']

    # see corroletione on heatmap after joining the columns
    X = tmp.iloc[:, 0:20]
    y = tmp.iloc[:, -1]
    corrmat = tmp.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10, 10))
    map3 = sns.heatmap(tmp[top_corr_features].corr(), annot=True)
    plt.tight_layout()
    plt.show()
    sns.scatterplot(data=tmp, x='alcohol * free sulfur dioxide', y='volatile acidity - citric acid', hue='quality')
    plt.show()

    # question 4
    # 1.GNB
    gnb(tmp, 'alcohol * free sulfur dioxide', 'volatile acidity - citric acid', 'quality')
    # gnb(tmp, 'free sulfur dioxide', 'volatile acidity - alcohol', 'quality')
    # gnb(tmp, 'free sulfur dioxide', '(alcohol - volatile acidity) - (density - citric acid)', 'quality')

    # 2
    # decision Tree for the original data
    wine.dropna(inplace=True)
    decision_Tree(wine, 'quality', 8)
    # decision Tree for cleanup data
    tree = decision_Tree(tmp, 'quality', 8)
    # create a decision tree png
    tree_png('tree.png', tree)
