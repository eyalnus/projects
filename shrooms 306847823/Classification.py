from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

CRED = '\033[91m'
CEND = '\033[0m'


def calcFeatureImportance(X, dt):
    features_list = X.columns.values
    feature_importance = dt.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(8, 7))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color="red")
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importance')
    plt.draw()
    # plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')
    plt.show()
    return features_list[sorted_idx[-13:]]


def performDT(df, preprocess=False):
    # Decision tree for mushrooms1.csv
    dt = tree.DecisionTreeClassifier(splitter='random', max_leaf_nodes=50, min_samples_split=15, max_depth=10,
                                     min_samples_leaf=2, random_state=10)

    if preprocess:
        # Drop NA values
        df = df.dropna()
        df = df.reset_index(drop=True)

        # Data encoding
        data_cols = df.applymap(lambda x: isinstance(x, str)).all(0)
        data_cols = df.columns[data_cols]
        label = LabelEncoder()
        for col in data_cols:
            df[col] = label.fit_transform(df[col])

    X = df.drop(['classes'], axis=1)
    y = df['classes']

    #################################################
    # Classification model

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    dt.fit(X_train, y_train)
    best_features = calcFeatureImportance(X, dt)

    if not preprocess:
        new_X = X[best_features]
        X_train, X_test, y_train, y_test = train_test_split(new_X, y, random_state=42, test_size=0.3)
        dt.fit(X_train, y_train)
        plt.close()
        fig = plt.figure(figsize=(35, 30))
        _ = tree.plot_tree(dt,
                           feature_names=X.columns,
                           class_names=['Edible', 'Poisonous'],
                           filled=True)

        fig.savefig("decision_tree.png")

    y_pred = dt.predict(X_test)

    print("Decision Tree Classifier report: \n\n", classification_report(y_test, y_pred))
    print(CRED + "Test Accuracy: {}%".format(round(dt.score(X_test, y_test) * 100, 2)) + CEND)

