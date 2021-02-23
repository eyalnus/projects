import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pydotplus
from sklearn import tree

"""
    FINAL PROJECT 
    SUBMITED BY : TSIBULSKY DAVD
                    ID: 309444065
"""

def decision_trea(dataFrame,target,critarion,depth):
    """
        create  decision trees function , draws the tree ,print report
    """
    y = dataFrame[target]
    X = dataFrame.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    clf = DecisionTreeClassifier(criterion=critarion, max_depth=depth, random_state=1)
    clf = clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # metrics = metrics(y_test,predicted,X_test)
    # print(metrics)

    feat_importance = clf.tree_.compute_feature_importances(normalize=False)
    coef = pd.DataFrame({'coef': feat_importance}, index=X.columns)
    coef = coef.sort_values(by='coef', ascending=False)
    print(coef)

    print(metrics.classification_report(y_test, prediction))

    fig = plt.figure(figsize=(15, 15))
    my_tree = tree.plot_tree(clf,feature_names=X.columns,filled=True)
    plt.savefig('tree5.png')
    # plt.show()


def bayes_plot(CS_DataFrame, className, model="gnb", spread=30):
    CS_DataFrame.dropna()
    colors = 'seismic'
    col1 = CS_DataFrame.columns[0]
    col2 = CS_DataFrame.columns[1]
    target = CS_DataFrame.columns[2]
    y = CS_DataFrame[target]
    X = CS_DataFrame.drop(target, axis=1)
    print(y)
    print(X)

    # Task 1/2-2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)

    prob = len(clf.classes_) == 2
    y_pred = clf.predict(X_test)

    # Task 1/2-5
    print(metrics.classification_report(y_test, y_pred))
    hueorder = clf.classes_

    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:, 1] - Z[:, 0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_) + 3)


    ypred = pd.Series(y_pred, name="prediction")
    predicted = pd.concat([X.reset_index(), y.reset_index(), ypred], axis=1)
    sns.scatterplot(data=predicted[::spread], x=col1, y=col2,
                    hue=target, hue_order=hueorder, palette=colors)

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    # plt.xlabel("bill_length_mm")
    # plt.ylabel("flipper_length_mm")

    plt.show()


if __name__ == "__main__":
    CS_DataFrame = pd.read_csv("csgo3.csv")
    #featurs data typy :
    org_df=CS_DataFrame.dropna() # print(CS_DataFrame.dtypes)

    # description :
    pd.set_option('display.max_columns', None)
    print(CS_DataFrame.describe())
    print(CS_DataFrame["round_winner"].describe())
    print(CS_DataFrame["round_winner"].unique())
    print(CS_DataFrame["map"].describe())
    print(CS_DataFrame["map"].unique())
    print(CS_DataFrame["bomb_planted"].describe())
    print(CS_DataFrame["bomb_planted"].unique())

    #cleansing data and save to csv file:
    for col in CS_DataFrame:
        dt = CS_DataFrame[col].dtype
        if dt == 'float64':
            CS_DataFrame[col] = CS_DataFrame[col].fillna(CS_DataFrame[col].mean().round())
        else:
            CS_DataFrame[col] = CS_DataFrame[col].fillna(CS_DataFrame[col].mode())
    CS_DataFrame['ct_health'] = np.where((CS_DataFrame.ct_health > 500), 500, CS_DataFrame.ct_health)
    CS_DataFrame['t_health'] = np.where((CS_DataFrame.t_health > 500), 500, CS_DataFrame.t_health)
    CS_DataFrame['t_players_alive'] = np.where((CS_DataFrame.t_players_alive > 5), 5, CS_DataFrame.t_players_alive)
    CS_DataFrame['ct_grenade_flashbang'] = np.where((CS_DataFrame.ct_grenade_flashbang > 5), 5, CS_DataFrame.ct_grenade_flashbang)
    CS_DataFrame['t_grenade_flashbang'] = np.where((CS_DataFrame.t_grenade_flashbang > 5), 5, CS_DataFrame.t_grenade_flashbang)
    CS_DataFrame['ct_grenade_smokegrenade'] = np.where((CS_DataFrame.ct_grenade_smokegrenade > 5), 5, CS_DataFrame.ct_grenade_smokegrenade)
    CS_DataFrame['t_grenade_smokegrenade'] = np.where((CS_DataFrame.t_grenade_smokegrenade > 5), 5, CS_DataFrame.t_grenade_smokegrenade)

    CS_DataFrame = pd.concat([CS_DataFrame, pd.get_dummies(CS_DataFrame['round_winner'],prefix="winner", prefix_sep='_')], axis=1)
    CS_DataFrame = pd.concat([CS_DataFrame, pd.get_dummies(CS_DataFrame['bomb_planted'], prefix="BP", prefix_sep='_')],axis=1)
    # save to csv file :
    # CS_DataFrame.to_csv('clean_data.csv', index=False)


    # matlop view settings  - colors ,axes , grid color , grid style , ticks, edges  ..
    colors = mpl.cycler('color',
                        ['orange', '3388BBb1', '9988DDb1',
                         'EECC55b1', '88BB44b1', 'FFBBBBb1'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')

    plt.rc('xtick', direction='out', color='k')
    plt.rc('ytick', direction='out', color='k')
    plt.rc('xtick.major', size=2, width=4, top=False)
    plt.rc('ytick.major', size=8, width=2, right=False)

    plt.rc('patch', edgecolor='r')

    #heat map :
    corr = CS_DataFrame.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, xticklabels=True, yticklabels=True, center=0,
                square=True, linewidths=.5)
    plt.title("Correlation Heat Map")
    plt.gcf().subplots_adjust(bottom=0.39)
    plt.show()

    #violin plots:
    plot = sns.violinplot(x="t_players_alive", y='t_health',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  t_health and the t_players_alive")
    plt.show()
    plot = sns.violinplot(x="ct_players_alive", y='ct_health',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  ct_health and the ct_players_alive")
    plt.show()
    plot = sns.violinplot(x="ct_helmets", y='ct_armor',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  ct_helmets and the ct_armor")
    plt.show()
    plot = sns.violinplot(x="t_helmets", y='t_armor',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  t_helmets and the t_armor")
    plt.show()
    plot = sns.violinplot(x="round_winner", y='t_armor',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  round_winner and the t_armor")
    plt.show()
    plot = sns.violinplot(x="round_winner", y='t_helmets',
                          data=CS_DataFrame)
    plot.set_title("Correlation between  round_winner and the t_helmets")
    plt.show()


    #scatter plot:
    CS_DataFrame.plot.scatter(x="t_armor", y="t_helmets")
    plt.title("Correlation between  t_armor and the t_helmets")
    plt.show()
    # CS_DataFrame.plot.scatter(x="t_grenade_smokegrenade", y="t_grenade_molotovgrenade")
    # plt.title("Correlation between  t_grenade_smokegrenade and the t_grenade_molotovgrenade")
    # plt.show()

    # plot = sns.violinplot(x="t_armor", y='t_helmets',
    #                       data=CS_DataFrame)
    # plot.set_title("Correlation between  ct_health and the ct_players_alive")
    # plt.show()


    #pivot table :
    my_pivot = CS_DataFrame.pivot_table(['winner_CT',"winner_T"], "map" ,aggfunc ="sum")
    print(my_pivot)
    my_pivot1 = CS_DataFrame.pivot_table(['winner_CT',"winner_T"], "bomb_planted", aggfunc="sum")
    print(my_pivot1)
    my_pivot2 = CS_DataFrame.pivot_table(['time_left'], ["map","bomb_planted"], aggfunc="mean")
    print(my_pivot2)
    my_pivot3 = CS_DataFrame.pivot_table(['winner_CT', "winner_T"], "ct_helmets", aggfunc="sum")
    print(my_pivot3)
    my_pivot4 = CS_DataFrame.pivot_table( "ct_armor",'winner_CT', aggfunc="mean")
    print(my_pivot4)
    my_pivot5 = CS_DataFrame.pivot_table(['winner_CT', "winner_T"], "ct_defuse_kits", aggfunc="sum")
    print(my_pivot5)
    my_pivot6 = CS_DataFrame.pivot_table("t_grenade_smokegrenade", 'winner_T', aggfunc="mean")
    print(my_pivot6)
    my_pivot6 = CS_DataFrame.pivot_table("t_grenade_molotovgrenade", 'winner_T', aggfunc="mean")
    print(my_pivot6)
    my_pivot7 = CS_DataFrame.pivot_table("ct_grenade_hegrenade", 'winner_CT', aggfunc="mean")
    print(my_pivot7)
    my_pivot8 = CS_DataFrame.pivot_table(["t_players_alive","ct_players_alive"], 'winner_T', aggfunc="mean")
    print(my_pivot8)



    #histograms:
    plt.hist([CS_DataFrame["ct_helmets"],CS_DataFrame["t_helmets"]], bins=20, alpha=0.5,
             histtype='stepfilled',label=['ct_helmets', 't_helmets'])
    plt.xlabel("helmets")
    plt.ylabel("amount")
    plt.legend(loc='upper center')
    plt.title("helmets histogramm")
    plt.show()

    # with pd.ExcelWriter('my_pivot.xlsx') as writer:
    #     my_pivot.to_excel(writer, sheet_name='pivot_table')



    """
        Machine learning: 
    """
    CS_DataFrame_learn = CS_DataFrame[["t_armor", "ct_armor", "ct_helmets", "t_helmets", "round_winner"]]

    CS_DataFrame_learn1 =  CS_DataFrame[["t_helmets","ct_defuse_kits","ct_players_alive","t_players_alive","round_winner"]]

    # graph = sns.pairplot(CS_DataFrame_learn1, hue='round_winner', height=1.5)
    # plt.show()

    y = CS_DataFrame['round_winner']
    x =  CS_DataFrame[["ct_armor","t_armor"]]

    #NaiveBase:
    bayes_plot(pd.concat([x, y], axis=1), 'round_winner', spread=1)

    #Decision Tree :
    bayes_plot(pd.concat([x, y], axis=1), 'round_winner',model= 5 , spread=1)

    CS_DataFrame3 = CS_DataFrame.drop(['map',"bomb_planted"], axis=1)
    decision_trea(CS_DataFrame_learn1,"round_winner","gini" ,5)
    # decision_trea(CS_DataFrame_learn1, "round_winner", "entropy", 3)




























