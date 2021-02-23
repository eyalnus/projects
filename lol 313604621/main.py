import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
from io import StringIO
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image


def full_correlation_heatmap(df):
    """ Create and display a full correlation heatmap for data,
        including both numerical and categorical values.
    """
    plt.figure(figsize=(12, 8))
    color_map = plt.cm.get_cmap('rocket')
    reversed_color_map = color_map.reversed()
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    df = df.select_dtypes(np.number)
    matrix = np.triu(df.corr())
    sns.heatmap(df.corr(), annot=True, mask=matrix, cmap=reversed_color_map)
    plt.tight_layout()
    plt.show()


def outlier_box_plots(df):
    """ Create and display box plots for the purpose of finding outliers in the numerical value columns."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)

    fig = sns.boxplot(y=df['gameDuraton'])
    fig.set_title('')
    fig.set_ylabel('gameDuraton')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['blueWardPlaced'])
    fig.set_title('')
    fig.set_ylabel('blueWardPlaced')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['blueWardkills'])
    fig.set_title('')
    fig.set_ylabel('blueWardkills')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['blueTotalMinionKills'])
    fig.set_title('')
    fig.set_ylabel('blueTotalMinionKills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['blueJungleMinionKills'])
    fig.set_title('')
    fig.set_ylabel('blueJungleMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['blueTotalHeal'])
    fig.set_title('')
    fig.set_ylabel('blueTotalHeal')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['redWardPlaced'])
    fig.set_title('')
    fig.set_ylabel('redWardPlaced')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['redWardkills'])
    fig.set_title('')
    fig.set_ylabel('redWardkills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['redTotalMinionKills'])
    fig.set_title('')
    fig.set_ylabel('redTotalMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['redJungleMinionKills'])
    fig.set_title('')
    fig.set_ylabel('redJungleMinionKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['redTotalHeal'])
    fig.set_title('')
    fig.set_ylabel('redTotalHeal')

    plt.show()


def outlier_dist_plot(df):
    """ Create and display distribution plots for the purpose of finding outliers in the numerical value columns."""
    sns.set_theme(style="whitegrid")

    sns.displot(df, x="gameDuraton", kde=True)
    plt.show()

    sns.displot(df, x="blueWardPlaced", kde=True)
    plt.show()

    sns.displot(df, x="blueWardkills", kde=True)
    plt.show()

    sns.displot(df, x="blueTotalMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="blueJungleMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="blueTotalHeal", kde=True)
    plt.show()

    sns.displot(df, x="redWardPlaced", kde=True)
    plt.show()

    sns.displot(df, x="redWardkills", kde=True)
    plt.show()

    sns.displot(df, x="redTotalMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="redJungleMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="redTotalHeal", kde=True)
    plt.show()


def first_blood_hist(df):
    """ Create and display kde plot of the categorical column: FirstBlood."""
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstBlood'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBlood'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBlood                  blue won')
    plt.show()


def first_tower_hist(df):
    """ Create and display kde plot of the categorical column: FirstTower."""
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstTower'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstTower'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstTower                  blue won')
    plt.show()


def first_baron_hist(df):
    """ Create and display kde plot of the categorical column: FirstBaron."""
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstBaron'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBaron'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBaron                  blue won')
    plt.show()


def first_dragon_hist(df):
    """ Create and display kde plot of the categorical column: FirstDragon."""
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstDragon'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstDragon'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstDragon                  blue won')
    plt.show()


def duration_hist(df):
    """ Create and display kde plot of the game duration, in reference to the winning team."""
    sns.kdeplot((df['gameDuration'][df['win'] == 'Red']), color='red')
    sns.kdeplot((df['gameDuration'][df['win'] == 'Blue']), color='blue')
    plt.show()


def combined_columns_bar_plots_cat(df):
    """ Create and display bar count plots of the categorical columns in reference to the winner."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.countplot(x="FirstBlood", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstBlood')

    plt.subplot(2, 2, 2)
    sns.countplot(x="FirstBaron", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstBaron')

    plt.subplot(2, 2, 3)
    sns.countplot(x="FirstTower", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstTower')

    plt.subplot(2, 2, 4)
    sns.countplot(x="FirstDragon", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstDragon')

    plt.show()


def combined_columns_box_plots_num(df):
    """ Create and display box plots of the numerical columns by comparing winner vs. loser."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerJungleMinionKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerJungleMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserJungleMinionKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserJungleMinionKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['winnerWardPlaced'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerWardPlaced')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['loserWardPlaced'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserWardPlaced')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerWardKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerWardKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserWardKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserWardKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['winnerTotalMinionKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerTotalMinionKills')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['loserTotalMinionKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserTotalMinionKills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerTotalHeal'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerTotalHeal')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserTotalHeal'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserTotalHeal')

    plt.show()


def red_data_pair_plot(df):
    """ Create and display scatter pair plot of the red team numerical columns."""
    red_data = df.drop(['blueWardPlaced', 'blueWardKills', 'blueTotalMinionKills',
                        'blueJungleMinionKills', 'blueTotalHeal'], axis=1)
    sns.pairplot(red_data, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'])
    plt.show()


def blue_data_pair_plot(df):
    """ Create and display scatter pair plot of the blue team numerical columns."""
    blue_data = df.drop(['redWardPlaced', 'redWardKills', 'redTotalMinionKills',
                         'redJungleMinionKills', 'redTotalHeal'], axis=1)
    sns.pairplot(blue_data, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'])
    plt.show()


def opponent_pair_plot(df, kind):
    """ Create and display pair plot of the numerical columns of the red team vs. the blue team.
        choose the kind of plot (scatter/kde'...) by 'kind' :argument
    """
    sns.pairplot(data=df, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'], kind=kind,
                 y_vars=['redWardPlaced', 'redWardKills', 'redTotalMinionKills',
                         'redJungleMinionKills', 'redTotalHeal'],
                 x_vars=['blueWardPlaced', 'blueWardKills', 'blueTotalMinionKills',
                         'blueJungleMinionKills', 'blueTotalHeal'])
    plt.show()


def six_zero_columns(df):
    """ Create and display a distribution plot of the row in the dataset that have 6 zero values
        in regards to the game duration. Also print the number of said rows.
    """
    six_zeros = df[(df["blueWardPlaced"] == 0) & (df["redWardPlaced"] == 0) &
                   (df["blueWardkills"] == 0) & (df["redWardkills"] == 0) &
                   (df["blueJungleMinionKills"] == 0) & (df["redJungleMinionKills"] == 0)]
    print(len(six_zeros))
    sns.displot(six_zeros, x="gameDuraton", kde=True)
    plt.show()


def clear_six_zero_rows(df):
    """ Remove the rows that contain 6 zeros from the dataset."""
    df.drop(df[(df["blueWardPlaced"] == 0) & (df["redWardPlaced"] == 0) &
               (df["blueWardkills"] == 0) & (df["redWardkills"] == 0) &
               (df["blueJungleMinionKills"] == 0) & (df["redJungleMinionKills"] == 0)].index, inplace=True)


def clear_outliers(df):
    """ Remove the rows that contain blatant outliers from the dataset."""
    df.drop(df[(df["redWardPlaced"] > 190)].index, inplace=True)
    df.drop(df[(df["redWardkills"] > 100)].index, inplace=True)
    df.drop(df[(df["redTotalMinionKills"] > 1200)].index, inplace=True)
    df.drop(df[(df["redJungleMinionKills"] > 350)].index, inplace=True)
    df.drop(df[(df["redTotalHeal"] > 150000)].index, inplace=True)
    df.drop(df[(df["blueWardPlaced"] > 200)].index, inplace=True)
    df.drop(df[(df["blueWardkills"] > 100)].index, inplace=True)
    df.drop(df[(df["blueTotalMinionKills"] > 1190)].index, inplace=True)
    df.drop(df[(df["blueTotalHeal"] > 130000)].index, inplace=True)


def replace_missing_values(df):
    """ Replace the missing values (NaN) with the mean (if numerical) or mode (if categorical) of the column."""
    df['blueWardPlaced'].fillna(round((df['blueWardPlaced'].mean()), 2), inplace=True)
    df['blueWardkills'].fillna(round((df['blueWardkills'].mean()), 2), inplace=True)
    df['blueTotalMinionKills'].fillna(round((df['blueTotalMinionKills'].mean()), 2), inplace=True)
    df['blueJungleMinionKills'].fillna(round((df['blueJungleMinionKills'].mean()), 2), inplace=True)
    df['blueTotalHeal'].fillna(round((df['blueTotalHeal'].mean()), 2), inplace=True)

    df['redWardPlaced'].fillna(round((df['redWardPlaced'].mean()), 2), inplace=True)
    df['redWardkills'].fillna(round((df['redWardkills'].mean()), 2), inplace=True)
    df['redTotalMinionKills'].fillna(round((df['redTotalMinionKills'].mean()), 2), inplace=True)
    df['redJungleMinionKills'].fillna(round((df['redJungleMinionKills'].mean()), 2), inplace=True)
    df['redTotalHeal'].fillna(round((df['redTotalHeal'].mean()), 2), inplace=True)

    df['FirstBlood'].fillna((df['FirstBlood'].mode()[0]), inplace=True)
    df['FirstTower'].fillna((df['FirstTower'].mode()[0]), inplace=True)
    df['FirstBaron'].fillna((df['FirstBaron'].mode()[0]), inplace=True)
    df['FirstDragon'].fillna((df['FirstDragon'].mode()[0]), inplace=True)


def combine_numerical_columns(df):
    """ Replace the ten columns of the team vs. team achievements with ten winner vs. loser achievements.
        The winner columns will contain the value that the winning team reached
        and the opposite fot the loser columns
    """
    df['winnerWardPlaced'] = df.apply(
        lambda row: compare_winning_team_num(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    df['winnerWardKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redWardKills'], row['blueWardKills'], row['win']), axis=1)
    df['winnerTotalMinionKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redTotalMinionKills'], row['blueTotalMinionKills'], row['win']),
        axis=1)
    df['winnerJungleMinionKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redJungleMinionKills'], row['blueJungleMinionKills'], row['win']),
        axis=1)
    df['winnerTotalHeal'] = df.apply(
        lambda row: compare_winning_team_num(row['redTotalHeal'], row['blueTotalHeal'], row['win']), axis=1)

    df['loserWardPlaced'] = df.apply(
        lambda row: compare_losing_team_num(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    df['loserWardKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redWardKills'], row['blueWardKills'], row['win']), axis=1)
    df['loserTotalMinionKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redTotalMinionKills'], row['blueTotalMinionKills'], row['win']),
        axis=1)
    df['loserJungleMinionKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redJungleMinionKills'], row['blueJungleMinionKills'], row['win']),
        axis=1)
    df['loserTotalHeal'] = df.apply(
        lambda row: compare_losing_team_num(row['redTotalHeal'], row['blueTotalHeal'], row['win']), axis=1)

    df.drop(['blueWardPlaced', 'redWardPlaced', 'blueWardKills', 'redWardKills',
             'blueTotalMinionKills', 'redTotalMinionKills', 'blueJungleMinionKills', 'redJungleMinionKills',
             'blueTotalHeal', 'redTotalHeal'], axis=1, inplace=True)

    return df


def compare_winning_team_num(red_val, blue_val, winner):
    """ Helper method for 'combine_numerical_columns' to decide the winner value."""
    if winner == 'Blue':
        return blue_val
    else:
        return red_val


def compare_losing_team_num(red_val, blue_val, winner):
    """ Helper method for 'combine_numerical_columns' to decide the loser value."""
    if winner == 'Blue':
        return red_val
    else:
        return blue_val


def combine_categorical_columns(df):
    """ Replace the four columns of the 'First' achievements
        with four columns determining if the winning team was the achiever or not.
        If the winning team was the first to achieve the topic then the new value will be '1',
        if it was the losing team then the value will be '0'.
    """
    df['FirstBlood'] = df.apply(lambda row: compare_team_cat(row['FirstBlood'], row['win']), axis=1)
    df['FirstTower'] = df.apply(lambda row: compare_team_cat(row['FirstTower'], row['win']), axis=1)
    df['FirstBaron'] = df.apply(lambda row: compare_team_cat(row['FirstBaron'], row['win']), axis=1)
    df['FirstDragon'] = df.apply(lambda row: compare_team_cat(row['FirstDragon'], row['win']), axis=1)

    return df


def compare_team_cat(val, winner):
    """ Helper method for 'combine_categorical_columns' to decide the value."""
    if winner == val:
        return 1
    else:
        return 0


def combine_tower_baron_and_jungles(df):
    """ Add two columns:
        'jungleDiff' - the difference in the number of jungle minions killed by the team,
         if the value is positive, then the red team was the one that killed more blue minions
         if the value is negative, then the blue team was the one that killed more red minions.
        'towerBaron' - If the winning team was able to achieve both 'FirstTower' and 'FirstBaron' then the value is 2,
         if the winning team was able to achieve only one, either 'FirstTower' or 'FirstBaron' then the value is 1,
         if the winning team was not able to achieve both 'FirstTower' and 'FirstBaron' then the value is 0,
    """
    df.replace('Red', 0, inplace=True)
    df.replace('Blue', 1, inplace=True)

    df['jungleDiff'] = df.apply(lambda row: row['redJungleMinionKills'] - row['blueJungleMinionKills'], axis=1)
    df['towerBaron'] = df.apply(lambda row: compare_tower_baron(row['FirstTower'], row['FirstBaron'], row['win']),
                                axis=1)

    return df


def compare_tower_baron(cat1, cat2, winner):
    """ Helper method for 'combine_tower_baron_and_jungles' to decide the value."""
    if cat1 == cat2:
        if cat1 == winner:
            return 2
        else:
            return 0
    else:
        return 1


def stat_diff_winner_loser_num(df):
    """ Calculate and print the mean of the new numerical columns and their increase percentage."""
    win_val = df['winnerWardPlaced'].mean()
    lose_val = df['loserWardPlaced'].mean()
    print('winnerWardPlaced:        ', round(win_val, 2))
    print('loserWardPlaced:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerWardKills'].mean()
    lose_val = df['loserWardKills'].mean()
    print('winnerWardKills:         ', round(win_val, 2))
    print('loserWardKills:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerTotalMinionKills'].mean()
    lose_val = df['loserTotalMinionKills'].mean()
    print('winnerTotalMinionKills:  ', round(win_val, 2))
    print('loserTotalMinionKills:   ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerJungleMinionKills'].mean()
    lose_val = df['loserJungleMinionKills'].mean()
    print('winnerJungleMinionKills: ', round(win_val, 2))
    print('loserJungleMinionKills:  ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerTotalHeal'].mean()
    lose_val = df['loserTotalHeal'].mean()
    print('winnerTotalHeal:         ', round(win_val, 2))
    print('loserTotalHeal:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)


def stat_diff_winner_loser_cat(df):
    """ Calculate and print the mean of the new categorical columns and their increase percentage."""
    win_val = df['FirstBlood'].mean()
    lose_val = 1 - win_val
    print('winnerFirstBlood:        ', round(win_val, 2))
    print('loserFirstBlood:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstTower'].mean()
    lose_val = 1 - win_val
    print('winnerFirstTower:        ', round(win_val, 2))
    print('loserFirstTower:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstBaron'].mean()
    lose_val = 1 - win_val
    print('winnerFirstBaron:        ', round(win_val, 2))
    print('loserFirstBaron:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstDragon'].mean()
    lose_val = 1 - win_val
    print('winnerFirstDragon:       ', round(win_val, 2))
    print('loserFirstDragon:        ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)


def calculate_percent(win_val, lose_val):
    """ Helper method for the two 'stat_diff' functions to decide the increase percent."""
    increase_percent = ((win_val - lose_val) / lose_val) * 100
    print('increase percent:        ', round(increase_percent, 2), '%\n')


def gaussian_naive_bayes(data, target):
    """ Create a Gaussian Naive Bayes classification model, with train size: 80%"""
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train.values.ravel())

    return gnb_model


def accuracy_score_comparison(df):
    """ Compare and print the accuracy scores of the Gaussian Naive Bayes classification model
        using different pairs of features
    """
    target = df[['win']]

    print("'FirstTower' and 'FirstBaron'")
    data = df[['FirstTower', 'FirstBaron']]
    print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')

    print("'redJungleMinionKills' and 'blueJungleMinionKills'")
    data = df[['redJungleMinionKills', 'blueJungleMinionKills']]
    print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')

    print("'FirstTower' and 'jungleDiff'")
    data = df[['FirstTower', 'jungleDiff']]
    print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')

    print("'towerBaron' and 'jungleDiff'")
    data = df[['towerBaron', 'jungleDiff']]
    print("accuracy score: ", round(gnb_accuracy_score(data, target), 4), '\n')


def gnb_accuracy_score(data, target):
    """ Calculate the accuracy score of the Gaussian Naive Bayes classification model"""
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train.values.ravel())

    y_model = gnb_model.predict(x_test)

    return metrics.accuracy_score(y_test, y_model)


def gnb_filled_contour(data, target):
    """ Create the visualization of the decision boundaries of the Gaussian Naive Bayes classification model"""
    clf = gaussian_naive_bayes(data, target)

    x_min, x_max = data.loc[:, 'towerBaron'].min() - 1, data.loc[:, 'towerBaron'].max() + 1
    y_min, y_max = data.loc[:, 'jungleDiff'].min() - 1, data.loc[:, 'jungleDiff'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    z = np.argmax(z, axis=1)
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap='Set1', alpha=0.5)
    plt.colorbar()
    plt.clim(0, 5)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.xlabel('towerBaron')
    plt.ylabel('jungleDiff')
    return plt


def gnb_contour_with_failed_scatter(data, target):
    """ Overlay the visualization of the failed predictions of the GNBs classification model in a scatter plot
        on the decision boundaries plot
    """
    clf = gaussian_naive_bayes(data, target)

    ax = gnb_filled_contour(data, target)
    y_pred_full = clf.predict(data)
    y_pred_series = pd.Series(y_pred_full)
    target_series = pd.Series(target['win'])

    failed_data = []
    for i in range(len(y_pred_series)):
        if y_pred_series[i] != target_series[i]:
            failed_data.append(i)

    new_data = pd.DataFrame(columns=['towerBaron', 'jungleDiff', 'win'])
    i = 0
    for f in failed_data:
        new_data.loc[i] = ([data.loc[f]['towerBaron']] + [data.loc[f]['jungleDiff']] + [target['win'].loc[f]])
        i += 1

    tmp = new_data.sample(2000)
    sns.scatterplot(data=tmp, x='towerBaron', y='jungleDiff', hue='win', palette='Set1')
    fig = ax.gcf()
    fig.set_size_inches(12, 8)

    return ax


def decision_tree_clf_report(data, target, depth, report):
    """ Create a Decision Tree Classifier model, with train size: 80%,
        depth size determined by the 'depth' :argument.
        Also print the classification report if 'report' :argument is True
    """
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    clf = DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    if report:
        print(metrics.classification_report(y_test, y_pred))

    return clf


def tree_permutation_importance_plot(data, target, depth=None):
    """ Create and display the permutation importance of the Decision Tree Classifier model,
        depth size determined by the 'depth' :argument.
    """
    clf = decision_tree_clf_report(data, target, depth, report=True)
    result = permutation_importance(clf, data, target, n_repeats=10, random_state=0)
    plt.bar(range(len(data.columns)), result['importances_mean'])
    plt.xticks(ticks=range(len(data.columns)), labels=data.columns, rotation=90)
    plt.tight_layout()
    plt.show()


def tree_visualization(data, target, img_name, depth=None):
    """ Create and write to file the tree of the Decision Tree Classifier model,
        depth size determined by the 'depth' :argument,
        image name determined by the 'img_name' :argument.
    """
    clf = decision_tree_clf_report(data, target, depth, report=False)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                    feature_names=data.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(img_name)
    Image(graph.create_png())


def intro():
    """ Run the intro chapter methods."""
    df = pd.read_csv("lol5.csv")
    print(df.shape)
    print(df.dtypes)


def initial_data_analysis():
    """ Run the initial data analysis chapter methods."""
    df = pd.read_csv("lol5.csv")
    df.drop(['gameId'], axis=1, inplace=True)

    print(df.isnull().sum())
    replace_missing_values(df)

    first_baron_hist(df)
    first_tower_hist(df)
    first_blood_hist(df)
    first_dragon_hist(df)

    print(df.describe())

    outlier_box_plots(df)
    outlier_dist_plot(df)

    six_zero_columns(df)

    clear_six_zero_rows(df)
    clear_outliers(df)

    outlier_box_plots(df)
    outlier_dist_plot(df)

    df.rename(columns={
        'gameDuraton': 'gameDuration',
        'blueWardkills': 'blueWardKills',
        'redWardkills': 'redWardKills'
    }, inplace=True)

    df.to_csv('clean_lol.csv', index=False)


def exploratory_data_analysis():
    """ Run the exploratory data analysis chapter methods."""
    df = pd.read_csv("clean_lol.csv")
    full_correlation_heatmap(df)
    duration_hist(df)

    red_data_pair_plot(df)
    blue_data_pair_plot(df)
    opponent_pair_plot(df, 'scatter')
    opponent_pair_plot(df, 'kde')

    df = combine_numerical_columns(df)
    combined_columns_box_plots_num(df)
    stat_diff_winner_loser_num(df)

    df = combine_categorical_columns(df)
    stat_diff_winner_loser_cat(df)
    combined_columns_bar_plots_cat(df)

    df = pd.read_csv("clean_lol.csv")
    df = combine_tower_baron_and_jungles(df)
    full_correlation_heatmap(df)


def classification_model():
    """ Run the classification model chapter methods."""
    df = pd.read_csv("clean_lol.csv")
    df = combine_tower_baron_and_jungles(df)

    df.replace('Red', 0, inplace=True)
    df.replace('Blue', 1, inplace=True)

    accuracy_score_comparison(df)

    sns.catplot(x="towerBaron", y="jungleDiff", hue="win", kind="bar", data=df, palette=['red', 'blue'])
    plt.tight_layout()
    plt.show()

    sns.catplot(x="towerBaron", y="jungleDiff", hue="win", kind="point", data=df, palette=['red', 'blue'])
    plt.tight_layout()
    plt.show()

    sns.stripplot(x="towerBaron", y="jungleDiff", hue="win", alpha=.5, data=df, palette=['red', 'blue'])
    plt.tight_layout()
    plt.show()

    data = df[['towerBaron', 'jungleDiff']]
    target = df[['win']]

    p = gnb_contour_with_failed_scatter(data, target)
    p.show()

    print('=============== Baseline Tree ===============')
    df = pd.read_csv("lol5.csv")
    df.drop(['gameId'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.replace('Red', 0, inplace=True)
    df.replace('Blue', 1, inplace=True)
    target = df[['win']]
    df.drop(['win'], axis=1, inplace=True)
    tree_permutation_importance_plot(df, target)

    print('=============== Manipulated Data Tree ===============')

    df = pd.read_csv("clean_lol.csv")
    df = combine_tower_baron_and_jungles(df)
    df.replace('Red', 0, inplace=True)
    df.replace('Blue', 1, inplace=True)
    target = df[['win']]
    df.drop(['win'], axis=1, inplace=True)
    four_feature_df = df[['towerBaron', 'jungleDiff', 'FirstTower', 'redTotalHeal']]

    print('-------- 17 features - full depth --------')
    tree_permutation_importance_plot(df, target)
    tree_visualization(df, target, 'full_tree.png')

    print('-------- 4 features - full depth --------')
    tree_permutation_importance_plot(four_feature_df, target)
    tree_visualization(four_feature_df, target, 'four_feature_tree.png')

    print('-------- 4 features - depth 8 --------')
    tree_permutation_importance_plot(four_feature_df, target, 8)

    print('-------- 4 features - depth 4 --------')
    tree_permutation_importance_plot(four_feature_df, target, 4)

    print('-------- 4 features - depth 5 --------')
    tree_permutation_importance_plot(four_feature_df, target, 5)
    tree_visualization(four_feature_df, target, 'four_feature_tree_depth_5.png', 5)


if __name__ == '__main__':
    print('1: ')
    intro()

    print('2: ')
    initial_data_analysis()

    print('3: ')
    exploratory_data_analysis()

    print('4: ')
    classification_model()
