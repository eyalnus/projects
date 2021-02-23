import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree

from io import StringIO
from IPython.display import Image
import pydotplus as pyd


class helper_functions:
    def getDescription(self, data):
        """
            general description of the dataset by sevral parameters
        """
        pd.set_option('max_columns', None)
        print('dataset description by pandas:\n', data.describe(include='all'))

        print('\ndataset missing values in percents:\n')
        for col in data.columns:
            pct_missing = np.mean(data[col].isnull())
            print('{}: {}%'.format(col, pct_missing * 100))

        print('\ndataset shape: ', data.shape)

        print(data.info())

    def main_correlation_heatmap(self, data):
        """
            helper function to show correlation between all
        """
        corrMatrix = data.corr()
        fig, ax = plt.subplots(figsize=(12, 12))
        mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corrMatrix, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, ax=ax, annot=True,
                    fmt=".2f")
        plt.show()
        plt.close()


class InitialDataAnalysis:
    def __init__(self):
        self.df = pd.read_csv('./lol2.csv')

    def get_data(self):
        return self.df

    def count_missing_rows(self):
        """
            this function presents the precentage of missing values at each column as a bar plot.
            used to demonstrate my work proccess
        """

        data_na = (self.df.isnull().sum() / len(self.df)) * 100
        data_miss = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
        f, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=data_miss.index, y=data_miss)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()

    def minion_correlation_heatmap(self):
        """
            helper function to show correlation between these columns
        """
        corrData = self.df[['blueTotalMinionKills', 'blueJungleMinionKills', 'redTotalMinionKills',
                            'redJungleMinionKills']].select_dtypes(np.number)
        corrMatrix = corrData.corr()
        sns.heatmap(corrMatrix, annot=True, linewidth=0.3, fmt='.1f')
        plt.show()
        plt.close()

    def change_categorials(self):
        """
            convert categorical data into numeric values.
            Red = 1
            Blue = 2
        """

        # set categorials to be numeric
        self.df['FirstDragon'] = np.where(self.df['FirstDragon'] == "Red", 1, 2)
        self.df['FirstBaron'] = np.where(self.df['FirstBaron'] == "Red", 1, 2)
        self.df['FirstTower'] = np.where(self.df['FirstTower'] == "Red", 1, 2)
        self.df['FirstBlood'] = np.where(self.df['FirstBlood'] == "Red", 1, 2)
        self.df['win'] = np.where(self.df['win'] == "Red", 1, 2)

    def clean_data(self):
        """
            this function cleans whole dataset.
            the function makes use of other functions such as clean_minion_kills.
            each column is treated differently.
        """

        # remove missing data at FirstBlood column
        self.df.dropna(subset=['FirstBlood'], inplace=True)

        # change categorical values to numeric
        self.change_categorials()

        # display distribution for each feature prior to cleansing
        self.histplot_each()

        # remove duplicates from the gameId column since each is uniq
        self.df = self.df.drop_duplicates(subset='gameId', keep='first')

        # change games with duration lower or equal to zero to median
        # first get mean after removing zeros to have correct values calculated
        duration_clean = self.df[self.df['gameDuraton'] > 0]
        self.df.loc[self.df['gameDuraton'] < 0, 'gameDuraton'] = duration_clean['gameDuraton'].mean()

        # remove all values in specified column containing zero
        self.df = self.df[self.df['blueWardPlaced'] > 0]
        self.df = self.df[self.df['blueWardkills'] > 0]

        # this part displays the correlation between these features
        self.joint_plot_minion_kills()  # check for linear correlation
        self.minion_correlation_heatmap()  # check for linear correlation

        # clean (Red & Blue) totalMinionKills and jungleMinionKills columns
        self.clean_minion_kills()

        ## clean the duration with columns correlated value
        self.clean_duration()

        # display the correlation between heals prior to fixing
        self.joint_plot_heals()  # check for linear relation between heal columns

        # #clean red and blue total heal section by populating the missing values with the correlated value
        self.clean_total_heal('redTotalHeal', 'redJungleMinionKills', 'redTotalMinionKills')
        self.clean_total_heal('blueTotalHeal', 'blueJungleMinionKills', 'blueTotalMinionKills')

        # display the correlation between wards prior to fixing
        self.joint_plot_wards()  # check for linear relation between ward columns

        # clean the redWardsPlaced with columns correlated value
        self.clean_red_wards_placed()

        # remove all remaining outliers
        self.remove_outliers()

        self.df.rename(columns={'gameDuraton': 'gameDuration', 'blueWardkills': 'blueWardKilled',
                                'blueTotalMinionKills': 'blueTotalMinionKilled',
                                'blueJungleMinionKills': 'blueJungleMinionKilled',
                                'redWardkills': 'redWardKilled', 'redTotalMinionKills': 'redTotalMinionKilled',
                                'redJungleMinionKills': 'redJungleMinionKilled'}, inplace=True)


        self.df.to_csv('lol_clean.csv', index=False)

    def clean_minion_kills(self):
        """
            this function cleans the data in the columns totalMinionKills and jungleMinionKills
            at both blue and red columns. it uses the correlation bettwen these columns in order
            to populate missing values in the best way possible. if no calculation optional - populate with the mean
        """

        total_ratio1 = self.df['blueTotalMinionKills'] / self.df['redTotalMinionKills']
        total_ratio2 = self.df['redTotalMinionKills'] / self.df['blueTotalMinionKills']

        total_ratio = (total_ratio1.mean() + total_ratio2.mean()) / 2

        total_jungle_ratio_red = (self.df['redJungleMinionKills'] / self.df['redTotalMinionKills']).mean()
        total_jungle_ratio_blue = (self.df['blueJungleMinionKills'] / self.df['blueTotalMinionKills']).mean()

        print('total ratio: ', total_ratio)
        print('red total to jungle ratio: ', total_jungle_ratio_red)
        print('blue total to jungle ratio: ', total_jungle_ratio_blue)

        red_total_mean = self.df['redTotalMinionKills'].dropna().mean()
        blue_total_mean = self.df['blueTotalMinionKills'].dropna().mean()
        red_jungle_mean = self.df['redTotalMinionKills'].dropna().mean()
        blue_jungle_mean = self.df['blueTotalMinionKills'].dropna().mean()

        self.df['redTotalMinionKills'] = self.df['redTotalMinionKills'].fillna(-1)
        self.df['blueTotalMinionKills'] = self.df['blueTotalMinionKills'].fillna(-1)
        self.df['blueJungleMinionKills'] = self.df['blueJungleMinionKills'].fillna(-1)
        self.df['redJungleMinionKills'] = self.df['redJungleMinionKills'].fillna(-1)

        self.df['redTotalMinionKills'] = self.df.apply(
            lambda row: self.calculate_total_minion(row['redTotalMinionKills'], row['blueTotalMinionKills'],
                                                    row['redJungleMinionKills'], red_total_mean),
            axis=1
        )

        self.df['blueTotalMinionKills'] = self.df.apply(
            lambda row: self.calculate_total_minion(row['blueTotalMinionKills'], row['redTotalMinionKills'],
                                                    row['blueJungleMinionKills'], blue_total_mean),
            axis=1
        )

        self.df['blueJungleMinionKills'] = self.df.apply(
            lambda row: self.calculate_jungle_minion(row['blueJungleMinionKills'], row['blueTotalMinionKills'],
                                                     row['redTotalMinionKills'], blue_jungle_mean),
            axis=1
        )

        self.df['redJungleMinionKills'] = self.df.apply(
            lambda row: self.calculate_jungle_minion(row['redJungleMinionKills'], row['redTotalMinionKills'],
                                                     row['blueTotalMinionKills'], red_jungle_mean),
            axis=1
        )

    def calculate_total_minion(self, goal_total, other_total, jungle, mean):
        if goal_total == -1:
            if other_total != -1:
                return other_total
            elif jungle != -1:
                return jungle * 4
            else:
                return mean

        else:
            return goal_total

    def calculate_jungle_minion(self, goal_jungle, first_total, second_total, mean):
        if goal_jungle == -1:
            if first_total != -1:
                return first_total / 4

            elif second_total != -1:
                return second_total / 4

            else:
                return mean
        else:
            return goal_jungle

    def clean_total_heal(self, heal_inp, jungle_inp, total_inp):
        """
            This function cleans the heal related columns
        """

        tmp_df = self.df.copy()
        self.df[heal_inp] = self.df[heal_inp].fillna(-1)

        tmp_df.loc[self.df[heal_inp] == 0, heal_inp] = 0.1
        tmp_df.loc[self.df[total_inp] == 0, total_inp] = 0.1
        tmp_df.loc[self.df[jungle_inp] == 0, jungle_inp] = 0.1
        tmp_df.loc[self.df['gameDuraton'] == 0, 'gameDuraton'] = 0.1

        total_rel = tmp_df[heal_inp] / tmp_df[total_inp]
        jungle_rel = tmp_df[heal_inp] / tmp_df[jungle_inp]
        duration_rel = tmp_df[heal_inp] / tmp_df['gameDuraton']

        total = (total_rel.mean() + total_rel.median()) / 2
        jungle = (jungle_rel.mean() + jungle_rel.median()) / 2
        duration = (duration_rel.mean() + duration_rel.median()) / 2

        self.df[heal_inp] = self.df.apply(
            lambda row: self.calculate_heal(row, duration, total, jungle, heal_inp, total_inp, jungle_inp),
            axis=1
        )

    def calculate_heal(self, row, duration_rel, total_rel, jungle_rel, heal, total, jungle):
        ans = []
        if row[heal] == -1:
            ans.append(row[total] * total_rel)
            ans.append(row[jungle] * jungle_rel)
            ans.append(row['gameDuraton'] * duration_rel)
            return sum(ans) / len(ans)

        return row[heal]

    def histplot_each(self):
        """
            this function used to plot histograms for each feature.
            used several times during this project to demonstrate data transformations
        """
        for name in self.df.columns:
            bins = round(abs(self.df[name].describe()['max'] - self.df[name].describe()['min']))
            bins = 100 if bins > 100 or name in ['FirstDragon', 'FirstBaron', 'FirstTower', 'FirstBlood',
                                                 'win'] else bins
            sns.histplot(self.df[name], bins=round(bins), kde='true')
            plt.show()
            plt.close()

    def boxplot_each(self):
        for name in self.df.columns:
            self.df.boxplot(column=[name])
            plt.show()
            plt.close()

    def joint_plot_minion_kills(self):
        """
            used to demonstrate the relation between these columns when cleaning the dataset
        """
        sns.jointplot(x='blueJungleMinionKills', y='blueTotalMinionKills', data=self.df)
        plt.show()
        plt.close()

        sns.jointplot(x='redJungleMinionKills', y='redTotalMinionKills', data=self.df)
        plt.show()
        plt.close()

        sns.jointplot(x='redTotalMinionKills', y='blueTotalMinionKills', data=self.df)
        plt.show()
        plt.close()

    def joint_plot_wards(self):
        """
            used to demonstrate the relation between these columns when cleaning the dataset
        """
        sns.jointplot(x='blueWardPlaced', y='redWardPlaced', data=self.df)
        plt.show()
        plt.close()

        sns.jointplot(x='blueWardkills', y='redWardPlaced', data=self.df)
        plt.show()
        plt.close()

    def clean_duration(self):
        """
            This function cleans the game duration column
        """
        red_rel_series = self.df['gameDuraton'] / self.df['redTotalMinionKills']
        blue_rel_series = self.df['gameDuraton'] / self.df['blueTotalMinionKills']

        red_rel = red_rel_series.mean()
        blue_rel = blue_rel_series.mean()

        self.df['gameDuraton'] = self.df['gameDuraton'].fillna(-1)

        self.df['gameDuraton'] = self.df.apply(
            lambda row: self.calculate_duration(row, red_rel, blue_rel),
            axis=1
        )

    def calculate_duration(self, row, red_rel, blue_rel):
        ans = []
        if row['gameDuraton'] == -1:
            ans.append(row['redTotalMinionKills'] * red_rel)
            ans.append(row['blueTotalMinionKills'] * blue_rel)
            return sum(ans) / len(ans)
        else:
            return row['gameDuraton']

    def clean_red_wards_placed(self):
        rel_series = self.df['redWardPlaced'] / self.df['blueWardPlaced']
        rel = rel_series.mean()

        self.df['redWardPlaced'] = self.df['redWardPlaced'].fillna(-1)

        self.df['redWardPlaced'] = self.df.apply(
            lambda row: self.calculate_red_wards_placed(row, rel),
            axis=1
        )

    def calculate_red_wards_placed(self, row, rel):
        if row['redWardPlaced'] == -1:
            return row['blueWardPlaced'] * rel
        else:
            return row['redWardPlaced']

    def remove_outliers(self):
        """
            This function removes the outliers.
            The outliers are the -0.13% top and bottom of each column.
        """
        filt_df = self.df[['gameDuraton', 'blueWardPlaced', 'blueWardkills',
                           'blueTotalMinionKills', 'blueJungleMinionKills', 'blueTotalHeal',
                           'redWardPlaced', 'redWardkills', 'redTotalMinionKills',
                           'redJungleMinionKills', 'redTotalHeal']].copy()

        self.df = self.df[(np.abs(stats.zscore(filt_df)) < 3).all(axis=1)]

    def joint_plot_heals(self):
        """
            used to demonstrate the relation between these columns when cleaning the dataset
        """
        sns.jointplot(y='blueTotalHeal', x='blueTotalMinionKills', data=self.df)
        plt.show()
        plt.close()

        sns.jointplot(y='blueTotalHeal', x='blueJungleMinionKills', data=self.df)
        plt.show()
        plt.close()

        sns.jointplot(y='blueTotalHeal', x='gameDuraton', data=self.df)
        plt.show()
        plt.close()


class ExploratoryDataAnalysis:
    def __init__(self):
        self.df = pd.read_csv('./lol_clean.csv')

    def get_data(self):
        """
            Returns the updated dataframe
        """
        return self.df

    def new_column(self):
        """
            This function is used to create the new columns i will be using later on
        """
        self.df['jungle_rel'] = self.df['blueJungleMinionKilled'] - self.df['redJungleMinionKilled']
        self.df['heal_rel'] = self.df['blueTotalHeal'] / 500 - self.df['redTotalHeal'] / 500
        self.df['ward_rel'] = self.df['blueWardKilled'] - self.df['redWardKilled']
        self.df['jungle_heal_ward_rel'] = self.df['jungle_rel'] + self.df['heal_rel'] + self.df['ward_rel']

        self.df['first_rel'] = self.df.apply(
            lambda row: self.calc_first(row),
            axis=1
        )

    def calc_first(self, row):
        """
            helper function to calculate the first_rel column
        """
        ans = 0
        red = 1

        ans += -1 if row['FirstBlood'] == red else 1
        ans += -1 if row['FirstDragon'] == red else 1
        ans += -2 if row['FirstTower'] == red else 2
        ans += -2 if row['FirstBaron'] == red else 2

        return ans

    def distribution_each(self):
        """
            This function presents the distribution regarding the win column for all other features
        """
        categorical = self.df.copy()
        categorical.replace(1, 'Red', inplace=True)
        categorical.replace(2, 'Blue', inplace=True)

        for name in self.df.columns:

            data = self.df
            if name in ['win', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon']:
                data = categorical

            sns.displot(data, x=name, hue="win", legend=False, palette=['red', 'blue'], fill=True)
            plt.legend(title='Winning team', labels=['Blue', 'Red'])
            plt.show()
            plt.close()

    def show_new_rels(self):
        """
            This function presents the plots of al the new columns i created
        """

        #this plot takes some time to be created
        sns.jointplot(x='blueJungleMinionKilled', y='redJungleMinionKilled', data=self.df, kind='kde', hue='win',
                      palette=['red', 'blue'], legend=False)
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()

        # this plot takes some time to be created
        sns.jointplot(x='blueTotalHeal', y='redTotalHeal', data=self.df, kind='kde', hue='win', palette=['red', 'blue'],
                      legend=False)
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()

        sns.displot(self.df, x="jungle_rel", hue="win", legend=False, palette=['red', 'blue'], fill=True)
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()

        sns.displot(self.df, x="first_rel", hue="win", legend=False, palette=['red', 'blue'], multiple="dodge")
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()

        sns.kdeplot(data=self.df, x="first_rel", hue="win", multiple="fill", palette=['red', 'blue'])
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()

        sns.displot(self.df, x="heal_rel", hue="win", legend=False, palette=['red', 'blue'], fill=True)
        plt.legend(title='Winning team', labels=['Blue', 'Red'])
        plt.show()
        plt.close()


class Classification():
    def __init__(self, data):
        self.df = data
        self.original = pd.read_csv('./lol2.csv')

    def naive_bayes(self):
        """
            This function creates a naive bayes model.
            The model is created out of two features - 'jungle_heal_ward_rel' and 'first_rel'.
            Then the data is displayed as decision regions with a scatter plot
            upon it displaying miss predicted data exclusively.

        """
        x = self.df[['jungle_heal_ward_rel', 'first_rel']]
        y = self.df['win']

        # train model based on bill_length & bill_depth
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        model = GaussianNB()
        model.fit(x_train, y_train)

        y_model = model.predict(x_test)

        # check model accuracy
        print('jungle_heal_ward_rel & first_rel:')
        print('accuracy: ', metrics.accuracy_score(y_test, y_model))

        x_min, x_max = x.loc[:, 'jungle_heal_ward_rel'].min() - 1, x.loc[:, 'jungle_heal_ward_rel'].max() + 1
        y_min, y_max = x.loc[:, 'first_rel'].min() - 1, x.loc[:, 'first_rel'].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # create contour plot
        plt.contourf(xx, yy, Z, cmap='Set1', alpha=0.5)
        plt.colorbar()
        plt.clim(0, len(model.classes_) + 3)

        fig = plt.gcf()
        fig.set_size_inches(12, 8)

        self.df['predicted_win'] = model.predict(x)
        incorrects = self.df[self.df['win'] != self.df['predicted_win']].copy()

        incorrects['win'] = np.where(incorrects['win'] == 1, 'Red', 'Blue')

        sns.scatterplot(data=incorrects, x='jungle_heal_ward_rel', y='first_rel', hue='win', hue_order=['Red', 'Blue'],
                        palette='Set1')
        plt.show()

    def basic_tree_classifier(self):
        """
            This function ceates a tree classifier based on the original data with na rows dropped.

        """
        self.original.dropna(inplace=True)

        self.original['FirstDragon'] = np.where(self.original['FirstDragon'] == "Red", 1, 2)
        self.original['FirstBaron'] = np.where(self.original['FirstBaron'] == "Red", 1, 2)
        self.original['FirstTower'] = np.where(self.original['FirstTower'] == "Red", 1, 2)
        self.original['FirstBlood'] = np.where(self.original['FirstBlood'] == "Red", 1, 2)
        self.original['win'] = np.where(self.original['win'] == "Red", 1, 2)

        x = self.original.drop(["gameId", 'win'], axis=1)
        y = self.original['win']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=1)
        clf = DecisionTreeClassifier(max_depth=6)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        print('Baseline tree classifier:')
        print(metrics.classification_report(y_test, y_pred, target_names=['Red', 'Blue']))

    def manipulated_tree_classifier(self):
        x = self.df[['jungle_heal_ward_rel', 'first_rel', 'FirstBaron', 'redWardPlaced']]
        y = self.df['win']
        y.replace(1, 'Red', inplace=True)
        y.replace(2, 'Blue', inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        clf = DecisionTreeClassifier(max_depth=4)
        # clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        result = permutation_importance(clf, x, y, n_repeats=10, random_state=0)

        plt.bar(range(len(x.columns)), result['importances_mean'])
        plt.xticks(ticks=range(len(x.columns)), labels=x.columns, rotation=90)
        plt.show()
        plt.close()

        y_pred = clf.predict(x_test)
        print(metrics.classification_report(y_test, y_pred, target_names=['Red', 'Blue']))

        print(tree.export_text(clf, feature_names=['jungle_heal_ward_rel', 'first_rel', 'FirstBaron', 'redWardPlaced']))

        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, feature_names=x.columns,
                        class_names=['Red', 'Blue'])
        graph = pyd.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('Tree.png')
        Image(graph.create_png())


def main():
    helpers = helper_functions()

    initial = InitialDataAnalysis()
    helpers.getDescription(initial.get_data())  # get initial description
    initial.count_missing_rows()  # display plot of missing data in percents
    initial.clean_data()

    explore = ExploratoryDataAnalysis()
    helpers.main_correlation_heatmap(explore.get_data())
    explore.new_column()
    explore.show_new_rels()
    explore.distribution_each()
    helpers.main_correlation_heatmap(explore.get_data())

    clf = Classification(explore.get_data())
    clf.naive_bayes()
    clf.basic_tree_classifier()
    clf.manipulated_tree_classifier()


if __name__ == "__main__":
    main()
