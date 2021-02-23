import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import Analysis

pd.set_option('display.max_columns', None)
correlation_threshold = 0.015

CRED = '\033[91m'
CEND = '\033[0m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'


def plotPCA(df):
    # PCA 2 components plot
    features = df.drop(columns='classes', axis=1)
    pca = PCA(n_components=2)
    pca.fit(features)
    projected = pca.fit_transform(features)
    plt.scatter(projected[:, 0], projected[:, 1], c=df['classes'], edgecolor='none', alpha=0.5, cmap='viridis')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    plt.close()


def var_threshold(df, threshold):
    features = df.drop(columns='classes', axis=1)
    vt = VarianceThreshold(threshold)
    vt.fit(features)
    low_var_features = [x for x in features.columns if x not in features.columns[vt.get_support()]]
    print("Feaures with variance of less than " + str(threshold) + ":", ", ".join(low_var_features))
    print("Number of remaining features:", sum(vt.get_support()))
    return low_var_features


def performEDA(df):
    #################################################
    # Remove features which corresponds to 1 class
    for col in df.columns.values:
        if len(df[col].unique()) == 1:
            print(len(df[col].unique()))
            print(f'Removing column {col}, which only contains the value: {df[col].unique()[0]}')
            df = df.drop(columns=col)

    mid = df['classes']
    df.drop(labels=['classes'], axis=1, inplace=True)
    df.insert(0, 'classes', mid)

    #################################################
    # Correlation
    print(CGREEN + "Calculating correlation between features and target: " + CEND)
    corr = df.corr()
    corr_y = abs(corr["classes"])
    print(corr_y)
    highest_corr = corr_y[corr_y > correlation_threshold]
    highest_corr.sort_values(ascending=True)
    print(f"Drop rows with correlation less then {correlation_threshold}. ")
    df = df[highest_corr.keys()]

    #################################################
    # Find pairs of highly correlated features
    print(Analysis.df_corr_coeff(df, 0.8))
    df = Analysis.del_corr(df, 0.95)

    for _col in df.columns.values:
        if len(df[_col].unique()) == 1:
            print(f'Removing column {_col}, which only contains the value: {df[_col].unique()[0]}')
            df = df.drop(columns=_col)

    #################################################
    plotPCA(df)

    #################################################
    import seaborn as sns
    # # Correlation Matrix plot
    # plt.figure(figsize=(23, 18))
    # sns.heatmap(df.corr(), linewidths=.2, cmap="Purples", annot=True)
    # plt.yticks(rotation=0)
    # plt.show()
    #
    # plt.close()

    # Distribution plots
    for i, col in enumerate(df.columns):

        if i == 0:
            plt.figure(figsize=(10, 5))
            plt.style.use(['seaborn-bright', 'dark_background'])
            sns.countplot(x=df[col], hue=df['classes'], data=df, palette='hsv')
            plt.title(col, fontsize=20, color='c')
            plt.show()

        plt.figure(figsize=(10, 5))
        plt.style.use(['seaborn-bright', 'dark_background'])
        sns.countplot(x=df[col], hue=df['classes'], data=df)
        plt.title(col, fontsize=20, color='c')
        plt.show()
        plt.close()

    # sns.pairplot(df[['classes', 'population', 'odor']], hue='classes', vars=['population', 'odor'], palette=['r', 'g'])
    # plt.savefig("pairplot1.png", format="png", dpi=500, bbox_inches="tight")

    # Runs a variance threshold algorithm and returns the features with a variance that is lower than the threshold
    low_variance = var_threshold(df, 0.1)
    df = df.drop(low_variance, axis=1)
    df.to_csv('data/cleaned2_mushrooms.csv')

    return df
