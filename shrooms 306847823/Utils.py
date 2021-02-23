import pandas as pd


# Gets a OneHotEncoder matrix and returns it as a dataframe with the dummy columns of df
def ohe_to_df(ohe, df):
    columns = pd.get_dummies(df, drop_first=True).columns
    ohe_df = pd.DataFrame(ohe, columns=columns[1:]).astype("int")
    ohe_df.insert(0, "classes", df["classes"])
    ohe_df["classes"].replace({"e": 1, "p": 0}, inplace=True)
    return ohe_df
