import numpy as np
import pandas as pd


# Prints pairs of features with a correlation that is greater than or equal to the given coefficient (in absolute value)
def df_corr_coeff(df, coeff):
    table = []
    upper = df.corr().where(np.triu(np.ones(df.corr().shape), k=1).astype(np.bool))
    for col in df.corr().columns:
        for i, val in enumerate(list(upper[col].dropna().values)):
            if abs(val) >= coeff:
                table.append((upper[col].dropna().name, df.corr().columns[i], val))
    table = pd.DataFrame(table, columns=["Feature 1", "Feature 2", "Correlation"])
    return table


# Deletes a feature for every pair of perfectly correlated features (disregarding the target feature)
def del_corr(df, score):
    features = df_corr_coeff(df, score)
    if features.shape[0] == 0:
        return df
    features_groups = [{features["Feature 1"][0], features["Feature 2"][0]}]
    for i in range(1, features.shape[0]):
        if features["Feature 1"][i] not in set.union(*features_groups) and features["Feature 2"][i] not in set.union(*features_groups):
            features_groups.append({features["Feature 1"][i], features["Feature 2"][i]})
        elif features["Feature 1"][i] in set.union(*features_groups) and features["Feature 2"][i] in set.union(*features_groups):
            continue
        else:
            for group in features_groups:
                if features["Feature 1"][i] in group or features["Feature 2"][i] in group:
                    group.add(features["Feature 1"][i])
                    group.add(features["Feature 2"][i])
                    break
    for group in features_groups:
        group.pop()
        df = df.drop(group, axis=1)
    return df