"""_summary_
Helper function for use in notebooks

"""
import pandas as pd
"""_summary_
  df: dataframe
  limit: value between 0 and 1
"""
def count_nan_features(df, limit):
  limitCount = limit * len(df)
  nanCountTotalFeatures = 0
  nanCountTotal = df.isna().sum().values

  for count in nanCountTotal:
    if count > limitCount:
      nanCountTotalFeatures += 1
  print(f"No of features with more than {round(limitCount*100/len(df),1)}% NaN's:", nanCountTotalFeatures)

NO_FEATURES_KEPT = 20



def feature_reduction_x(df):
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)
  values = X.corrwith(y1).abs()
  factors = values.nlargest(NO_FEATURES_KEPT).keys()
  return df[factors]

  #print(X.corrwith(y))


if __name__ == "__main__":
  df = pd.read_csv("../data/training/BERRatingData_aa.csv", sep=";", on_bad_lines="skip", low_memory=False)
  df = feature_reduction(df, 10)
  print(df.keys())