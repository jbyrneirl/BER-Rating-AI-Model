"""_summary_
Helper function for use in notebooks

"""

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
  print(f"No of features with more than {limitCount*100/len(df)}% NaN's:", nanCountTotalFeatures)