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
  #convert to X
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  #X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating'], axis='columns')
  #error - cannot drop last two columns

  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)
  values = X.corrwith(y1).abs()
  factors = values.nlargest(NO_FEATURES_KEPT).keys()
  return df[factors]

  #print(X.corrwith(y))


def rating_conversion(ber_rating):
  print(ber_rating.head())