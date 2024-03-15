"""_summary_
Helper function for use in notebooks

"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

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

NO_FEATURES_KEPT = 100



def feature_reduction_x(df):
  #convert to X
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)

  scaler = MinMaxScaler()
  Xnp = scaler.fit_transform(X)
  X = pd.DataFrame(Xnp, index=X.index, columns=X.columns)
  #print(len(list(X.columns)))
  values = X.corrwith(y1).abs()
  factors = values.nlargest(NO_FEATURES_KEPT).keys()

  return X[factors]


def feature_reduction_x2(df):
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)

  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp.fit(X)
  X = imp.transform(X)

  #scaler = MinMaxScaler()
  #Xnp = scaler.fit_transform(Xt)
  #X = pd.DataFrame(Xnp, index=X.index, columns=X.columns)

  

  return SelectKBest(f_classif, k=NO_FEATURES_KEPT).fit_transform(X, y2)



def rating_feature_conversion(ber_feature_numpy):
  rating_conversion_v = np.vectorize(rating_conversion)
  return rating_conversion_v(ber_feature_numpy)

  



def rating_conversion(ber_rating):
  output = ""
  if(ber_rating > 450):
    output = "G"
  elif(ber_rating > 380):
    output = "F"
  elif(ber_rating > 380):
    output = "F"
  elif(ber_rating > 340):
    output = "E2"
  elif(ber_rating > 300):
    output = "E1"
  elif(ber_rating > 260):
    output = "D2"
  elif(ber_rating > 225):
    output = "D1"
  elif(ber_rating > 200):
    output = "C3"
  elif(ber_rating > 175):
    output = "C2"
  elif(ber_rating > 150):
    output = "C1"
  elif(ber_rating > 125):
    output = "B3"
  elif(ber_rating > 100):
    output = "B2"
  elif(ber_rating > 75):
    output = "B1"
  elif(ber_rating > 50):
    output = "A3"
  elif(ber_rating > 25):
    output = "A2"
  elif(ber_rating <= 25):
    output = "A1"

  return output
