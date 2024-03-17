"""_summary_
Helper function for use in notebooks

"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

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
  #Can just alter this for quick change
  #return feature_reduction_corr(df)  #.4848 for decision tree   #.4168 for ridge regression  
  return feature_reduction_kBest(df)  #.4896 for decision tree  #.4252 for ridge regression
  #return feature_reduction_decision_tree(df)  #.41 for ridge regression - took a long time

def feature_reduction_corr(df):
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


def feature_reduction_kBest(df):
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)

  #imp = SimpleImputer(missing_values=np.nan, strategy='constant')
  imp = SimpleImputer(missing_values=np.nan, strategy='mean')  #.43
  imp.fit(X)
  #X = imp.transform(X)
  X = pd.DataFrame(imp.fit_transform(X), columns = imp.get_feature_names_out())

  scaler = MinMaxScaler()
  # X = scaler.fit_transform(X)
  X = pd.DataFrame(scaler.fit_transform(X), columns = imp.get_feature_names_out())

  res_mod = SelectKBest(f_classif, k=NO_FEATURES_KEPT).set_output(transform="pandas")
  res = res_mod.fit_transform(X, y2)
  return res

def feature_reduction_decision_tree(df):
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

  y1 = df.BerRating
  y2 = df.EnergyRating

  X = pd.get_dummies(X)

  X_train, X_test, y_train, y_test = train_test_split( X, y2, stratify=y2, random_state=2)
  best_score = 0
  best_depth = 0

  for d in range(2,15):
      model = DecisionTreeClassifier(max_depth=d)
      scores = cross_val_score(model, X_train, y_train, cv=5)
      mean = scores.mean()
      #print("Depth: ", d, "Accuracy:", mean)
      if(mean > best_score):
          best_score = mean
          best_depth = d

  model = DecisionTreeClassifier(max_depth=best_depth)
  model.fit(X_train,y_train)

  feature_model = SelectFromModel(model, prefit=True)

  return feature_model.transform(X)



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
