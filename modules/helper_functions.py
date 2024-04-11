import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

from scipy.special import softmax
from math import floor

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
  print(f"No of features with more than {round(limitCount*100/len(df),1)}% NaN's:", nanCountTotalFeatures)


def convert_ber_rating_to_energy_rating(y_pred):

  # classification metrics
  ber_label_mapping = {'25': 'A1', '50': 'A2', '75': 'A3', '100': 'B1', '125': 'B2', '150': 'B3', '175': 'C1', '200': 'C2', '225': 'C3', '260': 'D1', '300': 'D2', '340': 'E1', '380': 'E2', '450': 'F', '10000': 'G'}
  
  y_pred_list = y_pred.tolist()
  y_pred_energy_rating_labels_list = []

  counter=0
  for y_pred in y_pred_list:
    counter += 1
    start = -1000 # negative predictions
    for key, value in ber_label_mapping.items():
      if start < float(y_pred) and float(y_pred) <= float(key):
        y_pred_energy_rating_labels_list.append(value)
        break
      start = float(key)

  return np.asarray(y_pred_energy_rating_labels_list) # return as nparray 


def clean_up_features(df):
  """replace spaces in features names and trim all values for leading and trailing spaces

  Args:
      df (DataFrame): Original dataframe

  Returns:
      DataFrame: modified dataframe
  """

  # strip value from all features if feature is of type object
  for i in df.columns: 
    if df[i].dtype == 'object' or df[i].dtype == 'string': 
      df[i] = df[i].map(lambda x: x.strip() if isinstance(x, str) else x)
    else:
      pass

  # replace spaces in features names
  df.columns = df.columns.str.strip().str.replace(" ", "").str.replace("[^\w]", "", regex=True)
  return df

def features_to_drop():
  """ returns a list of features to drop
  Returns:
      List: list of features to drop
  """
  
  return [
      'CountyName',           # The county in which the dwelling is located
      #'Year_of_Construction', # The year the dwelling was originally constructed
      'TypeofRating',         # There are 3 types of BER Certificate: new dwelling, new dwelling - final BER, existing building BER
      'MultiDwellingMPRN',    # Indicates that the dwelling shares its electricity meter with another dwelling
      'TGDLEdition',          # The edition of the Building Regulations, Part L that applies to the dwelling
      'MPCDERValue',          # The Maximum Permitted Carbon Dioxide Emission Rating value
      'CPC', 
      'EPC', 
      'RER', 
      'RenewEPnren', 
      'RenewEPren', 
      'SA_Code', 
      'PurposeOfRating', 
      'HESSchemeUpgrade', 
      'DateOfAssessment', 
      'CO2Rating', 
      'CO2MainSpace', 
      'MPCDERValue',
      'FirstEnerConsumedComment',
      'SecondEnerConsumedComment',
      'ThirdEnerConsumedComment']

def drop_features(df):
  """Drop features not required/relevant. Not this function drops the features from the original dataframe. For performance reasons, it does not modify a copy of the dataframe

  Args:
      df (DataFrame)): Dataset to drop the features from

  Returns:
      DataFrame: The modified DataFrame
  """
  df = df.drop(features_to_drop(), axis='columns')

  print('drop_features end: ', df.shape)

  return df


def feature_main_heating_fuel(df):
  """Merge MainSpaceHeatingFuel and MainWaterHeatingFuel features and rename as MainHeatingFuel

  Args:
      df (DataFrame): DataFrame to modify

  Returns:
      DataFrame: modified dataframe
  """
  df.rename(columns={"MainSpaceHeatingFuel": "MainHeatingFuel"}, inplace=True)

  return df.drop(['MainWaterHeatingFuel'], axis='columns')


def feature_suspended_wooden_floor(df):
  """As per BER User Guide, air leakage through:
   - a solid floor is taken to be zero
   - a sealed suspended wooden floor has a leakage rate of 0.1 ac/h
   - an unsealed suspended wooden floor has a leakage rate of 0.2 ac/h

  Args:
      df (DataFrame): DataFrame to modify

  Returns:
      DataFrame: modified dataframe
  """
  # mapping
  suspended_wooden_floor_mapping = {'No': '0', 'Yes (Unsealed)': '0.1', 'Yes (Sealed)': '0.2'}

  # inner function
  def swr_replace__text_with_value(text):
    return suspended_wooden_floor_mapping.get(text, text)

  df_2 = df.copy()
  df_2['SuspendedWoodenFloor'] = df_2['SuspendedWoodenFloor'].map(suspended_wooden_floor_mapping)
  return df_2


def feature_structure_type(df):
  """As per BER User Guide, this information is used to estimate the amount of air leakage through cracks in the walls of the dwelling which is necessary if an air permeability test has not been performed on the dwelling:
  - air leakage through masonry walls contributes 0.35 ac/h
  - whereas timber/steel-frame and ICF walls have an assumed leakage rate of 0.25 ac/h
  - if no value selected then assume leakage rate of 0.35 ac/h
   
  Args:
      df (DataFrame): DataFrame to modify

  Returns:
      DataFrame: modified dataframe
  """
  # mapping
  structure_type_mapping = {
    'Masonry': '0.35', 
    'Please select': '0.35', 
    'Timber or Steel Frame': '0.25', 
    'Insulated Concrete Form': '0.25'
  }

  # inner function
  def st_replace__text_with_value(text):
    if len(text) == 0: # when not specified
      return '0.35'
    else:
      return structure_type_mapping.get(text, text)

  df_2 = df.copy()
  df_2['StructureType'] = df_2['StructureType'].map(structure_type_mapping)
  return df_2


def feature_ventilation_method(df):
  ventilation_method_mapping = {
    'Natural vent.': '0.52',
    'Pos input vent.- loft': '0.62',
    'Pos input vent.- outside': '0.50',
    'Whole house extract vent.': '0.50',
    'Bal.whole mech.vent no heat re': '0.71',
    'Bal. whole mech.vent heat recvr': '0.71',
    'Exhaust Air Heat Pump': '0.00'
  }

  # inner function
  def vm_replace__text_with_value(text):
    if len(text) == 0: # when not specified
      return '0.35'
    else:
      return ventilation_method_mapping.get(text, text)

  df_2 = df.copy()
  df_2['VentilationMethod'] = df_2['VentilationMethod'].map(ventilation_method_mapping)
  return df_2


def rus_ratio_multiplier(y):
  """function for exclusive use by feature_balance_data

  Args:
      y (DataFrame): dataframe

  Returns:
      dictionary: calculated total of rows for each EnergyRating label
  """

  target_stats = Counter(y)
  target_stats_dict = dict(target_stats)
  
  # sort dict by value
  keys = list(target_stats_dict.keys())
  values = list(target_stats_dict.values())
  sorted_value_index = np.argsort(values)
  target_stats_sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

  all_vals = target_stats_sorted_dict.values()
  
  # upper limit set to a multiple of the lowest number, i.e. 4 times
  val_iter = iter(all_vals)
  first_val = next(val_iter)
  #upper_limit = first_val * 4 # upper limit set to a multiple of the lowest number, i.e. 4 times

  # percentage of mean value
  print('all_vals:', type(list(all_vals)))
  print(len(all_vals))
  #upper_limit = np.mean(list(all_vals)) * 0.8 # 80%
  upper_limit = first_val # make the upper limit the same as the smallest label row count

  for key, value in target_stats.items():
    if value > upper_limit:
      target_stats[key] = int(upper_limit)
    else:
      target_stats[key] = value

  print('target_stats: ', target_stats)    
  return target_stats


def feature_balance_data(X_merged, y):

  X_merged_dtypes = X_merged.dtypes
  X_merged.columns = X_merged.columns.astype(str)

  X_res, y_res = RandomUnderSampler(sampling_strategy=rus_ratio_multiplier).fit_resample(X_merged,y)

  X_merged_dtypes_new = dict([(key, value.name) for key, value in X_merged_dtypes.to_dict().items()])

  # set column types back to original 
  for key, value in X_merged_dtypes_new.items():
    if key in X_res.columns:
      X_res[key] = X_res[key].astype(value) 

  return X_res, y_res



def feature_drop_more_than_category_values(df, number_to_drop=20): # DELETE
  # drop features with more than 20 category values
  categorical_cols = df.select_dtypes(include='O').keys()
  # unique values in each columns
  cats_to_drop = []
  for x in df.columns:
    if x in categorical_cols:
      if len(df[x].unique()) > number_to_drop:
        cats_to_drop.append(x) 
      #print(x ,':', len(df[x].unique()))

  df_2 = df.drop(cats_to_drop, axis=1)

  categorical_cols = df_2.select_dtypes(include='O').keys()
  #print(categorical_cols)

  df_3 = pd.get_dummies(df_2, columns = categorical_cols)
  # replace spaces in features names
  df_3.columns = df_3.columns.str.replace(" ", "_").str.replace("[^\w]", "", regex=True)
  
  return df_3


def features_convert_to_boolean(df):
  # check if columns contains YES, NO, nan and replace with binary 1,0

  categorical_cols = df.select_dtypes(include='O').keys()

  boolean_cols = []

  for column_name in categorical_cols:
    values = " ".join(str(x) for x in df[column_name].unique())
    if 'YES' in values or 'Yes' in values:
      df[column_name] = df[column_name].map({'YES':True ,'NO':False, 'Yes':True ,'No':False})
      df[column_name].fillna(False, inplace=True) # replace any NaN values with false
      df[column_name] = df[column_name].astype(bool)
      boolean_cols.append(column_name)

  print('boolean_cols end:', df.shape)

  return df


# 
NO_FEATURES_KEPT = 100



def feature_reduction_x(df):
  #Can just alter this for quick change
  #return feature_reduction_corr(df)  #.4848 for decision tree   #.4168 for ridge regression  
  return feature_reduction_kBest(df)  #.4896 for decision tree  #.4252 for ridge regression
  #return feature_reduction_decision_tree(df)  #.41 for ridge regression - took a long time

def feature_reduction_corr(df):
  #convert to X
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CountyName', 'Year_of_Construction', 'TypeofRating', 'MultiDwellingMPRN', 'TGDLEdition','CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

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

def feature_reduction_y_grid(df):
  X = df.drop(["BerRating", "EnergyRating"], axis='columns')
  X = X.drop(['CPC', 'EPC', 'RER', 'RenewEPnren', 'RenewEPren', 'SA_Code', 'PurposeOfRating', 'HESSchemeUpgrade', 'DateOfAssessment', 'CO2Rating', 'CO2MainSpace', 'MPCDERValue'], axis='columns')

  y1 = df.BerRating
  y2 = df.EnergyRating
  y_grid = pd.get_dummies(y2)

  X = pd.get_dummies(X)
  imp = SimpleImputer(missing_values=np.nan, strategy='mean')  #.43
  imp.fit(X)
  X = pd.DataFrame(imp.fit_transform(X), columns = imp.get_feature_names_out())

  scaler = MinMaxScaler()
  X = pd.DataFrame(scaler.fit_transform(X), columns = imp.get_feature_names_out())

  #res_mod = SelectKBest(f_classif, k=NO_FEATURES_KEPT).set_output(transform="pandas")

  names = set()
  k_num = floor(NO_FEATURES_KEPT/len(y_grid.columns))
  res_mod = SelectKBest(f_classif, k=k_num).set_output(transform="pandas")

  for (columnName, columnData) in y_grid.items():
    #print('Column Name : ', columnName)
    #print('Column Contents : ', columnData.values)
    res = res_mod.fit_transform(X, columnData)
    features = res.columns
    noFeatures = k_num + 1

    #errorchecking
    while(len(set(features).difference(names)) < k_num):
      new_k_best = SelectKBest(f_classif, k=noFeatures).set_output(transform="pandas")
      res = new_k_best.fit_transform(X, columnData)
      features = res.columns
      noFeatures += 1
    names.update(res.columns) 
  
  return X[list(names)]



def rating_feature_conversion(ber_feature_numpy):
  rating_conversion_v = np.vectorize(rating_conversion)
  return rating_conversion_v(ber_feature_numpy)

  



def rating_conversion(ber_rating):
  output = ""
  if(ber_rating > 450):
    output = "G"
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

def softmax(X, y, model):
  #Need to take the model type, and the data as an input
  #One=Hot the resulting categories?
  #Then, need to get probablities of each possible category
  y_grid = pd.get_dummies(y)
  model.fit(X , y)
  print(y_grid.head())
  return

if __name__=="__main__": 
  print(f"{__file__} can only be imported.") 
