import numpy as np

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

# 
if __name__=="__main__": 
  print(f"{__file__} can only be imported.") 