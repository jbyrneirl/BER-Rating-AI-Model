#from sklearn.compose import ColumnTransformer
#from sklearn.compose import make_column_selector

class NumericalColumnSelector:
  logging = False
  drop_features_transformer_singleton = None

  def __init__(self, drop_features_transformer_singleton=None, logging=False):
    super()
    self.logging = logging
    self.drop_features_transformer_singleton = drop_features_transformer_singleton

  #def make_column_selector():
  #  if self._logging:
  #    print('CategoricalColumnTransformer - make_column_selector:')
  #  return []

  def __call__(self, X):

    numerical_columns_final = []

    if self.logging:
      print('NumericalColumnSelector - __call__')

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    list_of_features_to_drop = []
    if self.drop_features_transformer_singleton:
      list_of_features_to_drop = self.drop_features_transformer_singleton.get_dropped_features()

    numerical_columns_final = [item for item in numerical_features if item not in list_of_features_to_drop]

    numerical_columns_final.remove('index')
    print('NumericalColumnSelector __call__ end: ', numerical_columns_final)    

    return numerical_columns_final  
