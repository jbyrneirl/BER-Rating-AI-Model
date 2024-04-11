class CategoricalColumnSelector:
  logging = False
  drop_features_transformer_singleton = None

  def __init__(self, drop_features_transformer_singleton=None, logging=False):
    self.logging = logging
    self.drop_features_transformer_singleton = drop_features_transformer_singleton

  def __call__(self, X):

    categorical_columns_final = []

    if self.logging:
      print('CategoricalColumnSelector - __call__')

    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    list_of_features_to_drop = []
    if self.drop_features_transformer_singleton:
      list_of_features_to_drop = self.drop_features_transformer_singleton.get_dropped_features()

    categorical_columns_final = [item for item in categorical_features if item not in list_of_features_to_drop]

    print('CategoricalColumnSelector __call__ end: ', categorical_columns_final)    

    return categorical_columns_final  
