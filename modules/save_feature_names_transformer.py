import pandas as pd
import hashlib
import time
from sklearn.base import BaseEstimator, TransformerMixin

class SaveFeatureNamesTransformerSingleton(BaseEstimator, TransformerMixin):
  _instance = None
  _logging = False
  _feature_names = {}
  
  def __new__(cls, logging=False, **kwargs):
    if cls._instance is None:
      if cls._logging:
        print('SaveFeatureNamesTransformerSingleton: Creating the object')
      cls._instance = super(SaveFeatureNamesTransformerSingleton, cls).__new__(cls, **kwargs)
      # Put any initialization here
      cls._logging = logging

      if cls._logging:
        print('SaveFeatureNamesTransformerSingleton constructor - self._feature_names length:', len(cls._feature_names))
    return cls._instance

  def transform(self, X, y=None):
    """Save feature names for reuse later

    Args:
        X (DataFrame): Dataset to get feature names from
        y (DataFrame): dependent variable

    Returns:
        DataFrame: The original DataFrame
    """

    if self._logging:
      print('SaveFeatureNamesTransformerSingleton - transform: initial', self._feature_names)

    feature_names = []
    if isinstance(X, pd.DataFrame):
      feature_names = X.columns.values.tolist()

    if feature_names:
      key = '_'.join((str(feature_names[0]), str(len(feature_names)), str(self.__get_hash())))
      self._feature_names[key] = feature_names

    if self._logging:
      print('SaveFeatureNamesTransformerSingleton - transform end: ', X.shape)
      print('SaveFeatureNamesTransformerSingleton - transform end: ', feature_names)

    return X

  def fit(self, X, y=None):
    if self._logging:
      print('SaveFeatureNamesTransformerSingleton - fit')
    return self # do nothing

  def get_feature_names(self):
    if self._logging:
      print('SaveFeatureNamesTransformerSingleton - get_feature_names:', self._feature_names)
    return sum(list(self._feature_names.values()), []) # return merged list of feature names

  def __get_hash(self):
    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    return hashlib.sha1().hexdigest()[:20]