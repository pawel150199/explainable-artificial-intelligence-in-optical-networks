from sklearn.tree import DecisionTreeRegressor as CART
from helpers.feature_selection import FeatureSelection

class DecisionTreeRegressor(CART):
    def __init__(self, feature_selection: bool=False, random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        self.feature_selection = feature_selection
    
    def fit(self, X, y, sample_weight=None):
        if self.feature_selection:
            self.fs = FeatureSelection(random_state=self.random_state)
            self.fs.fit(X, y)
            X = self.fs.transform(X)
        return super().fit(X, y, sample_weight)
    
    def predict(self, X):
        if self.feature_selection:
            X = self.fs.transform(X)
        return super().predict(X)
        