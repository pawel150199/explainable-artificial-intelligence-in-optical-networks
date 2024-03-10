from sklearn.neighbors import KNeighborsRegressor as KNN
from helpers.feature_selection import FeatureSelection

class KNeighborsRegressor(KNN):
    def __init__(self, feature_selection: bool=False, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_selection = feature_selection
        self.random_state = random_state
    
    def fit(self, X, y):
        if self.feature_selection:
            self.fs = FeatureSelection(random_state=self.random_state)
            self.fs.fit(X, y)
            X = self.fs.transform(X)
        return super().fit(X, y)
    
    def predict(self, X):
        if self.feature_selection:
            X = self.fs.transform(X)
        return super().predict(X)
        