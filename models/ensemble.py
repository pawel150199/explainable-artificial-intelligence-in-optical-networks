from sklearn.ensemble import RandomForestRegressor as RF

class RandomForestRegresor(RF):
    def __init__(self, feature_selection: bool=True):
        self.feature_selection = feature_selection
    