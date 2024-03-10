from sklearn.linear_model import LinearRegression as LR

class RandomForestRegresor(LR):
    def __init__(self, feature_selection: bool=True):
        self.feature_selection = feature_selection