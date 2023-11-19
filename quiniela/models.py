import pickle
from sklearn.ensemble import GradientBoostingClassifier

class QuinielaModel:
    def train(self, train_data):
        # Do something here to train the model
        features = ['away_team_rank','home_team_rank','matchday']
        target = ["match_result"]
        x_train = train_data[features]
        y_train = train_data[target].values.ravel()
        self.clf = GradientBoostingClassifier()
        self.clf.fit(x_train, y_train)


    def predict(self, predict_data):
        features = ['away_team_rank','home_team_rank','matchday']
        x_predict = predict_data[features]
        clf_y_pred = self.clf.predict(x_predict)
        return clf_y_pred

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
