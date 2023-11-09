import pickle
from sklearn.ensemble import GradientBoostingClassifier
from quiniela.transform_data import transform_data

class QuinielaModel:

    def train(self, train_data):
        # Do something here to train the model
        transformed_data = transform_data(train_data)
        features = ['away_team_rank','home_team_rank','matchday']
        target = ["match_result"]
        x_train = transformed_data[features]
        y_train = transformed_data[target]

        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)
        pass

    def predict(self, predict_data):
        # Do something here to predict
        return ["X" for _ in range(len(predict_data))]

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
