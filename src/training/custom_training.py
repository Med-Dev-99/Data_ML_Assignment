from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump
import pandas as pd

class CustomTrainingPipeline:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data['resume'],
            data['label'],
            test_size=0.2,
            random_state=0
        )

    def train(self, serialize=False, model_name='custom_model'):
        self.x_train = self.vectorizer.fit_transform(self.x_train)
        self.model.fit(self.x_train, self.y_train)

        if serialize:
            dump(self.model, f'models/{model_name}.joblib')
    
    def get_model_performance(self):
        x_test = self.vectorizer.transform(self.x_test)
        predictions = self.model.predict(x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        return accuracy, f1

    def render_confusion_matrix(self):
        x_test = self.vectorizer.transform(self.x_test)
        predictions = self.model.predict(x_test)
        cm = confusion_matrix(self.y_test, predictions)
        return cm
