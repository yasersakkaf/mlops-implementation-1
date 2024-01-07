
from sklearn.svm import SVC
import pandas as pd
import joblib
import os

if __name__=='__main__':
    training_container_train_data_path = os.environ.get('SM_CHANNEL_TRAIN')
    training_container_model_path = os.environ['SM_MODEL_DIR']
    X_train = pd.read_csv(os.path.join(training_container_train_data_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(training_container_train_data_path, "y_train.csv"))
    print("LOADED TRAIN DATA: ", training_container_train_data_path)
    classifier = SVC(kernel="linear", C=1, random_state=42)
    classifier.fit(X_train, y_train)
    model_path = os.path.join(training_container_model_path, "model.joblib")
    joblib.dump(classifier, model_path)
    print("SAVED TRAINED MODEL")

    
def model_fn(model_dir):
    print("model_fn-model_dir: ", model_dir)
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(input_data, content_type):
    print("input_fn-input_data: ", input_data)
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(input_data)
        return df

def predict_fn(input_data, model):
    print("predict_fn-input_data: ", input_data)
    prediction = model.predict(input_data)
    return prediction
