
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    
    # define file paths
    file_name = "heart.csv"
    processing_container_base_path = "/opt/ml/processing"
    processing_container_input_data_path = os.path.join(processing_container_base_path, "input")
    processing_container_train_data_path = os.path.join(processing_container_base_path, "train")
    processing_container_test_data_path = os.path.join(processing_container_base_path, "test")

    # read the csv inot dataframe
    df = pd.read_csv(os.path.join(processing_container_input_data_path, file_name))
    
    # define the columns to be encoded and scaled
    categorical_columns = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
    numerical_columns = ["age","trtbps","chol","thalachh","oldpeak"]
    
    # one hot encoding of categorical variables
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print('ONE-HOT ENCODING COMPLETED')
    
    X = df.drop(['output'], axis=1)
    y = df[['output']]
    print('FEATURES AND TARGETS SEPARATED')
    
    scaler = RobustScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    print('SCALING COMPLETED')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('DATA SPLIT COMPLETED')
    
    X_train.to_csv(os.path.join(processing_container_train_data_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processing_container_test_data_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processing_container_train_data_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processing_container_test_data_path, "y_test.csv"), index=False)
    print('SAVED TRANSFORMED DATA FILES')
