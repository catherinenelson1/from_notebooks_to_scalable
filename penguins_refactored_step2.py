def load_data(file_path):
    # download the dataset if it doesn't exist already
    # return a pandas DataFrame
    pass

def clean_data(df):
    # drop rows with missing values
    # return numpy arrays for features and labels
    pass


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(features, labels):
    # encode categorical variables
    # scale numerical features
    # split the data into train/test features and labels
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # train a model and save it 
    pass

def predict_new_data(X_new):
    # load the model and make predictions
    # return a string of the predicted class
    pass