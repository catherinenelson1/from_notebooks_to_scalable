import numpy as np
import pandas as pd
import numpy as np
import os
import joblib

from penguins_refactored import (
    clean_data, preprocess_data, train_model, predict_new_data
)

ENCODER_FILENAME = 'test_penguins_label_encoder.joblib'
MODEL_FILENAME = 'test_penguins_model.joblib'


def create_test_data():
    test_data = pd.DataFrame({
        'species': ['Adelie', 'Gentoo', 'Chinstrap'],
        'bill_length_mm': [39.1, 46.5, 49.7, None],
        'bill_depth_mm': [18.7, 14.3, 16.0, None],
        'flipper_length_mm': [181, 217, 193, None],
        'body_mass_g': [3750, 5200, 3800, None],
    })
    test_data.to_csv('test_penguins.csv', index=False)


def test_clean_data():
    # Arrange - create the data
    create_test_data()
    
    # Act - test the function
    features, labels = clean_data('test_penguins.csv')
    
    # Assert - check results
    assert features.shape == (3, 4)
    assert labels.shape == (3,)
    assert 'Adelie' in labels
    assert 'Gentoo' in labels
    assert 'Chinstrap' in labels
    
    # Clean up
    os.remove('test_penguins.csv')


def test_preprocess_data():
    features = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]])
    labels = np.array(['Adelie', 'Gentoo', 'Chinstrap'])

    X_train, X_test, y_train, y_test = preprocess_data(features, labels, ENCODER_FILENAME)

    assert X_train.shape == (2, 4)
    assert X_test.shape == (1, 4)
    assert y_train.shape == (2,)
    assert y_test.shape == (1,)

    assert y_train[0] in [0, 1, 2]
    assert y_test[0] in [0, 1, 2]

    assert os.path.exists(ENCODER_FILENAME)  # Encoder should be saved
    
    # Clean up
    os.remove(ENCODER_FILENAME)


def test_train_model():
    """Test train_model with simple inputs"""
    # Create simple test data
    X_train = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    X_test = np.array([[0.9, 0.8, 0.7, 0.6]])
    y_train = np.array([0, 1])
    y_test = np.array([1])
    
    # Test the function
    train_model(X_train, y_train, X_test, y_test, MODEL_FILENAME)
    
    # Check if model file was created
    assert os.path.exists(MODEL_FILENAME)
    
    # Clean up
    os.remove(MODEL_FILENAME)


def test_predict_new_data():
    """Test prediction with a pre-trained model"""
    # Create and save a simple model
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    
    # Train a simple model
    X = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.8, 0.7, 0.6]])
    y = np.array(['Adelie', 'Gentoo', 'Chinstrap'])
    
    # Create and save encoder
    encoder = LabelEncoder()
    encoder.fit(y)
    joblib.dump(encoder, ENCODER_FILENAME)
    
    # Create and save model
    model = LogisticRegression()
    model.fit(X, encoder.transform(y))
    joblib.dump(model, MODEL_FILENAME)
    
    # Test the prediction function
    result = predict_new_data(np.array([[0.1, 0.2, 0.3, 0.4]]))
    
    # Check the prediction is one of the valid species
    assert result in ['Adelie', 'Gentoo', 'Chinstrap']
    
    # Clean up
    os.remove(MODEL_FILENAME)
    os.remove(ENCODER_FILENAME)

