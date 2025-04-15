import numpy as np
from penguins_refactored_step2 import preprocess_data


def test_preprocess_data():
    features = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]])
    labels = np.array(['Adelie', 'Gentoo', 'Chinstrap'])

    X_train, X_test, y_train, y_test = preprocess_data(features, labels)

    assert X_train.shape == (2, 4)
    assert X_test.shape == (1, 4)
    assert y_train.shape == (2,)
    assert y_test.shape == (1,)

    assert y_train[0] in [0, 1, 2]
    assert y_test[0] in [0, 1, 2]