import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Download dataset
        """
    )
    return


@app.cell
def _():
    # dataset from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
    import requests

    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
    response = requests.get(url)

    with open('penguins_data.csv', 'wb') as file:
        file.write(response.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explore and clean the data
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('penguins_data.csv')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    df_1 = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    return (df_1,)


@app.cell
def _(df_1):
    len(df_1)
    return


@app.cell
def _(df_1):
    df_2 = df_1[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    return (df_2,)


@app.cell
def _(df_2):
    df_2.head()
    return


@app.cell
def _(df_2):
    df_2['species'].value_counts()
    return


@app.cell
def _(df_2):
    features = df_2[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()
    values = df_2['species'].to_numpy()
    return features, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Scale and encode the data
        """
    )
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    return LabelEncoder, StandardScaler, train_test_split


@app.cell
def _(features):
    features
    return


@app.cell
def _(StandardScaler, features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return (features_scaled,)


@app.cell
def _(LabelEncoder, values):
    # one-hot encode the species
    encoder = LabelEncoder()
    values_encoded = encoder.fit_transform(values)
    return encoder, values_encoded


@app.cell
def _(values_encoded):
    values_encoded
    return


@app.cell
def _(features_scaled, train_test_split, values_encoded):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, values_encoded, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Try out some models
        """
    )
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    return (
        LogisticRegression,
        RandomForestClassifier,
        classification_report,
        confusion_matrix,
    )


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(X_test, clf):
    y_pred = clf.predict(X_test)
    return (y_pred,)


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred, target_names=['Adelie', 'Gentoo', 'Chinstrap']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Make a prediction on new data
        """
    )
    return


@app.cell
def _(clf, encoder):
    encoder.inverse_transform(clf.predict([[40, 17, 190, 3500]]))[0]
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    return


@app.cell
def _(X_test, clf):
    clf.predict_proba(X_test[:4])
    return


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return (rf,)


@app.cell
def _(X_test, rf, y_test):
    rf.score(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explore the feature importance

        Plot logistic regression coefficients
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _(clf):
    coefficients = clf.coef_
    return (coefficients,)


@app.cell
def _(coefficients, np, plt):
    plt.figure(figsize=(10, 6))
    for i in range(coefficients.shape[0]):
        plt.plot(np.arange(coefficients.shape[1]), coefficients[i], marker='o', label=f'Class {i}')

    plt.xticks(np.arange(coefficients.shape[1]), ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Logistic Regression Coefficients')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

