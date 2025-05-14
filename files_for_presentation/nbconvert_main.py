
import requests
import pandas as pd

if __name__ == "__main__":
    # dataset from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
    response = requests.get(url)

    with open('penguins_data.csv', 'wb') as file:
        file.write(response.content)

    # ### Explore the data
    df = pd.read_csv('penguins_data.csv')

    df.head()

    df.isnull().sum()