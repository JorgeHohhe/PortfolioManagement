import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os


def create_directory(directory_name):
    if directory_name not in os.listdir():
        os.mkdir(directory_name)

def get_daily_return_value(stock_price, volatility):
    rnd = random.random()  # generate number, 0 <= x < 1.0
    change_percent = 2 * volatility * rnd
    if (change_percent > volatility):
        change_percent -= (2 * volatility)
    change_amount = stock_price * change_percent
    new_stock_price = stock_price + change_amount
    daily_return = (new_stock_price - stock_price) / stock_price
    return new_stock_price, daily_return * 100

def plot_graph_from_dataframe(df, days_of_data):
    x = list(range(days_of_data))
    for column in df.columns:
        y = df[column]
        plt.plot(x, y)
    
    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Daily Return (%)')
    plt.legend(df.columns)
    plt.savefig(f'output/Sample-{len(df.columns)}stocks-{days_of_data}days.png')

if __name__ == "__main__":
    create_directory('output')

    stock_names = ['TECH', 'HEALTH CARE','FINACIAL']
    volatility  = [0.10, 0.15, 0.02]

    days_of_data = 10  # (365 days * 10 years) of data
    df = pd.DataFrame({'Placeholder': [0]*days_of_data})

    for k, stock_name in enumerate(stock_names):
        stock_price = 1000
        df[stock_name] = pd.Series([], dtype=np.float64)
        for i in range(days_of_data): 
            stock_price, daily_return = get_daily_return_value(stock_price, volatility[k]) # 2%
            df.loc[i, stock_name] = daily_return 
    df = df.drop(['Placeholder'], axis=1)
    print(df)

    print('\nExpected Return:')
    for column in df.columns:
        print(column, df[column].mean())
    
    plot_graph_from_dataframe(df, days_of_data)

    print('\nPipeline Finished!')
