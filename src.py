import numpy as np
import pandas as pd
import matplotlib as plt
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

if __name__ == "__main__":
    create_directory('output')

    stock_names = ['TECH', 'HEALTH CARE','FINACIAL']
    volatility  = [0.10, 0.15, 0.02]

    days_of_data = 5  # (365 days * 10 years) of data
    df = pd.DataFrame({'Holder': ['a']*days_of_data})

    for k, stock_name in enumerate(stock_names):
        stock_price = 1000
        df[stock_name] = pd.Series([], dtype=np.float64)
        for i in range(days_of_data): 
            stock_price, daily_return = get_daily_return_value(stock_price, volatility[k]) # 2%
            df.loc[i, stock_name] = daily_return 
            # df_to_append = {stock_name: daily_return}
            # df = df.append(df_to_append, ignore_index = True)
    df = df.drop(['Holder'], axis=1)
    print(df)
    

    print('\nPipeline Finished!')
