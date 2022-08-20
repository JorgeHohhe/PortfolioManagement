import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def create_directory(directory_name):
    if directory_name not in os.listdir():
        os.mkdir(directory_name)

def get_daily_return_value(stock_price, volatility):
    rnd = np.random.random() - 0.5  # generate number, 0.5 <= x < 0.5
    change_percent = 2 * volatility * rnd
    change_amount = stock_price * change_percent
    new_stock_price = stock_price + change_amount
    # Simple heuristic to not lead with negative values of stock price (unreal scenario)
    if new_stock_price < 0:
        new_stock_price = abs(new_stock_price)
    # Calcutate daily return
    daily_return = (new_stock_price - stock_price) / stock_price
    return new_stock_price, daily_return * 100

def plot_graph_from_dataframe(df, days_of_data):
    fig, axs = plt.subplots(2)
    fig.suptitle('Stocks Data')

    x_daily_return = list(range(days_of_data))
    x_stock_price = x_daily_return[:]
    x_stock_price.append(len(x_daily_return))
    for column in df.columns:
        y_daily_return = df[column]
        axs[0].plot(x_daily_return, y_daily_return)
        axs[0].grid()
        axs[0].set_ylabel('Daily Return (%)')
        axs[0].legend(df.columns, loc=3)

        y_stock_price = [1000]
        for i, daily_return in enumerate(y_daily_return):
            y_stock_price.append(y_stock_price[i] * (1+daily_return/100))
        axs[1].plot(x_stock_price, y_stock_price)
        axs[1].plot(x_stock_price, [1000]*len(x_stock_price), 'k--')
        axs[1].grid()
        axs[1].set_xlabel('Days')
        axs[1].set_ylabel('Stock Price ($)')
    
    fig.savefig(f'output/Sample-{len(df.columns)}stocks-{days_of_data}days.png')

if __name__ == "__main__":
    # To grants reprodutibility
    np.random.seed(0)

    create_directory('output')

    stock_names = ['TECH', 'HEALTH CARE','FINACIAL']
    volatility  = [0.10, 0.15, 0.02]

    days_of_data = 100  # (365 days * 10 years) of data
    df = pd.DataFrame({'Placeholder': [0]*days_of_data})

    for k, stock_name in enumerate(stock_names):
        stock_price = 1000
        df[stock_name] = pd.Series([], dtype=np.float64)
        for i in range(days_of_data): 
            stock_price, daily_return = get_daily_return_value(stock_price, volatility[k])
            df.loc[i, stock_name] = daily_return
    df = df.drop(['Placeholder'], axis=1)
    print(df)

    print('\nExpected Return:')
    for column in df.columns:
        print(column, df[column].mean())
    
    plot_graph_from_dataframe(df, days_of_data)

    print('\nPipeline Finished!')
