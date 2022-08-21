import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def create_directory(directory_name):
    if directory_name not in os.listdir():
        os.mkdir(directory_name)

def get_mean_of_df_column(df_column):
    return np.mean(df_column)

def get_standard_deviation_of_df_column(df_column):
    return np.std(df_column)

def get_daily_return_value(df, stock_name):
    # GBM (Geometric Brownian Motion) + random volatility
    # volatility = np.random.random()
    # Correlation beetween stock and market (10.7)=10*(1+i)^365 for 7% a.a.
    # Parameters
    # drift coefficent
    mu = 0.0001854
    # number of steps
    n = 364
    # time in years
    T = 1
    # number of sims
    M = 1
    # initial stock price
    S0 = 100
    # volatility
    sigma = 0.3
    # calc each time step
    dt = T/n
    # simulation using numpy arrays
    St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    # include array of 1's
    St = np.vstack([np.ones(M), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path axis=0. 
    stock_price = S0 * St.cumprod(axis=0)

    # returns = np.random.normal(loc=0.0001854, scale=0.01, size=1)
    # new_stock_price = stock_price*(1+returns)
    # Simple heuristic to not lead with negative values of stock price (unreal scenario)
    stock_price = abs(stock_price)
    # Calcutate daily return
    daily_return = []
    for i in range(len(stock_price)-1):
        daily_return += [((stock_price[i+1] - stock_price[i]) / stock_price[i]) * 100]
    return stock_price, daily_return

def plot_graph_from_dataframe(df, days_of_data):
    fig, axs = plt.subplots(2)
    fig.suptitle('Stocks Data')

    x_daily_return = list(range(days_of_data))
    x_stock_price = x_daily_return[:]
    x_stock_price.append(len(x_daily_return))
    legend_columns = []
    for column in df.columns:
        if '_PRICE' not in column:
            legend_columns += [column]
            y_daily_return = df[column]
            axs[0].plot(x_daily_return, y_daily_return)
            
        else:
            y_stock_price = np.append([100], df[column])
            axs[1].plot(x_stock_price, y_stock_price)
    
    axs[0].grid()
    axs[0].set_ylabel('Daily Return (%)')
    axs[0].legend(legend_columns, loc=3)

    axs[1].plot(x_stock_price, [100]*len(x_stock_price), 'k--')
    axs[1].grid()
    axs[1].set_xlabel('Days')
    axs[1].set_ylabel('Stock Price ($)')

    fig.savefig(f'output/Sample-{len(df.columns)//2}stocks-{days_of_data}days-Oficial.png')

if __name__ == "__main__":
    # To grants code reproducibility
    np.random.seed(0)

    create_directory('output')

    stock_names = ['TECH', 'HEALTH CARE','FINACIAL']

    days_of_data = 365  # (365 days * 10 years) of data
    df = pd.DataFrame({'Placeholder': [0]*days_of_data})

    for k, stock_name in enumerate(stock_names):
        stock_price, daily_return = get_daily_return_value(df, stock_name)
        daily_return.append(np.nan)
        daily_return = np.array(daily_return, dtype=np.float64)
        df[stock_name] = daily_return
        df[stock_name + '_PRICE'] = stock_price 
    df = df.drop(['Placeholder'], axis=1)
    print(df)

    print('\nMean:')
    for column in df.columns:
        print(column, df[column].mean())
    
    plot_graph_from_dataframe(df, days_of_data)

    print('\nPipeline Finished!')
