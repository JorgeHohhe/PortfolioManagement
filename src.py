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

def get_daily_return_value(df, stock_price, stock_name):
    # GBM (Geometric Brownian Motion) + random volatility
    volatility = np.random.uniform(low=0.001, high=0.02)
    # Correlation between stock and market (10.7)=10*(1+i)^365 for 7% a.a.
    returns = np.random.normal(loc=0.0001854, scale=volatility, size=1)
    # Correlation between same market sector stocks
    if '_1' not in stock_name:
        if np.random.random() < 0.9: # chance of 90% to update returns
            if df[stock_name[:-2] + '_1'] > 0: # positive daily return
                    returns = abs(returns)
            else:
                returns = (-1) * abs(returns)
    # Update stock price
    new_stock_price = stock_price*(1+returns)
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
    axs[0].legend(legend_columns, loc=3, prop={'size': 6})

    axs[1].plot(x_stock_price, [100]*len(x_stock_price), 'k--')
    axs[1].grid()
    axs[1].set_xlabel('Days')
    axs[1].set_ylabel('Stock Price ($)')

    fig.savefig(f'output/Sample-{len(df.columns)//2}stocks-{days_of_data}days.png')
    print(f'Figure saved in: output/Sample-{len(df.columns)//2}stocks-{days_of_data}days.png')

if __name__ == "__main__":
    # To grants code reproducibility
    np.random.seed(0)

    create_directory('output')

    stock_names = ['CD_1', 'CD_2', 'CD_3', 'CS_1', 'CS_2', 'CS_3', 'TECH_1', 'TECH_2', 'UTILITIES_1', 'UTILITIES_2']

    days_of_data = 365  # (365 days * 10 years) of data
    df = pd.DataFrame({'Placeholder': [0]*days_of_data})

    for k, stock_name in enumerate(stock_names):
        stock_price = 100
        df[stock_name] = pd.Series([], dtype=np.float64)
        df[stock_name + '_PRICE'] = pd.Series([], dtype=np.float64)
        for i in range(days_of_data): 
            stock_price, daily_return = get_daily_return_value(df.loc[i], stock_price, stock_name)
            df.loc[i, stock_name] = daily_return
            df.loc[i, stock_name + '_PRICE'] = stock_price
    df = df.drop(['Placeholder'], axis=1)
    # print(df)

    print('\nMean:')
    for column in df.columns:
        print(column, df[column].mean())
    
    plot_graph_from_dataframe(df, days_of_data)

    print('\nPipeline Finished!')
