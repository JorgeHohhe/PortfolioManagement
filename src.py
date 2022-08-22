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

def calculete_overall_portfolio_return(df, stock_names, weights):
    overall_portfolio_return = 0
    for i, stock_name in enumerate(stock_names):
        overall_portfolio_return += weights[i] * df[stock_name].mean()
    return overall_portfolio_return

def calculete_portfolio_standard_deviation(df, stock_names, weights):
    portfolio_standard_deviation = 0
    for i, stock_name_i in enumerate(stock_names):
        for j, stock_name_j in enumerate(stock_names):
            portfolio_standard_deviation += weights[i] * weights[j] * get_standard_deviation_of_df_column(df[stock_name_i]) * get_standard_deviation_of_df_column(df[stock_name_j]) * np.corrcoef([df[stock_name_i], df[stock_name_j]])[0][1]
    portfolio_standard_deviation = np.sqrt(portfolio_standard_deviation)
    return portfolio_standard_deviation

def calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights):
    # Proxy for risk: standard deviation of the previous 21 days
    portfolio_standard_deviation = 0
    for i, stock_name_i in enumerate(stock_names):
        for j, stock_name_j in enumerate(stock_names):
            portfolio_standard_deviation += weights[i] * weights[j] * get_standard_deviation_of_df_column(df[stock_name_i][-21:]) * get_standard_deviation_of_df_column(df[stock_name_j][-21:]) * np.corrcoef([df[stock_name_i][-21:], df[stock_name_j][-21:]])[0][1]
    portfolio_standard_deviation = np.sqrt(portfolio_standard_deviation)
    return portfolio_standard_deviation

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_daily_return_value(df, stock_price, stock_name):
    # GBM (Geometric Brownian Motion) + random volatility
    volatility = np.random.uniform(low=0.0004, high=0.004)
    # Correlation between stock and market (10.7)=10*(1+i)^365 for 7% a.a.
    returns = np.random.normal(loc=0.0001854, scale=volatility, size=1)
    # Correlation between same market sector stocks
    if '_1' not in stock_name:
        if np.random.random() < 0.90: # chance of 90% to update returns
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

def plot_graph_from_dataframe(df, days_of_data, graph_name):
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

    fig.savefig(f'output/{graph_name}-{len(df.columns)//2}stocks-{days_of_data}days.png')
    print(f'Figure saved at: output/{graph_name}-{len(df.columns)//2}stocks-{days_of_data}days.png')

def update_portfolio_weights(weights, daily_dict_to_append, stock_names):
    for i, weight in enumerate(weights):
        weights[i] = weight * (1 + daily_dict_to_append[stock_names[i]]/100)
    weights = weights/np.sum(weights)
    return weights

if __name__ == "__main__":
    # To grants code reproducibility
    np.random.seed(0)

    create_directory('output')
    create_directory('data')

    stock_names = ['CD_1', 'CD_2', 'CD_3', 'CS_1', 'CS_2', 'CS_3', 'TECH_1', 'TECH_2', 'UTILITIES_1', 'UTILITIES_2']

    days_of_data = 365*10  # (365 days * 10 years) of data
    df = pd.read_csv('data/Sample-10stocks-3650days.csv', sep='|')
    # df = pd.DataFrame({'Placeholder': [0]*days_of_data})

    # for k, stock_name in enumerate(stock_names):
    #     stock_price = 100
    #     df[stock_name] = pd.Series([], dtype=np.float64)
    #     df[stock_name + '_PRICE'] = pd.Series([], dtype=np.float64)
    #     for i in range(days_of_data): 
    #         stock_price, daily_return = get_daily_return_value(df.loc[i], stock_price, stock_name)
    #         df.loc[i, stock_name] = daily_return
    #         df.loc[i, stock_name + '_PRICE'] = stock_price
    # df = df.drop(['Placeholder'], axis=1)
    # # Save the simutation
    # df.to_csv(f'data/Sample-{len(df.columns)//2}stocks-{days_of_data}days.csv', sep='|',index=False)
    # print(f'Stocks Simulated Data saved at: data/Sample-{len(df.columns)//2}stocks-{days_of_data}days.csv')

    # Overall information about stocks data
    weights = [0.1]*10
    overall_market_return = calculete_overall_portfolio_return(df, stock_names, weights)
    print(f'\nOverall Market Return: {overall_market_return} %')
    
    market_standard_deviation = calculete_portfolio_standard_deviation(df, stock_names, weights)
    print(f'\nMarket Standard Deviation: {market_standard_deviation}\n')

    # Plot graphs
    # plot_graph_from_dataframe(df, days_of_data, 'Sample')
    # for market_sector in ['CD', 'CS', 'TECH', 'UTILITIES']:
    #     plot_graph_from_dataframe(df[[stock for stock in df.columns if market_sector in stock]], days_of_data, market_sector)

    ### FIRST MONTH
    df_weights = pd.DataFrame(columns=['Weights'])
    # Initial Capital Allocation
    acceptable_risk = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
    print(f'Acceptable Risk: {acceptable_risk}')
    num_simulations = 5
    efficient_frontier = pd.DataFrame(columns=['Weights', 'Returns', 'Std_dev'])
    for i in range(num_simulations):
        weights = np.random.random(len(stock_names))
        weights = weights/np.sum(weights)
        returns = calculete_overall_portfolio_return(df, stock_names, weights)
        std_dev = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
        efficient_frontier = efficient_frontier.append({'Weights': weights, 'Returns': returns, 'Std_dev': std_dev}, ignore_index=True)
    index_nearest_to_acceptable_risk = find_nearest(efficient_frontier['Std_dev'], acceptable_risk)
    weights = efficient_frontier['Weights'][index_nearest_to_acceptable_risk]
    df_weights = df_weights.append({'Weights': weights}, ignore_index=True)

    first_month_weights = weights
    print('\nFirst month composition of my portfolio:')
    for i, stock_name in enumerate(stock_names):
        print(f'{stock_name}: {first_month_weights[i]:.6f}')

    ### FIRST TO SECOND MONTH
    # Simulate more 21 days and update portfolio weights each day
    days_of_data = 21
    start_position_index = len(df) - 1
    for i in range(days_of_data):
        daily_dict_to_append = {}
        stock_prices, daily_returns = [], []
        for stock_name in stock_names:
            stock_price = df[stock_name+'_PRICE'][len(df)-1]
            stock_price, daily_return = get_daily_return_value(df.loc[i+start_position_index], stock_price, stock_name)
            daily_dict_to_append[stock_name+'_PRICE'] = stock_price[0]
            daily_dict_to_append[stock_name] = daily_return[0]
        weights = update_portfolio_weights(weights, daily_dict_to_append, stock_names)
        df_weights = df_weights.append({'Weights': weights}, ignore_index=True)
        df = df.append(daily_dict_to_append, ignore_index=True)
    # Rebalance the portfolio
    acceptable_risk = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
    print(f'\nAcceptable Risk: {acceptable_risk}')
    num_simulations = 5
    efficient_frontier = pd.DataFrame(columns=['Weights', 'Returns', 'Std_dev'])
    for i in range(num_simulations):
        weights = np.random.random(len(stock_names))
        weights = weights/np.sum(weights)
        returns = calculete_overall_portfolio_return(df, stock_names, weights)
        std_dev = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
        efficient_frontier = efficient_frontier.append({'Weights': weights, 'Returns': returns, 'Std_dev': std_dev}, ignore_index=True)
    index_nearest_to_acceptable_risk = find_nearest(efficient_frontier['Std_dev'], acceptable_risk)
    weights = efficient_frontier['Weights'][index_nearest_to_acceptable_risk]
    # Remove last weight row because it will be rebalanced
    df_weights = df_weights.drop(df_weights.tail(1).index)
    df_weights = df_weights.append({'Weights': weights}, ignore_index=True)

    second_month_weights = weights
    print('\nSecond month composition of my portfolio:')
    for i, stock_name in enumerate(stock_names):
        print(f'{stock_name}: {second_month_weights[i]:.6f}')
    print('\nTrades:')
    for i, stock_name in enumerate(stock_names):
        print(f'{stock_name}: {100*(second_month_weights[i] - first_month_weights[i])/first_month_weights[i]:.6f} %')

    ### SECOND TO THIRD MONTH
    # Simulate more 21 days and update portfolio weights each day
    days_of_data = 21
    start_position_index = len(df) - 1
    for i in range(days_of_data):
        daily_dict_to_append = {}
        stock_prices, daily_returns = [], []
        for stock_name in stock_names:
            stock_price = df[stock_name+'_PRICE'][len(df)-1]
            stock_price, daily_return = get_daily_return_value(df.loc[i+start_position_index], stock_price, stock_name)
            daily_dict_to_append[stock_name+'_PRICE'] = stock_price[0]
            daily_dict_to_append[stock_name] = daily_return[0]
        weights = update_portfolio_weights(weights, daily_dict_to_append, stock_names)
        df_weights = df_weights.append({'Weights': weights}, ignore_index=True)
        df = df.append(daily_dict_to_append, ignore_index=True)
    # Rebalance the portfolio
    acceptable_risk = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
    print(f'\nAcceptable Risk: {acceptable_risk}')
    num_simulations = 5
    efficient_frontier = pd.DataFrame(columns=['Weights', 'Returns', 'Std_dev'])
    for i in range(num_simulations):
        weights = np.random.random(len(stock_names))
        weights = weights/np.sum(weights)
        returns = calculete_overall_portfolio_return(df, stock_names, weights)
        std_dev = calculete_portfolio_standard_deviation_previous_21_days(df, stock_names, weights)
        efficient_frontier = efficient_frontier.append({'Weights': weights, 'Returns': returns, 'Std_dev': std_dev}, ignore_index=True)
    index_nearest_to_acceptable_risk = find_nearest(efficient_frontier['Std_dev'], acceptable_risk)
    weights = efficient_frontier['Weights'][index_nearest_to_acceptable_risk]
    # Remove last weight row because it will be rebalanced
    df_weights = df_weights.drop(df_weights.tail(1).index)
    df_weights = df_weights.append({'Weights': weights}, ignore_index=True)

    third_month_weights = weights
    print('\nThird month composition of my portfolio:')
    for i, stock_name in enumerate(stock_names):
        print(f'{stock_name}: {third_month_weights[i]:.6f}')
    print('\nTrades:')
    for i, stock_name in enumerate(stock_names):
        print(f'{stock_name}: {100*(third_month_weights[i] - second_month_weights[i])/second_month_weights[i]:.6f} %')

    ### THIRD MONTH
    # Simulate more 21 days and update portfolio weights each day
    days_of_data = 21
    start_position_index = len(df) - 1
    for i in range(days_of_data):
        daily_dict_to_append = {}
        stock_prices, daily_returns = [], []
        for stock_name in stock_names:
            stock_price = df[stock_name+'_PRICE'][len(df)-1]
            stock_price, daily_return = get_daily_return_value(df.loc[i+start_position_index], stock_price, stock_name)
            daily_dict_to_append[stock_name+'_PRICE'] = stock_price[0]
            daily_dict_to_append[stock_name] = daily_return[0]
        weights = update_portfolio_weights(weights, daily_dict_to_append, stock_names)
        df_weights = df_weights.append({'Weights': weights}, ignore_index=True)
        df = df.append(daily_dict_to_append, ignore_index=True)

    ### PORTFOLIO PERFORMANCE
    # Return
    portfolio_return = 1
    month_portfolio_return = 1
    list_monthly_portfolio_return = []
    for day in range(21*3):
        daily_portfolio_return = 0
        for i, stock_name in enumerate(stock_names):
            daily_portfolio_return += df_weights['Weights'][day][i] * (1 + np.array([df[stock_name]])[0][-63+day]/100)
        month_portfolio_return *= daily_portfolio_return
        portfolio_return *= daily_portfolio_return
        if day%21==0:
            list_monthly_portfolio_return += [month_portfolio_return]
            month_portfolio_return = 1
    print(list_monthly_portfolio_return)
    print(f'\nReturn: {100*(portfolio_return-1):.6f} %')
    portfolio_volatility = np.std(list_monthly_portfolio_return)
    print(f'Monthly Volatility: {portfolio_volatility:.6f}')
    sharpe_ratio = (portfolio_return-1)/portfolio_volatility
    print(f'Sharpe Ratio: {sharpe_ratio:.6f}')

    print('\nPipeline Finished!')
