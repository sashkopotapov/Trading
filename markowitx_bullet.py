import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
# import sp
import matplotlib.patches as mpatches

selected = ['AAPL','AMZN','AXP','BA','BHP','CAT',
            'FB','GOOG','NFLX','MRK','JNJ','IVPAF',
            'UFPI','SAM','RIO','PFE','WST','VALE','UTX','UNH']


table_0 = pd.read_csv("stocks_data.csv")
table = table_0.set_index("Date")


# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()



returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
num_portfolios = 500

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)


#print(port_returns)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)
#print(df)


column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]


df = df[column_order]


min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()


sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]


plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Markowitz Bullet')
red_patch = mpatches.Patch(color='Red', label='Highest sharp ratio portfolio')
blue_patch = mpatches.Patch(color='Blue', label='Min volatility portfolio')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

#min variance portfolio with optimal return
# print("Min variance portfolio: \n", min_variance_port.T)

#highest sharp ratio portfolio
print("Highest sharp ratio: \n", sharpe_portfolio.T)

weights_sharp = sharpe_portfolio.as_matrix(columns = ['AAPL Weight','AMZN Weight',
                                                'AXP Weight','BA Weight',
                                                'BHP Weight','CAT Weight',
                                                'FB Weight','GOOG Weight',
                                                'NFLX Weight','MRK Weight',
                                                'JNJ Weight','IVPAF Weight',
                                                'UFPI Weight','SAM Weight',
                                                'RIO Weight','PFE Weight',
                                                'WST Weight','VALE Weight',
                                                'UTX Weight','UNH Weight'])

weights_minvol = min_variance_port.as_matrix(columns = ['AAPL Weight','AMZN Weight',
                                                'AXP Weight','BA Weight',
                                                'BHP Weight','CAT Weight',
                                                'FB Weight','GOOG Weight',
                                                'NFLX Weight','MRK Weight',
                                                'JNJ Weight','IVPAF Weight',
                                                'UFPI Weight','SAM Weight',
                                                'RIO Weight','PFE Weight',
                                                'WST Weight','VALE Weight',
                                                'UTX Weight','UNH Weight'])

matrix = table.as_matrix()

#multiple returns by weights and convert to list
sharp_portfolio_returns = np.inner(matrix,weights_sharp).tolist()
minvol_portfolio_returns = np.inner(matrix,weights_minvol).tolist()



returns_sharp  = list(map(lambda x: float(x[0]), sharp_portfolio_returns))
returns_minvol = list(map(lambda x: float(x[0]), minvol_portfolio_returns))

#get list of dates of the whole period
dates = table_0['Date'].tolist()

#create a dataframe with our portfolios
df = pd.DataFrame({'Date' : dates, 'Return sharp' : returns_sharp, 'Return Min Volatility' : returns_minvol})
df['Date'] = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)

#create plot here
plt.plot(df)
plt.gcf().autofmt_xdate()
plt.title('Returns of portfolios in retrospective')

red_patch = mpatches.Patch(color='Orange', label='Highest sharp ratio portfolio')
blue_patch = mpatches.Patch(color='Blue', label='Min volatility portfolio')
plt.legend(handles=[red_patch, blue_patch])

plt.ylabel('US Dollar')
plt.xlabel('Date')
plt.show()
