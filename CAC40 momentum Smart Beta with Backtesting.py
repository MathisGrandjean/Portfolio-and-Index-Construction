import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

"""we want to create a smart beta index that will replicate only the n assets of the cac40 that have the best 
returns on the d choosen day period
I used chatgpt in order to get the ticker and the name of each company and also to clean the code
"""

def calculate_momentum(price_data, lookback_period):
    """
    Calculates momentum for each asset over the lookback period
    """
    momentum = price_data.pct_change(lookback_period).iloc[-1]  # Performance over lookback period
    return momentum

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    """
    return np.dot(weights.T, returns)

def portfolio_vol(weights, covmat):
    """
    Computes the volatility of a portfolio from a covariance matrix and weights
    """
    return np.sqrt(np.dot(weights.T, np.dot(covmat, weights)))

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1



tickers_cac40 = [
    "AC.PA", "AIR.PA", "AI.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA",
    "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "DSY.PA", "ENGI.PA", "EL.PA",
    "ERF.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA",
    "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA",
    "SU.PA", "GLE.PA", "TEP.PA", "HO.PA", "TTE.PA",
    "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA"
]


#we import the data on yahoo finance
price = yf.download(tickers_cac40, start="2020-01-01", end="2023-12-31")["Adj Close"]
price = price.dropna()


#we choose the period that we want to lookback and then we calculate the momentum of each constituents on this period
lookback_period = 126
momentum_scores = calculate_momentum(price, lookback_period)

#we choose the 10 better assts and we create the df selected price with the price of our 10 assets
top_assets = momentum_scores.nlargest(10).index
selected_price = price[top_assets]

# We calculate the return of each assets and give weight to the index according to their return
returns_selected = selected_price.pct_change().dropna()
weights = momentum_scores[top_assets] / momentum_scores[top_assets].sum()

#we calculate the weight of the CAC40 according to the capitalisation of each assets
capitalisations = []
for symbol in tickers_cac40:
    stock = yf.Ticker(symbol)
    market_cap = stock.info.get("marketCap")  
    capitalisations.append(market_cap)

capitalisations = [cap for cap in capitalisations]
capitalisations = np.array(capitalisations)
weights_cac40 = capitalisations / np.sum(capitalisations)

#We plot the returns of our index and the return of the  CAC40
portfolio_returns = returns_selected.dot(weights)
cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
returns_cac40 = price.pct_change().dropna()
benchmark_returns = returns_cac40.dot(weights_cac40)
cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()



plt.figure(figsize=(10, 6))
plt.plot(cumulative_portfolio_returns, label="Smart Beta (Momentum)", linewidth=2)
plt.plot(cumulative_benchmark_returns, label="CAC40 Benchmark", linewidth=2)
plt.title("Smart Beta Portfolio vs. CAC40 Benchmark")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()
