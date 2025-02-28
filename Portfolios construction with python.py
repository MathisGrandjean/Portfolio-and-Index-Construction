import pandas as pd 
import yfinance as yf
from scipy.optimize import minimize
import numpy as np 

""" This is a final group assigment for the coursera course Introduction to portfolio construction with Python
The goal is to create several portfolios like the GMV, the MSR, the Equally Weighted and to compare them according to
their returns but also various measures of risk
"""

tickers = ["AAPL", "MSFT", "JNJ", "TSLA", "V", "SPY", "QQQ", "IEF", "LQD", "GLD"]
start_date = "2014-01-01"
end_date = "2024-12-01"
opening = pd.DataFrame()


for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)  
    opening[ticker] = data['Open']  


riskfree_rate=0.01
daily_returns = opening.pct_change().dropna()
expected_returns = daily_returns.mean()*252
cov_matrix = daily_returns.cov()

#we import all the function for edhec risk kit that is a module developped in class
def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns
def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5
def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x
def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year): 
    return r.std()*(periods_per_year**0.5)

"""
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef_with_portfolios(n_points, er, cov, riskfree_rate, w_msr, w_gmv, w_target):
    """
    Plots the efficient frontier and highlights MSR, GMV, and Target portfolios.
    """
    # Calculate the efficient frontier
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    
    # Create the efficient frontier DataFrame
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })


#we calculate the weigh of the gmv and msr portfolio
w_msr=msr(0.05,expected_returns.values,cov_matrix.values)
w_gmv =gmv(cov_matrix.values)
w_ew = np.repeat(1/10, 10)

# for each portfolio we calculate the return, the volatility, the sharpe ratio and the historic VaR
vol_msr=portfolio_vol(w_msr, cov_matrix)*np.sqrt(252)
vol_gmv=portfolio_vol(w_gmv, cov_matrix)*np.sqrt(252)
vol_ew=portfolio_vol(w_ew, cov_matrix)*np.sqrt(252)
return_msr= portfolio_return(w_msr, expected_returns)
return_gmv= portfolio_return(w_gmv, expected_returns)
return_ew= portfolio_return(w_ew, expected_returns)
sharpe_ratio_msr=(return_msr-riskfree_rate)/vol_msr
sharpe_ratio_gmv=(return_gmv-riskfree_rate)/vol_gmv
sharpe_ratio_ew=(return_ew-riskfree_rate)/vol_ew


#we calculate the return of each portfolio to calculate the historic VaR
Portfolio=pd.DataFrame()
Portfolio['portfolio MSR']= daily_returns.dot(w_msr)
Portfolio['portfolio GMV']= daily_returns.dot(w_gmv)
Portfolio['Returns MSR']=Portfolio['portfolio MSR'].pct_change()
Portfolio['Returns GMV']=Portfolio['portfolio GMV'].pct_change()
Portfolio['portfolio EW']= daily_returns.dot(w_ew)
Portfolio['Returns EW']=Portfolio['portfolio EW'].pct_change()
varhisto_MSR= var_historic(Portfolio['portfolio MSR'])
varhisto_GMV= var_historic(Portfolio['portfolio GMV'])
varhisto_EW= var_historic(Portfolio['portfolio EW'])


#we do the same for the portfolio that minimize the volatility with a target return of 15%
w_target_return= minimize_vol(0.15, expected_returns.values, cov_matrix.values)
return_target_return=portfolio_return(w_target_return, expected_returns)
vol_target_return =portfolio_vol(w_target_return, cov_matrix)*np.sqrt(252)
sharpe_ratio_target_return=(return_target_return-riskfree_rate)/vol_target_return
Portfolio['portfolio Target Return']= daily_returns.dot(w_target_return)
Portfolio['Returns Target Return']=Portfolio['portfolio Target Return'].pct_change()
varhisto_target_return= var_historic(Portfolio['portfolio Target Return'])



import matplotlib.pyplot as plt
# we plot the efficient frontier
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ef["Volatility"], ef["Returns"], label="Efficient Frontier", color="blue", linestyle="--")

# Plot MSR Portfolio
msr_return = portfolio_return(w_msr, er)
msr_volatility = portfolio_vol(w_msr, cov)
ax.scatter(msr_volatility, msr_return, color="red", label="MSR Portfolio", s=100)

# Plot GMV Portfolio
gmv_return = portfolio_return(w_gmv, er)
gmv_volatility = portfolio_vol(w_gmv, cov)
ax.scatter(gmv_volatility, gmv_return, color="green", label="GMV Portfolio", s=100)
   
ew_return = portfolio_return(w_ew, er)
ew_vol= portfolio_vol(w_ew, cov)
ax.scatter( ew_vol,ew_return, color="blue", label="EW Portfolio", s=100)

# Plot Target Portfolio
target_return = portfolio_return(w_target, er)
target_volatility = portfolio_vol(w_target, cov)
ax.scatter(target_volatility, target_return, color="orange", label="Target Portfolio", s=100)

 # Add labels, legend, and title
ax.set_title("Efficient Frontier with Portfolios")
ax.set_xlabel("Volatility (Risk)")
ax.set_ylabel("Expected Return")
ax.legend()
plt.grid()
plt.show()

# We plot the efficient frontier
plot_ef_with_portfolios(n_points=20, er=expected_returns.values, cov=cov_matrix.values, riskfree_rate=riskfree_rate,w_msr=w_msr,w_gmv=w_gmv,w_target=w_target_return)

# we summarize all the data in a table 
summary_data = {
    "Portfolio": ["MSR", "GMV", "Target Return","EW"],
    "Return": [return_msr, return_gmv, return_target_return,return_ew],
    "Volatility": [vol_msr, vol_gmv, vol_target_return,vol_ew],
    "Sharpe Ratio": [
        sharpe_ratio_msr,sharpe_ratio_gmv,sharpe_ratio_target_return,sharpe_ratio_ew],
    "Value-at-Risk (5%)": [
        var_historic(Portfolio['portfolio MSR']),
        var_historic(Portfolio['portfolio GMV']),
        varhisto_target_return,
        varhisto_EW
        
    ]
}
print(w_gmv)
weight_table ={
    "Assets":["AAPL", "MSFT", "JNJ", "TSLA", "V", "SPY", "QQQ", "IEF", "LQD", "GLD"],
    "GMV":w_gmv,
    "MSR":w_msr,
    "Target Return": w_target_return,
    "EW": w_ew
    }
df=pd.DataFrame(weight_table)
df['GMV'] = df['GMV'].apply(lambda x: round(x, 4))
df['MSR'] = df['MSR'].apply(lambda x: round(x, 4))
df['Target Return'] = df['Target Return'].apply(lambda x: round(x, 4))
print(df)

summary_df = pd.DataFrame(summary_data)
print(summary_df)
