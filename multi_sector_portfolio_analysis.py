#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 19:40:51 2025

@author: ravneetkaursaini
"""


# Multi-Sector Portfolio Analysis

#%%
# PART 1 . Data Acquisition & Visulalisation
# Objective : To gather and visualize 5 years of historical price data for a diversified set of U.S. assets — including 5 individual stocks and 5 sector ETFs — along with the benchmark SPY. This sets the foundation for analyzing performance, risk, and strategic allocation across sectors.
    
#  1a. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 


# 1b. Define tickers
stocks = ['VZ', 'F','XOM','CCI','JNJ']
etfs = ['XLV','XLE','XLRE','XLF','XLC']
benchmark = 'SPY'
tickers = stocks+etfs+[benchmark]


# 1c. Define data range 
start_date= '2020-01-01'
end_date= '2025-10-10'

# 1d. Download the data
data=yf.download(tickers,start=start_date, end=end_date,auto_adjust=False,group_by='tickers')

# 1e. Extract Ajusted Closing Prices
adj_close = pd.DataFrame({ticker:data[ticker]['Adj Close'] for ticker in tickers})

# 1f. Preview the data 
print(" Adjusted Close Sample:")
print(adj_close.head())


# 1g. Plot raw prices - Raw price trends to observe absolute growth.


plt.figure(figsize=(14,6))
for ticker in adj_close.columns:
    plt.plot(adj_close[ticker],label=ticker)
    
plt.title('Adjusted Closing Prices(2020-2025)')
plt.xlabel('Date')
plt.ylabel('Prices($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 1h. Plot normalised prices (start = 100)  - to compare relative growth across assets.
normalised = adj_close/adj_close.iloc[0]*100
normalised.plot(figsize=(14,6))
plt.title('Normalised Price Performance (Start=100)')
plt.xlabel('Date')
plt.ylabel('Indexed Price')
plt.grid(True)
plt.legend()
plt.show()

# CONCLUSION:
# Normalized performance revealed clear sector divergence:
# JNJ and XLV showed strong, steady growth — highlighting healthcare resilience.
# XOM and XLE displayed volatility tied to energy market cycles.
# CCI and XLRE moved in tandem, reflecting real estate trends.
# SPY provided a stable benchmark with moderate growth.
# Ford (F) showed high volatility, typical of consumer discretionary stocks.
# ETFs generally smoothed out sector-specific noise compared to individual stocks.


#%%

# PART 2 - RETURN & RISK CALCULATIONS
# OBJECTIVE : To calculate daily and annualized returns, volatility, and correlation across all assets. This helps us understand performance, risk, and relationships between sectors.

# 2a. Calculate daily returns
daily_returns = adj_close.pct_change().dropna()

# 2b. Calculate mean daily and annualised returns
mean_daily_returns = daily_returns.mean()
annualized_returns = mean_daily_returns* 252 

# 2c. Calculate daily and annualised volatility 
daily_volatility = daily_returns.std()
annualized_volatility = daily_volatility * np.sqrt(252)

# 2d. Calculate daily and annualized covariance matrix
daily_cov_matrix =daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

# 2e.Create Summary table
summary_stats = pd.DataFrame({
    'Annual Returns': annualized_returns,
    'Volatility': annualized_volatility,
    'Sharpe Ratio': annualized_returns/annualized_volatility
    })

print(" Summary Statistics")
print(summary_stats.round(4))



# 2f. Coorelation heatmap
import seaborn as sns
plt.figure(figsize=(14,6))
sns.heatmap(daily_returns.corr(),annot=True, cmap='coolwarm',fmt=".2f")
plt.title('Correlation Matrix of Daily Returns')
plt.legend()
plt.show()

# 2g. Rolling Volatility 
rolling_vot = daily_returns.rolling(window=30).std()
rolling_vot.plot(figsize=(14,6))
plt.title('30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.legend()
plt.show()


# CONCLUSION:
# Top Performers: JNJ and XLV showed consistent returns with relatively low volatility, making them attractive for defensive positioning.
# Volatile Assets: F and XOM exhibited higher volatility, reflecting sensitivity to macroeconomic and commodity cycles.
# Diversification Clarity: The correlation heatmap confirmed that combining ETFs and stocks from different sectors improves diversification.
# Risk Dynamics: Rolling volatility plots revealed that sector ETFs tend to smooth out short-term risk compared to individual stocks.



#%%


# PART 3- Equal-Weight Portfolio Construction

# OBJECTIVE = To build a portfolio with equal allocation across all selected assets (stocks + ETFs), calculate its performance metrics, and compare it against the benchmark (SPY). This helps assess how a simple diversified strategy performs across sectors.

# 3a. Define tickers ( excluidng SPY for portfolios)
portfolio_tickers =[ t for t in tickers if t !='SPY']
equal_weights = np.array([1/len(portfolio_tickers)]*len(portfolio_tickers))    # Equal allocation of weights

# 3b. Calculate daily portfolio returns
equal_daily_returns = daily_returns[portfolio_tickers].dot(equal_weights)

# 3c. Calculate cumulative returns 
equal_cumulative_returns = (1+equal_daily_returns).cumprod()

# 3d. Portfolio performance metrics
equal_daily_return = equal_daily_returns.mean()
equal_annual_return = equal_daily_return * 252
equal_annual_volatility = equal_daily_returns.std() * np.sqrt(252) 
equal_sharpe_ratio = equal_annual_return/equal_annual_volatility

#3e. Benchmark Metrics                                                      
spy_daily_return = daily_returns['SPY']
spy_cumulative_returns = (1+spy_daily_return).cumprod()
spy_annual_return = spy_daily_return.mean() * 252
spy_annual_volatility = spy_daily_return.std() * np.sqrt(252)
spy_sharpe_ratio = spy_annual_return/spy_annual_volatility

# 3f. Display results 
print("Equal-Weight Portfolio Performance:")
print(F"Annualised Return : {equal_annual_return:.4f}")
print(f"Annualized Volatility: {equal_annual_volatility:.4f}")
print(f"Sharpe Ratio: {equal_sharpe_ratio:.4f}")

print(" SPY Benchmark Performance:")
print(f"Annualized Return: {spy_annual_return:.4f}")
print(f"Annualized Volatility: {spy_annual_volatility:.4f}")
print(f"Sharpe Ratio: {spy_sharpe_ratio:.4f}")


# 3g. Cumulative Return Comparison 
plt.figure(figsize=(14, 6))
plt.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
plt.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
plt.title('Cumulative Returns: Equal-Weight vs SPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3h. Portfolio Allocation Pie Chart 
plt.figure(figsize=(6,6))
plt.pie(equal_weights, labels=portfolio_tickers, autopct='%1.1f%%', startangle=140)
plt.title('Equal-Weight Portfolio Allocation')
plt.tight_layout()
plt.show()


# 3i. Maximum Drawdown 
def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

equal_max_dd = max_drawdown(equal_cumulative_returns)
spy_max_dd = max_drawdown(spy_cumulative_returns)

print(f" Max Drawdown (Equal-Weight): {equal_max_dd:.2%}")
print(f" Max Drawdown (SPY): {spy_max_dd:.2%}")


# 3j. Summary Table 
summary_equal = pd.DataFrame({
    'Strategy': ['Equal Weight', 'SPY'],
    'Annual Return': [equal_annual_return, spy_annual_return],
    'Volatility': [equal_annual_volatility, spy_annual_volatility],
    'Sharpe Ratio': [equal_sharpe_ratio, spy_sharpe_ratio],
    'Max Drawdown': [equal_max_dd, spy_max_dd]
})

print(" Strategy Comparison Summary:")
print(summary_equal.round(4))


# CONCLUSION
# The equal-weight portfolio delivered competitive returns relative to SPY, with slightly higher volatility due to sector-specific exposure.
# Sharpe ratio showed strong risk-adjusted performance, validating the benefits of diversification.
# Maximum drawdown was deeper than SPY, highlighting vulnerability during market stress.
# Visuals confirmed steady growth and balanced allocation across sectors.


#%%

# PART 4 : Optimized Portfolio Construction
# OBJECTIVE : To build a portfolio that maximizes expected return while keeping volatility below a target threshold, using mean-variance optimization. This approach refines the equal-weight strategy by allocating capital based on historical return and risk characteristics.

import cvxpy as cp



# 4a. Define portfolio tickers (excluding SPY)
portfolio_tickers = [t for t in tickers if t !='SPY']
returns = daily_returns[portfolio_tickers]

# 4b. Calculate the expected returns and covariance metrics
mu = returns.mean().values 
cov = returns.cov().values

# Define variables
n = len(portfolio_tickers)
w = cp.Variable(n)

# Inputs
mu = returns.mean().values  # expected daily returns
cov = returns.cov().values  # daily covariance matrix
target_volatility = 0.20 / np.sqrt(252)  # target daily volatility (e.g. 20% annualized)

# Objective: maximize expected return
objective = cp.Maximize(mu @ w)

# Constraints: weights sum to 1, no shorting, volatility cap
constraints = [
    cp.sum(w) == 1,
    w >= 0,
    cp.sqrt(cp.quad_form(w, cov)) <= target_volatility
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve(qcp=True)

# Extract weights
opt_weights = w.value



# 4d. Extract optimized weights
opt_weights = w.value
opt_weights = np.round(opt_weights, 4)
opt_weights_df = pd.DataFrame({'Ticker': portfolio_tickers, 'Weight': opt_weights})
print("📊 Optimized Portfolio Weights:")
print(opt_weights_df)

# 4e. Calculate optimized portfolio returns
opt_daily_returns = returns.dot(opt_weights)
opt_cumulative_returns = (1 + opt_daily_returns).cumprod()

# 4f. Performance metrics
opt_annual_return = opt_daily_returns.mean() * 252
opt_annual_volatility = opt_daily_returns.std() * np.sqrt(252)
opt_sharpe_ratio = opt_annual_return / opt_annual_volatility

# 4g. Compare with equal-weight and SPY
comparison_df = pd.DataFrame({
    'Strategy': ['Optimized', 'Equal Weight', 'SPY'],
    'Annual Return': [opt_annual_return, equal_annual_return, spy_annual_return],
    'Volatility': [opt_annual_volatility, equal_annual_volatility, spy_annual_volatility],
    'Sharpe Ratio': [opt_sharpe_ratio, equal_sharpe_ratio, spy_sharpe_ratio]
})

print("\n📋 Strategy Comparison:")
print(comparison_df.round(4))

# 4h. Plot cumulative returns
plt.figure(figsize=(14, 6))
plt.plot(opt_cumulative_returns, label='Optimized Portfolio', linewidth=2)
plt.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
plt.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
plt.title('Cumulative Returns: Optimized vs Equal vs SPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4i. Plot optimized weights
plt.figure(figsize=(8, 6))
plt.bar(opt_weights_df['Ticker'], opt_weights_df['Weight'], color='teal')
plt.title('Optimized Portfolio Allocation')
plt.ylabel('Weight')
plt.grid(True)
plt.tight_layout()
plt.show()


# CONCLUSION:
# The optimized portfolio achieved a higher Sharpe ratio than both the equal-weight strategy and SPY, indicating better risk-adjusted performance.
# Allocation favored assets with strong return-to-risk profiles, reducing exposure to highly volatile or correlated sectors.
# Cumulative return plots showed smoother growth and better downside protection.
# Optimization introduced a more strategic and data-driven approach to portfolio design.



#%%

# PART 5 : Portfolio Risk Analysis, Minimum Volaitility & Scenerio Testing
# OBJECTIVE : To evaluate the risk profile of the optimized portfolio, construct a minimum volatility alternative, and compare strategy performance under various market conditions.
    
# 5a. Calculate Value at Risk 
from scipy.stats import norm

confidence_level = 0.95
portfolio_mean = opt_daily_returns.mean()
portfolio_std = opt_daily_returns.std()
VaR_95 = - (portfolio_mean + portfolio_std * norm.ppf(1 - confidence_level))
print(f" 95% Daily Value at Risk (VaR): {VaR_95:.4f}")

CVaR_95 = - (portfolio_mean + portfolio_std * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level))
print(f" 95% Conditional Value at Risk (CVaR): {CVaR_95:.4f}")

# 5b. Calculate risk contribution by asset
from numpy import dot

portfolio_vol = np.sqrt(dot(opt_weights.T, dot(cov, opt_weights)))
marginal_risk = dot(cov, opt_weights) / portfolio_vol
risk_contribution = opt_weights * marginal_risk
risk_contribution_pct = risk_contribution / portfolio_vol

risk_df = pd.DataFrame({
    'Ticker': portfolio_tickers,
    'Risk Contribution': risk_contribution_pct
})
print("📊 Risk Contribution by Asset:")
print(risk_df.round(4))


# 5c. Minimum Volatility Portfolio
n = len(portfolio_tickers)
w_minvol = cp.Variable(n)
objective_minvol = cp.Minimize(cp.quad_form(w_minvol,cov))
constraints_minvol = [cp.sum(w_minvol) == 1, w_minvol >= 0]
problem_minvol = cp.Problem(objective_minvol, constraints_minvol)
problem_minvol.solve()

minvol_weights = w_minvol.value
minvol_daily_returns = returns.dot(minvol_weights)
minvol_cumulative_returns = (1 + minvol_daily_returns).cumprod()
minvol_annual_return = minvol_daily_returns.mean() * 252
minvol_annual_volatility = minvol_daily_returns.std() * np.sqrt(252)
minvol_sharpe_ratio = minvol_annual_return / minvol_annual_volatility

# 5d. Scenario Testing: Plot Comparison
plt.figure(figsize=(14, 6))
plt.plot(opt_cumulative_returns, label='Optimized Portfolio', linewidth=2)
plt.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
plt.plot(minvol_cumulative_returns, label='Minimum Volatility Portfolio', linestyle='-.')
plt.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
plt.title('Cumulative Returns: All Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5e. Summary Table
summary_all = pd.DataFrame({
    'Strategy': ['Optimized', 'Equal Weight', 'Minimum Volatility', 'SPY'],
    'Annual Return': [opt_annual_return, equal_annual_return, minvol_annual_return, spy_annual_return],
    'Volatility': [opt_annual_volatility, equal_annual_volatility, minvol_annual_volatility, spy_annual_volatility],
    'Sharpe Ratio': [opt_sharpe_ratio, equal_sharpe_ratio, minvol_sharpe_ratio, spy_sharpe_ratio]
})

print("📋 Strategy Comparison Summary:")
print(summary_all.round(4))

# CONCLUSION
# The optimized portfolio delivered strong returns with balanced risk.
# The minimum volatility portfolio offered superior downside protection and smoother performance — ideal for conservative investors.
# VaR and CVaR highlighted potential losses under stress, reinforcing the importance of risk-aware design.
# Scenario testing confirmed that diversification and optimization can significantly improve portfolio resilience compared to passive benchmarks.






#%%

#   PART 6 : Efficient Frontier Visualization
# OBJECTIVE: To visualize the full spectrum of risk-return trade-offs across randomly generated portfolios and benchmark the performance of key strategies against the theoretical efficient frontier.


#  6a. Simulate random portfolios
num_portfolios = 100000
results = np.zeros((num_portfolios, 3))  # columns: return, volatility, Sharpe
weights_record = []

for i in range(num_portfolios):
    weights = np.random.rand(len(portfolio_tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)

    port_return = np.dot(weights, mu) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe = port_return / port_volatility

    results[i] = [port_return, port_volatility, sharpe]
    
#  6b. Convert to DataFrame
results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])

#  6c. Plot Efficient Frontier
plt.figure(figsize=(12, 6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Efficient Frontier')

# Highlight  portfolios
plt.scatter(opt_annual_volatility, opt_annual_return, color='red', marker='*', s=200, label='Optimized')
plt.scatter(minvol_annual_volatility, minvol_annual_return, color='blue', marker='D', s=100, label='Min Volatility')
plt.scatter(equal_annual_volatility, equal_annual_return, color='green', marker='o', s=100, label='Equal Weight')
plt.scatter(spy_annual_volatility, spy_annual_return, color='black', marker='X', s=100, label='SPY')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# CONCLUSION
# The efficient frontier revealed the boundary of optimal portfolios — those offering the highest return for a given level of risk.
# The optimized portfolio lies near the top edge of the frontier, confirming its superior risk-adjusted performance.
# The minimum volatility portfolio sits at the low-risk end, ideal for conservative investors.
# The equal-weight portfolio and SPY fall below the frontier, indicating room for improvement through strategic allocation.





# === Dashboard Functions ===

def get_price_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    for ticker in adj_close.columns:
        ax.plot(adj_close[ticker], label=ticker)
    ax.set_title('Adjusted Closing Prices (2020–2025)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_normalized_chart():
    normalized = adj_close / adj_close.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(14, 6))
    normalized.plot(ax=ax)
    ax.set_title('Normalized Price Performance (Start = 100)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Indexed Price')
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_summary_table():
    return summary_stats.round(4)

def get_correlation_heatmap():
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Daily Returns')
    plt.tight_layout()
    return fig

def get_rolling_volatility_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    rolling_vot.plot(ax=ax)
    ax.set_title('30-Day Rolling Volatility')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_cumulative_comparison_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
    ax.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
    ax.set_title('Cumulative Returns: Equal-Weight vs SPY')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_equal_weight_pie_chart():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(equal_weights, labels=portfolio_tickers, autopct='%1.1f%%', startangle=140)
    ax.set_title('Equal-Weight Portfolio Allocation')
    plt.tight_layout()
    return fig

def get_optimized_weights_chart():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(opt_weights_df['Ticker'], opt_weights_df['Weight'], color='teal')
    ax.set_title('Optimized Portfolio Allocation')
    ax.set_ylabel('Weight')
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_optimized_cumulative_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(opt_cumulative_returns, label='Optimized Portfolio', linewidth=2)
    ax.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
    ax.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
    ax.set_title('Cumulative Returns: Optimized vs Equal vs SPY')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_risk_contribution_table():
    return risk_df.round(4)

def get_strategy_comparison_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(opt_cumulative_returns, label='Optimized Portfolio', linewidth=2)
    ax.plot(equal_cumulative_returns, label='Equal-Weight Portfolio', linestyle='--')
    ax.plot(minvol_cumulative_returns, label='Minimum Volatility Portfolio', linestyle='-.')
    ax.plot(spy_cumulative_returns, label='SPY Benchmark', linestyle=':')
    ax.set_title('Cumulative Returns: All Strategies')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_strategy_summary_table():
    return summary_all.round(4)

def get_efficient_frontier_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.6)
    fig.colorbar(scatter, label='Sharpe Ratio')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Efficient Frontier')
    ax.scatter(opt_annual_volatility, opt_annual_return, color='red', marker='*', s=200, label='Optimized')
    ax.scatter(minvol_annual_volatility, minvol_annual_return, color='blue', marker='D', s=100, label='Min Volatility')
    ax.scatter(equal_annual_volatility, equal_annual_return, color='green', marker='o', s=100, label='Equal Weight')
    ax.scatter(spy_annual_volatility, spy_annual_return, color='black', marker='X', s=100, label='SPY')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# === Exposed Metrics ===
equal_sharpe = equal_sharpe_ratio
spy_sharpe = spy_sharpe_ratio
opt_sharpe = opt_sharpe_ratio
minvol_sharpe = minvol_sharpe_ratio
var_95 = VaR_95
cvar_95 = CVaR_95

