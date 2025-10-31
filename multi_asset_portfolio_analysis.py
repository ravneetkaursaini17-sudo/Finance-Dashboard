#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:43:14 2025

@author: ravneetkaursaini
"""

### Completed as a group by Guobin Gao , Ravneet Kaur Saini , Ayna Tovekelova and Shivaani Gurusankaran  ###


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot

#%%
# 1) Data Collection Parameters

START = '2020-10-01'
END = '2025-10-01'
TRADING_DAYS = 252
RF_ANNUAL = 0.03

# Selected 8 companies from distinct sectors; 
# SBUX - Starbucks
# UNP - Union Pacific
# JPM - JPMorgan Chase
# XOM - Exxon Mobil
# AMZN - Amazon
# JNJ - Johnson & Johnson
# CAT - Caterpillar
# MSFT - Microsoft

COMPANIES = ['SBUX', 'UNP', 'JPM', 'XOM', 'AMZN', 'JNJ', 'CAT', 'MSFT']
# Corresponding ETFs for each sector  
ETFS = ['XLY', 'XLU', 'XLF', 'XLE', 'XLC', 'XLV', 'XLI', 'XLK']
# Benchmark
BENCHMARK = ['SPY']

#%%
# Downloading data

data_companies = yf.download(COMPANIES, start=START, end=END, auto_adjust=False)
data_etfs = yf.download(ETFS, start=START, end=END, auto_adjust=False) 
data_benchmark = yf.download(BENCHMARK, start=START, end=END, auto_adjust=False)

# Extract adjusted close prices
adj_close_companies = data_companies.loc[:, ('Adj Close', slice(None))]
adj_close_companies.columns = adj_close_companies.columns.droplevel(0)
adj_close_companies = adj_close_companies.dropna(how='all')

adj_close_etfs = data_etfs.loc[:, ('Adj Close', slice(None))]
adj_close_etfs.columns = adj_close_etfs.columns.droplevel(0)
adj_close_etfs = adj_close_etfs.dropna(how='all')

adj_close_benchmark = data_benchmark.loc[:, ('Adj Close', slice(None))]
adj_close_benchmark.columns = adj_close_benchmark.columns.droplevel(0)
adj_close_benchmark = adj_close_benchmark.dropna(how='all')

# Resample to monthly frequency
monthly_returns_companies = adj_close_companies.resample('ME').last().pct_change().dropna(how='all')
monthly_returns_etfs = adj_close_etfs.resample('ME').last().pct_change().dropna(how='all') 
monthly_returns_benchmark = adj_close_benchmark.resample('ME').last().pct_change().dropna(how='all')







#%%
# 2) Portfolio Construction

# Portfolio 1: 8 selected companies, weighted equally
weights_companies = pd.Series(1/len(COMPANIES), index=COMPANIES)

# Portfolio 2: 8 selected ETFs, weighted equally  
weights_etfs = pd.Series(1/len(ETFS), index=ETFS)

# Portfolio 3: S&P 500 (SPY), weight of 1
weights_benchmark = pd.Series(1.0, index=BENCHMARK)


print("Company weights:", weights_companies)
print("ETF weights:", weights_etfs) 
print("Benchmark weight:", weights_benchmark)


#%%
# 3) Portfolio Growth Analysis

# Compute cumulative growth of $100,000 (GOAD) for each portfolio

initial_investment = 100000

# Portfolio 1: Companies
portfolio1_returns = monthly_returns_companies.dot(weights_companies)
portfolio1_growth = (1 + portfolio1_returns).cumprod() * initial_investment

# Portfolio 2: ETFs
portfolio2_returns = monthly_returns_etfs.dot(weights_etfs)
portfolio2_growth = (1 + portfolio2_returns).cumprod() * initial_investment

# Portfolio 3: Benchmark (SPY)
portfolio3_returns = monthly_returns_benchmark.dot(weights_benchmark)
portfolio3_growth = (1 + portfolio3_returns).cumprod() * initial_investment

# Combine into one DataFrame
growth_df = pd.DataFrame({
    'Companies': portfolio1_growth,
    'ETFs': portfolio2_growth,
    'Benchmark (SPY)': portfolio3_growth
})

# Plot growth
plt.figure(figsize=(10, 6))
growth_df.plot(title='Portfolio Growth of $100,000', figsize=(10, 6), grid=True)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.tight_layout()
plt.show()



#   Q1. Relative Performance of Portfolio - Over the five-year period, the Companies portfolio grew the most, ending at a little over $240,000 from the starting $100,000. The ETFs portfolio came in second, just ahead of the SPY benchmark, which had the lowest final value. This suggests that holding individual stocks gave higher returns, while ETFs and SPY were steadier but not as strong in overall growth.

#   Q2. Trends observed over the 5-year period - All three portfolios generally moved in the same direction, increasing along with the market. The Companies portfolio tended to rise more quickly during good market periods, especially from mid-2023 to 2024, while ETFs and SPY showed more gradual growth. Over time, the gap between the Companies portfolio and the others became clearer, showing higher growth but also more ups and downs.

#   Q3. Any anomalies or significant changes in growth trajectories - There were dips around late 2021 and mid-2022 across all portfolios, but the Companies portfolio dropped more sharply before bouncing back. Starting in early 2023, it recovered faster and moved well ahead of the ETFs and SPY. By 2025, the Companies portfolio had pulled away, while ETFs and SPY stayed on a smoother, more consistent path.


#%%
# 4) Beta and Alpha Comparison


import statsmodels.api as sm  


def calculate_beta_alpha(asset_returns, market_returns):
    market_returns.name = 'Market'
    X = sm.add_constant(market_returns)
    model = sm.OLS(asset_returns, X).fit()
    beta = model.params['Market']
    alpha = model.params['const']
    return beta, alpha

# Calculate Beta and Alpha
beta1, alpha1 = calculate_beta_alpha(portfolio1_returns, portfolio3_returns)
beta2, alpha2 = calculate_beta_alpha(portfolio2_returns, portfolio3_returns)
beta3, alpha3 = calculate_beta_alpha(portfolio3_returns, portfolio3_returns)

print("\nBeta and Alpha Comparison:")
print(f"Portfolio 1 (Companies): Beta = {beta1:.4f}, Alpha = {alpha1:.4f}")
print(f"Portfolio 2 (ETFs):      Beta = {beta2:.4f}, Alpha = {alpha2:.4f}")
print(f"Portfolio 3 (SPY):       Beta = {beta3:.4f}, Alpha = {alpha3:.4f}")

    

#   Q1. Which portfolio exhibits the lowest risk (safest portfolio)? - The ETFs portfolio has the lowest Beta (0.9409). That makes it the least risky and safest portfolio compared to Companies (0.9850) and SPY (1.0000).

#   Q2. Which portfolio has the highest excess return? - The Companies portfolio has the highest Alpha (0.0020). That means it delivered the greatest extra return beyond what its market exposure would predict.ETFs also had a small positive Alpha (0.0013), while SPY is ~0 (as expected for the benchmark).


#%%
# 5) Portfolio Performance Analysis

final_values = growth_df.iloc[-1]
best_portfolio_name = final_values.idxmax()
best_portfolio_returns = {
    'Companies': portfolio1_returns,
    'ETFs': portfolio2_returns,
    'Benchmark (SPY)': portfolio3_returns
}[best_portfolio_name]

# Mean return and CAGR
mean_return = best_portfolio_returns.mean()
cagr = (final_values[best_portfolio_name] / initial_investment) ** (1/5) - 1

# Additional metrics
volatility = best_portfolio_returns.std()
sharpe_ratio = (mean_return * TRADING_DAYS - RF_ANNUAL) / (volatility * np.sqrt(TRADING_DAYS))
max_drawdown = (growth_df[best_portfolio_name] / growth_df[best_portfolio_name].cummax() - 1).min()
positive_months = (best_portfolio_returns > 0).sum()
negative_months = (best_portfolio_returns < 0).sum()

print(f"\n Performance Tear Sheet for Best Portfolio: {best_portfolio_name}")
print(f"Mean Monthly Return: {mean_return:.4f}")
print(f"CAGR: {cagr:.4f}")
print(f"Volatility (Std Dev): {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
print(f"Positive Months: {positive_months}")
print(f"Negative Months: {negative_months}")


#   2. Select five additional metrics that you are unfamiliar with:
#      2a.  Explain their function , Interpret their values in relation to your portfolio , Determine whether your portfolio performs well or poorly based on these metrics.

#           - Volatility (standard deviation) - A volatility of 0.0491 indicates moderate variability in monthly returns. While some fluctuation exists, it’s not extreme. Acceptable risk level for the observed return, not overly volatile.
#           - Sharpe Ratio - A Sharpe Ratio of 5.1957 is considered excellent, suggesting the portfolio delivers strong returns relative to its risk. The portfolio performs very well on a risk-adjusted basis.
#           - Max Drawdown - A drawdown of -16.74% means the portfolio experienced a significant dip at some point. While the portfolio recovered, this highlights potential vulnerability during market downturns.
#           - Positivie Months - 38 out of 59 months were profitable. Indicates consistent performance  more than 64% of months were positive.
#           - Negative Months - 21 months showed losses.Losses were present but outweighed by gains, supporting overall strong performance.






# === Dashboard Functions ===

def get_portfolio_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    growth_df.plot(ax=ax, title='Portfolio Growth of $100,000', grid=True)
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_xlabel('Date')
    plt.tight_layout()
    return fig

def get_growth_dataframe():
    return growth_df


def get_summary_metrics():
    return {
        "Best Portfolio": best_portfolio_name,
        "Mean Monthly Return": mean_monthly_return,
        "CAGR": cagr_value,
        "Volatility": volatility,
        "Sharpe Ratio": optimized_sharpe,
        "Max Drawdown": max_portfolio_drawdown,
        "Positive Months": positive_months,
        "Negative Months": negative_months
    }

def get_beta_alpha():
    return {
        "Companies": {"Beta": beta1, "Alpha": alpha1},
        "ETFs": {"Beta": beta2, "Alpha": alpha2},
        "SPY": {"Beta": beta3, "Alpha": alpha3}
    }

# Expose key metrics
optimized_sharpe = sharpe_ratio
mean_monthly_return = mean_return
cagr_value = cagr
max_portfolio_drawdown = max_drawdown





