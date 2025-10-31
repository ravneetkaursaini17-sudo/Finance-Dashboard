#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:46:09 2025

@author: ravneetkaursaini
"""


#%%
# PART 1 - DATA ACQUISATION AND VISULALISATION
# Objective : To download 5 years of historical data and benchmark SPY and visualise price trends.

#  Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
import streamlit as st
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import plotting
from scipy.stats import norm
import seaborn as sns


# Define today's date
end_date = date.today().strftime('%Y-%m-%d')  # ✅ Define end_date here


# Sidebar selector
selected_project = st.sidebar.selectbox(
    "Choose a project",
    ["Portfolio Optimization and Construction", "Other Project"]
)


# === Dashboard Block ===
if selected_project == "Portfolio Optimization and Construction":

    st.markdown("## 📊 Portfolio Optimization and Construction")

    # 1. Define tickers and date range
    tickers = ['JPM', 'GS', 'BRK-B', 'WFC', 'C']
    benchmark_ticker = 'SPY'
    start_date = '2020-01-01'
    # end_date is already defined above
    

 # 2. Download data
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    spy_df = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)

    # 3. Extract adjusted close
adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
adj_close['SPY'] = spy_df['Adj Close'].reindex(adj_close.index).ffill()

# 4. Calculate daily returns
daily_returns = adj_close.pct_change().dropna()


    # 5. Price trend chart
fig, ax = plt.subplots(figsize=(12, 6))
for ticker in adj_close.columns:
    ax.plot(adj_close[ticker], label=ticker)
ax.set_title('Adjusted Closing Prices (2020–2025)')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
st.pyplot(fig)
    
    # 6. Normalized price chart
normalized_prices = adj_close / adj_close.iloc[0] * 100

fig, ax = plt.subplots(figsize=(12, 6))
for ticker in normalized_prices.columns:
    ax.plot(normalized_prices[ticker], label=ticker)

ax.set_title('Normalized Price Performance')
ax.set_xlabel('Date')
ax.set_ylabel('Indexed Price (Start = 100)')
ax.legend()

st.pyplot(fig)
    
    # 7. Daily and Annualized Returns
daily_returns = adj_close.pct_change().dropna()
mean_daily_returns = daily_returns.mean()
annualized_returns = mean_daily_returns * 252
daily_cov_matrix = daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

# 8. Summary Statistics
tickers_with_spy = tickers + ['SPY']
summary_stats = pd.DataFrame(index=tickers_with_spy)

for ticker in tickers_with_spy:
    returns = daily_returns[ticker]
    mode_series = returns.mode()
    summary_stats.loc[ticker, 'Mean'] = returns.mean()
    summary_stats.loc[ticker, 'Median'] = returns.median()
    summary_stats.loc[ticker, 'Mode'] = mode_series.iloc[0] if not mode_series.empty else np.nan
    summary_stats.loc[ticker, 'Skewness'] = returns.skew()
    summary_stats.loc[ticker, 'Volatility'] = returns.std()

st.markdown("### 📋 Summary Statistics")
st.dataframe(summary_stats.round(4))
    
    
# 9. Annual Returns Chart
annual_returns = adj_close.resample('Y').last().pct_change().dropna()

fig, ax = plt.subplots(figsize=(12, 6))
annual_returns.plot(kind='bar', ax=ax)
ax.set_title('Annual Returns by Ticker')
ax.set_ylabel('Return (%)')
ax.set_xlabel('Year')
ax.legend(title='Ticker')
st.pyplot(fig)


# 10. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Correlation Matrix of Daily Returns')
st.pyplot(fig)

# 11. Rolling Volatility
rolling_volatility = daily_returns.rolling(window=30).std()
fig, ax = plt.subplots(figsize=(12, 6))
rolling_volatility.plot(ax=ax)
ax.set_title('30-Day Rolling Volatility')
ax.set_ylabel('Volatility')
ax.set_xlabel('Date')

    
# 12. Rolling Sharpe Ratio
rolling_mean = daily_returns.mean(axis=1).rolling(window=30).mean()
rolling_std = daily_returns.std(axis=1).rolling(window=30).std()
rolling_sharpe = rolling_mean / rolling_std

fig, ax = plt.subplots(figsize=(12, 6))
rolling_sharpe.plot(ax=ax, color='purple')
ax.set_title('30-Day Rolling Sharpe Ratio')
ax.set_ylabel('Sharpe Ratio')
ax.set_xlabel('Date')
st.pyplot(fig)

# 13. Bar Charts for Summary Stats
for stat in ['Mean', 'Median', 'Volatility', 'Skewness']:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary_stats.index, summary_stats[stat], color='skyblue')
    ax.set_title(f'{stat} of Daily Returns')
    ax.set_ylabel(stat)
    ax.set_xlabel('Ticker')
    st.pyplot(fig)

# 14. Mean Return Pie Chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(mean_daily_returns[tickers_with_spy], labels=tickers_with_spy, autopct='%1.1f%%', startangle=140)
ax.set_title('Proportion of Mean Daily Returns')
st.pyplot(fig)

# 15. Histogram of Daily Returns
fig = plt.figure(figsize=(12, 8))
daily_returns.hist(bins=50)
plt.suptitle('Histogram of Daily Returns')
st.pyplot(fig)

# Calculate daily returns and annualized metrics
daily_returns = adj_close.pct_change().dropna()
mean_daily_returns = daily_returns.mean()
annualised_returns = mean_daily_returns * 252
daily_cov_matrix = daily_returns.cov()
annual_cov_matrix = daily_cov_matrix * 252

# 16. Equal Weight Portfolio Performance
weights_portfolio = np.array([0.2] * len(tickers))  # Dynamically match ticker count
portfolio_return = np.dot(weights_portfolio, annualised_returns[tickers])
portfolio_volatility = np.sqrt(np.dot(weights_portfolio.T, np.dot(annual_cov_matrix.loc[tickers, tickers], weights_portfolio)))
portfolio_sharpe = portfolio_return / portfolio_volatility

spy_return = annualised_returns['SPY']
spy_volatility = np.sqrt(annual_cov_matrix.loc['SPY', 'SPY'])
spy_sharpe = spy_return / spy_volatility

# Display metrics
st.markdown("### 📈 Equal Weight Portfolio Performance")
st.metric("Sharpe Ratio (Equal)", f"{portfolio_sharpe:.2f}")
st.metric("Sharpe Ratio (SPY)", f"{spy_sharpe:.2f}")

# 17. Equal Weight Allocation Pie Chart
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(weights_portfolio, labels=tickers, autopct='%1.1f%%', startangle=140)
ax.set_title('Portfolio Allocation (Equal Weights)')
st.pyplot(fig)

# 18. Optimized Portfolio (Max Sharpe Ratio)
mu = expected_returns.mean_historical_return(adj_close[tickers])
s = risk_models.sample_cov(adj_close[tickers])
ef = EfficientFrontier(mu, s)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
optimized_sharpe = ef.portfolio_performance()[2]

st.markdown("### 📊 Optimized Portfolio Performance")
st.metric("Sharpe Ratio (Optimized)", f"{optimized_sharpe:.2f}")

# Optimized Allocation Chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(list(cleaned_weights.keys()), list(cleaned_weights.values()), color='skyblue')
ax.set_title('Optimized Portfolio Allocation (Max Sharpe)')
st.pyplot(fig)

# Efficient Frontier Plot
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_efficient_frontier(EfficientFrontier(mu, s), ax=ax, show_assets=True)
ax.set_title('Efficient Frontier')
st.pyplot(fig)

# 19. Risk metrics
weights_array = np.array(list(cleaned_weights.values()))
selected_tickers = list(cleaned_weights.keys())
portfolio_variance = np.dot(weights_array.T, np.dot(annual_cov_matrix.loc[selected_tickers, selected_tickers], weights_array))
portfolio_volatility = np.sqrt(portfolio_variance)
confidence_level = 0.95
portfolio_mean = np.dot(weights_array, mu)
VaR = norm.ppf(1 - confidence_level) * portfolio_volatility - portfolio_mean

st.markdown("### 📉 Risk Metrics")
st.write({
    "Portfolio Variance": round(portfolio_variance, 6),
    "Portfolio Volatility": round(portfolio_volatility, 4),
    "Value at Risk (95%)": round(VaR, 6)
})

# 20. Strategy summary table
equal_portfolio_return = daily_returns[tickers].dot(weights_portfolio)
portfolio_daily_return = daily_returns[selected_tickers].dot(weights_array)
spy_daily_return = daily_returns['SPY']

equal_cumulative_return = (1 + equal_portfolio_return).cumprod()
optimized_cumulative_return = (1 + portfolio_daily_return).cumprod()
spy_cumulative_return = (1 + spy_daily_return).cumprod()

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

# 21. Strategy evaluation
def get_metrics(daily_ret, label):
    ann_return = np.mean(daily_ret) * 252
    ann_vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = ann_return / ann_vol
    st.markdown(f"### 📊 {label} Performance")
    st.write(f"Annualized Return: {ann_return:.4f}")
    st.write(f"Annualized Volatility: {ann_vol:.4f}")
    st.write(f"Sharpe Ratio: {sharpe:.4f}")
