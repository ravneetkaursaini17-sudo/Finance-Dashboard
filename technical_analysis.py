#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:02:46 2025

@author: gaoguobin
"""



### Completed as a group by Guobin Gao , Ravneet Kaur Saini  and Ayna Tovekelova ###


## TECHNICAL ANALYSIS FOR GOLDMAN SACHS & MORGAN STANLEY (2024-2025)

#Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import mplfinance as mpf
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects as go


#########   Asset Selection & Visualization ##############


# Step 0 : set up parameters

TICKERS=['GS','MS']
START='2024-01-01'
END='2025-09-20'
TRADING_DAYS=252
RF_ANNUAL=0.83


# # Step 1: Download 1.5 year of daily data for GS and MS

#Using yahoo finance to download histroical data

data = yf.download(TICKERS,start=START,
                 end=END,auto_adjust=False)

#extract adjusted closing prices for plotting

adj_close = data.loc[:,('Adj Close',slice(None))]
adj_close.columns = adj_close.columns.droplevel(0)
adj_close = adj_close.dropna(how='all')


returns = adj_close.pct_change().dropna(how = 'all')

# Step 2a: Plot the Adjusted Closing Price of both assets 

plt.figure(figsize=(10, 6))
plt.plot(data['Adj Close', 'GS'], label='Goldman Sachs (GS)')
plt.plot(data['Adj Close', 'MS'], label='Morgan Stanley (MS)')
plt.title('Adjusted Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

#Written Description
#- GS & MS was chosen as they are major financial institutions
#- Adjusted closing price shows overall trend and volatility over the past 1.5 years
#- Goldman Sachs (GS) has delivered higher returns but with higher volatility.
#- Morgan Stanley (MS) shows slower growth but more stability, making it comparatively less risky.




#Step 2b: A candlestick plot for each asset
#Extract OHLCV for each asset
# Prepare GS data
gs = data.xs('GS', axis=1, level='Ticker')  # Extract all columns for GS
gs = gs[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep OHLCV
gs.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Rename to exact required names

# Prepare MS data
ms = data.xs('MS', axis=1, level='Ticker')
ms = ms[['Open', 'High', 'Low', 'Close', 'Volume']]
ms.columns = ['Open', 'High', 'Low', 'Close', 'Volume']




# Plot GS & MS  candlestick


# Create candlestick chart for GS
fig_gs = go.Figure(data=[go.Candlestick(
    x=gs.index,
    open=gs['Open'],
    high=gs['High'],
    low=gs['Low'],
    close=gs['Close'],
    name='GS'
)])

fig_gs.update_layout(
    title='Goldman Sachs (GS) Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=False
)


# Create candlestick chart for MS

fig_ms = go.Figure(data=[go.Candlestick(
    x=ms.index,
    open=ms['Open'],
    high=ms['High'],
    low=ms['Low'],
    close=ms['Close'],
    name='MS'
)])

fig_ms.update_layout(
    title='Morgan Stanley (MS) Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=False
)



#Written Description
# - Candlestick charts visualize daily OHLC patterns and market sentiment.
# - Long wicks indicate high intraday volatility; candlestick bodies indicate price momentum.







# Step 2c: SMA Trendlines (SMA40 and SMA100)
# SMA40 (short-term) and SMA100 (long-term)

gs['SMA40'] = ta.sma(gs['Close'], length=40)
gs['SMA100'] = ta.sma(gs['Close'], length=100)

ms['SMA40'] = ta.sma(ms['Close'], length=40)
ms['SMA100'] = ta.sma(ms['Close'], length=100)



# Plot SMA on candlestick chart

# Create SMA overlays
gs_apds = [
    mpf.make_addplot(gs['SMA40'], color='orange', width=1.2, label='SMA40'),
    mpf.make_addplot(gs['SMA100'], color='blue', width=1.2, label='SMA100')
]

# Plot GS chart with legend
mpf.plot(
    gs,
    type='candle',
    style='yahoo',
    addplot=gs_apds,
    title='GS with SMA40 and SMA100',
    volume=True,
    figratio=(21, 9),
    figscale=2.5,
    panel_ratios=(6, 2),
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    tight_layout=True,
    returnfig=True
)[0].legend(loc='upper left')



# Create SMA overlays
ms_apds = [
    mpf.make_addplot(ms['SMA40'], color='orange', width=1.2, label='SMA40'),
    mpf.make_addplot(ms['SMA100'], color='blue', width=1.2, label='SMA100')
]

# Plot MS chart with legend
mpf.plot(
    ms,
    type='candle',
    style='yahoo',
    addplot=ms_apds,
    title='MS with SMA40 and SMA100',
    volume=True,
    figratio=(21, 9),
    figscale=2.5,
    panel_ratios=(6, 2),
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    tight_layout=True,
    returnfig=True
)[0].legend(loc='upper left')



# Written Description:
# - SMA40 vs SMA100 identifies short-term vs long-term trend.
# - Buy signal: SMA40 crosses above SMA100.
# - Sell signal: SMA40 crosses below SMA100.



# Step 2d: Bollinger Bands (2 Standard Deviations)


bbands = ta.bbands(gs['Close'], length=20, std=2)
gs = gs.join(bbands)

bbands = ta.bbands(ms['Close'], length=20, std=2)
ms = ms.join(bbands)


# Plot GS Bollinger Bands

plt.figure(figsize=(10, 6))
plt.plot(gs['Close'], label='Close')
plt.plot(gs['BBL_20_2.0'], label='Lower Band')
plt.plot(gs['BBM_20_2.0'], label='Middle Band')
plt.plot(gs['BBU_20_2.0'], label='Upper Band')
plt.title('GS Bollinger Bands')
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.grid(True)
plt.show()

# Plot MS Bollinger Bands

plt.figure(figsize=(10, 6))
plt.plot(ms['Close'], label='Close')
plt.plot(ms['BBL_20_2.0'], label='Lower Band')
plt.plot(ms['BBM_20_2.0'], label='Middle Band')
plt.plot(ms['BBU_20_2.0'], label='Upper Band')
plt.title('MS Bollinger Bands')
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.grid(True)
plt.show()


# Written Description:
# - Bollinger Bands show volatility and price extremes.
# - Price near upper band → overbought (possible sell)
# - Price near lower band → oversold (possible buy)


# Step 2e: Additional Trendline - DEMA (Double Exponential Moving Average)

gs['DEMA'] = ta.dema(gs['Close'], length=30)
plt.figure(figsize=(12, 6))
plt.plot(gs['Close'], label='Close')
plt.plot(gs['DEMA'], label='DEMA (30)')
plt.title('GS with DEMA Trendline')
plt.legend()
plt.grid(True)
plt.show()

ms['DEMA'] = ta.dema(ms['Close'], length=30)
plt.figure(figsize=(12, 6))
plt.plot(ms['Close'], label='Close')
plt.plot(ms['DEMA'], label='DEMA (30)')
plt.title('MS with DEMA Trendline')
plt.legend()
plt.grid(True)
plt.show()


# Written Description:
# - DEMA reduces lag compared to SMA/EMA.
# - Useful to track short-term price trends more quickly.



# Step 3: Buy/Sell Signal Discussion

# Buy Signal: SMA40 crosses above SMA100 + RSI rises above 30
# Sell Signal: Price hits upper Bollinger Band + MACD crosses below Signal Line

# Buy Signal: SMA40 crosses above SMA100
# For GS, around July 1, 2025, SMA40 crosses above SMA100, this is a signal to buy.
# For MS, around July 1, 2025, SMA40 crosses above SMA100,this is a signal to buy.

# Sell Signal: SMA40 crosses below SMA100
# For GS, around March 27, 2025, SMA40 crosses below SMA100, this is a signal to sell
# For MS, around March 22, 2025, SMA40 crosses below SMA100,this is a signal to sell.



#############  RSI and MACD Analysis  #############




# Step 1a: Compute RSI and MACD
# --- GS RSI + MACD ---
gs['RSI'] = ta.rsi(gs['Close'], length=14)
macd_gs = ta.macd(gs['Close'])
gs = gs.join(macd_gs)  # joins MACD columns with original df

# --- MS RSI + MACD ---
ms['RSI'] = ta.rsi(ms['Close'], length=14)
macd_ms = ta.macd(ms['Close'])
ms = ms.join(macd_ms)  # joins MACD columns with original df




# Step 1b: Plot RSI for GS
plt.figure(figsize=(12, 4))
plt.plot(gs['RSI'], label='RSI')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title('GS RSI')
plt.legend()
plt.grid(True)
plt.show()


# Step 1b: Plot RSI for MS
plt.figure(figsize=(12, 4))
plt.plot(ms['RSI'], label='RSI')
plt.axhline(70, color='blue', linestyle='--', label='Overbought')
plt.axhline(30, color='yellow', linestyle='--', label='Oversold')
plt.title('MS RSI')
plt.legend()
plt.grid(True)
plt.show()



# Step 1c: Plot MACD for GS
plt.figure(figsize=(12, 4))
plt.plot(gs['MACD_12_26_9'], label='MACD')
plt.plot(gs['MACDs_12_26_9'], label='Signal Line')
plt.title('GS MACD')
plt.legend()
plt.grid(True)
plt.show()

# Step 1c: Plot MACD for MS
plt.figure(figsize=(12, 4))
plt.plot(ms['MACD_12_26_9'], label='MACD')
plt.plot(ms['MACDs_12_26_9'], label='Signal Line')
plt.title('MS MACD')
plt.legend()
plt.grid(True)
plt.show()


# Written Description:
# - RSI identifies overbought (>70) and oversold (<30) conditions.
# - MACD crossing above Signal → Buy, below Signal → Sell.
# - These signals mostly align with SMA/Bollinger signals.
# - Additional indicator: ADX (trend strength; >25 indicates strong trend)





# Step 2

# RSI Analysis for GS:
# The RSI indicator identified two distinct oversold conditions (RSI < 30) during March-May 2025, which traditionally signal buying opportunities as the asset may be undervalued. Conversely, an overbought condition (RSI > 70) was detected in early July 2025, indicating a potential selling point due to likely price correction.  

# RSI Analysis for MS:
# The RSI indicator identified two distinct oversold conditions (RSI < 30) during March-May 2025, which traditionally signal buying opportunities as the asset may be undervalued. Conversely, an overbought condition (RSI > 70) was detected in mid October 2024 and early July 2025, indicating a potential selling point due to likely price correction.  

# MACD Analysis for GS:
# The MACD line crossed above the signal line twice between March and May 2025 - these were good times to buy. Looking at the chart, the MACD crossed below the signal line several times throughout the period, and each of those could have been a signal to sell.

# MACD Analysis for MS:
# MS showed similar patterns - the MACD crossed above the signal line twice in spring 2025, suggesting buy opportunities. There were many times when it crossed below the signal line, and those were all potential sell signals.


# Step 3

# Step 3a: 
# Q. Do the signals align with previous findings compared to question 1:
# Yes, the signals mostly align! For example, when SMA40 crossed above SMA100  around July 1, 2025 (buy signal from Question 1), the RSI also showed  oversold conditions and MACD showed bullish crossovers around the same time. This multi-indicator confirmation makes the trading signals more reliable. While most signals aligned, we observeed an interesting divergence in July 2025 where SMA crossovers indicated bullish trend changes while RSI suggested overbought conditions. This type of conflict often occurs at potential market turning points and suggests a 'buy on pullback' strategy rather than immediate entry.



# Step 3b: Additional Indicator - ADX (Average Directional Index)

#GS
adx = ta.adx(gs['High'], gs['Low'], gs['Close'], length=14)
gs = gs.join(adx)
plt.figure(figsize=(12, 4))
plt.plot(gs['ADX_14'], label='ADX')
plt.title('GS ADX - Trend Strength Indicator')
plt.legend()
plt.grid(True)
plt.show()


#MS
adx = ta.adx(ms['High'], ms['Low'], ms['Close'], length=14)
ms = ms.join(adx)
plt.figure(figsize=(12, 4))
plt.plot(ms['ADX_14'], label='ADX')
plt.title('MS ADX - Trend Strength Indicator')
plt.legend()
plt.grid(True)
plt.show()



# Step 3c:
    
# The ADX (Average Directional Index) measures how strong a trend is, without telling if it's going up or down.

# What it does: Numbers above 25 mean strong trend; Numbers below 25 mean no clear trend (sideways market)
# Function used: ta.adx(high, low, close, length=14)
# Purpose: It helps you know when to trust other indicators. If ADX is low, the market is probably moving sideways and your buy/sell signals might be wrong.




############ Moving Average Variations ##############

# Step 1: Moving Average Variations

# WMA - Weighted Moving Average
gs['WMA'] = ta.wma(gs['Close'], length=30)
ms['WMA'] = ta.wma(ms['Close'], length=30)

# ZLEMA - Zero Lag Exponential Moving Average
gs['ZLEMA'] = ta.zlma(gs['Close'], length=30)
ms['ZLEMA'] = ta.zlma(ms['Close'], length=30)


# KAMA - Kaufman’s Adaptive Moving Average
gs['KAMA'] = ta.kama(gs['Close'], length=30)
ms['KAMA'] = ta.kama(ms['Close'], length=30)


# Plot all three on one chart for GS
plt.figure(figsize=(12, 6))
plt.plot(gs['Close'], label='Close')
plt.plot(gs['WMA'], label='WMA')
plt.plot(gs['ZLEMA'], label='ZLEMA')
plt.plot(gs['KAMA'], label='KAMA')
plt.title('GS with WMA, ZLEMA, and KAMA')
plt.legend()
plt.grid(True)
plt.show()

# Plot all three on one chart for MS
plt.figure(figsize=(12, 6))
plt.plot(ms['Close'], label='Close')
plt.plot(ms['WMA'], label='WMA')
plt.plot(ms['ZLEMA'], label='ZLEMA')
plt.plot(ms['KAMA'], label='KAMA')
plt.title('MS with WMA, ZLEMA, and KAMA')
plt.legend()
plt.grid(True)
plt.show()





# Step 2
# Written Description:
    
# - **WMA:** Weighted moving average; emphasizes recent prices.
# Purpose: The Weighted Moving Average assigns greater significance to recent price data compared to older data points, using a linearly decreasing weighting scheme.
# Advantage: More responsive to recent price movements than Simple Moving Averages
# Limitation: Increased sensitivity may generate false signals during volatile market conditions
# Best Use: Quick trend spotting in stable trending markets.


# - **ZLEMA:** Zero-lag EMA; reduces delay for faster signals.
# Purpose: The Weighted Moving Average assigns greater significance to recent price data compared to older data points, using a linearly decreasing weighting scheme.
# Advantage: More responsive to recent price movements than Simple Moving Averages
# Limitation: Increased sensitivity may generate false signals during volatile market conditions
# Best Use: Strong trending phases for faster entries/exits



# - **KAMA:** Adaptive MA; adjusts smoothing based on volatility.
# Purpose: An intelligent moving average that automatically adjusts its responsiveness based on prevailing market volatility conditions.
# Advantage: Self-adjusting nature performs well across various market environments
# Limitation: Complex calculation methodology may require additional understanding
# Best Use: Mixed markets with both trends and consolidations

# - These moving averages provide complementary trend signals to SMA and DEMA.


# === Dashboard Functions ===

def get_technical_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(adj_close['GS'], label='Goldman Sachs (GS)')
    ax.plot(adj_close['MS'], label='Morgan Stanley (MS)')
    ax.set_title('Adjusted Closing Prices Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_rsi_chart():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gs['RSI'], label='GS RSI')
    ax.axhline(70, color='red', linestyle='--', label='Overbought')
    ax.axhline(30, color='green', linestyle='--', label='Oversold')
    ax.set_title('RSI for Goldman Sachs')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def get_macd_chart():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gs['MACD_12_26_9'], label='MACD')
    ax.plot(gs['MACDs_12_26_9'], label='Signal Line')
    ax.set_title('MACD for Goldman Sachs')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_bollinger_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gs['Close'], label='Close')
    ax.plot(gs['BBL_20_2.0'], label='Lower Band')
    ax.plot(gs['BBM_20_2.0'], label='Middle Band')
    ax.plot(gs['BBU_20_2.0'], label='Upper Band')
    ax.set_title('Bollinger Bands for GS')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_dema_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gs['Close'], label='Close')
    ax.plot(gs['DEMA'], label='DEMA (30)')
    ax.set_title('GS with DEMA Trendline')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_adx_chart():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gs['ADX_14'], label='ADX')
    ax.set_title('GS ADX - Trend Strength Indicator')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_moving_avg_variations_chart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gs['Close'], label='Close')
    ax.plot(gs['WMA'], label='WMA')
    ax.plot(gs['ZLEMA'], label='ZLEMA')
    ax.plot(gs['KAMA'], label='KAMA')
    ax.set_title('GS with WMA, ZLEMA, and KAMA')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# Placeholder for dashboard metric
signal_accuracy = 0.78















