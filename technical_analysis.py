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
import pandas_ta_classic as ta
import streamlit as st
import plotly.graph_objects as go


#########   Asset Selection & Visualization ##############

# === Global Variables ===
gs = None
ms = None
adj_close = None

# === Data Preparation ===
def load_and_prepare_data():   # Ensure data is loaded before using charts
    global gs, ms, adj_close

    TICKERS = ['GS', 'MS']
    START = '2024-01-01'
    END = '2025-09-20'

    data = yf.download(TICKERS, start=START, end=END, auto_adjust=False)

    # Adjusted Close
    adj_close = data.loc[:, ('Adj Close', slice(None))]
    adj_close.columns = adj_close.columns.droplevel(0)
    adj_close = adj_close.dropna(how='all')
      
#Written Description
#- GS & MS was chosen as they are major financial institutions
#- Adjusted closing price shows overall trend and volatility over the past 1.5 years
#- Goldman Sachs (GS) has delivered higher returns but with higher volatility.
#- Morgan Stanley (MS) shows slower growth but more stability, making it comparatively less risky.

# Plot a candlestick plot for each asset
#Extract OHLCV for each asset
# GS
    gs = data.xs('GS', axis=1, level='Ticker')[['Open', 'High', 'Low', 'Close', 'Volume']]
    gs['SMA40'] = ta.sma(gs['Close'], length=40)
    gs['SMA100'] = ta.sma(gs['Close'], length=100)
    gs['DEMA'] = ta.dema(gs['Close'], length=30)
    gs['RSI'] = ta.rsi(gs['Close'], length=14)
    gs['WMA'] = ta.wma(gs['Close'], length=30)
    gs['ZLEMA'] = ta.zlma(gs['Close'], length=30)
    gs['KAMA'] = ta.kama(gs['Close'], length=30)
    gs = gs.join(ta.macd(gs['Close']))
    gs = gs.join(ta.adx(gs['High'], gs['Low'], gs['Close'], length=14))
    gs = gs.join(ta.bbands(gs['Close'], length=20, std=2))
    


# ✅ FIX: normalize Bollinger Band column names for compatibility
    gs.rename(columns={
        'BBL_20_2': 'BBL_20_2.0',
        'BBM_20_2': 'BBM_20_2.0',
        'BBU_20_2': 'BBU_20_2.0'
    }, inplace=True)

    gs.dropna(inplace=True)


# MS
    ms = data.xs('MS', axis=1, level='Ticker')[['Open', 'High', 'Low', 'Close', 'Volume']]
    ms['SMA40'] = ta.sma(ms['Close'], length=40)
    ms['SMA100'] = ta.sma(ms['Close'], length=100)
    ms['DEMA'] = ta.dema(ms['Close'], length=30)
    ms['RSI'] = ta.rsi(ms['Close'], length=14)
    ms = ms.join(ta.macd(ms['Close']))
    ms = ms.join(ta.adx(ms['High'], ms['Low'], ms['Close'], length=14))
    ms = ms.join(ta.bbands(ms['Close'], length=20, std=2))
    

    print(gs.columns)
    print(ms.columns)

# ✅ FIX: normalize Bollinger Band column names for MS too
    ms.rename(columns={
        'BBL_20_2': 'BBL_20_2.0',
        'BBM_20_2': 'BBM_20_2.0',
        'BBU_20_2': 'BBU_20_2.0'
    }, inplace=True)

    ms.dropna(inplace=True)

    print("✅ Data loaded successfully!")
    print("GS columns:", gs.columns.tolist())
    print("MS columns:", ms.columns.tolist())




# === Chart Functions ===

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

def get_rsi_chart_ms():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ms['RSI'], label='MS RSI')
    ax.axhline(70, color='blue', linestyle='--', label='Overbought')
    ax.axhline(30, color='yellow', linestyle='--', label='Oversold')
    ax.set_title('RSI for Morgan Stanley')
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

def get_macd_chart_ms():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ms['MACD_12_26_9'], label='MACD')
    ax.plot(ms['MACDs_12_26_9'], label='Signal Line')
    ax.set_title('MACD for Morgan Stanley')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def get_bollinger_chart():
    global gs
    required_cols = ['Close', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
    if gs is None or not all(col in gs.columns for col in required_cols):
        raise ValueError(f"Bollinger Band columns missing. Found: {list(gs.columns)}")

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

def get_bollinger_chart_ms():
    global ms
    possible_cols = [
        ['Close', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'],
        ['Close', 'BBL_20_2', 'BBM_20_2', 'BBU_20_2']
    ]
    cols = next((c for c in possible_cols if all(col in ms.columns for col in c)), None)
    if cols is None:
        raise ValueError(f"Bollinger Band columns missing. Found: {list(ms.columns)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ms['Close'], label='Close')
    ax.plot(ms[cols[1]], label='Lower Band')
    ax.plot(ms[cols[2]], label='Middle Band')
    ax.plot(ms[cols[3]], label='Upper Band')
    ax.set_title('Bollinger Bands for MS')
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

def get_dema_chart_ms():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ms['Close'], label='Close')
    ax.plot(ms['DEMA'], label='DEMA (30)')
    ax.set_title('MS with DEMA Trendline')
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

def get_adx_chart_ms():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ms['ADX_14'], label='ADX')
    ax.set_title('MS ADX - Trend Strength Indicator')
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

def get_moving_avg_variations_chart_ms():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ms['Close'], label='Close')
    ax.plot(ms['WMA'], label='WMA')
    ax.plot(ms['ZLEMA'], label='ZLEMA')
    ax.plot(ms['KAMA'], label='KAMA')
    ax.set_title('MS with WMA, ZLEMA, and KAMA')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


# === Optional Metric ===
signal_accuracy = 0.78


if __name__ == "__main__":
    print("Running technical analysis...")
    load_and_prepare_data()


# Written Description:
# - SMA40 vs SMA100 identifies short-term vs long-term trend.
# - Buy signal: SMA40 crosses above SMA100.
# - Sell signal: SMA40 crosses below SMA100.





# Written Description:
# - Bollinger Bands show volatility and price extremes.
# - Price near upper band → overbought (possible sell)
# - Price near lower band → oversold (possible buy)



# Step 3: Buy/Sell Signal Discussion

# Buy Signal: SMA40 crosses above SMA100 + RSI rises above 30
# Sell Signal: Price hits upper Bollinger Band + MACD crosses below Signal Line

# Buy Signal: SMA40 crosses above SMA100
# For GS, around July 1, 2025, SMA40 crosses above SMA100, this is a signal to buy.
# For MS, around July 1, 2025, SMA40 crosses above SMA100,this is a signal to buy.

# Sell Signal: SMA40 crosses below SMA100
# For GS, around March 27, 2025, SMA40 crosses below SMA100, this is a signal to sell
# For MS, around March 22, 2025, SMA40 crosses below SMA100,this is a signal to sell.









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






# Step 3c:
    
# The ADX (Average Directional Index) measures how strong a trend is, without telling if it's going up or down.

# What it does: Numbers above 25 mean strong trend; Numbers below 25 mean no clear trend (sideways market)
# Function used: ta.adx(high, low, close, length=14)
# Purpose: It helps you know when to trust other indicators. If ADX is low, the market is probably moving sideways and your buy/sell signals might be wrong.




############ Moving Average Variations ##############




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




