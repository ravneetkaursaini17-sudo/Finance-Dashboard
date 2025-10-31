#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:25:52 2025

@author: ravneetkaursaini
"""



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import plotting
from scipy.stats import norm
import seaborn as sns
import os
from datetime import date
import numpy as np


# ✅ Set page layout — MUST be first Streamlit command
st.set_page_config(
    page_title="Ravneet Kaur Saini — Finance & Analytics Portfolio",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Now import your modules
import multi_asset_portfolio_analysis as ma
import multi_sector_portfolio_analysis as ms
import technical_analysis as ta
import nyc_flight_delay_analysis as nyc





st.markdown("""
    <style>
    /* 🔲 Dark background for main app */
    .stApp {
        background-color: #000000;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 🌒 Dark sidebar background */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #1a1a1a;
    }

    /* 🖋️ Header styling */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-family: 'Times New Roman', serif;
    }
    /* 📄 Markdown and text styling */
    .stMarkdown, .stText, .stTitle {
        color: #f0f0f0;
        font-family: 'Times New Roman', sans-serif;
    }

    /* 🔗 Link styling */
    a {
        color: #66ccff;
    }
    </style>
""", unsafe_allow_html=True)





# ✅ Sidebar
st.sidebar.image("ravneet_profile.jpeg", use_container_width=True)
st.sidebar.title("Ravneet Kaur Saini")
st.sidebar.markdown("**Finance Graduate** | **AP/AR Specialist** | **Python Financial Modelling Projects**")


# 📬 Contact Me
st.sidebar.subheader("📬 Contact Me")
st.sidebar.markdown("""
- 📧 [ravneetkaursaini17@gmail.com](mailto:ravneetkaursaini17@gmail.com)  
- 💼 [LinkedIn Profile](https://www.linkedin.com/in/ravneet-kaur-saini-412077339/)
""")


#skills
st.sidebar.info("""
### 🧠 Skills
- 📊 Valuation Techniques (DCF, DDM, Monte Carlo)  
- 🐍 Python for Analytics & Visualization  
- 📈 Excel Modeling & Bloomberg Terminal  
- 🎨 Streamlit Dashboard Customization  
- ⚖️ Derivatives Trading & Risk Management  
- 📝 Report Writing & Data Presentation
""")

# Main Section

st.title("📊 Ravneet Kaur Saini — Finance & Analytics Portfolio")
st.header("About Me")
st.write("""
Hello, I'm **Ravneet Kaur Saini**, a finance graduate with a passion for turning data into decisions. My journey spans from accounting fundamentals in India to advanced financial modeling in the U.S., and I thrive in roles that demand precision, integrity, and strategic thinking.

🔍 I specialize in:
- Financial reporting and reconciliation
- Python-based valuation and dashboarding
- Derivatives trading and risk analysis

💡 I’ve built dashboards that visualize portfolio performance, forecast risk, and uncover trading signals all grounded in real-world data.

🎯 I’m actively seeking entry-level roles where I can grow as a:
- Financial Analyst
- Credit/Risk Analyst
- AP/AR Specialist

Let’s connect and explore how I can contribute to your team with data-driven insights and a resilient mindset.
""")


st.header("💼 Professional Experience")

with st.expander("US Med-Equip (Houston, TX) — Accounts Receivable Specialist Intern"):
    st.write("""
    My internship at US Med-Equip gave me hands-on exposure to U.S. healthcare finance operations. I worked on ACH migration, AR collections, and vendor compliance — all within a fast-paced, data-driven environment. This experience sharpened my reporting skills and introduced me to real-world reconciliation challenges.
    """)
    st.markdown("""
    - 🗂️ Migrated ACH processes and streamlined AR collections  
    - 📊 Produced vendor compliance reports and reconciled balances  
    - 🌎 U.S. healthcare finance experience with operational impact  
    """)

with st.expander("Agarwal Mathur & Associates (India) — Finance Executive"):
    st.write("""
    Before moving to the U.S., I worked as a Finance Executive in India where I prepared financial statements, supported audit readiness, and contributed to business planning. This role built my foundation in accounting, compliance, and client-facing documentation — skills I now apply in portfolio analysis and financial modeling.
    """)
    st.markdown("""
    - 📄 Drafted financial statements and audit documentation  
    - 🧾 Ensured regulatory compliance and supported tax filings  
    - 📈 Contributed to business plans and financial forecasts  
    - 🇮🇳 Indian finance experience with strong accounting fundamentals  
    """)





st.header("Resume")
st.markdown("[Download Resume](https://docs.google.com/document/d/1not6MisdKmxFdy-ZN3Z1lRNY9FRd-IaW/edit?usp=sharing&ouid=100980197578796636924&rtpof=true&sd=true)")



st.header("🎓 My Education Journey")

st.markdown("### ✈️ Undergraduate Degree: Bhopal, India")

col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("🛫")
with col2:
    st.markdown("""
    **Bachelor of Commerce (Hons)**  
    *Bhopal School of Social Science*  
    📍 Location: Bhopal, India  
    🗓️ Duration: 3 Years  
    🧠 Focus: Accounting, Business Fundamentals  
    """)

st.markdown("---")

st.markdown("### 🛫 Post Graduation Degree: Bhopal, India")

col3, col4 = st.columns([1, 5])
with col3:
    st.markdown("🛫")
with col4:
    st.markdown("""
    **Master of Commerce (Accounting)**  
    *Bhopal School of Social Science* 
    📍 Location: Bhopal, India  
    🗓️ Duration: 2 Years  
    📊 Focus: Financial Reporting, Audit, Taxation  
    """)

st.markdown("---")

st.markdown("### 🛬 Post Graduation Degree: Buffalo, USA")

col5, col6 = st.columns([1, 5])
with col5:
    st.markdown("🛬")
with col6:
    st.markdown("""
    **MS in Finance**  
    *University at Buffalo, SUNY*  
    📍 Location: Buffalo, NY, USA  
    🗓️ Duration: 2 Years  
    💼 Focus: Financial Modeling, Derivatives, Portfolio Management  
    """)

st.markdown("---")
st.success("Building hands-on experience while completing MS in Finance, focused on practical applications in investment and analytics.")


st.header("📊 Projects")
st.subheader("🎓 Academic Projects & 🐍 Python Dashboards")

project_options = [
    "Select a Project",
    "Academic Valuation Projects",
    "NYC Flight Delay Analysis",
    "Multi Asset Portfolio Analysis",
    "Technical Analysis",
    "Portfolio Construction and Optimisation",
    "Multi-Sector Portfolio Analysis"
]

selected_project = st.radio("📂 Choose a Project", project_options)



               
# All Projects in One Chain

# Start the chain here
if selected_project == "Academic Valuation Projects":
    st.subheader("🏠 Home Depot Valuation – Group Project")
    st.write("""
    Conducted a comprehensive valuation of Home Depot using Discounted Cash Flow (DCF) and Comparable Company Analysis. Collaborated with team members to analyze financial statements, forecast future cash flows, and calculate intrinsic value.

    - 📄 [View Report](https://drive.google.com/file/d/1WCEaiDtXGtkwWAsXiu7LDMfTXS3uK3XP/view?usp=sharing)
    - 📊 [View Excel Model](https://docs.google.com/spreadsheets/d/1vCmRRhhedevGmg8NWFwyUHj1nXvHztjD/edit?usp=sharing)
    - 🛠️ Tools: Excel, PowerPoint
    - 📅 Status: Completed
    """)

    st.subheader("🍔 Shake Shack Valuation – Group Project")
    st.write("""
    Performed equity valuation for Shake Shack using Discounted Cash Flow (DCF) and market multiples. Assessed revenue drivers, cost structure, and growth potential. Presented findings in a detailed report and financial model.

    - 📄 [View Report](https://drive.google.com/file/d/1yS_BBCkAHN0ZZzJJwS09aiUl1l3UNd-M/view?usp=sharing)
    - 📊 [View Excel Model](https://docs.google.com/spreadsheets/d/1cZy0HUpumydOHF2u5wkxxblzXTAd_nkr/edit?usp=sharing)
    - 🛠️ Tools: Excel, PowerPoint
    - 📅 Status: Completed
    """)

if selected_project == "NYC Flight Delay Analysis":
    with st.expander("NYC Flight Delay Analysis"):
        st.markdown("""
        This project analyzes flight delay patterns across New York City airports using historical flight data.  
        The goal was to identify delay trends, highlight the most affected airlines and routes, and summarize key performance metrics.  
        Techniques used include data cleaning, aggregation, and visualization using Python libraries such as Pandas, Matplotlib, and Seaborn.
        """)

        st.pyplot(nyc.get_delay_chart())
        st.dataframe(nyc.get_top_delayed_flights())
        metrics = nyc.get_summary_metrics()
   
    

if selected_project == "Multi Asset Portfolio Analysis":
    with st.expander("Multi Asset Portfolio Analysis"):

        # 📘 Project Summary
        st.markdown("""
        ### 📊 Multi Asset Portfolio Analysis

        This project analyzes the performance and risk dynamics of three investment strategies over a five-year period (2020–2025), using historical market data:

        - **Portfolio 1:** Eight individual U.S. companies across diverse sectors  
          *(SBUX, UNP, JPM, XOM, AMZN, JNJ, CAT, MSFT)*

        - **Portfolio 2:** Eight sector-specific ETFs representing key segments of the economy  
          *(XLY, XLU, XLF, XLE, XLC, XLV, XLI, XLK)*

        - **Portfolio 3:** Benchmark index represented by SPY *(S&P 500 ETF)*

        The analysis includes:
        - 📈 Simulating portfolio growth from a $100,000 initial investment  
        - 📊 Evaluating performance metrics: CAGR, volatility, Sharpe ratio, max drawdown  
        - 📐 Comparing market sensitivity and excess returns using Beta and Alpha  
        - 📅 Measuring consistency through monthly return trends

        **Key insights:**
        - The **Companies portfolio** delivered the highest cumulative return and Alpha  
        - The **ETFs portfolio** showed the lowest Beta, indicating reduced market risk  
        - The **SPY benchmark** provided stable but lower returns

        **Tools used:** Python, Pandas, NumPy, Matplotlib, Statsmodels, yFinance  
        This project demonstrates practical application of portfolio theory, quantitative analysis, and financial modeling.
        """)
        
        # 📈 Portfolio Growth Chart
        st.pyplot(ma.get_portfolio_chart())

        # 📊 Portfolio Growth DataFrame
        st.subheader("Portfolio Growth Over Time")
        st.dataframe(ma.get_growth_dataframe())

        # 📋 Performance Summary
        st.subheader("Performance Summary")
        metrics = ma.get_summary_metrics()
        for label, value in metrics.items():
            formatted = (
                f"{value:.2%}" if isinstance(value, float) and ("Drawdown" in label or "CAGR" in label)
                else f"{value:.2f}" if isinstance(value, float)
                else str(value)
            )
            st.metric(label, formatted)

        # 📐 Beta and Alpha Comparison
        st.subheader("Beta and Alpha Comparison")
        beta_alpha = ma.get_beta_alpha()
        for portfolio, values in beta_alpha.items():
            st.write(f"**{portfolio}** — Beta: `{values['Beta']:.4f}`, Alpha: `{values['Alpha']:.4f}`")
        
if selected_project == "Technical Analysis":
    with st.expander("Technical Analysis"):
    
        # 📘 Project Summary
        st.markdown("""
        ### 📈 Technical Analysis of Goldman Sachs & Morgan Stanley

        This project applies technical analysis techniques to evaluate the price behavior and trading signals of two major financial institutions — Goldman Sachs (GS) and Morgan Stanley (MS) — over a 1.5-year period (2024–2025).

        **Key techniques used:**
        - 📊 Candlestick Charts to visualize daily OHLC patterns and market sentiment  
        - 📉 SMA & DEMA Trendlines to identify short-term and long-term momentum shifts  
        - 📎 Bollinger Bands to detect price extremes and volatility zones  
        - 📐 RSI & MACD Indicators to highlight overbought/oversold conditions and momentum crossovers  
        - 📊 ADX (Average Directional Index) to measure trend strength  
        - 🔁 Advanced Moving Averages including WMA, ZLEMA, and KAMA for adaptive trend tracking

        **Trading Signal Strategy:**
        - ✅ Buy Signal: SMA40 crosses above SMA100, RSI rises above 30, MACD crosses above Signal Line  
        - ❌ Sell Signal: Price hits upper Bollinger Band, SMA40 crosses below SMA100, MACD crosses below Signal Line

        **Insights:**
        - GS showed higher returns but greater volatility; MS was more stable with smoother price action  
        - Multi-indicator alignment (SMA, RSI, MACD) improved signal reliability  
        - ADX confirmed strong trend phases, helping validate entry/exit points

        **Tools used:** Python, yFinance, Pandas TA, Matplotlib, Plotly, mplfinance  
        This project demonstrates practical application of technical indicators for signal generation, trend analysis, and volatility assessment in equity markets.
        """)
        
        st.write("Applied RSI, MACD, and Bollinger Bands to identify trading signals.")
        st.pyplot(ta.get_technical_chart())
        st.metric("Signal Accuracy", f"{ta.signal_accuracy:.2%}")
        st.pyplot(ta.get_rsi_chart())
        st.pyplot(ta.get_macd_chart())
        st.pyplot(ta.get_bollinger_chart())
        st.pyplot(ta.get_dema_chart())
        st.pyplot(ta.get_adx_chart())
        st.pyplot(ta.get_moving_avg_variations_chart())
    



# Run analysis immediately when project is selected
if selected_project == "Portfolio Construction and Optimisation":
    with st.expander("📐 Portfolio Construction and Optimisation", expanded=True):
        st.markdown("## 📊 Portfolio Construction and Optimisation")
        
        # ✅ Define helper functions here
        def max_drawdown(cumulative_returns):
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()

        def get_metrics(daily_ret, label):
            ann_return = np.mean(daily_ret) * 252
            ann_vol = np.std(daily_ret) * np.sqrt(252)
            sharpe = ann_return / ann_vol
            st.markdown(f"### 📊 {label} Performance")
            st.write(f"Annualized Return: {ann_return:.4f}")
            st.write(f"Annualized Volatility: {ann_vol:.4f}")
            st.write(f"Sharpe Ratio: {sharpe:.4f}")

        st.markdown("""
            ### 📐 Portfolio Construction and Optimisation
                
                This project focuses on building and optimizing investment portfolios using historical market data and quantitative techniques.  
                The goal is to construct portfolios that balance risk and return across multiple assets, using modern portfolio theory and statistical modeling.
                
                **Key components of the analysis:**
                - 📊 Asset selection across diverse sectors and asset classes  
                - ⚖️ Equal-weight vs optimized allocation strategies  
                - 📈 Cumulative performance comparison against benchmark (e.g., SPY)  
                - 📉 Risk metrics including volatility, Value at Risk (VaR), and Conditional VaR (CVaR)  
                - 📐 Sharpe Ratio maximization and minimum volatility portfolio construction  
                - 🧠 Efficient frontier visualization to explore optimal risk-return combinations  
                - 🔍 Risk contribution analysis to understand asset-level impact
                
                **Optimization techniques used:**
                - Mean-variance optimization  
                - Risk parity and minimum volatility models  
                - Constraints on weights and diversification thresholds
                
                **Insights:**
                - The **optimized portfolio** consistently outperformed the benchmark on a risk-adjusted basis  
                - The **minimum volatility portfolio** offered smoother returns with lower drawdowns  
                - Risk contribution analysis revealed concentration risks and diversification benefits  
                - Efficient frontier plots helped visualize trade-offs and guide allocation decisions
                
                **Tools used:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, yFinance  
                This project demonstrates practical application of portfolio theory, optimization algorithms, and risk modeling in real-world investment strategy design.
                """)
                

        # 📈 Data Setup
        tickers = ['JPM', 'GS', 'BRK-B', 'WFC', 'C']
        benchmark_ticker = 'SPY'
        start_date = '2020-01-01'
        end_date = '2023-12-31'

        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
        spy_df = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)

        adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
        adj_close['SPY'] = spy_df['Adj Close'].reindex(adj_close.index).ffill()
        daily_returns = adj_close.pct_change().dropna()

        # 🔥 Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix of Daily Returns')
        st.pyplot(fig)

        # 📉 Rolling Volatility
        rolling_volatility = daily_returns.rolling(window=30).std()
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_volatility.plot(ax=ax)
        ax.set_title('30-Day Rolling Volatility')
        st.pyplot(fig)

        # 📈 Rolling Sharpe Ratio
        rolling_mean = daily_returns.mean(axis=1).rolling(window=30).mean()
        rolling_std = daily_returns.std(axis=1).rolling(window=30).std()
        rolling_sharpe = rolling_mean / rolling_std
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_sharpe.plot(ax=ax, color='purple')
        ax.set_title('30-Day Rolling Sharpe Ratio')
        st.pyplot(fig)

        # 📊 Summary Stats
        summary_stats = daily_returns.describe().T
        for stat in ['mean', '50%', 'std']:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(summary_stats.index, summary_stats[stat], color='skyblue')
            ax.set_title(f'{stat.capitalize()} of Daily Returns')
            st.pyplot(fig)

        skewness = daily_returns.skew()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(skewness.index, skewness.values, color='skyblue')
        ax.set_title('Skewness of Daily Returns')
        st.pyplot(fig)

        # ⚖️ Equal Weight Portfolio
        weights_portfolio = np.array([0.2] * 5)
        annualised_returns = daily_returns.mean() * 252
        annual_cov_matrix = daily_returns.cov() * 252

        portfolio_return = np.dot(weights_portfolio, annualised_returns[tickers])
        portfolio_volatility = np.sqrt(np.dot(weights_portfolio.T, np.dot(annual_cov_matrix.loc[tickers, tickers], weights_portfolio)))
        portfolio_sharpe = portfolio_return / portfolio_volatility

        spy_return = annualised_returns['SPY']
        spy_volatility = np.sqrt(annual_cov_matrix.loc['SPY', 'SPY'])
        spy_sharpe = spy_return / spy_volatility

        st.metric("Sharpe Ratio (Equal)", f"{portfolio_sharpe:.2f}")
        st.metric("Sharpe Ratio (SPY)", f"{spy_sharpe:.2f}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(weights_portfolio, labels=tickers, autopct='%1.1f%%', startangle=140)
        ax.set_title('Portfolio Allocation (Equal Weights)')
        st.pyplot(fig)

        # 🧠 Optimized Portfolio
        mu = expected_returns.mean_historical_return(adj_close[tickers])
        s = risk_models.sample_cov(adj_close[tickers])
        ef = EfficientFrontier(mu, s)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        ef.portfolio_performance(verbose=False)
        optimized_sharpe = ef.portfolio_performance()[2]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(cleaned_weights.keys()), list(cleaned_weights.values()), color='skyblue')
        ax.set_title('Optimized Portfolio Allocation (Max Sharpe)')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_efficient_frontier(EfficientFrontier(mu, s), ax=ax, show_assets=True)
        ax.set_title('Efficient Frontier')
        st.pyplot(fig)

        # 🔍 Risk Metrics
        weights_array = np.array(list(cleaned_weights.values()))
        selected_tickers = list(cleaned_weights.keys())
        portfolio_variance = np.dot(weights_array.T, np.dot(annual_cov_matrix.loc[selected_tickers, selected_tickers], weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        confidence_level = 0.95
        portfolio_mean = np.dot(weights_array, mu)
        VaR = norm.ppf(1 - confidence_level) * portfolio_volatility - portfolio_mean

        st.write({
            "Portfolio Variance": round(portfolio_variance, 6),
            "Portfolio Volatility": round(portfolio_volatility, 4),
            "Value at Risk (95%)": round(VaR, 6)
        })

        # ✅ Strategic Evaluation
        equal_portfolio_return = daily_returns[tickers].dot(weights_portfolio)
        portfolio_daily_return = daily_returns[selected_tickers].dot(weights_array)
        spy_daily_return = daily_returns['SPY']

        equal_cumulative_return = (1 + equal_portfolio_return).cumprod()
        optimized_cumulative_return = (1 + portfolio_daily_return).cumprod()
        spy_cumulative_return = (1 + spy_daily_return).cumprod()

        get_metrics(equal_portfolio_return, "Equal Weight Portfolio")
        get_metrics(portfolio_daily_return, "Optimized Portfolio")
        get_metrics(spy_daily_return, "SPY Benchmark")

        st.markdown("### 📉 Max Drawdown Comparison")
        st.write(f"Optimized Portfolio: {max_drawdown(optimized_cumulative_return):.2%}")
        st.write(f"Equal Weight Portfolio: {max_drawdown(equal_cumulative_return):.2%}")
        st.write(f"SPY Benchmark: {max_drawdown(spy_cumulative_return):.2%}")


           
            
            

if selected_project == "Multi-Sector Portfolio Analysis":
    with st.expander("🌐 Multi-Sector Portfolio Analysis"):
        # 📘 Project Summary
        st.markdown("""
        ### 🌐 Multi-Sector Portfolio Analysis

        This project evaluates the performance and risk characteristics of a diversified portfolio composed of sector-specific assets.  
        The analysis compares three strategies over a multi-year period:

        - **Equal-Weighted Portfolio:** Diversified allocation across multiple sectors  
        - **Optimized Portfolio:** Constructed using mean-variance optimization to maximize Sharpe Ratio  
        - **Benchmark Portfolio:** Represented by SPY (S&P 500 ETF)

        **Key components of the analysis:**
        - 📈 Price trends and normalized performance across sectors  
        - 🔗 Correlation heatmap to assess inter-asset relationships  
        - 📉 Rolling volatility to monitor risk over time  
        - 📊 Sharpe Ratios to compare risk-adjusted returns  
        - 🧮 Value at Risk (VaR) and Conditional VaR (CVaR) for downside risk assessment  
        - 🧠 Risk contribution analysis to understand asset-level impact  
        - 📐 Efficient frontier visualization to explore trade-offs between risk and return

        **Insights:**
        - The **optimized portfolio** achieved the highest Sharpe Ratio, outperforming both equal-weight and SPY  
        - The **equal-weight strategy** offered better diversification but slightly lower returns  
        - The **minimum volatility portfolio** provided the smoothest ride with moderate returns  
        - Sector correlations and volatility patterns revealed key diversification benefits

        **Tools used:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, yFinance  
        This project demonstrates practical application of modern portfolio theory, quantitative risk modeling, and optimization techniques in multi-asset investing.
        """)
        
        st.write("This section analyzes diversified assets across sectors, comparing equal-weight and optimized strategies against SPY.")
        st.pyplot(ms.get_price_chart())
        st.pyplot(ms.get_normalized_chart())
        st.dataframe(ms.get_summary_table())
        st.pyplot(ms.get_correlation_heatmap())
        st.pyplot(ms.get_rolling_volatility_chart())
        st.metric("Sharpe Ratio (Equal)", f"{ms.equal_sharpe:.2f}")
        st.metric("Sharpe Ratio (SPY)", f"{ms.spy_sharpe:.2f}")
        st.pyplot(ms.get_cumulative_comparison_chart())
        st.pyplot(ms.get_equal_weight_pie_chart())
        st.metric("Sharpe Ratio (Optimized)", f"{ms.opt_sharpe:.2f}")
        st.pyplot(ms.get_optimized_weights_chart())
        st.pyplot(ms.get_optimized_cumulative_chart())
        st.metric("95% VaR", f"{ms.var_95:.4f}")
        st.metric("95% CVaR", f"{ms.cvar_95:.4f}")
        st.dataframe(ms.get_risk_contribution_table())
        st.metric("Sharpe Ratio (Min Vol)", f"{ms.minvol_sharpe:.2f}")
        st.pyplot(ms.get_strategy_comparison_chart())
        st.dataframe(ms.get_strategy_summary_table())
        st.pyplot(ms.get_efficient_frontier_chart())
    
    




st.markdown("---")
st.success("Thanks for visiting my dashboard! Let’s connect and explore opportunities in finance and analytics.")

































