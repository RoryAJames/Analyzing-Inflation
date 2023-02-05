import os
import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import quandl as qd

key = os.environ.get('Quandl_API_Key')
qd.ApiConfig.api_key = key

st.set_page_config(layout="wide")

st.title('Analyzing Inflation, Earnings, and the S&P 500')

st.write("""The S&P 500 is a weighted stock market index that tracks the performance of the five-hundred leading publicly traded companies in the United States.
         Undoubtedly, it is the most followed index and is regarded as the best gauge of the overall stock market. The price of the S&P 500 can be broken down
         into two components: 1) The earnings per share (EPS) of the companies that make up the index. 2) The price to earnings multiple (P/E) that investors are willing
         to pay for EPS. Simply put, the value of the S&P 500 can be calculated with the following formula:""")

st.latex(r'''\text{S\&P 500 Value} = \text{Earnings Per Share} * \text{Price to Earnings}''')

st.write('')

#Get latest data points from Quandl and FRED

current_eps =  qd.get("MULTPL/SP500_EARNINGS_MONTH", rows = 1)
current_eps = current_eps.iloc[0].values[0]

current_pe =  qd.get("MULTPL/SP500_PE_RATIO_MONTH", rows = 1)
current_pe = current_pe.iloc[0].values[0]

close = pdr.DataReader(['sp500'], 'fred')
close = close.iloc[:, 0][len(close) - 1] #Get the last closing value of the S&P 500

#Display the latest data points in metric

st.write('The most recent data points show the following:')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Most recent monthly EPS", current_eps)
col2.metric("Most recent monthly PE", current_pe)
col3.metric("Expected S&P 500 Value", round(current_pe * current_eps,2))
col4.metric("Latest S&P 500 Close Value", close)

st.write("""So, if you want to know where the value of S&P 500 will trade all you need to know is two things! 1) What are earnings per share going to be. 2) What is
         the price to earnings multiple going to be. The problem is... no one knows what these are going to be.""")

st.subheader('Part 1: Inflation And Price to Earnings Multiple')

st.sidebar.info(
        """
        This app was built by Rory James
        
        Data for this project was sourced from the [Nasdaq Data Link](https://data.nasdaq.com/search)
        
        [Click Here For The Project Source Code](https://github.com/RoryAJames/Analyzing-Inflation) 
        
        Feel free to connect with me:        
        [GitHub](https://github.com/RoryAJames) | [LinkedIn](https://www.linkedin.com/in/rory-james-873493111/)
        
    """
    )