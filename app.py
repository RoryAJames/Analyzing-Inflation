import os
import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import quandl as qd

#A function that returns the most recent value of a symbol in Quandl
def get_quandl_data_latest(symobl):
    latest_value =  qd.get(symobl, rows = 1) #Filters down to the last row in the data frame
    latest_value = latest_value.iloc[0].values[0] #Returns the value of the last row
    return latest_value
 
 #A function that returns the most recent value of a symbol in FRED
def get_fred_data_latest(symbol):
    latest_value = pdr.DataReader([symbol], 'fred')
    latest_value = latest_value.iloc[:, 0][len(latest_value) - 1]
    return latest_value

key = os.environ.get('Quandl_API_Key')
qd.ApiConfig.api_key = key

st.set_page_config(layout="wide")

st.title('Analyzing Inflation, Earnings, and the S&P 500')

st.write("""The S&P 500 is a weighted stock market index that tracks the performance of the five-hundred leading publicly traded companies in the United States.
         Undoubtedly, it is the most followed index and is regarded as the best gauge of the overall stock market. The value of the S&P 500 can be broken down
         into two components: 1) The earnings per share (EPS) of the companies that make up the index. 2) The price to earnings multiple (P/E) that investors are willing
         to pay for EPS. Simply put, the value of the S&P 500 can be calculated with the following formula:""")

st.latex(r'''\text{S\&P 500 Value} = \text{Earnings Per Share} * \text{Price to Earnings}''')

st.write('')

#Get latest data points from Quandl and FRED

current_eps =  get_quandl_data_latest("MULTPL/SP500_EARNINGS_MONTH")

current_pe = get_quandl_data_latest("MULTPL/SP500_PE_RATIO_MONTH")

latest_close = get_fred_data_latest('sp500')

#Display the latest data points as Streamlit metrics

st.write('Let\'s see if this holds true. The latest data points show the following:')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monthly EPS", current_eps)
col2.metric("Monthly P/E", current_pe)
col3.metric("Expected S&P 500 Value", round(current_pe * current_eps,2))
col4.metric("Latest S&P 500 Close", latest_close)

st.write("""As you can see, the expected S&P 500 value trades around the latest close value. There will be some slight deviations between these figures
         since EPS and P/E are updated on a monthly basis, while the S&P 500 close value is updated daily. So, if you want to know where the value of S&P 500
         will trade all you need to know is two things! 1) What are earnings per share going to be. 2) What is the price to earnings multiple going to be. 
         The problem is... no one knows what these are going to be. Banks, investment firms, hedge funds, and the like hire vast teams of analysts to come
         up with models and predictions for these sort of figures. The reality is, none of them are able to predict these perfectly. However, I still think it is an
         important and informative exercise to analyze historical data and use it as a basis to come up with reasonable expectations of where the S&P 500 will trade.""")

st.write("This project consists of three parts: ")

st.markdown("- Part 1 - Exploring the relationship between inflation and the P/E multiple of the S&P 500.")
st.markdown("- Part 2 - Analyzing the historical EPS growth of the S&P 500.")
st.markdown("- Part 3 - Putting it all together and allowing a user to see where the S&P 500 may trade based on their inputs.")

st.write("""This project is for educational purposes only and should not be used for making investment decisions.""")

st.subheader('Part 1: Inflation And The P/E Multiple')

st.write("""Inflation has arguably been the largest topic of economic concern in the past year. In March 2022 the US Federal Reserve started to aggressively take
         measures through raising interest rates and quantitative tightening in an effort to cool the rate of inflation in the US. The reason for looking at the relationship between
         inflation and the P/E multiple is that historical data has shown there to be a [negative correlation between these variables](https://www.investopedia.com/ask/answers/123.asp#toc-review-of-the-pe-ratio).
         To see this relationship I created a scatter plot with a regression model using data points from the 1970's onward. Why did I start from the 70\'s? 
         """)

expected_inflation = get_fred_data_latest('EXPINF1YR')

current_inflation = get_quandl_data_latest("RATEINF/INFLATION_USA")

st.write(f"""The current US inflation rate is {round(current_inflation,2)}. Based on FRED estimates, the current [1-Year expected inflation rate](https://fred.stlouisfed.org/series/EXPINF1YR) 
         is {round(expected_inflation,2)}.""")

st.subheader('Part 2: Analyzing Historical Earnings')

st.subheader('Part 3: Putting It All Together')

st.sidebar.info(
        """
        This app was built by Rory James
        
        Data for this project was sourced from the [Nasdaq Data Link](https://data.nasdaq.com/search) and [FRED](https://fred.stlouisfed.org/)
        
        [Click Here For The Project Source Code](https://github.com/RoryAJames/Analyzing-Inflation) 
        
        Feel free to connect with me:        
        [GitHub](https://github.com/RoryAJames) | [LinkedIn](https://www.linkedin.com/in/rory-james-873493111/)
        
    """
    )