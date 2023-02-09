import streamlit as st
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import quandl as qd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

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

key = os.environ.get('Quandl_API_Key') #Retrieve Quandl key from environment variable
qd.ApiConfig.api_key = key

st.set_page_config(layout="wide")

st.title('Analyzing Inflation, Earnings, and the S&P 500')

st.write("""The S&P 500 is a weighted stock market index that tracks the performance of the five-hundred leading publicly traded companies in the United States.
         Undoubtedly, it is the most followed index and is regarded as the best gauge of the overall stock market. The value of the S&P 500 can be broken down
         into two components: 1) The earnings per share (EPS) value of the companies that make up the index. 2) The price to earnings multiple (P/E) that investors are willing
         to pay for the EPS value. Simply put, the value of the S&P 500 can be calculated with the following formula:""")

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
         since EPS and P/E are updated on a monthly basis, while the S&P 500 close is updated daily. So, if you want to know where the value of S&P 500
         will trade all you need to know is two things! 1) What are earnings going to be. 2) What is the P/E multiple going to be. The problem is... 
         no one knows what these are going to be. Banks, investment firms, hedge funds, and the like hire vast teams of analysts to come up with various models
         and predictions for these figures. The reality is, none of them are able to predict these perfectly. That being said, I still think it is an important 
         and informative exercise to analyze historical data, and use the analysis as a basis to come up with reasonable expectations of where the S&P 500 could trade.""")

st.write("This project consists of three parts: ")

st.markdown("- Part 1 - Exploring the relationship between inflation and the P/E multiple of the S&P 500.")
st.markdown("- Part 2 - Analyzing the historical EPS growth of the S&P 500.")
st.markdown("- Part 3 - Putting it all together and allowing a user to see where the S&P 500 may trade based on their inputs.")

st.write("""This project is for educational purposes only and should not be used for making investment decisions.""")

st.subheader('Part 1: Inflation And The P/E Multiple')

st.write("""Inflation has arguably been the largest topic of economic concern in the past year. In March 2022 the US Federal Reserve started to aggressively take
         measures to cool the rate of inflation in the US by raising interest rates. The reason for looking at the relationship between
         inflation and the P/E multiple is that historical data has shown there to be a [negative correlation between these variables](https://www.investopedia.com/ask/answers/123.asp#toc-review-of-the-pe-ratio).
         To see this relationship, I created a scatter plot with a regression model using data points from 1965 onward. Why did I choose 1965 as the starting year? This year was the beginning
         of what is regarded as [the period of great inflation](https://www.federalreservehistory.org/essays/great-inflation), where the US entered into a period of high
         inflation and economic hardship that lasted over a decade and a half. There has been a lot of commentary that the current period we are going through could resemble
         that of the great inflation period, should the US Federal Reserve not take adequate measures to bring down the current inflation rate. As such, I felt this was a logical
         place to start the data collection. After removing P/E outliers (values that exceeded the 95th percentile), it appears that the relationship between inflation and the
         P/E multiple isn't perfectly linear. A two degree polynomial regression model ended up being the best approach for capturing this relationship: 
         """)

### GET INFLATION AND P/E DATA FROM NASDAQ USING QUANDL

today = datetime.today().date()

today_day_num = today.strftime("%d") #Establishes today as a number

start_date = "1965-01-01"

end_date = today - timedelta(days=int(today_day_num) - 1) #Establish the end date as the first date of the current month

#Since the dates do not match you have to read the symbols in separately and then combine them using the pandas merge as of command

inflation_data = qd.get("RATEINF/INFLATION_USA", start_date = start_date, end_date = end_date)

pe_data = qd.get("MULTPL/SP500_PE_RATIO_MONTH", start_date = start_date, end_date = end_date)

inflation_df = pd.merge_asof(left=inflation_data, right=pe_data, right_index=True,left_index=True,direction='nearest') #Merge to the closest date since the days do not line up perfectly
inflation_df = inflation_df.rename(columns={'Value_x':'Inflation',
                                            'Value_y':'S&P500_PE'}) #Rename the columns after performing the pandas merge as of

### REMOVE OUTLIERS 

upper_limit = inflation_df['S&P500_PE'].quantile(0.95) #Establishes the cutoff for removing outliers

inflation_df = inflation_df[(inflation_df['S&P500_PE'] < upper_limit)] #Filters the data to values less than the cutoff

### Plot the scatter plot using a two degree polynomial regression model

fig = plt.figure(figsize=(14, 6))
sns.regplot(x='Inflation', y='S&P500_PE', data=inflation_df, order = 2, line_kws={"color":"black"})
sns.set(style="ticks")
sns.despine()
plt.xlabel("Inflation Rate", fontsize= 14, labelpad =12)
plt.ylabel("P/E Multiple", fontsize= 14, labelpad =12)
plt.title("Relationship Between Inflation and The P/E Multiple Of The S&P 500", fontsize=16, pad= 12)
plt.grid();

st.pyplot(fig)

# GET THE CURRENT AND EXPECTED INFLATION FIGURES

expected_inflation = get_fred_data_latest('EXPINF1YR')

current_inflation = get_quandl_data_latest("RATEINF/INFLATION_USA")

## Establish the polynomial regression model

x = inflation_df['Inflation']
y = inflation_df['S&P500_PE']

sm_poly = PolynomialFeatures(degree=2)
sm_poly_features = sm_poly.fit_transform(x.values.reshape(-1,1)) #Transform the inflation value to a two degree polynomial feature

model = sm.OLS(y, sm_poly_features).fit()

#Predict P/E based on current inflation rate

poly_inputs = sm_poly.fit_transform(np.array(current_inflation).reshape(-1,1)) #Transform the current inflation rate to polynomial inputs
prediction_intervals = model.get_prediction(poly_inputs) #Produces point estimate, upper, and lower prediction intervals of transformed polynomial inflation rate
intervals = prediction_intervals.summary_frame(alpha=0.05) #Produces a pandas dataframe of intervals that you can select from

#Create point estimate, upper, and lower bounds from the output of intervals summary frame

point_estimate = round(intervals['mean'][0],2)
lower_estimate = round(intervals['mean_ci_lower'][0],2)
upper_estimate = round(intervals['mean_ci_upper'][0],2)

#If the current P/E is above the point estimate write a blurb about it

if current_pe > point_estimate:
    
    st.write(f"""The current US inflation rate is {round(current_inflation,2)}%. Based on the above regression model, this would equate to a P/E point estimate of {point_estimate},
             with a lower confidence interval of {lower_estimate}, and an upper confidence interval of {upper_estimate}.""")
    
#If the current P/E is lower than the write a separate blurb about it

else:
    
    st.write("Need to come up with a sentence for when the P/E multiple is less than the estimate.")

st.write(f"Based on FRED estimates, the current [1-Year expected inflation rate](https://fred.stlouisfed.org/series/EXPINF1YR) is {round(expected_inflation,2)}.")

#DISPLAY THE RESULTS OF THE MODEL SUMMARY   

st.write(model.summary())

st.subheader('Part 2: Analyzing Historical Earnings')

st.subheader('Part 3: Putting It All Together')

st.sidebar.info(
        """
        This app was built by Rory James
        
        Data for this project was sourced from the [Nasdaq Data Link](https://data.nasdaq.com/) and [FRED](https://fred.stlouisfed.org/)
        
        [Click Here For The Project Source Code](https://github.com/RoryAJames/Analyzing-Inflation) 
        
        Feel free to connect with me:        
        [GitHub](https://github.com/RoryAJames) | [LinkedIn](https://www.linkedin.com/in/rory-james-873493111/)
        
    """
    )