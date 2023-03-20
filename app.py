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
from DataCleaning import OutlierRemover

#A function that returns the most recent value of a symbol in Quandl
def get_quandl_data_latest(symobl):
    latest_value =  qd.get(symobl, rows = 1) #Filters down to the last row in the data frame
    latest_value = latest_value.iloc[0].values[0] #Returns the value of the last row
    return round(latest_value,2)
 
 #A function that returns the most recent value of a symbol in FRED
def get_fred_data_latest(symbol):
    latest_value = pdr.DataReader([symbol], 'fred')
    latest_value = latest_value.iloc[:, 0][len(latest_value) - 1]
    return round(latest_value,2)

# A function that calculates future value of earnings based on current earnings and a growth rate
def eoy_estimate(current_value,growth_rate):
    estimate = current_value * (1+round(growth_rate/100,2))
    return round(estimate,2)

#A function that calculates the percent difference between two numbers
def percent_diff(current_value,future_value):
    value = ((future_value-current_value)/current_value)*100
    return round(value,2)

key = os.environ.get('Quandl_API_Key') #Retrieve Quandl key from environment variable
qd.ApiConfig.api_key = key

#Set app to a wide layout

st.set_page_config(layout="wide")

#App header

st.title('Analyzing Inflation, Earnings, and the S&P 500')

try:
    
    #Get latest data points from Quandl and FRED

    current_eps =  get_quandl_data_latest("MULTPL/SP500_EARNINGS_MONTH")

    current_pe = get_quandl_data_latest("MULTPL/SP500_PE_RATIO_MONTH")

    latest_close = get_fred_data_latest('sp500')

    st.write("""The S&P 500 is a weighted stock market index that tracks the performance of the leading five-hundred publicly traded companies in the United States.
            Undoubtedly, it is the most followed index and is regarded as the best gauge of the overall stock market. The value of the S&P 500 can be broken down
            into two components: 1) The earnings per share (EPS) value of the companies that make up the index, 2) The price to earnings multiple (P/E) that investors are willing
            to pay for EPS. Simply put, the value of the S&P 500 can be calculated with the following formula:""")

    st.latex(r'''\text{S\&P 500 Value} = \text{Earnings Per Share} * \text{Price to Earnings}''')

    st.write('')
    
    #Display the latest data points as Streamlit metrics

    st.write('Let\'s see if this holds true. The latest data points show the following:')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EPS", current_eps)
    col2.metric("P/E", current_pe)
    col3.metric("Expected S&P 500 Value", round(current_pe * current_eps,2))
    col4.metric("Latest S&P 500 Close", latest_close)

    st.write("""As you can see, the latest S&P 500 close value is currently trading pretty close to the expected value. There will be some slight deviations between these figures
            since EPS and P/E are updated on a quarterly and monthly basis respectively, while the S&P 500 close is updated daily. So, if you want to know where the value of S&P 500
            will trade at the end of a given year all you need to know is two things! What are earnings and the P/E multiple going to be? The problem is... 
            no one knows what these are going to be. Banks, investment firms, hedge funds, and the like hire vast teams of analysts to come up with various models
            and predictions for these figures. The reality is, none of them are able to predict them perfectly. That being said, I still think it is an important 
            and informative exercise to analyze historical data, and use the analysis as a basis to come up with reasonable expectations of where the S&P 500 could trade.""")

    st.write("This project consists of three parts: ")

    st.markdown("- Part 1 - Exploring the relationship between inflation and the P/E multiple of the S&P 500.")
    st.markdown("- Part 2 - Analyzing the historical EPS growth rate of the S&P 500.")
    st.markdown("- Part 3 - Putting it all together and allowing a user to see where the S&P 500 may trade based on their inputs.")

    st.write("""This project is for educational purposes only and should not be used for making investment decisions.""")

    st.subheader('Part 1: Inflation And The P/E Multiple')

    st.write("""Inflation has arguably been the largest topic of economic concern in the past year. In March 2022, the US Federal Reserve started to aggressively take
            measures to cool the rate of inflation in the US by raising interest rates. The reason for looking at the relationship between
            inflation and the P/E multiple is that there is supposed to be a [negative correlation between these data points](https://www.investopedia.com/ask/answers/123.asp#toc-review-of-the-pe-ratio).
            To see this relationship, I created a scatter plot with a regression model using data points from 1965 onward. Why did I choose 1965 as the starting year? This year was the beginning
            of what is regarded as [the period of great inflation](https://www.federalreservehistory.org/essays/great-inflation), where the US entered into a period of high
            inflation and economic hardship that lasted for over a decade and a half. There has been a lot of commentary that the current period we are going through could resemble
            the great inflation era if the US Federal Reserve does not take adequate measures to bring down the current inflation rate. As such, I felt this was a logical
            place to start the data collection and analysis. After removing P/E outliers (values that exceeded the 95th percentile), it appears that the relationship between inflation and the
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

    ### REMOVE P/E OUTLIERS - Values that exceed the 95th percentile

    upper_limit = inflation_df['S&P500_PE'].quantile(0.95) #Establishes the cutoff for removing outliers

    inflation_df = inflation_df[(inflation_df['S&P500_PE'] < upper_limit)] #Filters the data to values less than the cutoff

    ### PLOT THE SCATTER PLOT USING A TWO DEGREE POLYNOMIAL REGRESSION MODEL

    fig = plt.figure(figsize=(14, 6))
    sns.regplot(x='Inflation', y='S&P500_PE', data=inflation_df, order = 2, line_kws={"color":"black"})
    sns.set(style="ticks")
    sns.despine()
    plt.xlabel("Inflation Rate (%)", fontsize= 14, labelpad =12)
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

    # A function that provides model estimates for P/E based on an inflation rate that is provided

    def model_estimates(inflation_rate):
        
        poly_inputs = sm_poly.fit_transform(np.array(inflation_rate).reshape(-1,1)) #Transform the provided inflation rate to polynomial inputs
        prediction_intervals = model.get_prediction(poly_inputs) #Produces point estimate, upper, and lower prediction intervals of transformed polynomial inflation rate
        intervals = prediction_intervals.summary_frame(alpha=0.05) #Produces a pandas dataframe of intervals that you can select from

        #Create point estimate, upper, and lower bounds from the output of intervals summary frame

        point_estimate = round(intervals['mean'][0],2)
        lower_estimate = round(intervals['mean_ci_lower'][0],2)
        upper_estimate = round(intervals['mean_ci_upper'][0],2)
        
        return point_estimate, lower_estimate, upper_estimate

    #Get model estimates for current inflation rate

    current_point_estimate, current_lower_estimate, current_upper_estimate = model_estimates(current_inflation)

    #Get model estimates for expected inflation rate

    exp_point_estimate, exp_lower_estimate, exp_upper_estimate = model_estimates(expected_inflation)

    #SUMMARY OF THE REGRESSION MODELS, PREDICTIONS AND CONFIDENCE INTERVALS
    
    st.write(f"""Based on the scatter plot and regression line, there is a negative correlation between inflation and the P/E multiple. As inflation rises,
            investors have lower market return expectations, thus the P/E multiple decreases. While the regression model is not perfect, it does an exceptional job
            at predicting the P/E multiple during the excessively high inflationary periods. If we take the current US inflation rate of {round(current_inflation,2)}%
            and apply it to the regression model, we get a P/E point estimate of {current_point_estimate}. But this prediction is way off from the current P/E multiple!
            This could be attributed to current inflation being a backwards looking data point as markets are typically priced using forward looking predictions and expectations. 
            Fortunately, FRED provides a [1-Year expected inflation rate](https://fred.stlouisfed.org/series/EXPINF1YR) which is currently sitting at {expected_inflation}%.
            If we apply this rate to the regression model we get a P/E point estimate of {exp_point_estimate}.""")

    #A SUMMARY STATEMENT BASED ON THE CURRENT P/E MULTIPLE AND WHETHER IT IS WITHIN THE CONFIDENCE INTERVAL RANGE
    statment = " "

    if current_pe > exp_upper_estimate:
        statment = """At this moment, the current P/E multiple is greater than the upper confidence interval. While it is not a certainty, this does suggest that the current P/E multiple
        may decrease in the near future and could revert to the expected range of values."""
        
    elif exp_lower_estimate <= current_pe <= exp_upper_estimate:
        statment = """At this moment, the current P/E multiple is within the confidence interval range."""

    elif  current_pe < exp_lower_estimate:
        statment = """At this moment, the current P/E multiple is less than the lower confidence interval. While it is not a certainty, this does suggest that the current P/E multiple
        may increase in the near future and could revert to the expected range of values."""

    st.write(f"""It is worth noting that regression models also provide a range of values, known as confidence intervals, that represent where the true prediction value
            is expected to fall between. This is typically done at a 95% confidence interval. In other words, we are 95% confident that the true value will fall between a lower and upper limit.
            Rather than using a point estimate, we can use the confidence intervals to get a more complete picture of where the P/E multiple might be. In the case of the expected inflation
            rate, we get a P/E multiple range between {exp_lower_estimate} and {exp_upper_estimate}. {statment}
            """)

    #DISPLAY THE RESULTS OF THE MODEL SUMMARY   
        
    with st.expander("Click Here To See The Full Regression Model Summary"):
        st.write(model.summary())
        
    ## HISTORICAL EARNINGS

    st.subheader('Part 2: Analyzing Historical Earnings Growth')

    #Get historical earnings data

    historical_earnings = qd.get("MULTPL/SP500_EARNINGS_MONTH", start_date = start_date, end_date = end_date).pct_change(periods=12)*100 #Apply percent change with 12 periods to see annual
    historical_earnings.dropna(inplace=True) #Drop null values after calculating percent change
    historical_earnings = historical_earnings.rename(columns={'Value':'Historical_Earnings'})

    #Remove outliers in historical earnings using IQR method

    outlier_remover = OutlierRemover(historical_earnings)
    outlier_remover.remove_outliers_iqr('Historical_Earnings')
    historical_earnings = outlier_remover.df #Returns the historical earnings dataframe as df where the outliers have been removed

    #Create bad, median, and good years of EPS growth based on quantiles

    bad_year = round(historical_earnings['Historical_Earnings'].quantile(0.25),2)
    median_year = round(historical_earnings['Historical_Earnings'].median(),2)
    good_year = round(historical_earnings['Historical_Earnings'].quantile(0.75),2)

    st.write("""To determine where earnings will be at the end of a given year, I analyzed the historical EPS growth rate of the S&P 500 from 1965 onwards.
            Rather than trying to estimate a specific value for the EPS growth rate, I figured the best approach would be to determine a range of reasonable values.
            After removing outliers (using the IQR method), I decided that the EPS growth rate can fall into three options - bad, typical, and good years. For the sake of simplicity,
            I assigned these years to the quartile cutoff points in the EPS growth rate. This produced the following EPS growth rate options:         
            """)

    st.markdown(f"- Bad Year (25th Percentile) - {bad_year}%")
    st.markdown(f"- Typical Year (Median) - {median_year}%")
    st.markdown(f"- Good Year (75th Percentile) - {good_year}%")

    # Historical EPS growth rate box plot

    earnings_fig = plt.figure(figsize=(14, 6))
    sns.boxplot(y= 'Historical_Earnings', data=historical_earnings)
    sns.despine()
    plt.ylabel("EPS Growth Rate (%)", fontsize= 14, labelpad =12)
    plt.title("Historical EPS Growth Rate Of The S&P 500 Since 1965", fontsize=16, pad= 12);

    st.pyplot(earnings_fig)

    ## PUTTING IT ALL TOGETHER

    st.subheader('Part 3: Putting It All Together')

    st.write("""Now that we have analyzed the relationship between inflation and the P/E multiple, along with the the historical EPS growth rate options, I want to put the two together
            and let you see where the S&P 500 might trade based on your inflation and EPS growth rate inputs.
            """)

    row1_col1, row1_col2 = st.columns([1,1])

    #List of options for inflation rate

    inflation_rates = [None,"1 Year Expected Inflation Rate", "Manual Input"]

    ##USER SELECTION FOR INFLATION RATE

    with row1_col1:

        inflation_selection = st.selectbox("Where Do You Think Inflation Will Be In One Year",inflation_rates)

        if inflation_selection:

            if inflation_selection == "1 Year Expected Inflation Rate":
                inflation_selection = expected_inflation
                
            elif inflation_selection == "Manual Input":
                inflation_input = st.number_input("Enter An Inflation Rate (%)")
                inflation_selection = inflation_input
                
            #Apply the user selection to the regression model

            user_point_estimate, user_lower_estimate, user_upper_estimate = model_estimates(inflation_selection)

    ##USER SELECTION FOR EPS GROWTH
        
    #List of options for EPS growth rate

    eps_growth_rate = [None,"Bad Year - 25th Percentile", "Typical Year - Median", "Good Year - 75th Percentile", "Manual Input"]

    with row1_col2:

        eps_growth_rate_selection = st.selectbox("What Do You Think The One Year EPS Growth Rate Will Be",eps_growth_rate)
        
        if eps_growth_rate_selection:

            if eps_growth_rate_selection == "Bad Year - 25th Percentile":
                eps_growth_rate_selection = bad_year
                
            elif eps_growth_rate_selection == "Typical Year - Median":
                eps_growth_rate_selection = median_year
            
            elif eps_growth_rate_selection == "Good Year - 75th Percentile":
                eps_growth_rate_selection = good_year
                
            elif eps_growth_rate_selection == "Manual Input":
                eps_growth_rate_input = st.number_input("Enter An EPS Growth Rate (%)")
                eps_growth_rate_selection = eps_growth_rate_input

    #Button that will display final calculations. If None is selected for inflation or EPS it results in an error and displays the prompt in the except

    ok = st.button("Calculate S&P 500 Valuation")

    try:
        if ok:
            
            #FINAL CALCULATIONS BASED ON USER SELECTIONS
            eps_estimate = eoy_estimate(current_eps, eps_growth_rate_selection)
            final_value = round(user_point_estimate * eps_estimate,2)
            percent_difference = percent_diff(latest_close, final_value)
            
            st.write("Here is a breakdown based on your inputs: ")
            
            st.markdown(f""" - An inflation rate of {inflation_selection}% produces a P/E point estimate of {user_point_estimate}, with a lower confidence interval of {user_lower_estimate},
                        and an upper confidence interval of {user_upper_estimate}.""")
            
            st.markdown(f""" - An EPS growth rate of {eps_growth_rate_selection}% would bring the year end EPS value to {eps_estimate}.""")
                        
            st.markdown(f""" - These figures would result in the S&P 500 ending the year at {final_value}. This would produce a {percent_difference}% difference
                    from the most recent S&P 500 close value.""")
        
    except:
            st.write("Please make a valid selection.")

except:
    st.write("Apologies, I am experiencing higher than normal traffic volumes. Please check back later.")
    
## INFORMATION ON THE SIDE OF THE APP

st.sidebar.info(
        """
        This app was built by Rory James
        
        Data for this project was sourced from the [Nasdaq Data Link](https://data.nasdaq.com/) and [FRED](https://fred.stlouisfed.org/)
        
        [Click Here For The Project Source Code](https://github.com/RoryAJames/Analyzing-Inflation) 
        
        Feel free to connect with me:        
        [GitHub](https://github.com/RoryAJames) | [LinkedIn](https://www.linkedin.com/in/rory-james-873493111/)
        
    """)