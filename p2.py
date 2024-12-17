import pandas_datareader as web
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import datetime as dt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score,roc_curve,auc)
import itertools
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import os
import tweepy as tw
import requests
import config
from prophet import Prophet
import yfinance as yf


#import talib as ta

auth = tw.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

def buy_sell(signal):
    buy=[]
    sell=[]
    flag=-1
    for i in range(0,len(signal)):
        if signal['MACD'][i]> signal['Signal'][i]:
            sell.append(np.nan)
            if flag!= 1:
                buy.append(signal['Close'][i])
                flag =1
            else:
                buy.append(np.nan)
        elif signal['MACD'][i]< signal['Signal'][i]:
            buy.append(np.nan)
            if flag !=-0:
                sell.append(signal['Close'][i])
                flag=0
            else:
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return (buy,sell)

def bol_band(newdf):
    buy = []
    sell = []
    for i in range((len(newdf['Close']))):
        if newdf['Close'][i] > newdf['Upper'][i]:
            buy.append(np.nan)
            sell.append(newdf['Close'][i])
        elif newdf['Close'][i]< newdf['Lower'][i]:
            buy.append(newdf['Close'][i])
            sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return buy,sell
        
def ema_buy_sell(df):
    buy = []
    sell = []
    flag_long =False
    flag_short = False
    for i in range(len(df)):
        if df['mid'][i]< df['long'][i] and df['short'][i] < df['mid'][i] and flag_long ==False and flag_short==False:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            flag_short =True
        elif flag_short == True and df['short'][i] > df['mid'][i]:
            sell.append(df['Close'][i])
            buy.append(np.nan)
            flag_short =False
        elif df['mid'][i]> df['long'][i] and df['short'][i] > df['mid'][i] and flag_long ==False and flag_short==False:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            flag_long =True
        elif flag_long == True and df['short'][i] < df['mid'][i]:
            sell.append(df['Close'][i])
            buy.append(np.nan)
            flag_long =False        
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return buy ,sell
        

    
        


def get_mape(y_true, y_pred): 
    """
    Compute Mean Absolute Percentage Error (MAPE)
    
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    
    OUTPUT:
    mape - Mean Absolute Percentage Error (%)
    
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return mape

def get_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE)
    
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    
    OUTPUT:
    rmse - Root Mean Squared Error
    
    """
    rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))
                   
    return rmse

def get_x_y(data, N, offset):
    """
    Split data into input variable (X) and output variable (y)
    
    INPUT:
    data - dataset to be splitted
    N - time frame to be used
    offset - position to start the split
    
    OUTPUT:
    X - input variable
    y - output variable
    """
    X, y = [], []
    
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y




st.write("""
         #  Stock Recommendation Model
         **Visualize all the Stock Data ** 
         """)
         

# Sider bar header
st.sidebar.header('User Input')

def get_input():
    start = st.sidebar.text_input("Start Date", "2020-01-01")
    start_date = pd.to_datetime(start).date()
    end = dt.datetime.now().date()
    end_da = st.sidebar.text_input("End Date", end)
    end_date = pd.to_datetime(end_da).date()
    
    
    return start_date , end_date 


#sidebar
comp_name = st.sidebar.text_input('Ticker Name','GOOG')

start,end = get_input()
strategy = st.sidebar.selectbox("Select Strategy",('Home',"Machine Learning","Technical Analysis"))


def get_dataset(comp_name,start_date,end_date):

    comp = yf.Ticker(comp_name)
    
    df = comp.history(start=start_date,end=end_date)
    return df


df = get_dataset(comp_name, start, end)
df=pd.DataFrame(df)

def functionalities(comp_name,df,strategy):
    if strategy =='Home':
        st.header('Guildlines')
        st.write('1. To get Started First Select which stock you want check and update it in sidecar using Ticker of that Stock')
        st.write('2. Update the dates according to your needs')
        st.write("""3. Once entered all required data,  select the option you wanna go in 'Select Strategy' dropbox""")
        st.header("HOME")
        st.write('______________________________________________')
        st.write(comp_name+" Close Price\n")
        st.line_chart(df['Close'])
        
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high = df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        st.write('___________________________________')
        st.write(comp_name+" Candlestick Chart")  
        fig.update_layout(height=650)
        st.plotly_chart(fig)
        st.write('_________________________________')
        st.header(comp_name+' Fundamentals')
        st.write('                               ')

        
        # metric = [       'priceToBookRatioTTM',
        #           'priceEarningsRatioTTM',
        #           'debtEquityRatioTTM',
        #           'returnOnEquityTTM',
        #           'currentRatioTTM',
        #           'quickRatioTTM',
        #           'operatingCashFlowPerShareTTM',
        #           'cashRatioTTM',
        #           'interestCoverageTTM',
        #           'returnOnAssetsTTM',
        #           'priceToSalesRatioTTM',
        #           'dividendYielTTM',
        #           'grossProfitMarginTTM'
        #           ]
        
        
        
        r = requests.get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{comp_name}?apikey=8e394203572a076787825eca3419fb2a")
        data = r.json()
        # df1 = {}
        # for i in range(len(metric)):
        #     if metric[i] in data[0]:
        #         df1[metric[i]] = data[0][metric[i]]
        st.write('return On Equity - (ROE) - '+ str(data[0]['returnOnEquityTTM']))
        st.write('price To Book Ratio - (P/B) - '+ str(data[0]['priceToBookRatioTTM']))
        st.write('price Earnings Ratio - (P/E) - '+ str(data[0]['priceEarningsRatioTTM']))
        st.write('debt to Equity Ratio - (D/E) - '+ str(data[0]['debtEquityRatioTTM']))
        st.write('current Ratio - '+ str(data[0]['currentRatioTTM']))
        st.write('quick Ratio- '+ str(data[0]['quickRatioTTM']))
        st.write('operating Cash Flow Per Share - '+ str(data[0]['operatingCashFlowPerShareTTM']))
        st.write('cash Ratio - '+ str(data[0]['cashRatioTTM']))
        st.write('interest Coverage - '+ str(data[0]['interestCoverageTTM']))
        st.write('return On Assets TTM - '+ str(data[0]['returnOnAssetsTTM']))
        st.write('price To Sales Ratio TTM - '+ str(data[0]['priceToSalesRatioTTM']))
        st.write('dividend Yield TTM - '+ str(data[0]['dividendYielTTM']))
        st.write('gross Profit Margin TTM - '+ str(data[0]['grossProfitMarginTTM']))



        



   
     
            



        
    elif strategy == 'Machine Learning':
        ml_model = st.sidebar.selectbox("Select Model", ('LSTM', 'Tree Classifier', 'Prophet'))
        st.header('Machine Learning Model - ' + ml_model)
        future_days = 90
        new_df1 = pd.DataFrame()
        new_df1['Close'] = df['Close']
        new_df1['Prediction'] = df[['Close']].shift(-future_days)
        X = np.array(new_df1.drop(['Prediction'], axis=1))[:-future_days]
        y = np.array(new_df1['Prediction'])[:-future_days]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        x_future = new_df1.drop(['Prediction'], axis=1)[:-future_days]
        x_future = x_future.tail(future_days)
        x_future = np.array(x_future)





       
            
            
            

        if ml_model == 'Tree Classifier':
            st.write('_____________________________')
            
            st.write('Decision trees in Machine Learning are used for building classification and regression models to be used in data mining and trading. A decision tree algorithm performs a set of recursive actions before it arrives at the end result ')
            st.write('_____________________________')
            tree = DecisionTreeRegressor().fit(x_train, y_train)
            tree_prediction = tree.predict(x_future)
            predictions = tree_prediction
            valid =  new_df1[X.shape[0]:]
            valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices

            plt.subplot(2,1,1)
            plt.ylabel('Close Price USD ($)',fontsize=8)
            plt.plot(new_df1['Close'])
            plt.subplot(2,1,2)
            plt.plot(valid[['Close','Predictions']])
            plt.ylabel('Close Price USD ($)',fontsize=8)
            plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
            plt.xticks(rotation=45)

            plt.show()
            st.pyplot()
            st.write(tree.score(x_test, y_test))
        
        elif ml_model == 'LSTM':
            st.write('_____________________________')
            
            st.write('Long-Short-Term Memory Recurrent Neural Network belongs to the family of deep learning algorithms. It is a recurrent network because of the feedback connections in its architecture. It has an advantage over traditional neural networks due to its capability to process the entire sequence of data.')
            st.write('_____________________________')
            plt.plot( df['Close'], label = 'Closing Price History')
            plt.legend(loc = "upper left")
            plt.xlabel('Year')
            plt.ylabel('Stock Price ($)')
            plt.xticks(rotation=45)
            plt.show()
            st.pyplot()
            
            test_size = 0.05
            training_size = 1 - test_size
            test_num = int(test_size * len(df))
            train_num = int(training_size * len(df))
            train = df[:train_num][[ 'Close']]
            test = df[train_num:][[ 'Close']]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])
            scaled_data_train = scaled_data[:train.shape[0]]
            X_train, y_train = get_x_y(scaled_data_train, 60, 60)
            lstm_units = 50 
            optimizer = 'adam'
            epochs = 1
            batch_size = 1
            model = Sequential()
            model.add(LSTM(units = lstm_units, return_sequences = True, input_shape = (X_train.shape[1],1)))
            model.add(LSTM(units = lstm_units))
            model.add(Dense(1))
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')
            model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)
            inputs = df['Close'][len(df) - len(test) - 60:].values
            inputs = inputs.reshape(-1,1)
            inputs = scaler.transform(inputs)
            X_test = []
            for i in range(60, inputs.shape[0]):
                X_test.append(inputs[i-60:i,0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            closing_price = model.predict(X_test)
            closing_price = scaler.inverse_transform(closing_price)
            test['Predictions_lstm'] = closing_price
            plt.figure(figsize = (20,10))
            plt.xlabel('Year')
            plt.ylabel('Stock Price ($)')
            plt.plot( train['Close'], label = 'Closing Price History [train data]')
            plt.plot( test['Close'], label = 'Closing Price History [test data]')
            plt.plot( test['Predictions_lstm'], label = 'Closing Price - Predicted')
            plt.legend(loc = "upper left")
            plt.xticks(rotation=45)
            plt.show()
            st.pyplot()
            rmse_lstm = get_rmse(np.array(test['Close']), np.array(test['Predictions_lstm']))
            mape_lstm = get_mape(np.array(test['Close']), np.array(test['Predictions_lstm']))
            st.write('Root Mean Squared Error: ' + str(rmse_lstm))
            st.write('Mean Absolute Percentage Error (%): ' + str(mape_lstm))
            
            
        else:
            st.write('_________________________________')
            st.write('Prophet, designed and pioneered by Facebook, is a time series forecasting library that requires no data preprocessing and is extremely simple to implement.Prophet tries to capture the seasonality in the past data and works well when the dataset is large.')
            st.write('_________________________________')
            
            model = Prophet()
            period  = 60
            new_df2 = pd.DataFrame()
            new_df2['y']= df['Close']
            new_df2 = new_df2.reset_index()
            
            new_df2['ds'] = pd.to_datetime(new_df2['Date']).dt.tz_localize(None)
            train = new_df2[:len(new_df2)-period]
            valid = new_df2[len(new_df2)-period:]
            model.fit(train)
            future = model.make_future_dataframe(period)
            forecast = model.predict(future)
            forecast_valid = forecast['yhat'][len(new_df2)-period:]
            rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
            
            
            valid['Predictions'] = forecast_valid.values
            plt.plot(train['y'] , label = 'Close Price')
            plt.plot(valid[['y']], label = 'Actual Price')
            plt.plot(valid[['Predictions']],label='Predicted Price')
            plt.xticks(rotation=45)
            plt.legend()
            plt.show()
            st.pyplot()
            st.write('RMS = ' + str(rms))
            

        
        
        
    elif strategy == 'Technical Analysis':
       ta_indicator = st.sidebar.selectbox("Select Indicator",('EMA','RSI','MACD','Bollinger Band'))
       if ta_indicator == 'EMA':
           st.write('_________________________________')

           st.header('EMA - Expotential Moving Average')
           st.write('An exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance on the most recent data points. The exponential moving average is also referred to as the exponentially weighted moving average. An exponentially weighted moving average reacts more significantly to recent price changes than a simple moving average (SMA), which applies an equal weight to all observations in the period.')
           st.write('_________________________________')          
           short_ema = df.Close.ewm(span = 5, adjust=False ).mean()
           mid_ema = df.Close.ewm(span = 21, adjust=False ).mean()
           long_ema = df.Close.ewm(span = 63, adjust=False ).mean()
           plt.plot(df['Close'], label='Close price')
           plt.plot(short_ema, label='Short EMA')
           plt.plot(mid_ema , label = 'Middle EMA')
           plt.plot(long_ema,label='Long EMA',color = 'grey')
           plt.legend()
           plt.xticks(rotation=45)

           plt.show()
           st.pyplot()
           st.write('_________________________________')

           new_df = pd.DataFrame()
           new_df['Close'] = df['Close']
           new_df['short'] = short_ema
           new_df['mid'] = mid_ema
           new_df['long'] = long_ema
           new_df['Buy'] = ema_buy_sell(new_df)[0]
           new_df['Sell'] = ema_buy_sell(new_df)[1]     
           st.write('BUY/SELL Indicator')
           plt.plot(df['Close'], label='Close price', alpha=0.4)
           plt.plot(short_ema, label='Short EMA',alpha=0.4)
           plt.plot(mid_ema , label = 'Middle EMA',alpha=0.4)
           plt.plot(long_ema,label='LOng EMA',color = 'grey',alpha=0.4)
           
           plt.scatter(new_df.index,new_df['Buy'],label='Buy' ,color ='green' ,marker='^',alpha=1)
           plt.scatter(new_df.index,new_df['Sell'],label='Sell' ,color ='red' ,marker='v',alpha=1)
           plt.legend()
           plt.xticks(rotation=45)
           plt.show()
           st.pyplot()
           
           
       elif ta_indicator== 'RSI':        
           st.write('______________________________________________________________________')
           st.header('RSI-Relative Strength Index ')
           st.write('______________________________________________________________________')
           st.write('Relative Strength Index (RSI) measures the speed and change of price movements to evaluate if a stock is overbought or oversold.')
           st.write('Relative Strength Index (RSI) is a momentum indicator. It measures how well a stock is performing against itself by comparing the strength of price, speed, and magnitude of recent price changes. From this, RSI can evaluate overbought and oversold market conditions. If a stock is trading above its true value, it is considered overbought. If a stock is trading below its true value, it is considered oversold. RSI can also be used to reveal price trends and suggest points to enter and exit trades.')
           st.write('Relative Strength Index (RSI) is also an oscillator.Oscillators are indicators that are used when viewing charts that are ranging (non-trending) to determine overbought or oversold conditions. Its value is displayed as a number from 0–100. The traditional interpretation of the RSI sees the areas below 30% and above 70% as significant territories. Where the RSI is greater than 70%, this indicates that the stock is overbought or overvalued. This suggests that the stock is primed for a corrective pullback. Where the RSI is lower than 30%, this shows that the stock and oversold or undervalued, suggesting that the stock may be due for a reversal.')
           
           delta = df['Close'].diff()
           delta = delta[1:]
           up = delta.copy()
           down = delta.copy()
           up[up<0]= 0
           down[down>0]=0
           period = 14
           avg_gain = up.rolling(window=period).mean()
           avg_loss=abs(down.rolling(window=period).mean())
           RS = avg_gain / avg_loss
           RSI = 100.0 - (100.0 / (1.0 + RS))
           st.write('______________________________________________________________________')
           st.write()
           st.write(comp_name + ' Close Price')
           plt.plot(df['Close'])
           plt.xticks(rotation=45)

           plt.show()
           st.pyplot()
           st.write('______________________________________________________________________')
           
           st.write('RSI Chart')
           plt.plot(RSI.index,RSI)
           plt.axhline(30,linestyle='--' , alpha= 0.5, color='red')
           plt.axhline(70,linestyle='--' , alpha= 0.5, color='red')
           plt.xticks(rotation=45)
           plt.show()           
           st.pyplot()       
       elif ta_indicator== 'MACD':
           st.write('______________________________________________________________________')
           st.header('MACD')
           st.write('______________________________________________________________________')
           st.write('The Moving Average Convergence Divergence (MACD) crossover is a technical indicator that uses the difference between exponential moving averages (EMA) to determine the momentum and the direction of the market. The MACD crossover occurs when the MACD line and the signal line intercept, often indicating a change in the momentum/trend of the market.')
           st.write('MACD Line: This line is the difference between two given Exponential Moving Averages. To calculate the MACD line, one EMA with a longer period known as slow length and another EMA with a shorter period known as fast length is calculated. The most popular length of the fast and slow is 12, 26 respectively. The final MACD line values can be arrived at by subtracting the slow length EMA from the fast length EMA.')
           st.write('Signal Line: This line is the Exponential Moving Average of the MACD line itself for a given period of time. The most popular period to calculate the Signal line is 9. As we are averaging out the MACD line itself, the Signal line will be smoother than the MACD line.')
           st.write('''IF MACD LINE > SIGNAL LINE => BUY THE STOCK\n
IF SIGNAL LINE > MACD LINE => SELL THE STOCK
''')
           
           
           short_ema = df['Close'].ewm(span=12, adjust=False).mean()
           long_ema = df['Close'].ewm(span=26, adjust=False).mean()
           MACD = short_ema - long_ema
           signal = MACD.ewm(span=9,adjust=False).mean()
           new_df = pd.DataFrame()       
           new_df['MACD']= MACD
           new_df['Signal'] = signal
           new_df['Close'] = df['Close']
           st.write('_______________________________________________________________________')
           plt.plot(df.index,MACD,label = 'MACD', color='red')
           plt.plot(df.index,signal  , label='Signal LIne' , color = 'blue')
           plt.legend()
           plt.xticks(rotation=45)
           
           plt.show()
           st.pyplot()
           st.write('_______________________________________________________________________')

           st.write('BUY/SELL Indicator')
           a = buy_sell(new_df)
           new_df['Buy_signal_price']= a[0]
           new_df['Sell_signal_price'] = a[1]
           plt.scatter(new_df.index,new_df['Buy_signal_price'] , color='green' , label = 'BUY' , marker='^', alpha=1)
           plt.scatter(new_df.index,new_df['Sell_signal_price'] , color='red' , label = 'SELL' , marker='^', alpha=1)
           plt.plot(new_df['Close'], label ='Close Price', alpha= 0.3)
           plt.legend()
           plt.xticks(rotation=45)
           plt.show()
           st.pyplot()
       
       elif ta_indicator== 'Bollinger Band':
           st.write('_________________________________')
           st.header('Bollinger Band')
           st.write("""Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.
Bollinger Bands use 2 parameters, Period and Standard Deviations, StdDev. The default values are 20 for period, and 2 for standard deviations""")
           st.write('____________________')
           period =20
           newdf = pd.DataFrame()
           newdf['Close'] = df['Close']
           newdf['SMA'] = newdf['Close'].rolling(window=period).mean()
           newdf['STD'] = newdf['Close'].rolling(window=period).std()
           newdf['Upper'] = newdf['SMA'] + (newdf['STD']*2)
           newdf['Lower'] = newdf['SMA'] - (newdf['STD']*2)
           plt.plot(df.index,df['Close'] , label="Close Price", color='black')
           plt.plot(newdf.index,newdf['SMA'] , label="Simple Moving Average", color='blue')
           plt.plot(newdf.index,newdf['Upper'] , label="Upper Band", color='green')
           plt.plot(newdf.index,newdf['Lower'] , label="Lower Band", color='red')
           plt.legend()
           plt.xticks(rotation=45)
           plt.show()
           st.pyplot()
           a = bol_band(newdf)
           newdf['Buy']= a[0]
           newdf['Sell'] = a[1]
           plt.plot(newdf.index,newdf['Close'] , label="Close Price", color='grey', alpha=0.4)
           plt.scatter(newdf.index,newdf['Buy'], label="Buy", color='green', marker='^',alpha=1)
           plt.scatter(newdf.index,newdf['Sell'] , label="Sell", color='red', marker='^',alpha=1)
           plt.legend()
           plt.xticks(rotation=45)
           plt.show()
           st.pyplot()

           
         

           
           
           
    
                                 

functionalities(comp_name,df,strategy)
