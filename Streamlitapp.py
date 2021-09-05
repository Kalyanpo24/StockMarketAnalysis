import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
import math
import tweepy
import preprocessor as p
from textblob import TextBlob
import regex as re
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-20,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(5030).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return df
def LSTM_ALGO(df):
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(df['Date'],df['Close'],c = 'r')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend('Close',loc = 'upper right')
        st.pyplot(fig)
        FullData=df[['Close']].values
        st.write('Original Prices')
        st.write(FullData[-30:])
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler()
 
        DataScaler = sc.fit(FullData)
        X=DataScaler.transform(FullData)
 
        # Printing last 10 values of the scaled data which we have created above for the last model
        # Here I am changing the shape of the data to one dimensional array because
        # for Multi step data preparation we need to X input in this fashion
        X=X.reshape(X.shape[0],)
        X_samples = list()
        y_samples = list()
        NumerOfRows = len(X)
        TimeSteps=30  # next few day's Price Prediction is based on last how many past day's prices
        FutureTimeSteps=7 # How many days in future you want to predict the prices
        # Iterate thru the values to create combinations
        for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
            x_sample = X[i-TimeSteps:i]
            y_sample = X[i:i+FutureTimeSteps]
            X_samples.append(x_sample)
            y_samples.append(y_sample)
        X_data=np.array(X_samples)
        X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
        y_data=np.array(y_samples)
        TestingRecords=5
        X_train=X_data[:-TestingRecords]
        X_test=X_data[-TestingRecords:]
        y_train=y_data[:-TestingRecords]
        y_test=y_data[-TestingRecords:] 
        TimeSteps=X_train.shape[1]
        TotalFeatures=X_train.shape[2]
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        regressor = Sequential()
        regressor.add(LSTM(units = 20, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
        regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
        regressor.add(LSTM(units = 10, activation = 'relu', return_sequences=False ))
        regressor.add(Dense(units = FutureTimeSteps))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        import time
        StartTime=time.time()
        regressor.fit(X_train, y_train, batch_size = 7, epochs = 30)
 
        EndTime=time.time()
        st.write("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')
        predicted_Price = regressor.predict(X_test)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        st.write('#### Predicted Prices ####')
        st.write(predicted_Price)
         
        # Getting the original price values for testing data
        orig=y_test
        orig=DataScaler.inverse_transform(y_test)
        st.write('\n#### Original Prices ####')
        st.write(orig)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(orig,c = 'r')
        plt.plot(predicted_Price,c = 'g')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('### Accuracy of the predictions:'+ str(100 - (100*(abs(orig-predicted_Price)/orig)).mean().round(2))+'% ###')
        plt.legend(['Orig','Predict'],loc = 'lower right')
        st.pyplot(fig)
        st.write("Based on these last 30 Closing Prices", FullData[-30:])
        Last30DaysPrices=FullData[-30:]
        Last30DaysPrices=Last30DaysPrices.reshape(-1, 1)
        X_test=DataScaler.transform(Last30DaysPrices)
        NumberofSamples=1
        TimeSteps=X_test.shape[0]
        NumberofFeatures=X_test.shape[1]
        
        X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
        Next7DaysPrice = regressor.predict(X_test)
        Next7DaysPrice = DataScaler.inverse_transform(Next7DaysPrice)
        st.write("Next 7 Days Price",Next7DaysPrice)
        error_lstm = math.sqrt(mean_squared_error(orig, predicted_Price))
        st.write("LSTM RMSE:",error_lstm)

def ARIMA_ALGO(df):
        uniqueVals = df["Code"].unique()  
        len(uniqueVals)
        df=df.set_index("Code")
        #for daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat[0])
                obs = test[t]
                history.append(obs)
            return predictions
        for company in uniqueVals[:10]:
            data=(df.loc[company,:]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price','Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'],axis =1)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(Quantity_date)
            #plt.savefig('static/Trends.png')
            st.pyplot(fig)
            
            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            #fit in model
            predictions = arima_model(train, test)
            
            #plot graph
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(test,label='Actual Price', c='r')
            plt.plot(predictions,label='Predicted Price',c='y')
            plt.title('Stock Prediction Graph using ARIMA model')
            plt.legend(loc=4)
            #plt.savefig('static/ARIMA.png')
            st.pyplot(fig)
            print()
            st.write("##############################################################################")
            arima_pred=predictions[-2]
            st.write("Tomorrow's",user_input," Closing Price Prediction by ARIMA:",arima_pred)
            #rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            st.write("ARIMA RMSE:",error_arima)
            st.write("##############################################################################")
            return arima_pred, error_arima
def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test,label='Actual Price' )
        plt2.plot(y_test_pred,label='Predicted Price')
        plt2.title('Stock Prediction Graph using Linear Regression model')
        plt2.legend(loc=4)
        #plt2.savefig('static/LR.png')
        st.pyplot(fig)
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        st.write("##############################################################################")
        st.write("Tomorrow's ",user_input," Closing Price Prediction by Linear Regression: ",lr_pred)
        st.write("Linear Regression RMSE:",error_lr)
        st.write("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr
def retrieving_tweets_polarity(symbol):
        num_of_tweets = 200
        
        #companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
        #symbols = companies['Name'].tolist()                                    
        stock_ticker_map = pd.read_csv('https://raw.githubusercontent.com/Kalyanpo24/StockMarketAnalysis/main/Yahoo-Finance-Ticker-Symbols.csv')
        stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
        symbol = stock_full_form['Name'].to_list()[0][0:]

        auth = tweepy.OAuthHandler('7x7w8Ti8GmMtuhvj7IXBuvZmP', 'kpFGcKiGxP9dJtlQzZ99p2OK9HM0IjA1I8n23N40VJyctOzvkG')
        auth.set_access_token('1365544389498474497-97lzyZE9lroLNDzLhvW1dLS3MDKfV5', '8mF9CrBt4SvmewPZaapgL7kkh2iJus824SvhBJOvDpXlw')
        user = tweepy.API(auth)
        
        tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(num_of_tweets)
        
        tweet_list = [] #List of tweets alongside polarity
        global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
        tw_list=[] #List of tweets only => to be displayed on web page
        #Count Positive, Negative to plot pie chart
        pos=0 #Num of pos tweets
        neg=1 #Num of negative tweets
        for tweet in tweets:
            count=20 #Num of tweets to be displayed on web page
            #Convert to Textblob format for assigning polarity
            tw2 = tweet.full_text
            tw = tweet.full_text
            #Clean
            tw=p.clean(tw)
            #print("-------------------------------CLEANED TWEET-----------------------------")
            #print(tw)
            #Replace &amp; by &
            tw=re.sub('&amp;','&',tw)
            #Remove :
            tw=re.sub(':','',tw)
            #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
            #print(tw)
            #Remove Emojis and Hindi Characters
            tw=tw.encode('ascii', 'ignore').decode('ascii')

            #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
            #print(tw)
            blob = TextBlob(tw)
            polarity = 0 #Polarity of single individual tweet
            for sentence in blob.sentences:
                   
                polarity += sentence.sentiment.polarity
                if polarity>0:
                    pos=pos+1
                if polarity<0:
                    neg=neg+1
                
                global_polarity += sentence.sentiment.polarity
            if count > 0:
                tw_list.append(tw2)
                
            tweet_list.append((tw, polarity))
            count=count-1
        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)
        else:
            global_polarity = global_polarity
        neutral=num_of_tweets-pos-neg
        if neutral<0:
        	neg=neg+neutral
        	neutral=20
        print()
        st.write("##############################################################################")
        st.write("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
        st.write("##############################################################################")
        labels=['Positive','Negative','Neutral']
        sizes = [pos,neg,neutral]
        explode = (0, 0, 0)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.title('Sentiment Analysis Graph based on tweets')
        fig, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        #plt.savefig('static/SA.png')
        #plt.close(fig)
        #plt.show()
        st.write(fig)
        if global_polarity>0:
            st.write()
            st.write("##############################################################################")
            st.write("Tweets Polarity: Overall Positive")
            st.write("##############################################################################")
            tw_pol="Overall Positive"
        else:
            st.write()
            st.write("##############################################################################")
            st.write("Tweets Polarity: Overall Negative")
            st.write("##############################################################################")
            tw_pol="Overall Negative"
        return global_polarity,tw_list,tw_pol,pos,neg,neutral

def recommending(df, global_polarity,today_stock,mean):
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea="RISE"
                decision="BUY"
                st.write()
                st.write("##############################################################################")
                st.write("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",user_input,"stock is expected => ",decision)
            elif global_polarity <= 0:
                idea="FALL"
                decision="SELL"
                st.write()
                st.write("##############################################################################")
                st.write("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",user_input,"stock is expected => ",decision)
        else:
            idea="FALL"
            decision="SELL"
            st.write()
            st.write("##############################################################################")
            st.write("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",user_input,"stock is expected => ",decision)
        return idea, decision
st.title('Stock Trend Prediction')
st.markdown("The dashboard will help a researcher to get to know \
more about the given datasets and it's output")
user_input=st.text_input('Enter Valid Stock Symbol','')
try:
     get_historical(user_input)
except:
    st.write("Data Haven't found Please enter correct symbol")
else:
    df = pd.read_csv(''+user_input+'.csv')
    st.write("##############################################################################")
    st.write("Today's",user_input,"Stock Data: ")
    today_stock=df.iloc[-1:]
    st.write(today_stock)
    st.write("##############################################################################")
    df = df.dropna()
    code_list=[]
    for i in range(0,len(df)):
        code_list.append(user_input)
    df2=pd.DataFrame(code_list,columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df=df2
    LSTM_ALGO(df)
    arima_pred, error_arima=ARIMA_ALGO(df)
    df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
    polarity,tw_list,tw_pol,pos,neg,neutral = retrieving_tweets_polarity(user_input)
    idea, decision=recommending(df, polarity,today_stock,mean)
    st.write(tw_list)
    
