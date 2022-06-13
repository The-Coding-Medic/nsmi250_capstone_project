# import and alias streamlit and date for stock range
import streamlit as sl
from datetime import date

# import and alias pandas and altair for data representation
import pandas as pd
import altair as alt

# import and alias yahoo finance to gather stock data
import yfinance as yf

# import fbprophet and plotting tools to train and graph ML data
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# date range for data to be trained
START_DATE = "2017-05-15"
TODAY = date.today().strftime("%Y-%m-%d")

# title of application
sl.title("FIG Stock Price Prediction in ML")

# top 10 stocks by trade name imported from yfinance
available_stocks = ("CSCO", "INTC", "CRM", "NVDA", "UBER", "AAPL", "GOOGL", "FB", "MSFT", "AMZN")

# selection for stock to be predicted
selected_stock = sl.selectbox("Select dataset for prediction", available_stocks)

# creates a slider to allow years of data to be trained and predicted
assistance_message = sl.text("Please select the number of years to predict using the slider below.")
n_years = sl.slider("Years of prediction:", 1, 5)
period = n_years * 365

# loads data from yfinance given date ranges
def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data

# loads data from chosen stock from select box
data = load_data(selected_stock)

# presents a 'tail' snapshot of data from yfinance related to stock
sl.subheader('Historical Stock Data')
sl.write(data.tail())

# plots data parameters for stock from yfinance and isolates open and closing prices
def data_plot():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening Price'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing Price'))
    fig1.layout.update(title_text="Data Over Time", xaxis_rangeslider_visible=True)
    sl.plotly_chart(fig1)

# calls data plot method
data_plot()

# assigns data range and parameter for training
train_data = data[['Date', 'Close']]
train_data = train_data.rename(columns={"Date": "ds", "Close": "y"})

# assigns data to be trained by Prophet for prediction
ml_algo = Prophet()
ml_algo.fit(train_data)
future_prediction = ml_algo.make_future_dataframe(periods=period)
data_prediction = ml_algo.predict(future_prediction)

# titles and presents a snapshot of predicted stock data from ML training
sl.subheader('Predicted Stock Data')
sl.write(data_prediction.tail())

# plots predictions in an interactive plotly chart
sl.write('Stock Predictions')
fig2 = plot_plotly(ml_algo, data_prediction)
sl.plotly_chart(fig2)

# lists top 3 stock returns over 5 years
# to be manually updated monthly by FIG members for consistency 
"Top 3 Stock Returns at 5 Years"
source = pd.DataFrame({
    'Avg Return (%)': [341, 232, 147],
    'Stock': ['NVDA', 'AAPL', 'MSFT']
})

bar_chart = alt.Chart(source).mark_bar().encode(
    y='Avg Return (%):Q',
    x='Stock:O',
)

sl.altair_chart(bar_chart, use_container_width=True)