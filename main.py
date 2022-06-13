import streamlit as sl
from datetime import date

import pandas as pd
import altair as alt

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START_DATE = "2017-05-15"
TODAY = date.today().strftime("%Y-%m-%d")

sl.title("FIG Stock Price Prediction in ML")

available_stocks = ("CSCO", "INTC", "CRM", "NVDA", "UBER", "AAPL", "GOOGL", "FB", "MSFT", "AMZN")
selected_stock = sl.selectbox("Select dataset for prediction", available_stocks)

assistance_message = sl.text("Please select the number of years to predict using the slider below.")
n_years = sl.slider("Years of prediction:", 1, 5)
period = n_years * 365


def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data


data = load_data(selected_stock)

sl.subheader('Historical Stock Data')
sl.write(data.tail())


def data_plot():
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening Price'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing Price'))
    fig1.layout.update(title_text="Data Over Time", xaxis_rangeslider_visible=True)
    sl.plotly_chart(fig1)


data_plot()

train_data = data[['Date', 'Close']]
train_data = train_data.rename(columns={"Date": "ds", "Close": "y"})

ml_algo = Prophet()
ml_algo.fit(train_data)
future_prediction = ml_algo.make_future_dataframe(periods=period)
data_prediction = ml_algo.predict(future_prediction)

sl.subheader('Predicted Stock Data')
sl.write(data_prediction.tail())

sl.write('Stock Predictions')
fig2 = plot_plotly(ml_algo, data_prediction)
sl.plotly_chart(fig2)

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