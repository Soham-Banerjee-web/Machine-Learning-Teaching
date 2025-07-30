import yfinance as yf
import streamlit as st

st.write(
         "# Welcome to the Stock Price App")

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start='2010-01-01', end='2025-01-01')

# Check if data was retrieved successfully
if not tickerDf.empty:
    st.line_chart(tickerDf.Close)
    st.line_chart(tickerDf.Volume)
else:
    st.error(f"No data found for {tickerSymbol}. Please check the ticker symbol.")  