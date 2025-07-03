
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from train_model import predict_category

st.title("💸 AI Expense Categorizer")

file = st.file_uploader("Upload Bank Statement (CSV)", type="csv")
if file:
    df = pd.read_csv(file)
    df['Category'] = df['Description'].apply(predict_category)
    st.write(df)

    chart = df['Category'].value_counts()
    st.subheader("📊 Expense Breakdown")
    st.bar_chart(chart)
