# frontend - streamlit app
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
import altair as alt


def display_results(disp_results: dict):
    """
    Displays the results using altair graphs.

    :param disp_results: results from predictions with keys, values as returned by the api.
    :return: nothing
    """
    date_data = list(disp_results["results"].keys())
    date_data.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'))
    num_receipts = [round(disp_results["results"][date], 3) for date in date_data]
    data_df = pd.DataFrame(
        {
            "date": date_data,
            "number of receipts": num_receipts,
        }
    )
    alt_chart = alt.Chart(data_df).mark_line(point=True).encode(
        x=alt.X("date", sort=None).scale(zero=False),
        y=alt.Y("number of receipts").scale(zero=False))
    st.altair_chart(alt_chart)


window = 15
year = st.number_input('Year:', value=2022, format="%d", min_value=2022)
month = st.number_input('Month:', value=1, format="%d", min_value=1, max_value=12)
text = st.text_input('Past data (comma separated values):',
                     f"enter at least {window} values to predict using this data")

prev_data = []
if len(text.strip()) > 0:
    try:
        prev_data = [int(data_str.strip()) for data_str in text.split(',')]
    except:
        st.error(f"Please enter {window} valid numbers in comma separated format.")

if st.button("Predict", type="primary"):
    # get_predictions_from_service = False
    if len(prev_data) != window and year > 2023:
        st.error("Without entering previous data, you can only predict for 2022 and 2023. \
        Because the data used for developing the model is not extensive to predict meaningfully \
        beyond that point.")
    elif len(prev_data) == 0:
        url = 'http://localhost:5000/predict_yy_mm'
        data = {"year": year, "month": month}
        results = requests.post(url, json=data).json()
        display_results(results)
    elif len(prev_data) == window:
        url = 'http://localhost:5000/predict_with_data'
        data = {
            "prev_data": prev_data,
            "year": year,
            "month": month}
        results = requests.post(url, json=data).json()
        display_results(results)
