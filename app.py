import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('linear_regression_model.pkl')

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("🏡 Prediksi Harga Rumah dengan Linear Regression")

st.markdown("Silakan masukkan informasi rumah:")

col1, col2 = st.columns(2)

with col1:
    ms_subclass = st.number_input("MS SubClass", min_value=0, value=20)
    lot_area = st.number_input("Lot Area", min_value=0, value=8000)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
    year_built = st.number_input("Year Built", min_value=1800, value=2000)
    year_remod = st.number_input("Year Remod/Add", min_value=1800, value=2005)
    tot_rms = st.number_input("Total Rooms Above Ground", min_value=0, value=6)
    fireplaces = st.number_input("Fireplaces", min_value=0, value=1)
    garage_area = st.number_input("Garage Area", min_value=0, value=400)
    yr_sold = st.number_input("Year Sold", min_value=2006, max_value=2010, value=2010)

with col2:
    street = st.selectbox("Street", ["Pave", "Grvl"])
    lot_config = st.selectbox("Lot Config", ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
    house_style = st.selectbox("House Style", ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl'])
    bldg_type = st.selectbox("Building Type", ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'])
    foundation = st.selectbox("Foundation", ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'])
    sale_type = st.selectbox("Sale Type", ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'Oth', 'Con'])
    sale_condition = st.selectbox("Sale Condition", ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'])

if st.button("🔍 Prediksi Harga"):
    input_dict = {
        'MS SubClass': [ms_subclass],
        'Lot Area': [lot_area],
        'Overall Qual': [overall_qual],
        'Overall Cond': [overall_cond],
        'Year Built': [year_built],
        'Year Remod/Add': [year_remod],
        'TotRms AbvGrd': [tot_rms],
        'Fireplaces': [fireplaces],
        'Garage Area': [garage_area],
        'Yr Sold': [yr_sold],
        'Street_' + street: [1],
        'Lot Config_' + lot_config: [1],
        'House Style_' + house_style: [1],
        'Bldg Type_' + bldg_type: [1],
        'Foundation_' + foundation: [1],
        'Sale Type_' + sale_type: [1],
        'Sale Condition_' + sale_condition: [1]
    }

    dummy_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.coef_.shape[0]
    all_columns = model.feature_names_in_
    input_data = pd.DataFrame(columns=all_columns)
    input_data.loc[0] = 0  # set semua ke 0

    for key, val in input_dict.items():
        if key in input_data.columns:
            input_data.at[0, key] = val[0]

    # Prediksi
    pred = model.predict(input_data)[0]
    st.success(f"💰 Estimasi Harga Rumah: **${pred:,.2f}**")
