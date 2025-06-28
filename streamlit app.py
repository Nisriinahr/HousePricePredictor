import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# Judul
st.title('🏠 Prediksi Harga Rumah (Model Linear Regression)')

# Form input user
col1, col2 = st.columns(2)

with col1:
    MS_SubClass = st.number_input('MS SubClass', value=20)
    Lot_Area = st.number_input('Lot Area', value=8000)
    Overall_Qual = st.number_input('Overall Quality (1-10)', value=5)
    Overall_Cond = st.number_input('Overall Condition (1-10)', value=5)
    Year_Built = st.number_input('Year Built', value=2000)
    TotRms_AbvGrd = st.number_input('Total Rooms Above Ground', value=6)
    Fireplaces = st.number_input('Fireplaces', value=1)
    Garage_Area = st.number_input('Garage Area', value=400)

with col2:
    Yr_Sold = st.number_input('Year Sold', value=2010)
    Year_Remod_Add = st.number_input('Year Remod/Add', value=2005)
    Street_Pave = st.selectbox('Street Type', ['Pave', 'Grvl']) == 'Pave'
    Lot_Config_Corner = st.selectbox('Lot Config (Corner?)', ['Yes', 'No']) == 'Yes'
    House_Style_1Story = st.selectbox('House Style (1 Story?)', ['Yes', 'No']) == 'Yes'
    Bldg_Type_1Fam = st.selectbox('Building Type (1 Family?)', ['Yes', 'No']) == 'Yes'
    Foundation_PConc = st.selectbox('Foundation Type (PConc?)', ['Yes', 'No']) == 'Yes'
    Sale_Type_WD = st.selectbox('Sale Type (WD?)', ['Yes', 'No']) == 'Yes'
    Sale_Condition_Normal = st.selectbox('Sale Condition (Normal?)', ['Yes', 'No']) == 'Yes'

# Tombol prediksi
if st.button('🔍 Prediksi Harga'):
    # Format sesuai urutan fitur setelah one-hot encoding
    input_data = np.array([[
        MS_SubClass, Lot_Area, Overall_Qual, Overall_Cond, Year_Built,
        Year_Remod_Add, TotRms_AbvGrd, Fireplaces, Garage_Area, Yr_Sold,
        int(Street_Pave),
        int(Lot_Config_Corner),
        int(House_Style_1Story),
        int(Bldg_Type_1Fam),
        int(Foundation_PConc),
        int(Sale_Type_WD),
        int(Sale_Condition_Normal)
    ]])

    # Prediksi harga
    prediction = model.predict(input_data)[0]
    st.success(f'💰 Estimasi Harga Rumah: ${prediction:,.2f}')
