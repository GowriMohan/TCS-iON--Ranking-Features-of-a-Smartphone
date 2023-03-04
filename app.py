# Smart Phone classifier final
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image

# Creating Front end of application
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url('https://www.techprevue.com/wp-content/uploads/2013/11/smartphone-technology.jpg');
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


add_bg_from_url()

st.set_option('deprecation.showPyplotGlobalUse', False)


# Loading up the Classification model we created
model = pickle.load(open('ranking.pkl', 'rb'))

# Loading scaler
scaler = pickle.load(open("scaling_features.pkl", "rb"))


# Caching the model for faster loading
#@st.cache

nav = st.sidebar.radio("Menu", ["Home", "Prediction"])

if nav == "Home":
    st.title("Smart Phone Classification")
    st.subheader("Welcome to Smart Phone Price Classification.  ")
    st.write("* For Classification of Price According To Feature values:        Select Prediction page.")

    st.write('* For Ranking The Smartphone Features Based On Price:             Select Ranking Features page.')

if nav == "Prediction":

    st.title("Do you want to know the expected price of the Smartphone?")
    st.markdown("For predicting the price range category of the smartphone please fill the details in the left pane.")

    st.sidebar.subheader('Enter the details:')

    battery_power = st.sidebar.number_input("What is the desired battery power?")

    blue = st.sidebar.radio("Does the phone have bluetooth or not?", ["Yes", "No"])
    if blue == "Yes":
        blue = 1
    else:
        blue = 0

    clock_speed = st.sidebar.slider('How much is the expected clock_speed?', min_value=0.5, max_value=3.0, value=0.0, step=0.1)

    dual_sim = st.sidebar.radio("Has dual sim support or not?", ["Yes", "No"])
    if dual_sim == "Yes":
        dual_sim = 1
    else:
        dual_sim = 0

    fc = st.sidebar.slider('How much is the expected Front Camera mega pixels?', min_value=0, max_value=20, value=0, step=1)

    four_g = st.sidebar.radio("Does the phone have 4G or not?", ["Yes", "No"])
    if four_g == "Yes":
        four_g = 1
    else:
        four_g = 0

    int_memory = st.sidebar.number_input("What is the desired Internal Memory (in GB)?")

    m_dep = st.sidebar.slider('How much is the expected Mobile Depth (in cm)?', min_value=0.0, max_value=1.0, value=0.0,
                      step=0.1)

    mobile_wt = st.sidebar.number_input("What is the desired Weight of mobile phone?")

    n_cores = st.sidebar.slider('What is the number of cores of processor?', min_value=1, max_value=8, value=1, step=1)

    pc = st.sidebar.slider('How much is the expected Primary Camera mega pixels?', min_value=0, max_value=20, value=0, step=1)

    px_height = st.sidebar.number_input("How much is the desired Pixel Resolution Height?")

    px_width = st.sidebar.number_input("How much is the desired Pixel Resolution Width?")

    ram = st.sidebar.number_input("How much is the desired RAM (in MB)?")

    sc_h = st.sidebar.slider('How much is the desired Screen Height of mobile (in cm)?', min_value=0, max_value=20, value=0,
                     step=1)

    sc_w = st.sidebar.slider('How much is the desired Screen Width of mobile (in cm)?', min_value=0, max_value=20, value=0,
                     step=1)

    talk_time = st.sidebar.slider('What is the talk - time (longest time that a single battery charge will last)?', min_value=0,
                          max_value=20, value=0, step=1)

    three_g = st.sidebar.radio("Does the phone have 3G or not??", ["Yes", "No"])
    if three_g == "Yes":
        three_g = 1
    else:
        three_g = 0

    touch_screen = st.sidebar.radio("Has touch screen or not?", ["Yes", "No"])
    if touch_screen == "Yes":
        touch_screen = 1
    else:
        touch_screen = 0

    wifi = st.sidebar.radio("Does the phone have wifi or not???", ["Yes", "No"])
    if wifi == "Yes":
        wifi = 1
    else:
        wifi = 0


    d = {'battery_power':[battery_power], 'blue':[blue], 'clock_speed':[clock_speed], 'dual_sim':[dual_sim], 'fc':[fc], 'four_g':[four_g],
       'int_memory':[int_memory], 'm_dep':[m_dep], 'mobile_wt':[mobile_wt], 'n_cores':[n_cores], 'pc':[pc], 'px_height':[px_height],
       'px_width':[px_width], 'ram':[ram], 'sc_h':[sc_h], 'sc_w':[sc_w], 'talk_time':[talk_time], 'three_g':[three_g],
       'touch_screen':[touch_screen], 'wifi':[wifi]}
    df = pd.DataFrame(data=d)

    print(df)

    input_data = scaler.transform(df)
    print(input_data)


    pred_x = pd.DataFrame(input_data , columns=['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi'])
    print(pred_x)

    result = model.predict(pred_x)
    print(result)

    # when 'Predict' is clicked, make the prediction and store it
    st.caption('For knowing the prince range category of your smartphone:')
    if st.button('Click Here'):
        if result == 0:
            st.success('Your expected price range for this Smartphone is: Low Cost')
        elif result == 1:
            st.success('Your expected price range for this Smartphone is: Medium Cost')
        elif result == 2:
            st.warning('Your expected price range for this Smartphone is: High Cost')
        else:
            st.error('Your expected price range for this Smartphone is: Very High Cost')

    # Calculate feature importances
    importances = pd.DataFrame({'feature': pred_x.columns, 'importance': np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False)

    st.caption('Do you want to rank the features???')
    if st.button('Rank the features'):
        st.write('Ranking features of smartphone on basis of its importance.')
        st.dataframe(importances)

    st.caption('Do you want to plot a graph, ranking the features???')
    # Create a bar chart of feature importances using Matplotlib
    if st.button('Click here to plot important features'):

        max_value = 100
        progress_bar = st.progress(0)
        for i in range(max_value):
            # Update the progress bar
            progress_bar.progress(i + 1)

            # Sleep for a short period of time
            time.sleep(0.01)

        st.write('Bar chart of feature importances.')
        fig, ax = plt.subplots(figsize=(28, 20))
        ax.bar(importances['feature'], importances['importance'], color='b')
        ax.set_xticklabels(importances['feature'], rotation=90, fontsize=18)
        ax.set_xlabel('Features', fontsize=22)
        ax.set_ylabel('Importance', fontsize=22)
        ax.set_title('Feature Importances', fontsize=30)
        plt.show()

        # Display the chart using Streamlit
        st.pyplot(fig)
