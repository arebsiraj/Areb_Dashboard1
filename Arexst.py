import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import mplcursors
import altair as alt




# HEADINGS
st.subheader('Well come to Areb_sat Diabetes prediction dashboard!')

# Load and display the image
image = Image.open("C:/Users/Areb_Sat/Documents/diabetes_dashboard/mine.jpg")
st.image(image, caption='mine', use_column_width=True)

# Add CSS styling to position the image to the most right corner
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the diabetes dataset

df=pd.read_csv("c:/Users/Areb_Sat/Documents/diabetes_dashboard/diabetes.csv")




st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())




# Data Visualization

st.title('Data Visualization')
st.subheader('Correlation Heatmap')
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask=mask)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()



# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# Function to collect user report
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3, key="pregnancies_slider" )
  glucose = st.sidebar.slider('Glucose', 0,200, 120, key="glucose_slider" )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70, key="bp_slider" )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20, key="skinthickness_slider" )
  insulin = st.sidebar.slider('Insulin', 0,846, 79, key="insulin_slider" )
  bmi = st.sidebar.slider('BMI', 0,67, 20, key="bmi_slider")
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47, key="dpf_slider")
  age = st.sidebar.slider('Age', 21,88, 33, key="age_slider")

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# Collect patient data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)



# Train the model
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)

# Convert feature names to match the training feature names
user_data_updated = {
    'Pregnancies': user_data['pregnancies'],
    'Glucose': user_data['glucose'],
    'BloodPressure': user_data['bp'],
    'SkinThickness': user_data['skinthickness'],
    'Insulin': user_data['insulin'],
    'BMI': user_data['bmi'],
    'DiabetesPedigreeFunction': user_data['dpf'],
    'Age': user_data['age']
     
    
}

# Perform the prediction
user_result = rf.predict(pd.DataFrame(user_data_updated, index=[0]))




# VISUALISATIONS

st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

import mplcursors


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'twilight')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)





# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)




# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)




# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)




# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)



# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT

st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = '<span style="color:green; font-size:28px;">You are not Diabetic</span>'
else:
    output = '<span style="color:red; font-size:28px;">You are Diabetic</span>'
st.markdown(output, unsafe_allow_html=True)

# Alerts and Notifications
# Image for Diabetes Positive
if user_result[0] == 1:
    st.warning('Attention: You are predicted to have diabetes!')
    diabetes_image = Image.open("C:/Users/Areb_Sat/Documents/diabetes_dashboard/diabates.jpg")
    st.image(diabetes_image, caption="Diabetes Positive", use_column_width=True)

st.subheader('Performance Metrics')
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')







# Placeholder values for user_data and color
# Age vs Pregnancies
# Set up the scatter plot using Altair with a different palette
scatter_plot = alt.Chart(df).mark_circle(size=100).encode(
    x=alt.X('Age', title='Age'),
    y=alt.Y('Pregnancies', title='Pregnancy Count'),
    color=alt.Color('Outcome:N', scale=alt.Scale(scheme='category10')),
    tooltip=['Age', 'Pregnancies', 'Outcome']
) 

# Add the user data point
user_point = alt.Chart(user_data).mark_circle(size=200).encode(
    x=alt.X('age', title='Age'),
    y=alt.Y('pregnancies', title='Pregnancy Count'),
    color=alt.value(color)
)

# Combine the scatter plot and the user data point
chart = (scatter_plot + user_point).properties(
    width=600,
    height=500,
    title='Pregnancy count Graph (Others vs Yours)'
)

# Display the scatter plot
st.altair_chart(chart, use_container_width=True)









