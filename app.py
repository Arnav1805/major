# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from feature_engine.outliers import ArbitraryOutlierCapper
import streamlit as st

# Read dataset
df = pd.read_csv('insurance_major.csv')

# Data info and description
st.header("Data Overview")
st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Dataset Info")
buffer = []
df.info(buf=buffer)
st.text('\n'.join(buffer))

st.write("### Descriptive Statistics")
st.dataframe(df.describe())

st.write("### Null Values")
st.write(df.isnull().sum())

# Exploratory Visualization
features = ['age', 'bmi']
st.write("### Scatter Plots")
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for i, col in enumerate(features):
    sns.scatterplot(data=df, x=col, y='charges', hue='smoker', ax=axes[i])
st.pyplot(fig)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Boxplots Before Capping
st.write("### Boxplots Before Outlier Treatment")
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(df['age'], ax=ax2[0])
sns.boxplot(df['bmi'], ax=ax2[1])
st.pyplot(fig2)

# Capping Outliers in BMI
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
iqr = Q3 - Q1
lowlim = Q1 - 1.5 * iqr
upplim = Q3 + 1.5 * iqr

st.write(f"Outlier Cap: Lower = {lowlim}, Upper = {upplim}")

arb = ArbitraryOutlierCapper(min_capping_dict={'bmi': 13.6749}, max_capping_dict={'bmi': 47.315})
df[['bmi']] = arb.fit_transform(df[['bmi']])

# Boxplot After
st.write("### Boxplot After Outlier Treatment")
fig3 = plt.figure()
sns.boxplot(df['bmi'])
st.pyplot(fig3)

# Skewness
st.write(f"Skewness of BMI: {df['bmi'].skew()}")
st.write(f"Skewness of Age: {df['age'].skew()}")

# Encoding
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# Correlation Matrix
st.write("### Correlation Matrix")
st.dataframe(df.corr())

# Splitting
X = df.drop(['charges'], axis=1)
Y = df[['charges']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=62)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Prediction and Evaluation
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(Y_test, y_pred_rf)
r2_rf = r2_score(Y_test, y_pred_rf)

st.write(f"### Model Evaluation\n**Random Forest - MSE:** {mse_rf:.2f}, **R2 Score:** {r2_rf:.2f}")

# Save Model
joblib.dump(rf_model, 'major.pkl')

# ------------------------------
# Streamlit Prediction Interface
# ------------------------------
st.header("Healthcare Insurance Cost Prediction (Live Model)")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", ['male', 'female'])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

# Encoding for input
sex = 1 if sex == 'female' else 0
smoker = 1 if smoker == 'yes' else 0
region_map = {'northeast': 1, 'northwest': 0, 'southeast': 2, 'southwest': 3}
region = region_map[region]

input_data = np.array([[age, sex, bmi, children, smoker, region]])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict Insurance Cost"):
    model = joblib.load('major.pkl')
    prediction = model.predict(input_data_scaled)
    st.success(f"Estimated Insurance Cost: ${prediction[0]:.2f}")
