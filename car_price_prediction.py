# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%

# Load the dataset
df = pd.read_csv('./data.csv/data.csv')
print(df.dtypes)


# %%
df.head()

# %% [markdown]
# ## Data Cleaning

# %%
# Data cleaning
# Drop rows with missing values
df.dropna(inplace=True)
df.head()


# %%
# Drop irrelevant columns
df = df.drop(['Engine Cylinders', 'Market Category', 'Popularity', 'highway MPG', 'city mpg'], axis=1)
df.head()

# %%
# Rename columns
df = df.rename(columns={'Make': 'Brand', 'Engine HP': 'HP', 'Transmission Type': 'Transmission', 'Driven_Wheels': 'Drive Mode',
                        'MSRP': 'Price'})
df.head()

# %%
# Drop duplicates
df.drop_duplicates(keep='first', inplace=True)
df.head()

# %% [markdown]
# ## EDA

# %%
# EDA
st.title('Car Prices EDA')

st.sidebar.title('Visualization Settings')

# %%
# Create a sidebar for choosing visualization options
fig_type = st.sidebar.selectbox('Select a type of chart', ('Histogram', 'Boxplot', 'Scatterplot', 'Heatmap'))

# %%
if fig_type == 'Histogram':
    # Create a histogram of prices
    st.header('Price distribution')
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Price', kde=True, bins=30)
    st.pyplot(fig)

# %%
if fig_type == 'Boxplot':
    # Create a boxplot of prices by brand
    st.header('Price by brand')
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Brand', y='Price')
    plt.xticks(rotation=90)
    st.pyplot(fig)

# %%
if fig_type == 'Scatterplot':
    # Create a scatterplot of prices and horsepower
    st.header('Price by horsepower')
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='HP', y='Price', hue='Drive Mode')
    st.pyplot(fig)

# %%
# Heatmap of correlations
if fig_type == 'Heatmap':
    corr = df.corr()
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(fig)


# %%
# Show the data table
st.header('Data Table')
st.dataframe(df)

# %% [markdown]
# ## Building our Model

# %%
# Encode the categorical variables using OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Encode the target variable
le = LabelEncoder()
df['Price'] = le.fit_transform(df['Price'])

# Split the dataset into training and testing sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer to encode categorical variables
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), ['Brand', 'Model', 'Engine Fuel Type', 'Transmission', 'Vehicle Size', 'Drive Mode', 'Vehicle Style'])],
    remainder='passthrough'
)

# Fit the column transformer on the training data and transform both the training and testing data
X_train_encoded = ct.fit_transform(X_train)
X_test_encoded = ct.transform(X_test)

# Scale the data using StandardScaler
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)



# %%
from sklearn.linear_model import LinearRegression

# Train a linear regression model on the training data
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Model Evaluation

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the testing data and evaluate the model
y_pred = regressor.predict(X_test_scaled)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R^2 Score:', r2_score(y_test, y_pred))

# Write in streamlit

st.sidebar.title('Linear Regression Model Evaluation')

st.sidebar.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
st.sidebar.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
st.sidebar.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
st.sidebar.write('R^2 Score:', r2_score(y_test, y_pred))

