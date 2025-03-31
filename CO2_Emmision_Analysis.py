import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------Reading from file----------------------------------
data = pd.read_excel('visualizing_global_co2_data.xlsx')


# -------------------------------------------------Summary---------------------------------------------
#Summarize the class-type, range-index or range of Index, column details, data type, memory
print(data.info()) 
#Retrieve First 5 top values
print(data.head())  
#Retrieve Last 5 bottom values                                     
print(data.tail())    
  
#-------------------------------------------------Stastical Data----------------------------------------
print(data.describe()) 

# ------------------------------------------------- Handling missing values-----------------------------
print(data.isna().sum())
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# ----------------------------------------------------Again checking the dataset------------------------
print(data.info())

# ------------------------------------------Checking correlation to get the insight----------------------

# Calculate correlations with target variable (e.g., CO2)
correlations = data.corr(numeric_only=True)['CO2'].sort_values(ascending=False)
print(correlations)

# Visualize correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()