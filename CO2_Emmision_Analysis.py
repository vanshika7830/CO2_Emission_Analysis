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

# -------------------------------------------OBJECTIVES------------------------------------------


# 1. How have global CO2 emissions changed year-over-year?
plt.figure(figsize=(12,6))
data.groupby('Year')['CO2'].sum().plot(marker='o', color='red')
plt.title('Global CO2 Emissions Trend (Yearly)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.grid()
plt.show()

# Conclusion: The time-series plot revealed a steady increase in global CO2 emissions, with noticeable dips during economic recessions 
# (e.g., 2008, 2020). This confirms the urgent need for policy interventions to decouple emissions from economic growth.



# 2. Which countries contribute the most to CO2 emissions in 2020?
top_emitters = data[data['Year']==2020].nlargest(10, 'CO2')[['Country','CO2']]
plt.figure(figsize=(10,6))
sns.barplot(x='CO2', y='Country', data=top_emitters)
plt.title('Top 10 CO2 Emitting Countries (2020)')
plt.show()

# why 2020? - 2020 is likely the newest year with reliable data in the dataset.
# 2020 had unique emission drops due to lockdowns (↓transport/industry).

# Conclusion: China, the US, and Russia dominated emissions. This highlights the disproportionate impact of a few nations and the need for targeted mitigation strategies