import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------Reading from file----------------------------------
data = pd.read_excel('visualizing_global_co2_data.xlsx')


# -------------------------------------------------Summary---------------------------------------------
#Retrieve First 5 top values
print(data.head())  
#Retrieve Last 5 bottom values                                     
print(data.tail())    
#Summarize the class-type, range-index or range of Index, column details, data type, memory
print(data.info())   


#-------------------------------------------------Stastical Data----------------------------------------
print(data.describe()) 

# ------------------------------------------------- Handling missing values-----------------------------
print(data.isna().sum())
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# ----------------------------------------------------Again checking the dataset------------------------
print(data.info())