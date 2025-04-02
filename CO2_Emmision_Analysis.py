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


# 2. Which countries contribute the least and most to CO2 emissions in 2020?


top_emitters = data[data['Year'] == 2020].nlargest(5, 'CO2')[['Country', 'CO2']]

plt.figure(figsize=(10, 6))
sns.barplot(x='CO2', y='Country', data=top_emitters, palette=sns.color_palette("husl", len(top_emitters)))
plt.title('Top 10 CO2 Emitting Countries (2020)')
plt.show()

# Least emmitor

least_emitters = data[data['Year'] == 2020].nsmallest(5, 'CO2')[['Country', 'CO2']]

plt.figure(figsize=(10, 6))
sns.barplot(x='CO2', y='Country', data=least_emitters, palette=sns.color_palette("husl", len(least_emitters)))
plt.title('Top 10 CO2 Emitting Countries (2020)')
plt.show()


# why 2020? - 2020 is likely the newest year with reliable data in the dataset.
# 2020 had unique emission drops due to lockdowns (↓transport/industry).

# Conclusion: China, the US, and Russia dominated emissions. This highlights the disproportionate impact of a few nations and the need for targeted mitigation strategies
# Cape verde, malta, malawi, rwanda, chad have least emmision


# 3. Energy Source Contribution
# What percentage of emissions come from coal, oil, and gas?
sources = ['coal_co2','oil_co2','gas_co2']
source_contribution = data.groupby('Year')[sources].sum().iloc[-1] # Latest year
source_contribution.plot.pie(autopct='%1.1f%%', colors=['#FFC20A','#00668E','#17BECF'])

plt.title('Global CO2 Emissions by Energy Source (2020)')
plt.ylabel('')
plt.show()

# Conclusion: Coal accounted for 66% of emissions, followed by oil (21.9%) and gas (12.1%). This underscores coal as the most critical target for transitioning to cleaner energy.



# 4. Temperature Change Relationship
# Is there a visible relationship between CO2 and temperature change?

plt.figure(figsize=(9,6))
sns.scatterplot(x='CO2', y='temperature_change_from_co2', data=data, hue='Year', palette='coolwarm')
plt.title('CO2 Emissions vs Temperature Change')
plt.show()

# Conclusion: The strong positive correlation (r=0.0.89) between CO2 and temperature change empirically validates the link between emissions and global warming.



#5. Despite near-identical correlation scores (oil_co2: 0.955, coal_co2: 0.918), how do their emission patterns differ in the 10 highest-emitting countries?

top_emitters = data[data['Year']==2020].nlargest(5, 'CO2')
# Stacked area plot for fuel composition
plt.figure(figsize=(12,6))
plt.stackplot(
    top_emitters['Country'], 
    top_emitters['coal_co2'], 
    top_emitters['oil_co2'], 
    top_emitters['gas_co2'],
    labels=['Coal','Oil','Gas'],
    colors=['#333333','#8B0000','#4682B4']
)
plt.title('Fossil Fuel Composition in Top 5 Emitters (2020)')
plt.ylabel('CO₂ Emissions (Mt)')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()

# Conclusion - Oil's slightly higher correlation may mask critical regional differences (e.g., coal-heavy China vs. oil-dependent Saudi Arabia). This reveals where decarbonization efforts should prioritize fuel switching.


# 6. The Energy Efficiency Paradox
# Why does primary_energy_consumption (0.026 correlation) show almost no link to CO₂, despite energy being the main emission source?

# Calculate coal's share of total CO₂ emissions
data['coal_share'] = (data['coal_co2'] / data['CO2']) * 100
print(data[['Country', 'Year', 'coal_co2', 'primary_energy_consumption', 'coal_share']].head())

# Summary statistics for verification
print(data['coal_share'].describe())

# Example: Convert coal CO₂ to energy units (assuming 1 MtCO₂ = 0.4 Mtoe)
data['coal_energy'] = data['coal_co2'] * 0.4
data['coal_share'] = (data['coal_energy'] / data['primary_energy_consumption']) * 100

data.dropna(subset=['coal_co2', 'primary_energy_consumption'])

