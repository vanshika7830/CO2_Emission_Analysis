import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
# --------------------------------------------------Reading from file----------------------------------
data = pd.read_excel('visualizing_global_co2_data.xlsx')


# -------------------------------------------------Summary---------------------------------------------
#Summarize the class-type, range-index or range of Index, column details, data type, memory
print("Information regarding null values\n")
print(data.info()) 
#Retrieve First 5 top values
print("\n\n")
print("First 5 values\n")
print(data.head())  
print("\n\n")
#Retrieve Last 5 bottom values
print("First 5 values\n")                                     
print(data.tail())    
  
#-------------------------------------------------Stastical Data----------------------------------------
print("\n\n")
print("Summary of dataset")
print(data.describe()) 

# ------------------------------------------------- Handling missing values-----------------------------
print(data.isna().sum())
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Dropping the unneccessary column
# data = data.drop(['ISO_Code','temperature_change_from_ghg']) 

# ----------------------------------------------------Again checking the dataset------------------------
print("\n\n")
print(data.info())

# ------------------------------------------Checking correlation to get the insight----------------------

# Calculate correlations with target variable (e.g., CO2)
print("\n\n")
print("Correlation")
correlations = data.corr(numeric_only=True)['CO2'].sort_values(ascending=False)
print(correlations)

# Visualize correlation matrix
plt.figure(figsize=(16, 12))
corr_data = data[['oil_co2', 'coal_co2', 'gas_co2', 'temperature_change_from_co2', 'Cement_CO2', 'primary_energy_consumption']]
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Outliers Detection
print("\n\n")
# Columns to check for outliers
columns = ['GDP', 'Cement_CO2', 'primary_energy_consumption', 
           'temperature_change_from_ghg', 'CO2','coal_co2','gas_co2','oil_co2']

outlier_iqr_summary = {}

for col in columns:
    if col not in data.columns:
        continue
    series = data[col].dropna()  # Remove NaN
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    outlier_iqr_summary[col] = {
        'IQR': round(IQR, 2),
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outlier Count': outliers.count(),
        'Outlier %': round(outliers.count() / len(series) * 100, 2)
    }

# Display result as a DataFrame
iqr_outlier_df = pd.DataFrame(outlier_iqr_summary).T
print("Outliers detected using IQR method:\n")
print(iqr_outlier_df)

# -------------------------------------------OBJECTIVES------------------------------------------

print("\n\n")
# 1. How have global CO2 emissions changed year-over-year?
plt.figure(figsize=(16,12))
data.groupby('Year')['CO2'].sum().plot(marker='o', color='red')
plt.title('Global CO2 Emissions Trend (Yearly)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Million Tonnes)')
plt.grid()
plt.show()

# Conclusion: The time-series plot revealed a steady increase in global CO2 emissions, with noticeable dips during economic recessions 
# (e.g., 2008, 2020). This confirms the urgent need for policy interventions to decouple emissions from economic growth.


# 2. Which countries contribute the most and least to CO2 emissions in 2020?
top_emitters = data[data['Year'] == 2020].nlargest(5, 'CO2')[['Country', 'CO2']]
plt.figure(figsize=(16,12))
sns.barplot(x='CO2', y='Country', data=top_emitters, hue='Country',palette=sns.color_palette("husl", len(top_emitters)))
plt.title('Top 5 CO2 Emitting Countries (2020)')
plt.show()

# Least emmitor
least_emitters = data[data['Year'] == 2020].nsmallest(5, 'CO2')[['Country', 'CO2']]
plt.figure(figsize=(16,12))
sns.barplot(x='CO2', y='Country', data=least_emitters, hue='Country',palette=sns.color_palette("husl", len(least_emitters)))
plt.title('Bottom 5 CO2 Emitting Countries (2020)')
plt.show()
# why 2020? - 2020 is likely the newest year with reliable data in the dataset.
# 2020 had unique emission drops due to lockdowns (↓transport/industry).

# Conclusion: China, the US, and Russia dominated emissions. This highlights the disproportionate impact of a few nations and the need for targeted mitigation strategies
# Cape verde, malta, malawi, rwanda, chad have least emmision


# 3. Energy Source Contribution
# What percentage of emissions come from coal, oil, and gas and its impacts?
plt.figure(figsize=(16,12))
sources = ['coal_co2','oil_co2','gas_co2']
source_contribution = data.groupby('Year')[sources].sum().iloc[-1] # Latest year
source_contribution.plot.pie(autopct='%1.1f%%', colors=['#FFC20A','#00668E','#17BECF'])

plt.title('Global CO2 Emissions by Energy Source (2020)')
plt.ylabel('')
plt.show()
# Conclusion: Coal accounted for 66% of emissions, followed by oil (21.9%) and gas (12.1%). This underscores coal as the most critical target for transitioning to cleaner energy.



# 4. Temperature Change Relationship
#Relationship between CO2 and temperature change?
plt.figure(figsize=(16,12))
sns.scatterplot(x='CO2', y='temperature_change_from_co2', data=data, hue='Year', palette='coolwarm')
plt.title('CO2 Emissions vs Temperature Change')
plt.show()
# Conclusion: The strong positive correlation (r=0.0.89) between CO2 and temperature change empirically validates the link between emissions and global warming.



#5. Despite near-identical correlation scores (oil_co2: 0.955, coal_co2: 0.918), how do their emission patterns differ in the 5 highest-emitting countries?
top_emitters = data[data['Year']==2020].nlargest(5, 'CO2')
# Stacked area plot for fuel composition
plt.figure(figsize=(16,12))
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
plt.figure(figsize=(16,12))
sns.scatterplot(
    data=data[data['Year'] == 1997],
    x='primary_energy_consumption',
    y='co2_per_gdp',
    hue='coal_share',  
    size='GDP',        
    palette='viridis_r',
    sizes=(50, 500),
    alpha=0.7
)
plt.title("Energy Use vs. CO₂ Intensity Colored by Coal Dependency (1997)")
plt.xlabel("Primary Energy Consumption (Mtoe)")
plt.ylabel("CO₂ per GDP (kg/$)")
plt.legend(title='% Coal in Energy Mix', bbox_to_anchor=(1.05, 1))
plt.grid(alpha=0.2)
plt.show()   # Yellow coal independent yet high consumption of energy



# 7. To identify which fuel types are most distinctive/significant for each country's emissions profile, beyond just absolute emission values.
fuels = ['coal_co2', 'oil_co2', 'gas_co2']
df_fuels = data[['Country'] + fuels].dropna()

#Calculate TF (Term Frequency) - Fuel share per country
tf = df_fuels.groupby('Country')[fuels].sum()
tf = tf.div(tf.sum(axis=1), axis=0)  # Convert to percentages

#Calculate IDF (Inverse Document Frequency)
n_countries = len(tf)
idf = np.log(n_countries / (tf > 0).sum(axis=0)) + 1  # +1 to avoid division by zero

#Calculate TF-IDF scores
tfidf_scores = tf * idf.values

# 5. Get top fuel for each country
tfidf_scores['dominant_fuel'] = tfidf_scores.idxmax(axis=1)
print(tfidf_scores.sort_values('dominant_fuel'))

plt.figure(figsize=(12, 16))
sns.scatterplot(
    data=data,
    x='coal_co2',
    y='oil_co2',
    size='gas_co2',  # Bubble size = gas emissions
    hue='Country',   # Color by country
    sizes=(20, 200),
    alpha=0.7
)
plt.title('Coal vs. Oil CO₂ Emissions (Size = Gas Emissions)')
plt.xlabel('Coal CO₂ (Mt)')
plt.ylabel('Oil CO₂ (Mt)')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
