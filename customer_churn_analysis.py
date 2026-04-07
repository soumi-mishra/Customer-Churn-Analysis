# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv("C:\\Users\\hp\\Downloads\\Bank+Customer+Churn\\Bank_Churn.csv")
print("Dataset loaded successfully!\n")

# Show basic information
print(df.info())
print(df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------
# Improvement 1: Data Preprocessing
# ---------------------------
# Encode categorical variables for modeling later
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# ---------------------------
# Objective 1: Customer Churn Overview
# ---------------------------
plt.figure(figsize=(8,6))
sns.countplot(x='Exited', data=df, palette='Set2')
plt.title('Customer Churn Distribution')
plt.xlabel('Exited (0=Stayed, 1=Churned)')
plt.ylabel('Count')

# Add percentage labels
total = len(df)
for p in plt.gca().patches:
    height = p.get_height()
    percentage = f'{100 * height/total:.1f}%'
    plt.gca().annotate(percentage, 
                       (p.get_x() + p.get_width()/2., height), 
                       ha='center', va='bottom', fontsize=12, color='black')
plt.show()

# Pie chart
churn_counts = df['Exited'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=['Stayed', 'Churned'], autopct='%1.1f%%',
        colors=['orange','#66b3ff'])
plt.title('Churn Percentage')
plt.show()

# ---------------------------
# Objective 2: Demographic Impact
# ---------------------------
plt.figure(figsize=(8,6))
sns.countplot(x='Gender', hue='Exited', data=df, palette='pastel')
plt.title('Churn by Gender')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Geography_France', hue='Exited', data=df, palette='muted')
plt.title('Churn by Geography (France vs Others)')
plt.show()

# Age buckets for clearer trends
df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=['18-30','31-45','46-60','60+'])
plt.figure(figsize=(8,6))
sns.countplot(x='AgeGroup', hue='Exited', data=df, palette='coolwarm')
plt.title('Churn by Age Group')
plt.show()

# ---------------------------
# Objective 3: Customer Behavior
# ---------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='Tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='NumOfProducts', hue='Exited', data=df, palette='deep')
plt.title('Products vs Churn')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='IsActiveMember', hue='Exited', data=df, palette='coolwarm')
plt.title('Active Member vs Churn')
plt.show()

# ---------------------------
# Objective 4: Financial Factors
# ---------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score vs Churn')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Balance vs Churn')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
plt.title('Salary vs Churn')
plt.show()

# ---------------------------
# Correlation Analysis
# ---------------------------
corr = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# Pearson correlation examples
corr_val, _ = pearsonr(df['CreditScore'], df['Exited'])
print(f"Correlation between Credit Score and Churn: {corr_val:.2f}")

corr_val_age, _ = pearsonr(df['Age'], df['Exited'])
print(f"Correlation between Age and Churn: {corr_val_age:.2f}")

corr_val_tenure, _ = pearsonr(df['Tenure'], df['Exited'])
print(f"Correlation between Tenure and Churn: {corr_val_tenure:.2f}")