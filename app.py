import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("C:/Users/mishr/Desktop/CODE/Data Analysis/Customer-Churn-Analysis/Bank_Churn.csv")

# Preprocessing
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=['18-30','31-45','46-60','60+'])

# Map churn labels
df['ChurnLabel'] = df['Exited'].map({0: "Not Churned", 1: "Churned"})

# Sidebar filters
st.sidebar.header("Filters")
gender_filter = st.sidebar.selectbox("Select Gender", options=[None, "Male", "Female"])
age_filter = st.sidebar.multiselect("Select Age Group", options=df['AgeGroup'].unique())
active_filter = st.sidebar.selectbox("Active Member", options=[None, 0, 1])

# Apply filters
filtered_df = df.copy()
if gender_filter:
    filtered_df = filtered_df[filtered_df['Gender'] == (0 if gender_filter=="Male" else 1)]
if age_filter:
    filtered_df = filtered_df[filtered_df['AgeGroup'].isin(age_filter)]
if active_filter is not None:
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == active_filter]

st.title("📊 Bank Customer Churn Dashboard")

# Churn distribution
fig_churn = px.histogram(
    filtered_df,
    x="ChurnLabel",
    color="ChurnLabel",
    title="Customer Churn Distribution",
    labels={"ChurnLabel": "Churn Status"}
)
# Add count labels on top of bars
fig_churn.update_traces(texttemplate='%{y}', textposition='outside')
st.plotly_chart(fig_churn)

# Pie chart
fig_pie = px.pie(
    filtered_df,
    names="ChurnLabel",
    title="Churn Percentage",
    color="ChurnLabel",
    color_discrete_map={"Not Churned":"orange","Churned":"#66b3ff"}
)
fig_pie.update_traces(textinfo="label+percent", textfont_size=14)
st.plotly_chart(fig_pie)
# Churn by Age Group
fig_age = px.histogram(filtered_df, x="AgeGroup", color="ChurnLabel", barmode="group",
                       title="Churn by Age Group")
st.plotly_chart(fig_age)

# Tenure vs Churn
fig_tenure = px.box(filtered_df, x="ChurnLabel", y="Tenure", title="Tenure vs Churn")
st.plotly_chart(fig_tenure)

# Products vs Churn
fig_products = px.histogram(filtered_df, x="NumOfProducts", color="ChurnLabel", barmode="group",
                            title="Products vs Churn")
st.plotly_chart(fig_products)

# Financial factors
fig_balance = px.box(filtered_df, x="ChurnLabel", y="Balance", title="Balance vs Churn")
st.plotly_chart(fig_balance)

fig_salary = px.box(filtered_df, x="ChurnLabel", y="EstimatedSalary", title="Salary vs Churn")
st.plotly_chart(fig_salary)

# Correlation heatmap
corr = filtered_df[['CreditScore','Age','Tenure','Balance','EstimatedSalary','Exited']].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="YlGnBu", title="Correlation Heatmap")
st.plotly_chart(fig_corr)