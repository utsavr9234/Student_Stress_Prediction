import streamlit as st
import pandas as pd

df=pd.read_csv("Student LifeStyle.csv")

#df.shape
#df.info()
#df.head()
#df.tail()
#df.min()
#df.max()
#df.mean()
#df.median()
#df.std()
#df.count()
#df.describe()

df["Student_Type"]=df["Student_Type"].map({"school":1,"college":2,"working_student":3})

print(df.isnull().sum())

student_type_mode = df["Student_Type"].mode()[0]
df["Student_Type"] = df["Student_Type"].fillna(student_type_mode)
df["Month"] = df["Month"].ffill().bfill()
numeric_cols = [
    "Sleep_Hours",
    "Study_Hours",
    "Social_Media_Hours",
    "Attendance",
    "Exam_Pressure",
    "Family_Support",
]

for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print(df.isnull().sum())
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
df =df.reset_index(drop=True)
print(df.duplicated().sum())


st.write(df)

st.title("Hello World")

st.button("Button")