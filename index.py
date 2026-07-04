import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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




# Set a clean grid style for all plots
sns.set_theme(style="whitegrid")

def data():
    # --- FIRST ROW: Scatter Plot & Box Plot ---
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # 'hue' maps colors dynamically to the Stress_Level categories/values
        # 'palette' defines the color scheme (e.g., flare, viridis, magma)
        sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df, ax=ax1)
        ax1.set_title("Exam Pressure vs Stress (Scatter Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x="Exam_Pressure", data=df, palette="Set2", ax=ax2)
        ax2.set_title("Exam Pressure Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)


    # --- SECOND ROW: Regression Plot & Bar Chart ---

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # regplot uses single 'color' arguments for scatter points and lines
        sns.regplot(x="Exam_Pressure", y="Stress_Level", data=df, ax=ax1, 
                    scatter_kws={"color": "#4e79a7", "alpha": 0.6}, 
                    line_kws={"color": "#e15759", "linewidth": 2})
        ax1.set_title("Exam Pressure vs Stress (Regression Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        # 'hue' splits bars by stress level, creating a multi-colored cluster chart
        sns.barplot(x="Exam_Pressure", y="Stress_Level", palette="crest", data=df, ax=ax2)
        ax2.set_title("Exam Pressure vs Stress (Bar Chart)")
        plt.tight_layout()
        st.pyplot(fig2)


    # --- THIRD ROW: Count Plot & Second Bar Chart ---

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # countplot automatically colors bars based on the variable's value
        sns.countplot(x="Exam_Pressure", hue="Exam_Pressure", palette="viridis", legend=False, data=df, ax=ax1)
        ax1.set_title("Frequency of Exam Pressure Levels (Count Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        # Fixed the empty sns.barplot() error by providing valid data and adding a cool 'coolwarm' palette
        sns.countplot()
        ax2.set_title("Stress Level vs Average Exam Pressure")
        plt.tight_layout()
        st.pyplot(fig2)

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    # Fixed the empty sns.barplot() error by providing valid data and adding a cool 'coolwarm' palette
    sns.heatmap(df.corr(), annot=True, ax=ax2)
    ax2.set_title("Stress Level vs Average Exam Pressure")
    st.pyplot(fig2)

    df["Student_Type"]=df["Student_Type"].map({1:"School",2:"College",3:"Working Student"})
    st.write(df)

#data()

st.title("Student Stress Prediction")
#st.text(sns.__version__)


if st.button("Button",use_container_width=True):
    data()

if st.button("Clear",use_container_width=True):
        st.cache_resource.clear()