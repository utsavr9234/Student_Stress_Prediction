# improved.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Data Engine (Optimized & Cached) ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Student LifeStyle.csv")
    if "Student_Type" in df.columns:
        df["Student_Type"] = df["Student_Type"].map({"school": 1, "college": 2, "working_student": 3})
        df["Student_Type"] = df["Student_Type"].fillna(df["Student_Type"].mode()[0])
    if "Month" in df.columns:
        df["Month"] = df["Month"].ffill().bfill()
        
    numeric_cols = ["Sleep_Hours", "Study_Hours", "Social_Media_Hours", "Attendance", "Exam_Pressure", "Family_Support"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    df.drop_duplicates(inplace=True)
    return df.reset_index(drop=True)

# Global variables accessible by visualizations
df = load_and_clean_data()
df1 = df[["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Stress_Level"]]
sns.set_theme(style="whitegrid")

# --- Function 1: ML Engine ---
def predict_stress(exam_p, sleep, study, attend, family):
    feature_names = ["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support"]
    X = df[feature_names]
    Y = df["Stress_Level"]
    
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.30, random_state=10)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(random_state=10)
    model.fit(X_train_scaled, Y_train)

    input_data = pd.DataFrame([[exam_p, sleep, study, attend, family]], columns=feature_names)
    scaled_input = scaler.transform(input_data)
    
    prediction = int(model.predict(scaled_input)[0])
    return prediction

# --- Function 2: Individual Student Metrics ---
def display_student_metrics(exam_p, sleep, study, attend, family, pred_stress):
    st.markdown("---")
    st.subheader("🎯 Your Personalized Stress Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=["Your Profile"], y=[exam_p], color="#e15759", ax=ax)
        ax.set_ylabel("Rating (1-10)")
        ax.set_ylim(0, 10)
        ax.set_title("Your Inputted Exam Pressure")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        metrics_df = pd.DataFrame({"Metric": ["Study Hours", "Sleep Hours"], "Hours": [study, sleep]})
        sns.barplot(x="Metric", y="Hours", data=metrics_df, palette="muted", ax=ax)
        ax.set_ylim(0, 24)
        ax.set_title("Your Allocation: Study vs Sleep")
        st.pyplot(fig)

    st.subheader("📊 Where You Stand Compared to Other Students")
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, alpha=0.3, ax=ax3)
    ax3.scatter(exam_p, pred_stress, color="gold", s=400, marker="*", edgecolor="black", linewidth=1.5, label="Your Profile", zorder=5)
    ax3.set_title("Your Placement on the Global Dataset Map")
    ax3.legend()
    st.pyplot(fig3)

# --- Function 3: Global Visualizations ---
def display_global_data():
    st.markdown("---")
    st.header("📊 Overall Dataset Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(x="Exam_Pressure", y="Stress_Level", data=df1, ax=ax, 
                    scatter_kws={"color": "#4e79a7", "alpha": 0.5}, 
                    line_kws={"color": "#e15759", "linewidth": 2})
        ax.set_title("Exam Pressure vs Stress Trend")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Stress_Level", y="Sleep_Hours", data=df1, palette="Set2", ax=ax)
        ax.set_title("Sleep Hours Distribution by Stress Level")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Stress_Level", hue="Stress_Level", palette="viridis", data=df1, ax=ax, legend=False)
        ax.set_title("Dataset Balance: Stressed vs Non-Stressed Count")
        st.pyplot(fig)
        
    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)