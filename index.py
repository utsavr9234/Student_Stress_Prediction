import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

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

df1 = df[["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Stress_Level"]]


# Set a clean grid style for all plots
sns.set_theme(style="whitegrid")

def data():
    # --- FIRST ROW: Scatter Plot & Box Plot ---
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # 'hue' maps colors dynamically to the Stress_Level categories/values
        # 'palette' defines the color scheme (e.g., flare, viridis, magma)
        sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, ax=ax1)
        ax1.set_title("Exam Pressure vs Stress (Scatter Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # 'hue' maps colors dynamically to the Stress_Level categories/values
        # 'palette' defines the color scheme (e.g., flare, viridis, magma)
        sns.scatterplot(x="Exam_Pressure", y="Study_Hours", hue="Study_Hours", palette="flare", data=df1, ax=ax1)
        ax1.set_title("Exam Pressure vs Study Hours (Scatter Plot)")
        plt.tight_layout()
        st.pyplot(fig1)


    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # 'hue' maps colors dynamically to the Stress_Level categories/values
        # 'palette' defines the color scheme (e.g., flare, viridis, magma)
        sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, ax=ax1)
        ax1.set_title("Exam Pressure vs Stress (Scatter Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x="Exam_Pressure", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Exam Pressure Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)


    # --- SECOND ROW: Regression Plot & Bar Chart ---

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # regplot uses single 'color' arguments for scatter points and lines
        sns.regplot(x="Exam_Pressure", y="Stress_Level", data=df1, ax=ax1, 
                    scatter_kws={"color": "#4e79a7", "alpha": 0.6}, 
                    line_kws={"color": "#e15759", "linewidth": 2})
        ax1.set_title("Exam Pressure vs Stress (Regression Plot)")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        # 'hue' splits bars by stress level, creating a multi-colored cluster chart
        sns.barplot(x="Exam_Pressure", y="Stress_Level", palette="crest", data=df1, ax=ax2)
        ax2.set_title("Exam Pressure vs Stress (Bar Chart)")
        plt.tight_layout()
        st.pyplot(fig2)


    # --- THIRD ROW: Count Plot & Second Bar Chart ---

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # countplot automatically colors bars based on the variable's value
        sns.countplot(x="Exam_Pressure", hue="Exam_Pressure", palette="viridis", legend=False, data=df1, ax=ax1)
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
    sns.heatmap(df1.corr(), annot=True, ax=ax2)
    ax2.set_title("Stress Level vs Average Exam Pressure")
    st.pyplot(fig2)

    df["Student_Type"]=df["Student_Type"].map({1:"School",2:"College",3:"Working Student"})
    st.write(df)


def stressCal(exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input):
    X = df[["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support"]]
    Y = df["Stress_Level"]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=10)

    
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X, Y)
    model1 = DecisionTreeClassifier()
    model1.fit(X, Y)
    model2 = RandomForestClassifier()
    model2.fit(X, Y)
    model3 = SVC()
    model3.fit(X, Y)
    model4 = KNeighborsClassifier()
    model4.fit(X, Y)
    model5 = GradientBoostingClassifier()
    model5.fit(X, Y)

    scaled_input1 = scaler.transform([[exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input]])
    LR=model.predict(scaled_input1)
    st.success(f"🧠 Predicted Stress Score: **{LR.item():.2f} / 10**")

    st.metric(
        label="Estimated Stress Level", 
        value=f"{LR.item():.2f}",
        delta="High Risk" if LR.item() == 1 else "Normal"  # Optional: adds a visual status indicator
    )

#data()

st.title("Student Stress Prediction")
#st.text(sns.__version__)
exam_pressue_input = st.number_input('Rate Exam Pressure (1(Min)-10(Max)):', min_value=1, value=1, step=1, max_value=10)
sleep_hours_input = st.number_input('Sleep Duration (Hours)::', min_value=0.00, value=0.00, step=0.01, max_value=24.00)
study_hours_input = st.number_input('Study Duration (Hours)::', min_value=0.00, value=0.00, step=0.01, max_value=24.00)
attendance_input = st.number_input('Attendance (Percentage)::', min_value=0.00, value=0.00, step=0.01, max_value=100.00)
family_support_input = st.radio("Do you have family support?",options=["Yes","No"],horizontal=True)

if family_support_input=="Yes":
    family_support_input=1
elif family_support_input=="No":
    family_support_input=0
else:
    family_support_input=-1

if st.button("Button",use_container_width=True):
    stressCal(exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input)
    data()

if st.button("Clear",use_container_width=True):
        st.cache_resource.clear()