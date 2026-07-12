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
from improved import display_student_metrics,display_global_data
from sklearn.metrics import accuracy_score,classification_report

def clear_form_callback():
    st.cache_resource.clear()
    st.session_state.exam_pressue_input = 1
    st.session_state.sleep_hours_input = 0.0
    st.session_state.study_hours_input = 0.0
    st.session_state.attendance_input = 0.0
    st.session_state.family_support_input = 1
    st.session_state.Social_Media_input=0.0

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

target_cols = ["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Social_Media_Hours", "Stress_Level"]

outliers = []
outlier_indices = set()

# 2. Run the IQR scan on these specific columns
for col in target_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        count = outlier_mask.sum()
        percent = (count / len(df)) * 100

        outliers.append({
            'Column': col,
            'Outlier Count': count,
            'Outlier %': round(percent, 2)
        })
        
        outlier_indices.update(df[outlier_mask].index)

# 3. Print the summary report
outlier_df = pd.DataFrame(outliers)
# st.write("Outlier Detection Summary:")
# st.write(outlier_df)

Q1 = df["Sleep_Hours"].quantile(0.25)
Q3 = df["Sleep_Hours"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 1. Identify the outliers
df_outliers = df[(df['Sleep_Hours'] > upper) | (df['Sleep_Hours'] < lower)]

# 2. Drop the outlier indices and save the result into df_new (NO inplace=True)
df_new = df.drop(index=df_outliers.index).reset_index(drop=True)

# 3. Now your print statements will work perfectly!
print("Lower Fence:", lower)
print("Upper Fence:", upper)
print("Dropped Outlier Indices:", list(df_outliers.index))
print("Cleaned DataFrame Preview:\n", df_new.head())

# 4. Drop the unified outlier indices from the master dataframe
df_cleaned = df.drop(index=list(outlier_indices)).reset_index(drop=True)

# 5. Extract your final subset (including non-numeric ones like Student_Type)
final_features = ["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Student_Type", "Social_Media_Hours", "Stress_Level"]
df1 = df_cleaned[final_features]
print(df1)
df1_outliers = df1[(df1['Sleep_Hours'] > upper) | (df1['Sleep_Hours'] < lower)]
print(df1_outliers)
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

    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x="Sleep_Hours", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Sleep Hours Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x= "Study_Hours", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Study Hours Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)
    
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x="Attendance", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Attendance Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x= "Family_Support", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Family Support Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)
    
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x= "Student_Type", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Student Type Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        # Changed figsize to 6, 4.5 to keep heights perfectly uniform!
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x= "Social_Media_Hours", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Social Media Hours Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5)) 
        # 'palette' applies distinct colors across different exam pressure scores
        sns.boxplot(x= "Stress_Level", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Stress Level Distribution (Box Plot)")
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # countplot automatically colors bars based on the variable's value
        sns.countplot(x="Exam_Pressure", hue="Exam_Pressure", palette="viridis", legend=False, data=df1, ax=ax1)
        ax1.set_title("Frequency of Exam Pressure Levels (Count Plot)")
        plt.tight_layout()
        st.pyplot(fig1)


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


    #fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    # Fixed the empty sns.barplot() error by providing valid data and adding a cool 'coolwarm' palette
    #sns.heatmap(df1.corr(), annot=True, ax=ax2)
    #ax2.set_title("Stress Level vs Average Exam Pressure")
    #st.pyplot(fig2)

    #df["Student_Type"]=df["Student_Type"].map({1:"School",2:"College",3:"Working Student"})
    #st.write(df)
    # num_cols = df.select_dtypes(include='number').columns

    # for col in num_cols:
    #     plt.figure(figsize=(8, 2))
    #     sns.boxplot(x=df[col])
    #     plt.title(f'Box Plot of {col}')
    # plt.show()



def StudentData(exam_pressure_input, sleep_hours_input, study_hours_input, attendance_input, family_support_input, Stress_Level):
    st.subheader("🎯 Your Personalized Stress Metrics")
    
    # 1. Convert the user's current inputs into a single-row DataFrame
    user_df = pd.DataFrame([{
        "Exam_Pressure": exam_pressure_input,
        "Sleep_Hours": sleep_hours_input,
        "Study_Hours": study_hours_input,
        "Attendance": attendance_input,
        "Family_Support": family_support_input,
        "Stress_Level": Stress_Level
    }])

    # 2. Set up the display layout
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        # Plotting ONLY the user's data point as a bar chart
        sns.barplot(x="Exam_Pressure", y="Stress_Level", data=user_df, color="#e15759", ax=ax1)
        ax1.set_title("Your Current Exam Pressure vs Stress Level")
        ax1.set_ylim(0, 10)  # Keeps the y-axis standard up to max stress score
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        # Plotting another user metrics context: Study Hours vs Sleep Hours
        sns.barplot(x="Study_Hours", y="Sleep_Hours", data=user_df, color="#4e79a7", ax=ax2)
        ax2.set_title("Your Allocation: Study Hours vs Sleep Hours")
        ax2.set_ylim(0, 24)  # Maximum possible hours in a day
        plt.tight_layout()
        st.pyplot(fig2)

    # --- An Elegant Alternative: Showing the user where they stand against the group ---
    st.subheader("📊 Where You Stand Compared to Other Students")
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    
    # Plot the background historical distribution data using the main database (df1)
    sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, alpha=0.4, ax=ax3)
    
    # Overlay the current user's specific input onto the group chart as a huge, bright star
    ax3.scatter(exam_pressure_input, Stress_Level, color="gold", s=300, marker="*", edgecolor="black", linewidth=1.5, label="Your Profile", zorder=5)
    
    ax3.set_title("Your Placement on the Global Exam Pressure vs Stress Map")
    ax3.legend()
    st.pyplot(fig3)




def stressCal(exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input,student_type,Social_Media_input):
    X = df1[["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Student_Type", "Social_Media_Hours"]]
    Y = df1["Stress_Level"]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=10)

    
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train, Y_train)
    model1 = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model1.fit(X_train, Y_train)
    model2 = RandomForestClassifier(
        n_estimators=250,          # More trees provide a smoother decision boundary
        max_depth=15,              # Slightly deeper to catch complex trait patterns
        min_samples_split=4,       # Balanced threshold for splitting branches
        min_samples_leaf=1,        # Allows the model to catch subtle outlier combinations
        class_weight="balanced",   # Adjusts automatically if you have more stressed than non-stressed rows
        random_state=42            # Keeps your results reproducible
    )
    model2.fit(X_train, Y_train)
    model3 = SVC(
        kernel='rbf',
        C=100,
        gamma='scale',
        
    )
    model3.fit(X_train, Y_train)
    model4 = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        metric='minkowski',
        p=2
    )
    model4.fit(X_train, Y_train)
    model5 = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model5.fit(X_train, Y_train)

    scaled_input1 = scaler.transform([[exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input,student_type,Social_Media_input]])
    LR_pred=model.predict(scaled_input1)
    DTC_pred=model1.predict(scaled_input1)
    RFC_pred=model2.predict(scaled_input1)
    SVC_pred=model3.predict(scaled_input1)
    KNN_pred=model4.predict(scaled_input1)
    GBC_pred=model5.predict(scaled_input1)
    #logistic=model.predict(scaler.transform([[9.0, 7.419726,6.442748,81.452460,5.0]]))
    #st.text(logistic)
    # Extract the binary prediction value (0 or 1)
    prediction = int(LR_pred.item())
    Y_pred=model.predict(X_test)
    score=accuracy_score(Y_pred,Y_test)
    Y_pred1=model1.predict(X_test)
    score1=accuracy_score(Y_pred1,Y_test)
    Y_pred2=model2.predict(X_test)
    score2=accuracy_score(Y_pred2,Y_test)
    Y_pred3=model3.predict(X_test)
    score3=accuracy_score(Y_pred3,Y_test)
    Y_pred4=model4.predict(X_test)
    score4=accuracy_score(Y_pred4,Y_test)
    Y_pred5=model5.predict(X_test)
    score5=accuracy_score(Y_pred5,Y_test)

    # Map the binary outputs to user-friendly text labels
    if prediction == 1:
        status_text = "Stressed 😖🫩"
        status_delta = "High Risk"
        st.error(f"⚠️ Prediction: The model indicates you are currently **{status_text}**.")
        st.write("LR",score,"DTC",score1,"RFC",score2,"SVC",score3,"KNN",score4,"GBC",score5)
    else:
        status_text = "Not Stressed 🥳 "
        status_delta = "Normal / Low Risk"
        st.success(f"✅ Prediction: The model indicates you are **{status_text}**.")
        st.write("LR",score,"DTC",score1,"RFC",score2,"SVC",score3,"KNN",score4,"GBC",score5)
        

    # Display clean metric cards without confusing decimals
    st.metric(
        label="Estimated Stress Status", 
        value=status_text,
        delta=status_delta,
        delta_color="inverse" if prediction == 1 else "normal"  # Colors red for High Risk, green for Normal
    )
    return LR_pred

#data()

st.title("Student Stress Prediction 🧠🤓")
#st.text(sns.__version__)
#"Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Student_Type", "Social_Media_Hours"
StType=["School","College","Working Student"]
exam_pressue_input = st.number_input('Rate Exam Pressure (1(Min)-10(Max)):', min_value=1, max_value=10, step=1, key="exam_pressue_input")
sleep_hours_input = st.number_input('Sleep Duration (Hours):', min_value=0.0, max_value=24.0, step=0.01, key="sleep_hours_input")
study_hours_input = st.number_input('Study Duration (Hours):', min_value=0.0, max_value=24.0, step=0.01, key="study_hours_input")
attendance_input = st.number_input('Attendance (Percentage):', min_value=0.0, max_value=100.0, step=0.01, key="attendance_input")
family_support_input = st.number_input('Rate Family Support (1(Min)-10(Max)):', min_value=1, max_value=10, step=1, key="family_support_input")
student_type=st.selectbox('Student Type',StType)
Social_Media_input = st.number_input('Social Media Duration (Hours):', min_value=0.0, max_value=24.0, step=0.01, key="Social_Media_input")

if student_type=="School":
    student_type=1
elif student_type=="College":
    student_type=2
else:
    student_type=3

# Initialize session state defaults so the app doesn't crash on first load
if "exam_pressue_input" not in st.session_state:
    st.session_state.exam_pressue_input = 1
if "sleep_hours_input" not in st.session_state:
    st.session_state.sleep_hours_input = 0.0
if "study_hours_input" not in st.session_state:
    st.session_state.study_hours_input = 0.0
if "attendance_input" not in st.session_state:
    st.session_state.attendance_input = 0.0
if "family_support_input" not in st.session_state:
    st.session_state.family_support_input = 1

if family_support_input=="Yes":
    family_support_input=1
elif family_support_input=="No":
    family_support_input=0
else:
    family_support_input=-1

if st.button("Button",use_container_width=True):
    LR=stressCal(exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input,student_type,Social_Media_input)
    StudentData(exam_pressue_input,sleep_hours_input ,study_hours_input ,attendance_input ,family_support_input,LR.item())
    data()
    if st.button("Run Advanced Visuals",use_container_width=True):
        display_student_metrics(exam_pressue_input, sleep_hours_input, study_hours_input, attendance_input, family_support_input, LR.item())
        display_global_data()

# if st.button("Clear",use_container_width=True):
#     st.cache_resource.clear()
#     st.session_state.exam_pressue_input = 1
#     st.session_state.sleep_hours_input = 0.00
#     st.session_state.study_hours_input = 0.00
#     st.session_state.attendance_input = 0.00
#     st.session_state.family_support_input = 1
#     st.rerun()

st.button("Clear", use_container_width=True, on_click=clear_form_callback)