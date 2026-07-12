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
from sklearn.metrics import accuracy_score

#Page Configuration
st.set_page_config(
    page_title="Student Stress Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

#set the intial state 
if "exam_pressue_input" not in st.session_state:
    st.session_state.exam_pressue_input = 1
if "sleep_hours_input" not in st.session_state:
    st.session_state.sleep_hours_input = 6.0
if "study_hours_input" not in st.session_state:
    st.session_state.study_hours_input = 4.0
if "attendance_input" not in st.session_state:
    st.session_state.attendance_input = 75.0
if "family_support_input" not in st.session_state:
    st.session_state.family_support_input = 5
if "Social_Media_input" not in st.session_state:
    st.session_state.Social_Media_input = 2.0

def clear_form_callback():
    st.cache_resource.clear()
    st.session_state.exam_pressue_input = 1
    st.session_state.sleep_hours_input = 6.0
    st.session_state.study_hours_input = 4.0
    st.session_state.attendance_input = 75.0
    st.session_state.family_support_input = 5
    st.session_state.Social_Media_input = 2.0

@st.cache_data
def load_and_preprocess_data():
    #read the csv file and mapping the values like school =1 etc
    df = pd.read_csv("Student LifeStyle.csv")
    df["Student_Type"] = df["Student_Type"].map({"school": 1, "college": 2, "working_student": 3})
    student_type_mode = df["Student_Type"].mode()[0]
    df["Student_Type"] = df["Student_Type"].fillna(student_type_mode)
    df["Month"] = df["Month"].ffill().bfill()
    
    numeric_cols = ["Sleep_Hours", "Study_Hours", "Social_Media_Hours", "Attendance", "Exam_Pressure", "Family_Support"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    df.drop_duplicates(inplace=True)
    
    #IQR Calculation and removal of outliers
    target_cols = ["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Social_Media_Hours", "Stress_Level"]
    outlier_indices = set()
    for col in target_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outlier_indices.update(df[outlier_mask].index)
            
    df_cleaned = df.drop(index=list(outlier_indices)).reset_index(drop=True)
    final_features = ["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Student_Type", "Social_Media_Hours", "Stress_Level"]
    return df_cleaned[final_features]

df1 = load_and_preprocess_data()
sns.set_theme(style="whitegrid")
#training the model and analysing the situation with 6 models below 
@st.cache_resource
def train_ml_models(dataframe):
    X = dataframe[["Exam_Pressure", "Sleep_Hours", "Study_Hours", "Attendance", "Family_Support", "Student_Type", "Social_Media_Hours"]]
    Y = dataframe["Stress_Level"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=10)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "DTC": DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42),
        "RFC": RandomForestClassifier(n_estimators=250, max_depth=15, min_samples_split=4, class_weight="balanced", random_state=42),
        "SVC": SVC(kernel='rbf', C=100, gamma='scale', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GBC": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, Y_train)
        preds = model.predict(X_test_scaled)
        scores[name] = accuracy_score(Y_test, preds)
        
    return models, scaler, scores

models_dict, global_scaler, model_scores = train_ml_models(df1)

# --- Dynamic Counseling Engine ---
def generate_recommendations(exam_p, sleep, study, attend, family, social):
    st.markdown("---")
    st.header("🌱 Personalized Action Plan & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Vulnerability Areas Identified")
        risk_counter = 0
        if sleep < 6.5:
            st.warning("🔴 **Sleep Deprivation Alert**: Your sleep window falls below the therapeutic biological threshold of 6.5 hours. Chronic sleep deficits actively limit prefrontal cortex performance, skyrocketing exam anxiety.")
            risk_counter += 1
        if social > 3.0:
            st.warning("🔴 **Social Media Overstimulation**: Spending more than 3 hours daily on dynamic algorithmic loops forces high dopamine turn-over, resulting in decreased attention spans and elevated background cortisol.")
            risk_counter += 1
        if exam_p > 7 and family < 5:
            st.warning("🔴 **Isolation Pressure Trap**: High academic tension combined with perceived moderate-to-low systemic family backing forms a highly toxic isolation dynamic for performance pressure.")
            risk_counter += 1
        if attend < 75.0:
            st.warning("🔴 **Attendance Risk Boundary**: Attendance below 75% triggers systemic academic alienation, multiplying hidden stressors near exam blocks.")
            risk_counter += 1
            
        if risk_counter == 0:
            st.success("🍏 **Excellent Risk Mitigation**: Your baseline lifestyle parameters indicate resilient structural routines!")

    with col2:
        st.subheader("🚀 Scientific Recovery Roadmap")
        recommendations = [
            "**Implement Non-Sleep Deep Rest (NSDR)**: If structural study durations are rigid, implement a 20-minute protocol to clear adenosine loads during heavy afternoons.",
            "**Strategic Digital Fasting**: Move social media interactions strictly past 6:00 PM. Never check notifications within the first 45 minutes of waking to preserve structural dopamine setups.",
            "**Box Breathing Protocol**: Before studying, execute 4 cycles of box breathing (Inhale 4s, Hold 4s, Exhale 4s, Hold 4s) to drop systemic autonomic arousal levels down.",
            "**Active Recall Anchoring**: Minimize passive reading loops. Utilize active retrieval spaces (flashcards, blurting techniques) to cut unnecessary study hours by up to 25% while boosting memory consolidation."
        ]
        for idx, tip in enumerate(recommendations, 1):
            st.markdown(f"{idx}. {tip}")

# --- Advanced Visualizations Panel ---
def display_global_analytics():
    st.markdown("---")
    st.header("📊 Overall Cohort Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, ax=ax1)
        ax1.set_title("Exam Pressure vs Stress Trend")
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(6, 4.37))
        sns.boxplot(x="Stress_Level", y="Sleep_Hours", data=df1, palette="Set2", ax=ax2)
        ax2.set_title("Sleep Distribution vs Stress Grouping")
        st.pyplot(fig2)
        
    with col2:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="Exam_Pressure", y="Study_Hours", hue="Stress_Level", palette="viridis", data=df1, ax=ax3)
        ax3.set_title("Exam Pressure vs Study Hours Dynamics")
        st.pyplot(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # Fixed system correlation matrix to prevent visual overlay clusters
        sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4, cbar=False)
        ax4.set_title("Dynamic Feature Inter-Correlation Matrix")
        st.pyplot(fig4)

def display_student_metrics(exam_p, sleep, study, attend, family, social, pred_stress):
    st.markdown("---")
    st.header("🎯 Your Personalized Stress Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        # Visual 1: Structural Allocation Split (Pie Chart)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        time_categories = ["Study Hours", "Sleep Hours", "Social Media", "Other Activities"]
        allocated = study + sleep + social
        remainder = max(0, 24.0 - allocated)
        time_values = [study, sleep, social, remainder]
        
        ax1.pie(time_values, labels=time_categories, autopct='%1.1f%%', colors=["#4e79a7", "#59a14f", "#e15759", "#bab0ac"], startangle=90, wedgeprops={'edgecolor': 'w'})
        ax1.set_title("Your 24-Hour Behavioral Metric Allocation")
        st.pyplot(fig1)
        
    with col2:
        # Visual 2: Metric Comparison Axis Against Resilient Benchmarks
        fig2, ax2 = plt.subplots(figsize=(6, 4.64))
        # Finding the typical averages for students who are marked "Not Stressed"
        cohort_baseline = df1[df1["Stress_Level"] == 0].mean()
        
        metrics_comparison = pd.DataFrame({
            "Metric": ["Sleep Time", "Study Time", "Social Media"],
            "Your Profile": [sleep, study, social],
            "Healthy Cohort Average": [cohort_baseline["Sleep_Hours"], cohort_baseline["Study_Hours"], cohort_baseline["Social_Media_Hours"]]
        }).melt(id_vars="Metric", var_name="Profile Group", value_name="Hours")
        
        sns.barplot(x="Metric", y="Hours", hue="Profile Group", data=metrics_comparison, palette="muted", ax=ax2)
        ax2.set_title("Your Configuration vs. Healthy Cohort Baselines")
        st.pyplot(fig2)
        
    st.subheader("📊 Where You Stand Compared to Other Students")
    fig3, ax3 = plt.subplots(figsize=(10, 4.2))
    sns.scatterplot(x="Exam_Pressure", y="Stress_Level", hue="Stress_Level", palette="flare", data=df1, alpha=0.3, ax=ax3)
    ax3.scatter(exam_p, pred_stress, color="gold", s=400, marker="*", edgecolor="black", linewidth=1.5, label="Your Location", zorder=5)
    ax3.legend()
    st.pyplot(fig3)

# --- User Interface Structure ---
st.title("🎓 Student Stress Prediction & Insights Dashboard 🧠")
st.markdown("Analyze baseline biological indicators, behavioral configurations, and track systematic risk margins via Machine Learning frameworks.")

st.sidebar.header("📋 Input Lifestyle Factors")
exam_pressue_input = st.sidebar.slider('Rate Exam Pressure (1-10):', 1, 10, int(st.session_state.exam_pressue_input), key="exam_pressue_input")
sleep_hours_input = st.sidebar.slider('Sleep Duration (Hours):', 0.0, 24.0, float(st.session_state.sleep_hours_input), step=0.5, key="sleep_hours_input")
study_hours_input = st.sidebar.slider('Study Duration (Hours):', 0.0, 24.0, float(st.session_state.study_hours_input), step=0.5, key="study_hours_input")
attendance_input = st.sidebar.slider('Attendance (Percentage):', 0.0, 100.0, float(st.session_state.attendance_input), step=1.0, key="attendance_input")
family_support_input = st.sidebar.slider('Rate Family Support (1-10):', 1, 10, int(st.session_state.family_support_input), key="family_support_input")

StType = ["School", "College", "Working Student"]
student_type = st.sidebar.selectbox('Current Academic Tier', StType)
Social_Media_input = st.sidebar.slider('Social Media Duration (Hours):', 0.0, 24.0, float(st.session_state.Social_Media_input), step=0.5, key="Social_Media_input")

type_mapping = {"School": 1, "College": 2, "Working Student": 3}
numeric_student_type = type_mapping[student_type]

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    predict_clicked = st.button("Analyze & Predict Stress Status", use_container_width=True, type="primary")
with col_btn2:
    st.button("Clear Form Inputs", use_container_width=True, on_click=clear_form_callback)

if predict_clicked:
    scaled_input = global_scaler.transform([[
        exam_pressue_input, sleep_hours_input, study_hours_input, 
        attendance_input, family_support_input, numeric_student_type, Social_Media_input
    ]])
    
    # Run dynamic prediction step
    prediction = int(models_dict["LR"].predict(scaled_input)[0])
    
    st.markdown("---")
    st.subheader("🔮 Machine Learning Diagnostic Results")
    
    if prediction == 1:
        status_text = "Stressed 😖🫩"
        status_delta = "High Risk Level"
        st.error(f"⚠️ **Diagnostic Alert**: The tracking pipeline registers your metrics as: **{status_text}**.")
    else:
        status_text = "Not Stressed 🥳"
        status_delta = "Normal / Low Risk"
        st.success(f"✅ **Diagnostic Summary**: The tracking pipeline registers your metrics as: **{status_text}**.")
        
    # Frame Accuracy Matrix Metrics Layout
    st.markdown("#### ⚙️ Framework Pipeline Benchmarks (Accuracy Scores)")
    metric_cols = st.columns(6)
    for idx, (m_name, score) in enumerate(model_scores.items()):
        with metric_cols[idx]:
            st.metric(label=f"{m_name} Engine", value=f"{score*100:.1f}%")
            
    st.metric(
        label="Final Calculated Assessment Status", 
        value=status_text,
        delta=status_delta,
        delta_color="inverse" if prediction == 1 else "normal"
    )
    
    # Render customized charts & recommendations pipeline sequentially
    display_student_metrics(exam_pressue_input, sleep_hours_input, study_hours_input, attendance_input, family_support_input, Social_Media_input, prediction)
    generate_recommendations(exam_pressue_input, sleep_hours_input, study_hours_input, attendance_input, family_support_input, Social_Media_input)
    display_global_analytics()