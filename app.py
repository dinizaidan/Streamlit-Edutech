import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Load the Model and Preprocessor ---
try:
    model = joblib.load('logistic_regression_model.pkl')
    ohe = joblib.load('onehot_encoder.joblib')
    scaler = joblib.load('standard_scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model atau preprocessor (encoder/scaler) tidak ditemukan.")
    st.info("Make sure 'logistic_regression_model.pkl', 'onehot_encoder.joblib', and 'standard_scaler.joblib' are located in the same folder.")
    st.stop() # Stop if the file is not found

# --- Define Numerical and Categorical Columns ---
numerical_cols = ['Age_at_enrollment']
categorical_cols = [
    'Marital_status', 'Application_mode', 'Course',
    'Daytime_evening_attendance', 'Nationality', 'Gender',
    'Academic_Success_Score', 'Academic_Engagement_Level', 'Parental_SES_Score',
    'Financial_Risk', 'socioeconomic_risk_score', 'Credit_Recognition_Rate',
    'First_Choice_Application', 'Performance_Shift', 'Macroeconomic_Stress'
]

# --- APPLICATION ---
st.set_page_config(layout="wide") # Make the layout wide
st.title("Student Performance Prediction Application")
st.markdown("---")
st.write("""
This Application predicts whether a student will *Graduate* or *Dropout* based on the data provided.
""")
st.markdown("---")

# --- User Interface (UI) for Data Input ---
st.header("Demography and Academics")
col1, col2 = st.columns(2)
with col1:
    age_at_enrollment = st.slider("Age at enrollment", min_value=16, max_value=80, value=20, step=1)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    nationality = st.selectbox("Nationality", [
        'Portuguese', 'European Countries', 'Brazilian', 'PALOP', 'Latin American'
    ])
with col2:
    marital_status = st.selectbox("Marital Status", [
        'Never Married (yet)', 'Currently Married/In Union', 'Previously Married'
    ])
    application_mode = st.selectbox("Application Mode", [
        'General', 'International', 'Special', 'Change'
    ])
    daytime_evening_attendance = st.selectbox("Attendance Time", [
        'Daytime', 'Evening'
    ])

course = st.selectbox("Course", [
    'Engineering/Technology', 'Arts/Design/Communication', 'Social Sciences',
    'Agriculture/Environment', 'Health Sciences', 'Business', 'Education', 'Other'
])


st.subheader("Academic and Performance Details")

# Academic Success Level
col_asl_input, col_asl_desc = st.columns([1, 1.5])
with col_asl_input:
    academic_success_score = st.selectbox("Academic Success Level", [
        'Unknown', 'Good', 'No Performance', 'Low', 'Excellent'
    ])
with col_asl_desc:
    st.write(
    """
    **How is 'Academic Success Level' Calculated?**
    The 'Academic Success Level' is determined by an **Academic Success Score**, 
    which is calculated based on your **Approval Rate** and **Average Grade**.
    
    Here's a breakdown of the calculation:
    
    1.  **Total Credits Attempted:**
        * `Total_Credits_Attempted` = `Curricular units 1st sem enrolled` + `Curricular units 2nd sem enrolled`
        
    2.  **Total Credits Approved:**
        * `Total_Credits_Approved` = `Curricular units 1st sem approved` + `Curricular units 2nd sem approved`
        
    3.  **Approval Rate:**
        * `Approval_Rate` = `Total_Credits_Approved` / `Total_Credits_Attempted`
        
    4.  **Average Grade:**
        * `Average_Grade` = (`Curricular units 1st sem grade` + `Curricular units 2nd sem grade`) / 2
        
    5.  **Academic Success Score:**
        * `Academic_Success_Score` = `Approval_Rate` * `Average_Grade`
        
    Finally, your 'Academic Success Level' is categorized as follows:
    
    * **Excellent:** If `Academic_Success_Score` >= 15
    * **Good:** If `Academic_Success_Score` >= 10
    * **Low:** If `Academic_Success_Score` > 0 and < 10
    * **No Performance:** If `Academic_Success_Score` is 0 (or undefined due to no credits attempted/approved)
    * **Unknown:** If data is unavailable or could not be processed.
    """)

# Academic Engagement Level
col_ael_input, col_ael_desc = st.columns([1, 1.5])
with col_ael_input:
    academic_engagement_level = st.selectbox("Academic Engagement Level", [
        'Unknown', 'Very High', 'High', 'Moderate', 'Low', 'Very Low'
    ])
with col_ael_desc:
    st.write("""
    **How is 'Academic Engagement Level' Calculated?**
    Your 'Academic Engagement Level' reflects how actively you've participated in evaluations. 
    It's calculated using your **Academic Engagement Score**, which considers your total evaluations 
    against the sum of all evaluations (those with and without grades).
    
    Here's how it's determined:
    
    1.  **Total Evaluations:**
        * `Total_Evaluations` = `Curricular units 1st sem evaluations` + `Curricular units 2nd sem evaluations`
        
    2.  **Evaluations Without Grade:**
        * `Evaluations_Without_Grade` = `Curricular units 1st sem without evaluations` + `Curricular units 2nd sem without evaluations`
        
    3.  **Academic Engagement Score:**
        * `Academic_Engagement_Score` = `Total_Evaluations` / (`Total_Evaluations` + `Evaluations_Without_Grade`)
        
    Based on this score, your 'Academic Engagement Level' is categorized as follows:
    
    * **Very High:** If `Academic_Engagement_Score` $\\geq$ 0.9
    * **High:** If `Academic_Engagement_Score` is between 0.75 and 0.9 (exclusive of 0.9)
    * **Moderate:** If `Academic_Engagement_Score` is between 0.5 and 0.75 (exclusive of 0.75)
    * **Low:** If `Academic_Engagement_Score` is between 0.25 and 0.5 (exclusive of 0.5)
    * **Very Low:** If `Academic_Engagement_Score` $<$ 0.25     
    """)

# Credit Recognition Rate
col_crr_input, col_crr_desc = st.columns([1, 1.5])
with col_crr_input:
    credit_recognition_rate = st.selectbox("Credit Recognition Rate", [
        'Unknown', 'No Credits Recognized', 'Low Recognition',
        'Moderate Recognition', 'High Recognition', 'Exceptional (Overcredited)'
    ])
with col_crr_desc:
    st.write("""
    **How is 'Credit Recognition Rate' Calculated?**
    The 'Credit Recognition Rate' indicates the proportion of credits you've successfully 
    had recognized (e.g., from previous studies or transfers) compared to the total credits you've attempted.
    
    Here's the breakdown of how it's calculated:
    
    1.  **Total Credits Credited:**
        * `Total_Credits_Credited` = `Curricular units 1st sem credited` + `Curricular units 2nd sem credited`
        
    2.  **Total Credits Attempted:**
        * (This value is carried over from the 'Academic Success Score' calculation)
        * `Total_Credits_Attempted` = `Curricular units 1st sem enrolled` + `Curricular units 2nd sem enrolled`
        
    3.  **Credit Recognition Rate:**
        * `Credit_Recognition_Rate` = `Total_Credits_Credited` / `Total_Credits_Attempted`
        
    **Important Note on Calculation:**
    To prevent errors, if `Total_Credits_Attempted` is zero, the `Credit_Recognition_Rate` will be considered 'Unknown' (represented as `NaN`).
    
    Based on this rate, your 'Credit Recognition Level' is categorized as follows:
    
    * **Unknown:** If the rate cannot be calculated (e.g., no credits attempted).
    * **No Credits Recognized:** If `Credit_Recognition_Rate` is exactly 0.
    * **Low Recognition:** If `Credit_Recognition_Rate` $<$ 0.5 (but greater than 0).
    * **Moderate Recognition:** If `Credit_Recognition_Rate` is between 0.5 and less than 0.8 (0.5 $\\leq$ rate $<$ 0.8).
    * **High Recognition:** If `Credit_Recognition_Rate` is between 0.8 and 1.0, inclusive (0.8 $\\leq$ rate $\\leq$ 1.0).
    * **Exceptional (Overcredited):** If `Credit_Recognition_Rate` $>$ 1.0 (This might indicate special circumstances where more credits are recognized than initially attempted for a period).
    """)

# First Choice Application 
st.subheader("Application Details") 
col_app_input, col_perf_input = st.columns([1, 1.5])
with col_app_input:
    first_choice_application = st.selectbox("Application", [
        'First Choice', 'Second/more Choice'
    ])
with col_perf_input:
    performance_shift = st.selectbox("Performance Compared with Previous Qualification", [
        'Performed Better', 'Performed Worse', 'Maintained Performance'
    ])
    st.write("""
    **What is 'Performance Shift' and How is it Calculated?**
    The 'Performance Shift' metric helps you understand how your academic performance 
    in your current program compares to your performance in your previous qualification. 
    It's a ratio that gives insight into whether you've improved, maintained, or declined.

    Here's the simple calculation:

    * **Performance Shift** = `Admission grade` / `Previous qualification grade`

    **Interpreting Your Performance Shift:**

    * **$>$ 1.0: Performed Better Than Before**
        * This means your admission grade for the current program is higher than your previous qualification grade. Great job – you're doing even better!
    * **$\\approx$ 1.0: Maintained Performance**
        * If the ratio is close to 1.0, it suggests you've maintained a similar level of academic performance from your previous studies. Consistent effort!
    * **$<$ 1.0: Performed Worse Than Prior Education**
        * A ratio less than 1.0 indicates that your admission grade is lower than your previous qualification grade. This could highlight areas where you might need to focus more or adjust your study habits.
    """)

st.header("External Risks and Factors")
# Financial Risk
col_fr_input, col_fr_desc = st.columns([1, 1.5])
with col_fr_input:
    financial_risk = st.selectbox("Financial Risk", [
        'financially safe', 'moderate risk', 'high risk'
    ])
with col_fr_desc:
    st.write("""
    **How is Your 'Financial Risk' Assessed?**
    Your 'Financial Risk' level is an indicator of potential financial challenges 
    that might impact your academic journey. It's calculated based on two key factors: 
    whether you are a **Debtor** and the status of your **Tuition Fees**.

    Here's the calculation and what each component means:

    * **Debtor Status:**
        * This is a binary value (0 or 1).
        * **1** if you are classified as a debtor (meaning you have outstanding payments).
        * **0** if you are not a debtor.

    * **Tuition Fees Up-to-Date Status:**
        * This is also a binary value (0 or 1).
        * **1** if your tuition fees are up-to-date.
        * **0** if your tuition fees are NOT up-to-date.
        * In the calculation, we use `(1 - Tuition_fees_up_to_date)` so that a '0' (not up-to-date) contributes to the risk.

    The **Financial Risk Score** is calculated as:
    `Financial_Risk_Score` = `Debtor` + (1 - `Tuition_fees_up_to_date`)

    This score will result in one of three values (0, 1, or 2), which are then mapped to a risk level:

    * **0: Financially Safe**
        * This means you are not a debtor AND your tuition fees are up-to-date.
    * **1: Moderate Risk**
        * This indicates one of the following: you are a debtor, OR your tuition fees are not up-to-date.
    * **2: High Risk**
        * This signifies that you are both a debtor AND your tuition fees are not up-to-date.
    """)

# Socioeconomic Risk Score
col_ses_input, col_ses_desc = st.columns([1, 1.5])
with col_ses_input:
    socioeconomic_risk_score = st.selectbox("Socioeconomic Risk Assessment", [
        'No identified risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk' # Reordered for common perception
    ])
with col_ses_desc:
    st.write("""
    **How is Your 'Socioeconomic Level' Assessed?**
    Your 'Socioeconomic Level' is calculated as a **Socioeconomic Risk Score** that considers several factors that might indicate a need for additional support 
    or resources. Each contributing factor adds to your risk score.

    The score is based on the sum of the following indicators (where each contributes 1 if true, 0 if false):

    * **Displaced:**
        * Are you a displaced student? (e.g., due to circumstances requiring relocation)
    * **Educational Special Needs:**
        * Do you have recognized educational special needs?
    * **International:**
        * Are you an international student? (This can sometimes imply unique challenges)
    * **Scholarship Holder:**
        * Are you a scholarship holder? (While often positive, sometimes scholarships are tied to specific socioeconomic criteria or needs)

    The **Socioeconomic Risk Score** is calculated as:
    `Socioeconomic_Risk_Score` = `Displaced` + `Educational_special_needs` + `International` + `Scholarship_holder`

    Based on this cumulative score, your 'Socioeconomic Level' is categorized:

    * **0: No Identified Risk**
        * None of the above factors are present.
    * **1: Low Risk**
        * Only one of the above factors is present.
    * **2: Moderate Risk**
        * Two of the above factors are present.
    * **$>$ 2: High Risk**
        * Three or more of the above factors are present.         
    """)

# Parental Socioeconomic Status
col_pses_input, col_pses_desc = st.columns([1, 1.5])
with col_pses_input:
    parental_ses_score = st.selectbox("Parental Socioeconomic Status", [
        'Low SES', 'Medium SES', 'High SES'
    ])
with col_pses_desc:
    st.write("""
    **How is Parental Socioeconomic Status** Determined?**
    Your 'Parental Socioeconomic Status is an indicator of your parents' 
    educational attainment and professional background. It helps us understand the socioeconomic context 
    that might influence your academic journey.

    The SES score is categorized into three levels based on your parents' highest education level and their occupation type:

    * **Low SES (Low Socioeconomic Status):**
        * Neither of your parents has achieved a higher education degree (e.g., university level) **nor** do they hold a white-collar job.
        * This suggests a more limited access to resources typically associated with higher education and professional occupations.

    * **Medium SES (Medium Socioeconomic Status):**
        * **One** of your parents has either attained a higher education degree **or** holds a white-collar job.
        * This indicates a mixed socioeconomic background, with one parent contributing to higher educational or professional standing.

    * **High SES (High Socioeconomic Status):**
        * **Both** of your parents have either achieved a higher education degree **or** hold a white-collar job.
        * Alternatively, if **one parent** possesses **both** a higher education degree and a white-collar job, this also qualifies for High SES.
        * This typically suggests a stronger foundation of resources and opportunities stemming from higher education and professional careers.
    """)

# Macroeconomic Stress Level
col_msl_input, col_msl_desc = st.columns([1, 1.5])
with col_msl_input:
    macroeconomic_stress = st.selectbox("Macroeconomic Stress Level", [
        'Low Stress', 'Moderate Stress', 'High Stress' 
    ])
with col_msl_desc:
    st.write("""
    **How is 'Macroeconomic Stress Level' Determined?**
    Your 'Macroeconomic Stress Level' is an indicator that assesses the impact of broader 
    economic conditions on your academic environment. It's calculated using key economic indicators: 
    **Unemployment Rate**, **Inflation Rate**, and **Gross Domestic Product (GDP)**.

    **Understanding the Components:**

    Before combining them, each of these economic indicators is first **normalized**. 
    This means their values are scaled to a range between 0 and 1, making them comparable 
    regardless of their original units or scales. This normalization is done using a 
    min-max scaling method:

    $Normalized\\_Value = (Current\\_Value - Minimum\\_Value) / (Maximum\\_Value - Minimum\\_Value)$

    Once normalized:

    * **Unemployment Rate:** A higher unemployment rate generally indicates more economic stress.
    * **Inflation Rate:** A higher inflation rate signifies rising costs and typically points to increased economic stress.
    * **GDP (Gross Domestic Product):** GDP represents the total economic output. A higher GDP usually suggests a healthier economy, thus reducing stress.

    **Calculating the Macroeconomic Stress Level:**

    The `Macroeconomic_Stress_Level` is derived from these normalized values using the following formula:

    `Macroeconomic_Stress_Level` = `Normalized Unemployment Rate` + `Normalized Inflation Rate` - `Normalized GDP`

    **Interpreting Your Macroeconomic Stress Level:**

    Based on the calculated score (let's call it 'msi' for macroeconomic stress index), 
    your macroeconomic environment is categorized as:

    * **Low Stress:** If the `msi` score is $<$ 5.
    * **Moderate Stress:** If the `msi` score is between 5 (inclusive) and $<$ 10.
    * **High Stress:** If the `msi` score is $\\geq$ 10.
    """)

st.markdown("---")


# Prediction Button and Logic
if st.button("Predict Student Performance", help="Click to get performance prediction based on input"):
    # --- Input Data Preparation for Prediction ---
    # Create a dictionary from user inputs
    user_data = {
    'Age_at_enrollment': age_at_enrollment,
    'Marital_status': marital_status,
    'Application_mode': application_mode,
    'Course': course,
    'Daytime_evening_attendance': daytime_evening_attendance,
    'Nationality': nationality,
    'Gender': gender,
    'Academic_Success_Score': academic_success_score,
    'Academic_Engagement_Level': academic_engagement_level,
    'Parental_SES_Score': parental_ses_score,
    'Financial_Risk': financial_risk,
    'socioeconomic_risk_score': socioeconomic_risk_score,
    'Credit_Recognition_Rate': credit_recognition_rate,
    'First_Choice_Application': first_choice_application,
    'Performance_Shift': performance_shift,
    'Macroeconomic_Stress': macroeconomic_stress
}

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    st.subheader("Your Input Data:")
    st.dataframe(input_df)

    # 1. Separate numerical and categorical inputs
    numerical_input = input_df[numerical_cols]
    categorical_input = input_df[categorical_cols]

    # 2. Scale numerical features
    scaled_numerical_input = scaler.transform(numerical_input)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=numerical_cols, index=input_df.index)

    # 3. One-hot encode categorical features
    try:
        encoded_categorical_input = ohe.transform(categorical_input)
        encoded_categorical_df = pd.DataFrame(encoded_categorical_input,
                                               columns=ohe.get_feature_names_out(categorical_cols),
                                               index=input_df.index)
    except ValueError as e:
        st.error(f"Error during one-hot encoding: {e}. This might happen if new categories are present in your input that the encoder hasn't seen during training. Ensure your encoder is fitted on all possible categories.")
        st.stop()

    # 4. Concatenate preprocessed features and ensure exact column order
    preprocessed_input = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1) # THIS LINE IS CRUCIAL
    expected_features_for_model = numerical_cols + list(ohe.get_feature_names_out(categorical_cols))

    final_input_for_prediction = preprocessed_input.reindex(columns=expected_features_for_model, fill_value=0)

    st.subheader("Processed Data for Model (Final for Prediction):")
    st.dataframe(final_input_for_prediction)

    # Make prediction
    prediction_proba_all = model.predict_proba(final_input_for_prediction)[0]
    prediction = model.predict(final_input_for_prediction)[0]

    if 0 in model.classes_ and 1 in model.classes_:
        graduate_idx = np.where(model.classes_ == 0)[0][0]
        dropout_idx = np.where(model.classes_ == 1)[0][0]

        proba_graduate = prediction_proba_all[graduate_idx]
        proba_dropout = prediction_proba_all[dropout_idx]
    else:
        st.error("Model classes do not contain expected values (0 and 1). Cannot determine probabilities correctly.")
        st.stop()


    # --- Display Prediction Result ---
    st.subheader("Prediction Result:")     
    if prediction == 0: # If the prediction is 0 (Graduate)
        st.success(f"This student is predicted to: **Graduate**")
    else: # If the prediction is 1 (Dropout)
        st.error(f"This student is predicted to: **Dropout**")
        
    st.write(f"Probability of 'Graduate': **{proba_graduate*100:.2f}%**")
    st.write(f"Probability of 'Dropout': **{proba_dropout*100:.2f}%**")
    st.markdown("---")
    st.info("Note: This prediction is based on a trained Machine Learning model.")


st.markdown("---")
st.markdown("Created by Dinny Zaidan Nadwah")
st.markdown("dininadwah@gmail.com")