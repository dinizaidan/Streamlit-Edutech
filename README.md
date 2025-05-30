# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
This project is a submission for Dicoding's Data Science Level Expert class. The dataset contains information about Jaya Jaya Institute, a fictitious higher education institution which known for producing graduates with an excellent reputation. The problem is that the current dropout rate is considerably high, which presents a critical challenge. This project aims to identify the factors contributing to student dropouts, enabling the institute to offer targeted support and reduce attrition.

### Permasalahan Bisnis
The high student dropout rate at Jaya Jaya Institute threatens its reputation and long-term sustainability.

### Cakupan Proyek
This project explores key contributors to student dropout using machine learning techniques—Random Forest and Logistic Regression—and communicates findings through an interactive dashboard.

### Persiapan

**Setup environment**:
Create and use the `sub2` environment, then install the necessary libraries:

```bash
conda create -n sub2 python=3.10    # Create environment named sub2 with python 3.10
conda activate sub2                 # Activate the environment
pip install -r requirements.txt     
pip install pandas scikit-learn sqlalchemy matplotlib seaborn pymysql streamlit joblib
```

**Data Source**: [Data of Jaya Jaya Maju Institute](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv)
This dataset created from a higher education institution (acquired from several disjoint databases) related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes information known at the time of student enrollment (academic path, demographics, and social-economic factors) and the students' academic performance at the end of the first and second semesters. The data is used to build classification models to predict students' dropout and academic sucess.

**Database Configuration**
1. Connect to MySQL using SQLAlchemy:
```python
engine = create_engine('mysql+pymysql://username:password@localhost:3306/database_name')
```

2. Store data in MySQL
```python
table_name = 'student'
data_dashboard.to_sql(table_name, con=engine, if_exists='replace', index=False)
```

3. Launch Metabase via Docker
```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

4. Access the dashboard:
- URL: http://localhost:3000
- Credentials:
email:: dininadwah@gmail.com
password : adin02


## Machine Learning Prototype
This section details the core machine learning workflow, which involves data processing, model training, and evaluation. All steps are primarily implemented and documented within the `notebook.ipynb` file.

### 1. Data Preprocessing (notebook.ipynb)
- Simplifying categories (mapping)
- Aggregate relevant columns
- Drop columns used in aggregation
- Format and define categorical variables
- Visualize key variables
- Check for missing and duplicate values
- Duplicate data for analysis and dashboard
- Encode, scale, and split data

### 2. Modeling and Evaluation (notebook.ipynb)
- Build models using Random Forest and Logistic Regression
- Evaluate model performance
- Save the trained models

### 3. Streamlit Prediction Application (app.py)
This project features an interactive web application built with Streamlit, which serves as a deployment interface for the trained machine learning model. This application allows users to predict the likelihood of a student Graduating or Dropping out based on custom input data, providing real-time insights.

**Run the Streamlit Application**
The saved model files (`logistic_regression_model.pkl`, `onehot_encoder.joblib`, `standard_scaler.joblib`) must located in the same directory as `app.py`.

```bash
streamlit run app.py
```
The application will automatically open in web browser (typically at `http://localhost:8501`)

**Application Features**
The Streamlit application provides an intuitive user interface where users can input various student demographic, academic, and external factor details:

- *Demography & Academics*: Age at enrollment, gender, nationality, marital status, application mode, attendance time (daytime/evening), and course of study.
- *Academic and Performance Details*: Academic Success Level, Academic Engagement Level, and Credit Recognition Rate. Each of these includes detailed explanations on how their underlying scores are calculated.
- *Application Details*: Whether the application was a 'First Choice' and the 'Performance Shift' compared to previous qualifications, also with calculation explanations.
- *External Risks and Factors*: Financial Risk, Socioeconomic Risk Assessment, Parental Socioeconomic Status, and Macroeconomic Stress Level, each accompanied by explanations of their respective calculation methodologies.

After entering the data, users can click the "Predict Student Performance" button to receive a prediction (Graduate or Dropout) along with the probabilities for each outcome.

This application serves as a practical demonstration of how machine learning models can be utilized to deliver actionable insights and predictions to stakeholders.

## Business Dashboard
**Key Insights from dashboard**
✔ Students most likely to graduate are financially stable, enroll early, and have strong socioeconomic support.
✔ Dropout risk increases with financial stress, moderate socioeconomic risk, and later age at enrollment.
✔ Male students have slightly higher dropout rates compared to females.
✔ Students with “no identified risk” still show notable dropout rates, indicating hidden challenges.


## Conclusion

### High Importance Features (Random Forest)

**1. Academic Success Score**
Students with a "Good" academic success score are significantly less likely to drop out. While students with a "Low" and "No Performance" academic success score are highly vulnerable to dropping out.

**2. Age at Enrollment**
Younger students (especially those aged 18–22) tend to have better retention rates. Dropout risk appears to increase with age.

**3. Financial Risk**
Financial instability significantly increases dropout risk. Many graduates are financially safe. Most dropouts come from moderate to high financial risk backgrounds.

**4. Academic Engagement Level**
Academic involvement is a protective factor. Students with very high engagement mostly graduate. Those with low engagement are more likely to drop out.


### Key Factors Influencing student's dropout (Logistic Regression)

**Top Risk Factors for Dropout** (Green Bars)

**1. Academic Success Score: No Performance**
The strongest positive predictor. Students with no academic performance score are highly likely to drop out.

**2. Financial Risk: High**
Students in the high financial risk category face a significant dropout risk.

**3. Course: Engineering/Technology**
Students in engineering/technology fields show higher dropout risk.

**4. Nationality: Latin American**
Latin American students show a higher tendency to drop out (possibly due to cultural or support barriers).

**5. Academic Success Score: Low**
Students with low academic scores are at elevated risk.


**Protective Factors Against Dropout** (Red Bars)

**1. Academic Success Score: Good/Excellent**
Good scores are highly protective.

**2. Financial Risk: Financially Safe**
Students with strong financial status are less likely to drop out.

**3. Course: Social Sciences/Health Sciences**
Students in these programs tend to have lower dropout rates.

**4. Academic Engagement Level: Very High**
Higher engagement correlates with staying enrolled.

**5. Performance Shift: Maintained or Improved**
Students maintaining or improving performance are less likely to drop out.



## Recommendations

Based on the analysis using Random Forest and Logistic Regression, the following data-driven recommendations are proposed to reduce student dropout rates at Jaya Jaya Institute:

**1. Early Academic Intervention**
- Closely monitor students with low or missing academic performance scores.
- Provide tutoring programs, study skills workshops, and regular academic progress tracking.

**2. Financial Support Programs**
- Expand scholarship opportunities and offer flexible tuition payment plans for students at moderate to high financial risk.
- Implement early identification systems to detect students experiencing financial stress.

**3. Enhancing Academic Engagement**
- Track class attendance and participation to flag disengaged students.
- Develop mentoring programs and learning communities to foster a supportive academic environment.

**4. Targeted Support for High-Risk Programs**
- Students in Engineering/Technology programs are more prone to dropout.
- Evaluate curriculum load and provide additional academic counseling for these students.

**5. Personalized Support for International Students**
- Students from Latin American countries and certain nationalities are at higher risk of dropout.
- Provide cultural integration programs and language support services to improve retention.



## Project Overview

This project leverages data science to identify key drivers of student dropout at Jaya Jaya Institute. The workflow includes:

- **Business Understanding**: Analyzing the critical issue of increasing dropout rates within the institution.
- **Data Preparation**: Cleaning and preparing the dataset for machine learning analysis.
- **Modeling**: Building and interpreting two machine learning models — Random Forest and Logistic Regression — to uncover high-impact features related to student dropout.
- **Deployment**: Visualizing model insights through an interactive Business Dashboard using Metabase to support strategic decision-making.

Tools & Technologies:
- **Python** (pandas, scikit-learn, seaborn, matplotlib, SQLAlchemy)
- **MySQL** for storing cleaned and processed data
- **Metabase** for interactive dashboards (deployed via Docker)
- **VSCode** as the development environment

This project serves as a strategic initiative to help the institution design preventive measures against student dropout using actionable, data-driven insights.
