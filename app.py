import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('student_dropout_model.pkl')
scaler = joblib.load('student_dropout_scaler.pkl')

# Page config
st.set_page_config(
    page_title='Student Dropout Prediction',
    page_icon='🎓',
    layout='wide'
)

st.title('🎓 Student Dropout Prediction System')
st.markdown('Enter student details below to predict dropout risk.')
st.divider()

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Personal Info')
    marital_status         = st.selectbox('Marital Status', [1,2,3,4,5,6], help='1=Single 2=Married 3=Widower 4=Divorced 5=Facto union 6=Separated')
    gender                 = st.selectbox('Gender', [1,0], help='1=Male 0=Female')
    age_enrollment         = st.slider('Age at Enrollment', 17, 70, 20)
    international          = st.selectbox('International Student', [0,1], help='0=No 1=Yes')
    displaced              = st.selectbox('Displaced', [0,1], help='0=No 1=Yes')
    nacionality            = st.number_input('Nacionality Code', min_value=1, max_value=109, value=1)

with col2:
    st.subheader('Academic Info')
    application_mode       = st.number_input('Application Mode', min_value=1, max_value=57, value=1)
    application_order      = st.slider('Application Order', 0, 9, 1)
    course                 = st.number_input('Course Code', min_value=33, max_value=9991, value=9991)
    daytime_attendance     = st.selectbox('Attendance', [1,0], help='1=Daytime 0=Evening')
    previous_qualification = st.number_input('Previous Qualification', min_value=1, max_value=43, value=1)
    educational_needs      = st.selectbox('Educational Special Needs', [0,1], help='0=No 1=Yes')

with col3:
    st.subheader('Financial Info')
    debtor                 = st.selectbox('Debtor', [0,1], help='0=No 1=Yes')
    tuition_fees           = st.selectbox('Tuition Fees Up to Date', [1,0], help='1=Yes 0=No')
    scholarship_holder     = st.selectbox('Scholarship Holder', [0,1], help='0=No 1=Yes')
    mothers_qualification  = st.number_input("Mother's Qualification", min_value=1, max_value=44, value=1)
    fathers_qualification  = st.number_input("Father's Qualification", min_value=1, max_value=44, value=1)
    mothers_occupation     = st.number_input("Mother's Occupation", min_value=0, max_value=194, value=1)
    fathers_occupation     = st.number_input("Father's Occupation", min_value=0, max_value=194, value=1)

st.divider()
st.subheader('📚 Academic Performance')

col4, col5 = st.columns(2)

with col4:
    st.markdown('**1st Semester**')
    cu1_credited     = st.slider('Units Credited', 0, 20, 0, key='c1')
    cu1_enrolled     = st.slider('Units Enrolled', 0, 26, 6, key='e1')
    cu1_evaluations  = st.slider('Evaluations', 0, 45, 6, key='ev1')
    cu1_approved     = st.slider('Units Approved', 0, 26, 6, key='a1')
    cu1_grade        = st.number_input('Grade (0-20)', min_value=0.0, max_value=20.0, value=12.0, key='g1')
    cu1_no_eval      = st.slider('Without Evaluations', 0, 12, 0, key='ne1')

with col5:
    st.markdown('**2nd Semester**')
    cu2_credited     = st.slider('Units Credited', 0, 20, 0, key='c2')
    cu2_enrolled     = st.slider('Units Enrolled', 0, 23, 6, key='e2')
    cu2_evaluations  = st.slider('Evaluations', 0, 33, 6, key='ev2')
    cu2_approved     = st.slider('Units Approved', 0, 20, 6, key='a2')
    cu2_grade        = st.number_input('Grade (0-20)', min_value=0.0, max_value=20.0, value=12.0, key='g2')
    cu2_no_eval      = st.slider('Without Evaluations', 0, 12, 0, key='ne2')

st.divider()
st.subheader('🌍 Economic Factors')

col6, col7, col8 = st.columns(3)
with col6:
    unemployment_rate = st.number_input('Unemployment Rate', min_value=7.6, max_value=16.2, value=10.8)
with col7:
    inflation_rate    = st.number_input('Inflation Rate', min_value=-0.8, max_value=3.7, value=1.4)
with col8:
    gdp               = st.number_input('GDP', min_value=-4.06, max_value=3.51, value=1.74)

st.divider()

# Predict button
if st.button('🔍 Predict Dropout Risk', use_container_width=True):

    input_data = np.array([[
        marital_status, application_mode, application_order,
        course, daytime_attendance, previous_qualification,
        nacionality, mothers_qualification, fathers_qualification,
        mothers_occupation, fathers_occupation, displaced,
        educational_needs, debtor, tuition_fees,
        gender, scholarship_holder, age_enrollment, international,
        cu1_credited, cu1_enrolled, cu1_evaluations, cu1_approved,
        cu1_grade, cu1_no_eval,
        cu2_credited, cu2_enrolled, cu2_evaluations, cu2_approved,
        cu2_grade, cu2_no_eval,
        unemployment_rate, inflation_rate, gdp
    ]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    confidence   = round(max(probability) * 100, 2)
    dropout_prob = round(probability[1] * 100, 2)

    # Show result
    st.divider()
    if prediction == 1:
        st.error(f'⚠️ HIGH DROPOUT RISK')
        st.metric('Dropout Probability', f'{dropout_prob}%')
        st.markdown('### Recommended Actions')
        st.markdown('- Contact student immediately for counseling')
        st.markdown('- Check financial situation — offer aid if needed')
        st.markdown('- Provide academic support and tutoring')
        st.markdown('- Assign a mentor for regular follow-up')
    else:
        st.success(f'✅ LOW DROPOUT RISK')
        st.metric('Dropout Probability', f'{dropout_prob}%')
        st.markdown('### Student Status')
        st.markdown('- Student is likely to complete the course')
        st.markdown('- Continue regular academic monitoring')