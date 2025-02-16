import pandas as pd
import gradio as gr
import joblib


le=joblib.load('le_col.pkl')
std=joblib.load('std_col.pkl')
lr=joblib.load('model.pkl')


le_col=['Family History','Smoking Status','Diabetes Risk']
std_col=['Age','BMI','Blood Pressure','Physical Activity (hours/week)']









def predictio_model_C(age,bmi,bp,pa,fh,ss):
    try:
        input_data=pd.DataFrame({
            'Age':[age],
            'BMI':[bmi],
            'Blood Pressure':[bp],
            'Physical Activity (hours/week)':[pa],
            'Family History':[fh],
            'Smoking Status':[ss]
        })
        for col in['Family History','Smoking Status']:
            input_data[col]=le[col].transform(input_data[col])
        input_data[std_col]=std.transform(input_data[std_col])
        prediction=lr.predict(input_data)
        if prediction[0]==1:
            return 'Yes, have a diabets'
        else:
            return 'No diabets'
    except Exception as e:
        return str(e)
gr.Interface(
    fn=predictio_model_C,
    inputs=[
        gr.Number(label='Age'),
        gr.Number(label='BMI'),
        gr.Number(label='Blood Pressure'),
        gr.Number(label='Physical Activity (hours/week)'),
        gr.Dropdown(choices=['Yes','No'],label='Family History'),
        gr.Dropdown(choices=['Yes','No'],label='Smoking Status')
    ],
    outputs=gr.Textbox(label='Prediction')
).launch()