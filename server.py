import pickle
import streamlit as st
import json
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

file = open("model/randomforest.pickle" ,'rb')
rf_classifier = pickle.load(file)


def load_lottifiel(filepath:str):
    with open(filepath,'r') as f:return json.load(f)

st_lottie( load_lottifiel("Animation - 1704726019196.json"),height=250)



file = open("pipeline/label encoding/prognosis_label.pickle" ,'rb')
test_result_label_encoding = pickle.load(file)

columns = ['itching', 'nodal_skin_eruptions', 'chills', 'joint_pain',
       'stomach_pain', 'vomiting', 'spotting_ urination', 'fatigue',
       'weight_loss', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness',
       'sweating', 'dehydration', 'headache', 'yellowish_skin', 'dark_urine',
       'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
       'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellowing_of_eyes',
       'swelled_lymph_nodes', 'malaise', 'chest_pain', 'neck_pain',
       'extra_marital_contacts', 'slurred_speech', 'knee_pain',
       'muscle_weakness', 'movement_stiffness', 'loss_of_balance',
       'unsteadiness', 'bladder_discomfort', 'continuous_feel_of_urine',
       'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
       'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
       'abnormal_menstruation', 'dischromic _patches', 'family_history',
       'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
       'receiving_blood_transfusion', 'coma', 'stomach_bleeding', 'scurring',
       'blister', 'yellow_crust_ooze']

selected_option = []
for option in columns:
   option_modi = ""
   try:
      for h in option.split("_"):
         option_modi +=f"{h} "
   except:option_modi = option
   select = st.radio(f"Have you {option_modi}?", ["Yes", "No"],horizontal=True)
   selected_option.append(select)
   

if st.button('Submit'):
   selected_option_modi = []
   for i in selected_option:
      if i == "Yes":selected_option_modi.append(1)
      else:selected_option_modi.append(0)
   result = rf_classifier.predict([selected_option_modi])[0]
   result = test_result_label_encoding.inverse_transform([result])[0]
   st.write(f'Test Results : {result}')