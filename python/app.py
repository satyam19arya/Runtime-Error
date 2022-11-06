from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from tensorflow.keras.models import Sequential

# Keras

from keras.models import load_model
import tensorflow as tf

model_wlf=load_model('./wlf.h5')
model_cancer=load_model('./cancer.h5')
model_heart=load_model('./heart.h5')
model_stroke=load_model('./stroke (1).h5')
model_fat=load_model('./bodyfat.h5')

# Load your trained model



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras

#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')
m={'Air Pollution':0,'Alcohol use':1,'Dust Allergy':2,'chronic Lung Disease':3,'Balanced Diet':4,'Smoking':5,'Passive Smoker':6,
     'Chest Pain':7,'Coughing of Blood':8,'Fatigue':9,'Weight Loss':10,'Shortness of Breath':11,'Wheezing':12,'Dry Cough':13,'Snoring':14,
     'FRUITS_VEGGIES':15,'PLACES_VISITED':16,'CORE_CIRCLE':17,'SUPPORTING_OTHERS':18,'SOCIAL_NETWORK':19,'ACHIEVEMENT':20,
     'DAILY_STEPS':21,'LIVE_VISION':22,'SLEEP_HOURS':23,'SUFFICIENT_INCOME':24,'PERSONAL_AWARDS':25,'TIME_FOR_PASSION':26,
     'Female':27,'Male':28,'Age':29,'Weight':30,'Height':31,'Neck':32, 'Chest':33,'Abdomen':34,'Hip':35,'Thigh':36,'Knee':37,
     'Ankle':38,'Biceps':39, 'Forearm':40,'Wrist':41,'work_type_Govt_job':42,'work_type_Never_worked':43,'work_type_Private':44,
     'work_type_Self-employed':45, 'work_type_children':46,'hypertension':47,'heart_disease':48,'ever_married':49,
     'Residence_type':50,'avg_glucose_level':51,'bmi':52,'HighBP':53, 'HighChol':54, 'CholCheck':55, 'Diabetes':56,'PhysActivity':57, 'HvyAlcoholConsump':58,'AnyHealthcare':59,'Education':60,'Income':61
     }
arr=[]
for a in range(62):
    arr.append(0)
def abc(inp):
    arr.append(inp)
    if(len(arr)==62):
        model_predict(arr,model_cancer,model_fat,model_wlf,model_heart,model_stroke)



def model_predict(arr,model_cancer,model_fat,model_wlf,model_heart,model_stroke):
    cancer=model_cancer.predict([[arr[m['Air Pollution']],arr[m['Alcohol use']],arr[m['Dust Allergy']],arr[m['chronic Lung Disease']],
                                  arr[m['Balanced Diet']],arr[m['Smoking']],arr[m['Passive Smoker']],arr[m['Chest Pain']],arr[m['Coughing of Blood']]
                                     ,arr[m['Fatigue']],arr[m['Weight Loss']],arr[m['Shortness of Breath']],arr[m['Wheezing']],arr[m['Dry Cough']],arr[m['Snoring']]]])
    fat=model_fat.predict([[arr[m['Age']],arr[m['Weight']],arr[m['Neck']],arr[m['Chest']],arr[m['Abdomen']],arr[m['Hip']],
                            arr[m['Thigh']],arr[m['Knee']],arr[m['Ankle']],arr[m['Biceps']],arr[m['Forearm']],arr[m['Wrist']]]])
    wlf=0
    if(arr[m['Age']]<=20):
     wlf=model_wlf.predict([[arr[m['FRUITS_VEGGIES']],arr[m['PLACES_VISITED']],arr[m['CORE_CIRCLE']],arr[m['SUPPORTING_OTHERS']],
                             arr[m['SOCIAL_NETWORK']],arr[m['ACHIEVEMENT']],arr[m['DAILY_STEPS']],arr[m['LIVE_VISION']],arr[m['SLEEP_HOURS']],
                             arr[m['SUFFICIENT_INCOME']],arr[m['PERSONAL_AWARDS']],arr[m['TIME_FOR_PASSION']],arr[m['Female']],arr[m['Male']],0,0,0,1]])
    elif arr[m['Age']]<=35:
        wlf = model_wlf.predict(
            [[arr[m['FRUITS_VEGGIES']], arr[m['PLACES_VISITED']], arr[m['CORE_CIRCLE']], arr[m['SUPPORTING_OTHERS']],
              arr[m['SOCIAL_NETWORK']], arr[m['ACHIEVEMENT']], arr[m['DAILY_STEPS']], arr[m['LIVE_VISION']],
              arr[m['SLEEP_HOURS']],
              arr[m['SUFFICIENT_INCOME']], arr[m['PERSONAL_AWARDS']], arr[m['TIME_FOR_PASSION']], arr[m['Female']],
              arr[m['Male']], 1, 0, 0, 0]])
    elif arr[m['Age']]<=50:
        wlf = model_wlf.predict(
            [[arr[m['FRUITS_VEGGIES']], arr[m['PLACES_VISITED']], arr[m['CORE_CIRCLE']], arr[m['SUPPORTING_OTHERS']],
              arr[m['SOCIAL_NETWORK']], arr[m['ACHIEVEMENT']], arr[m['DAILY_STEPS']], arr[m['LIVE_VISION']],
              arr[m['SLEEP_HOURS']],
              arr[m['SUFFICIENT_INCOME']], arr[m['PERSONAL_AWARDS']], arr[m['TIME_FOR_PASSION']], arr[m['Female']],
              arr[m['Male']], 0, 1, 0, 0]])
    else:
        wlf = model_wlf.predict(
            [[arr[m['FRUITS_VEGGIES']], arr[m['PLACES_VISITED']], arr[m['CORE_CIRCLE']], arr[m['SUPPORTING_OTHERS']],
              arr[m['SOCIAL_NETWORK']], arr[m['ACHIEVEMENT']], arr[m['DAILY_STEPS']], arr[m['LIVE_VISION']],
              arr[m['SLEEP_HOURS']],
              arr[m['SUFFICIENT_INCOME']], arr[m['PERSONAL_AWARDS']], arr[m['TIME_FOR_PASSION']], arr[m['Female']],
              arr[m['Male']], 0, 0, 1, 0]])
    heart=model_heart.predict([[arr[m['HighBP']],arr[m['HighChol']],arr[m['CholCheck']],arr[m['bmi']],0,arr[m['Diabetes']],arr[m['PhysActivity']],
                                arr[m['FRUITS_VEGGIES']]/3,arr[m['FRUITS_VEGGIES']]/3,arr[m['HvyAlcoholConsump']],arr[m['AnyHealthcare']],arr[m['Male']],arr[m['Age']],arr[m['Education']],arr[m['Income']]]])
    stroke=model_stroke.predict([[arr[m['work_type_Govt_job']],arr[m['work_type_Never_worked']],arr[m['work_type_Private']],arr[m['work_type_Self-employed']],
                                  arr[m['work_type_children']],arr[m['Female']],arr[m['Male']],arr[m['Female']],arr[m['Age']],
                                  arr[m['hypertension']],arr[m['heart_disease']],arr[m['ever_married']] ,arr[m['Residence_type']],arr[m['avg_glucose_level']],arr[m['bmi']]]])

    g=200*heart[0]+200*stroke[0]

    g+=200-wlf[0]/5
    g+=fat[0]*4
    g+=cancer[0][0]*66+cancer[0][1]*66+cancer[0][2]*66
    return 1000-g





@app.route('/', methods=['GET','POST'])
def main():
    # Main page
    if request.method=="POST":
        jrr=[]
        for a in range(61):

            pp=request.form.get("points"+str(a+1))
            

            if(a==27):
                if pp==1:
                    jrr.append(0)
                    jrr.append(1)
                else:
                    jrr.append(1)
                    jrr.append(0)
            else:

                jrr.append(int(pp))
            

        opp=model_predict(jrr,model_cancer,model_fat,model_wlf,model_heart,model_stroke)
        return render_template('home.html',message=" Your Health Score is "+str(int(opp[0])))




    return render_template('home.html')




if __name__ == '__main__':
    app.run(debug=True)