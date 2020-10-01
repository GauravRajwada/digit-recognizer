# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:02:24 2020

@author: Gaurav
"""

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import seaborn as sb
import pandas as pd

model = tf.keras.models.load_model("E:/AI Application Implementation/Show/Digit_minist/handwritten_digits.model")

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('try.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        img=request.files.get("file","")
        print(kk)
    else:
        return render_template('try.html')

if __name__=="__main__":
    app.run()
