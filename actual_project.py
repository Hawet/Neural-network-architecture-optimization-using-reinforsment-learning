import tensorflow as tf
import numpy as np
import tensorboard
import xlrd
import csv
from xlrd import open_workbook
import warnings
warnings.filterwarnings('ignore')
Y=np.array([],dtype=float)
X=np.zeros((1,48),dtype=float)

import datetime
#log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir=r"C:\Users\hawet\Desktop\хацкерство\Diploma\logs\fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

import sklearn.model_selection as sk


import matplotlib.pyplot as plt
h=np.array([])
wb = open_workbook(r'C:\Users\hawet\Desktop\хацкерство\Diploma\diplom_normalised.xlsx')
for s in wb.sheets():
  for row in range(s.nrows):
      for i in range(s.ncols):
          if i==0:
              Y=np.append(Y, s.cell(row, i).value)
          else:
              h=np.append(h, s.cell(row, i).value)
      X=np.vstack((X,h))
      h=np.array([])

X=np.delete(X,0,axis=0)
X_train, X_test, y_train, y_test = sk.train_test_split(X,Y,test_size=0.10, random_state = 42)

from gen_alg import *
#mselos, pred =construct(X,Y,X_test,y_test,[40, 15])
import keras_applications
new_model = tf.keras.models.load_model('new_iter.h5')
pred=new_model.predict(X)
new_model.summary()
import pandas as pd
import matplotlib.pyplot as plt



dti = pd.date_range('2003-07-1', periods=len(Y), freq='Q')


plt.plot(dti,Y,dti,pred,linewidth=2)
plt.ylabel("Мдрд. сомов")
plt.legend(['Реальные значения', 'Модельные значения'])
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.show()


seas = np.array([],dtype=float)
real_gdp = np.array([],dtype=float)

wb = open_workbook(r'C:\Users\hawet\Desktop\хацкерство\Diploma\season.xlsx')
for s in wb.sheets():
  for row in range(s.nrows):
      for i in range(s.ncols):
          if i==0:
              seas=np.append(seas, s.cell(row, i).value)
          else:
              real_gdp=np.append(real_gdp, s.cell(row, i).value)


seas_pred=[]
for i in range(0,len(seas)):
    seas_pred.append(pred[i]/seas[i])

seas_for_forecast=np.array([],dtype=float)
X_forecast=np.zeros((1,48),dtype=float)
h=np.array([],dtype=float)
wb = open_workbook(r'C:\Users\hawet\Desktop\хацкерство\Diploma\seas_forecast.xlsx')
for s in wb.sheets():
  for row in range(s.nrows):
      for i in range(s.ncols):
          if i==0:
              seas_for_forecast=np.append(seas_for_forecast, s.cell(row, i).value)
          else:
              h=np.append(h, s.cell(row, i).value)
      X_forecast=np.vstack((X_forecast,h))
      h=np.array([])


forecast=new_model.predict(X_forecast)
seas_forecast=[]

for i in range(0,len(seas_for_forecast)):
    seas_forecast.append(forecast[i]/seas_for_forecast[i])

forecast_dates = pd.date_range('2019q3', periods=10, freq='Q')

plt.plot(forecast_dates , seas_forecast , linewidth=2)
plt.ylabel("Мдрд. сомов")
plt.legend(['Прогноз'])
plt.xticks(rotation='45')
plt.show()
print(seas_forecast,"прогноз")

plt.plot(dti,real_gdp,dti,seas_pred,linewidth=2)
plt.ylabel("Мдрд. сомов")
plt.legend(['Реальные значения', 'Модельные значения'])
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.show()