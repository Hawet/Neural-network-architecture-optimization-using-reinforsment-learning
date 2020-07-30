import tensorflow as tf
import numpy as np
import tensorboard
import xlrd
import csv
from xlrd import open_workbook
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import tensorboard
import xlrd
import csv
from xlrd import open_workbook
import warnings
import random
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
genetical_algorythm(X,Y,X_test,y_test)













