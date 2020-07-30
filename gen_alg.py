import tensorflow as tf
import numpy as np
import tensorboard
import xlrd
import csv
from xlrd import open_workbook
import warnings
import random

def crossover(mother1,mother2):
   if random.uniform(0,1)<0.5:
       child=[mother1[0], mother2[1]]
   else:
       child = [mother2[0], mother1[1]]
   return child

def mutate(child, prob_mut, max_number):
 for c in range(0, len(child)-1):
        if random.uniform(0,1)<=prob_mut:
            child[0]=round(random.uniform(0.01,1)*100)
            child[1]=round(random.uniform(0.01,1)*1000)
 return child

def construct(X,Y, X_test, y_test,structure,):
    import datetime
    log_dir = r"C:\Users\hawet\Desktop\хацкерство\Diploma\logs\fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

    model =tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[48, ])),
    for i in range(0,structure[0]):
        model.add(tf.keras.layers.Dense(structure[1], activation = 'linear'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='Adam', loss='mse', metrics=['mse'])
    model.summary()
    model.fit(X, Y, epochs=27, callbacks=[tensorboard_callback], validation_data=(X_test, y_test))
    pred = model.predict(X)
    Y=np.array(Y)
    pred=np.reshape(pred,(65,))
    print(np.shape(Y))
    print(np.shape(pred))
    mseloss = (tf.keras.losses.mean_squared_error(Y,pred))
    mseloss=int(mseloss)
    print('Архитектура', structure)
    print('loss', mseloss)
    model.save('new_iter.h5')
    return mseloss , pred

def genetical_algorythm(X,Y,X_test,y_test):
    mating_pool=[[1, 400],
                 [30, 300],
                 [15, 400],
                 [20, 10],
                 [5,50],
                 [38,30],
                 [24,40],
                 [14,7],
                 [13,20],
                 [13,50],
                 [21,20]]
    losses=[25.5,
            35,
            40,
            20,
            30,
            38,
            24,
            14,
            13,
            13,
            21]
    print(mating_pool[1])
    print(len(losses))
    current_loss = 500
    while current_loss >= 5:
        parent1 = random.randint(0,len(losses)-1)
        parent2 = random.randint(0,len(losses)-1)
        parent_params1 = [mating_pool[parent1][0] , mating_pool[parent1][1]]
        parent_params2 = [mating_pool[parent2][0] , mating_pool[parent2][1]]
        loss1 = losses[parent1]
        loss2 = losses[parent2]
        child = crossover(parent_params1,parent_params2)
        print(child,"ребенок")
        child=mutate(child,0.3,50)
        current_loss, pred = construct(X,Y,X_test,y_test,[child[0],child[1]])
        compared_loss=loss1*loss2/2
        if current_loss < compared_loss:
            mating_pool.append(child)
            losses.append(current_loss)

