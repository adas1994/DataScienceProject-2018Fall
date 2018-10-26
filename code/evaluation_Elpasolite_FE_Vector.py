#load necessary package
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd

def two_layer_model(X,y,seed=6,patience=100):
    #split the dataset into old-out 10 to (80%,10%,10% for training/validation/test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    input_dim=X.shape[1]
    
    callback=EarlyStopping( monitor='val_loss',
                            min_delta=0,
                            patience=patience,
                            verbose=0, mode='auto')

    model = Sequential()
    model.add(Dense(10, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(lr=0.01,decay=1e-4),loss='mae',metrics=['mae'])
    
    history=model.fit(X_train, y_train, validation_data=[X_val,y_val],
                      epochs=1000,batch_size=32,callbacks=[callback])
    
    score = model.evaluate(X_test, y_test)
    
    return_dict={"history":history,"test":score[0]}
    
    return return_dict


def evaluation_Elpasolite_FE_Vector(hold_out_N=10,
    
    elap_formation_energy_path="../data_set/evaluation_dataset/FE_elapsolite.txt",
    input_file_path="../data_set/test_output/LPE_input_test.txt",
    
    output_file_path="../data_set/test_output/Evaluation_test.txt",
    save_result=False
    ):


    
    #hold number 10 is not implemented yet
    X=np.loadtxt(input_file_path)

    data = pd.read_csv(elap_formation_energy_path,index_col=0)
    y=data["formation_energy"]

    #10-hold out method
    hold_out_N=hold_out_N
    history_list=[]
    test_list=[]

    for i in range(hold_out_N):
        inter_dict=two_layer_model(X,y,seed=i)
        history_list.append(inter_dict["history"])
        test_list.append(inter_dict["test"])

    if save_result:
        np.savetxt(output_file_path,np.array(test_list))
        
    return_dict={"history_list":history_list,"test_list":test_list}
    return return_dict


