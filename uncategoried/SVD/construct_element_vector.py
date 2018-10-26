#import needed package
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS,TSNE,SpectralEmbedding,LocallyLinearEmbedding,Isomap

from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csc_matrix

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import re
import time

def Element_Vector_PCA(data_matrix_nor,vector_lenth):
    #this function can not be calculated on local computer
    
    #minus mean to prepare for PCA
    mean_1=np.mean(data_matrix_nor,axis=0)
    obj_raw_norm_center=odata_matrix_nor-mean_1

    #time consuming step, perform following 2steps on CRC front(take ~1 minute)
    C=np.dot(obj_raw_norm_center.T,obj_raw_norm_center)
    eig_val,eig_vector=eigs(C,vector_lenth)

    #get the real part(complex part is all 0)
    eig_val=np.real(eig_val)
    eig_vector=np.real(eig_vector)

    #get final value and store them in the file
    final=np.dot(obj_raw_norm_center,eig_vector)
    return_dict={"eig_value":eig_val,"eig_vector":eig_vector,"element_vector":final}
    
    return return_dict

def Element_Vector_SVD(data_matrix_nor,vector_lenth):
    a=csc_matrix(data_matrix_nor, dtype=float)
    u, s, vt = svds(a, k=vector_lenth)
    
    s_diagonal=np.diag(s)
    F=np.dot(u,s_diagonal)
    
    return_dict={"u":u,"s":s,"vt":vt,"element_vector":F}
    return return_dict


def Element_Vector_LEP(data_matrix_nor,vector_lenth,n_neighbors=None):
    #class sklearn.manifold.SpectralEmbedding(n_components=2, affinity=’nearest_neighbors’, gamma=None, 
    #                                         random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=None)
    
    #http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
    
    embedding = SpectralEmbedding(n_components=vector_lenth,n_neighbors=n_neighbors)
    data_transformed = embedding.fit_transform(data_matrix_nor)
    return data_transformed

def Construc_Element_Vector(
    pair_matrix_path="../data_set/element_pair_matrix/first_Matrix_09_13_2018.csv",
    
    n_neighbors=0,
    vector_lenth=20,
    method="LEP",
    
    output_folder_pre_path="../data_set/element_vector_generation",
    file_name="",
    
    add_time=True,
    resutl_norm=False,
    save_result=False
    ):


    #import element-envrionment pairwise matrix
    dataframe = pd.read_csv(pair_matrix_path)

    #get matrix and elements
    data_matrix = dataframe.values
    elements = np.asarray(data_matrix[:,0]).astype(str)
    data_matrix_raw = data_matrix[:,1:].astype(float)

    #do normalization of data_matrix_raw
    data_matrix_nor=normalize(data_matrix_raw, norm='l2', axis=1, copy=True)

    #different methods implementation here
    if method=="PCA":
        return_dict=Element_Vector_PCA(data_matrix_nor,vector_lenth)
        element_vector=return_dict["element_vector"]
        
    elif method=="SVD":
        return_dict=Element_Vector_SVD(data_matrix_nor,vector_lenth)
        element_vector=return_dict["element_vector"]
        
    elif method=="LEP":
        if n_neighbors==0:
            element_vector=Element_Vector_LEP(data_matrix_nor,vector_lenth)
        else:
            element_vector=Element_Vector_LEP(data_matrix_nor,vector_lenth,n_neighbors)
        return_dict={"element_vector":element_vector}
            
    else:
        print("unkonw method\n")
        return None
    
    #normalization
    if resutl_norm:
        vector_MAX=np.max(element_vector)
        vector_MIN=np.min(element_vector)

        element_vector=(element_vector-(vector_MAX+vector_MIN)/2)/(vector_MAX-vector_MIN)
    
    #build output_folder and file
    time_slice=""
    if add_time:
        time_slice+=time.asctime()
        
    #build output path
    if save_result:
        output_path=output_folder_pre_path+"/"+method+str(vector_lenth)+time_slice+".txt"
        np.savetxt(output_path,mds_transformed)
    
    return return_dict
    