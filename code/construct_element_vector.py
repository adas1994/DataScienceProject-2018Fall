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

#linear PCA and SVD
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

def Element_Vector_TSNE(data_matrix_nor,vector_lenth):
    #(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, 
    #n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, 
    #init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
    #http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    data_transformed = TSNE(n_components=vector_lenth,perplexity=4).fit_transform(data_matrix_nor)
    return data_transformed

def Element_Vector_LEP(data_matrix_nor,vector_lenth,n_neighbors=None):
    #class sklearn.manifold.SpectralEmbedding(n_components=2, affinity=’nearest_neighbors’, gamma=None, 
    #random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=None)
    
    #http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
    
    embedding = SpectralEmbedding(n_components=vector_lenth,n_neighbors=n_neighbors)
    data_transformed = embedding.fit_transform(data_matrix_nor)
    return data_transformed

def Element_Vector_ISOMAP(data_matrix_nor,vector_lenth,n_neighbors=None):
    #class sklearn.manifold.Isomap(n_neighbors=5, n_components=2, 
    #eigen_solver=’auto’, tol=0, max_iter=None, path_method=’auto’, 
    #neighbors_algorithm=’auto’, n_jobs=None
    isomap=Isomap(n_neighbors=n_neighbors,n_components=vector_lenth,n_jobs=4)
    data_transformed = isomap.fit_transform(data_matrix_nor)
    return data_transformed
    
def Element_Vector_MDS(data_matrix_nor,vector_lenth):
    #class sklearn.manifold.MDS(n_components=2, metric=True, n_init=4, 
    #max_iter=300, verbose=0, eps=0.001, n_jobs=None, random_state=None, dissimilarity=’euclidean’)
    embedding=MDS(n_components=vector_lenth,n_jobs=4)
    data_transformed = embedding.fit_transform(data_matrix_nor)
    return data_transformed
    
def Element_Vector_LLE(data_matrix_nor,vector_lenth,n_neighbors=None):
    #sklearn.manifold.locally_linear_embedding(X, n_neighbors, n_components, reg=0.001, 
    #eigen_solver=’auto’, tol=1e-06, max_iter=100, method=’standard’, hessian_tol=0.0001, 
    #modified_tol=1e-12, random_state=None, n_jobs=None)
    embedding=LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=vector_lenth,n_jobs=4)
    data_transformed = embedding.fit_transform(data_matrix_nor)
    return data_transformed

def Element_Vector_RANDOM(data_shape,method="uniform",seed=None):
    np.random.seed(seed)
    data_random=np.random.uniform(-0.5,0.5,data_shape)
    return data_random

def Construc_Element_Vector(
    pair_matrix_path="../data_set/element_pair_matrix/first_Matrix_09_13_2018.csv",
    
    n_neighbors=4,
    vector_lenth=20,
    method="LEP",
    
    output_folder_pre_path="../data_set/element_vector_generation",
    file_name="",
    
    add_time=True,
    result_norm=False,
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
            
    elif method=="TSNE":
        element_vector=Element_Vector_TSNE(data_matrix_nor,vector_lenth)
        return_dict={"element_vector":element_vector}
        
    elif method=="ISOMAP":
        if n_neighbors==0:
            element_vector=Element_Vector_ISOMAP(data_matrix_nor,vector_lenth)
        else:
            element_vector=Element_Vector_ISOMAP(data_matrix_nor,vector_lenth,n_neighbors)
        return_dict={"element_vector":element_vector}
    
    elif method=="MDS":
        if n_neighbors==0:
            element_vector=Element_Vector_MDS(data_matrix_nor,vector_lenth)
        else:
            element_vector=Element_Vector_MDS(data_matrix_nor,vector_lenth)
        return_dict={"element_vector":element_vector}
    
    elif method=="LLE":
        if n_neighbors==0:
            element_vector=Element_Vector_LLE(data_matrix_nor,vector_lenth)
        else:
            element_vector=Element_Vector_LLE(data_matrix_nor,vector_lenth,n_neighbors)
        return_dict={"element_vector":element_vector}
    elif method=="RANDOM":
        element_vector=Element_Vector_RANDOM((data_matrix_nor.shape[0],vector_lenth))
        return_dict={"element_vector":element_vector}
    
    else:
        print("unkonw method\n")
        return None
    
    #normalization
    if result_norm:
        vector_MAX=np.max(element_vector)
        vector_MIN=np.min(element_vector)

        element_vector=(element_vector-(vector_MAX+vector_MIN)/2)/(vector_MAX-vector_MIN)
    
    #build output_folder and file
    time_slice=""
    if add_time:
        time_slice+=time.asctime()
        
    #build output path
    if save_result:
        if len(file_name)>0:
            output_path=output_folder_pre_path+"/"+file_name+".txt"
        else:
            output_path=output_folder_pre_path+"/"+method+str(vector_lenth)+time_slice+".txt"
            
        np.savetxt(output_path,element_vector)
    
    return return_dict
    