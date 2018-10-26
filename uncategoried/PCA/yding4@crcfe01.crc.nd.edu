import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigs

#retrieve element-envrionment pair matrix from file
obj=pd.read_csv('first_Matrix_09_13_2018.csv',index_col=0)

#L2 normalize in atrribution
obj_raw_norm=normalize(obj, norm='l2', axis=1)

#minus mean to prepare for PCA
mean_1=np.mean(obj_raw_norm,axis=0)
obj_raw_norm_center=obj_raw_norm-mean_1

#time consuming step, perform following 2steps on CRC front(take ~1 minute)
C=np.dot(obj_raw_norm_center.T,obj_raw_norm_center)
eig_val,eig_vector=eigs(C,k=30)

#get the real part(complex part is all 0)
eig_val=np.real(eig_val)
eig_vector=np.real(eig_vector)

#get final value and store them in the file
final=np.dot(obj_raw_norm_center,eig_vector)
np.savetxt("PCA_vector.csv",final,delimiter=",")

np.savetxt("PCA_Eigen_Value.txt",eig_val,delimiter="\t")
