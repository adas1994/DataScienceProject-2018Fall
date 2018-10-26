from sklearn.metrics.pairwise import euclidean_distances as ed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS,TSNE,SpectralEmbedding,LocallyLinearEmbedding,Isomap
import pandas as pd
import re

input_file = "first_Matrix_09_13_2018.csv"
dataframe = pd.read_csv(input_file)
data_matrix = dataframe.values

elements = np.asarray(data_matrix[:,0]).astype(str)
num_elements = len(elements)
data_matrix = data_matrix[:,1:].astype(float)
input_dim = data_matrix.shape[1]

# Converting Raw Sparse Vectors to 20 and 30 dimensional embeddings using Laplace Eigenmap
embedding_20dim = SpectralEmbedding(n_components=20)
embedding_30dim = SpectralEmbedding(n_components=30)
x_transformed1 = embedding_20dim.fit_transform(data_matrix)
x_transformed2 = embedding_20dim.fit_transform(data_matrix)

# Converting 20/30-dimensional embedding to 2/4dimensional embedding for scatterplot
embedding_2d = TSNE(n_components=2,perplexity=4)
embedding_4d = TSNE(n_components=4,perplexity=4)
TSNEprojection2d = embedding_2d.fit_transform(x_transformed1)#x_transformed2
#TSNEprojection4d = embedding_4d.fit_transform(x_transformed1)#x_transformed2
np.savetxt("tsne_embedding.txt",TSNEprojection2d,delimiter='\t')
# Other nonlinear embedding methods

isomap = Isomap(n_neighbors=4,n_components=20,n_jobs=4)
isomap_transformedX = isomap.fit_transform(data_matrix)

mds = MDS(n_components=20)
mds_transformedX = mds.fit_transform(data_matrix)

hLLE = LocallyLinearEmbedding(n_neighbors=4,n_components=20,max_iter=500)
hLLE_transformedX = hLLE.fit_transform(data_matrix)
 





#np.savetxt("laplace_eigenmap20d.txt",x_transformed1,delimiter='\t')
fe_file = np.genfromtxt("FE_TRUE.txt",dtype=str)
elements = np.genfromtxt("elements.txt",dtype=str)
formulae,formation_energy = fe_file[:,0],np.asfarray(fe_file[:,1])
chem = []
for i in formulae:
    i = re.split('1|2|6',i)
    i.pop()
    chem.append(i)

vector_list1 = []
for formula_id in range(len(chem)):
    formula = chem[formula_id]
    el1,el2,el3,el4 = formula[0],formula[1],formula[2],formula[3]
    sp1,sp2,sp3,sp4 = np.where(elements==el1)[0],np.where(elements==el2)[0],np.where(elements==el3)[0],np.where(elements==el4)[0]
    sp1,sp2,sp3,sp4=sp1[0],sp2[0],sp3[0],sp4[0]
    v1,v2,v3,v4 = x_transformed1[sp1],x_transformed1[sp2],x_transformed1[sp3],x_transformed1[sp4]
    v = np.concatenate((v1,v2,v3,v4))
    form_e = formation_energy[formula_id]
    v = np.concatenate((v,np.array([form_e])))
    vector_list1.append(v)

vector_list1 = np.asfarray(vector_list1)
np.savetxt("inputvecforstep2_fromLaplace.txt",vector_list1,delimiter='\t')
print('laplace saved')
vector_list1 = []
for formula_id in range(len(chem)):
    formula = chem[formula_id]
    el1,el2,el3,el4 = formula[0],formula[1],formula[2],formula[3]
    sp1,sp2,sp3,sp4 = np.where(elements==el1)[0],np.where(elements==el2)[0],np.where(elements==el3)[0],np.where(elements==el4)[0]
    sp1,sp2,sp3,sp4=sp1[0],sp2[0],sp3[0],sp4[0]
    v1,v2,v3,v4 = isomap_transformedX[sp1],isomap_transformedX[sp2],isomap_transformedX[sp3],isomap_transformedX[sp4]
    v = np.concatenate((v1,v2,v3,v4))
    form_e = formation_energy[formula_id]
    v = np.concatenate((v,np.array([form_e])))
    vector_list1.append(v)

vector_list1 = np.asfarray(vector_list1)
np.savetxt("inputvecforstep2_fromIsoMap.txt",vector_list1,delimiter='\t')
print('isomap saved')
vector_list1 = []
for formula_id in range(len(chem)):
    formula = chem[formula_id]
    el1,el2,el3,el4 = formula[0],formula[1],formula[2],formula[3]
    sp1,sp2,sp3,sp4 = np.where(elements==el1)[0],np.where(elements==el2)[0],np.where(elements==el3)[0],np.where(elements==el4)[0]
    sp1,sp2,sp3,sp4=sp1[0],sp2[0],sp3[0],sp4[0]
    v1,v2,v3,v4 = mds_transformedX[sp1],mds_transformedX[sp2],mds_transformedX[sp3],mds_transformedX[sp4]
    v = np.concatenate((v1,v2,v3,v4))
    form_e = formation_energy[formula_id]
    v = np.concatenate((v,np.array([form_e])))
    vector_list1.append(v)

vector_list1 = np.asfarray(vector_list1)
np.savetxt("inputvecforstep2_fromMDS.txt",vector_list1,delimiter='\t')
print('mds saved')
vector_list1 = []
for formula_id in range(len(chem)):
    formula = chem[formula_id]
    el1,el2,el3,el4 = formula[0],formula[1],formula[2],formula[3]
    sp1,sp2,sp3,sp4 = np.where(elements==el1)[0],np.where(elements==el2)[0],np.where(elements==el3)[0],np.where(elements==el4)[0]
    sp1,sp2,sp3,sp4=sp1[0],sp2[0],sp3[0],sp4[0]
    v1,v2,v3,v4 = hLLE_transformedX[sp1],hLLE_transformedX[sp2],hLLE_transformedX[sp3],hLLE_transformedX[sp4]
    v = np.concatenate((v1,v2,v3,v4))
    form_e = formation_energy[formula_id]
    v = np.concatenate((v,np.array([form_e])))
    vector_list1.append(v)

vector_list1 = np.asfarray(vector_list1)
np.savetxt("inputvecforstep2_fromhLLE.txt",vector_list1,delimiter='\t')
print('hessian LLE saved')
