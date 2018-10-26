import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import pymatgen as mg
from pymatgen.core import periodic_table, Element

element_file = open("elements.txt","r")
latent_file = np.genfromtxt("encoded_rep.txt",delimiter='\t')
elements = []
for line in element_file:
    elements.append(line.rstrip('\n'))
    

svd = TruncatedSVD(n_components=4,algorithm="randomized",n_iter=50)
svdTransformed_lat = svd.fit_transform(latent_file)
pca = PCA(n_components=4)
pcaTransformed_lat = pca.fit_transform(latent_file)

first2svd = svdTransformed_lat[:,:2]
last2svd  = svdTransformed_lat[:,2:]
first2pca = pcaTransformed_lat[:,:2]
last2pca  = pcaTransformed_lat[:,2:]

test=["Li","Be","B","C","N","O","F"]
group_main_list=[]
for element_string in test:
    Ele=Element(element_string)
    group_main_list.append(Ele.group)

elements_list_main=[]
for group_number in group_main_list:
    #creat subset
    elements_sublist=[]
    for raw_element in elements:
        if Element(raw_element).group==group_number:
            if Element(raw_element).number<57 or Element(raw_element).number>71:
                elements_sublist.append(raw_element)
    #sort subset
    elements_sublist.sort(key=lambda x:Element(x).number)
    #add subset to elements_list_main
    elements_list_main+=elements_sublist
    #elements_list_main.append("{}".format(group_number))


#apply the raw number to mapping value
raw_number_list=[]
for element_select in elements_list_main:
    for raw_number in range(len(elements)):
        if element_select==elements[raw_number]:
            raw_number_list.append(raw_number)

data_main = first2pca[raw_number_list,:]
data_main.shape
data_main_flip=np.flip(data_main,axis=1)
fig=plt.imshow(data_main_flip, interpolation='nearest', cmap="rainbow")
plt.show(fig)

fig = plt.figure()
ax = fig.add_subplot(111)



ax.scatter(data_main[:,0],data_main[:,1],label='1st two svd components')
plt.xlim(-1,1)
plt.ylim(-1,1)
for i in range(len(elements_list_main)):
    txt = elements_list_main[i]
    ax.annotate(txt, (data_main[i,0], data_main[i,1]))
plt.savefig('elements_scatterplot.pdf')
#plt.show()



