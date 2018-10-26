#load necessary package
import numpy as np
import pandas as pd
from pymatgen import Element,Composition

def get_Elpasolite_FE_Vector(
    element_vector_path="../data_set/element_vector_generation/laplace_eigenmap/laplace_eigenmap_20.txt",
    vector_length=20,
    Elpasolite_FE_path="../data_set/evaluation_dataset/FE_elapsolite.txt",

    element_list_path="../data_set/element_list/elements.txt",
                                 
    test_output_path="../data_set/test_output/",
    input_file_name="LPE_input_test.txt",  
    
    result_norm=False,
    save_result=False
                            ):

    ##do max-min normalization for vector matrix?? if do ,then would be 0-1？feel bad
    #can do like x_=(x-(max+min)/2)/(max-min)/2 to put x_ to [-1,1]

    #load elaposolite formation energy file
    data = pd.read_csv(Elpasolite_FE_path,index_col=0)

    #load elements_list
    element_list_path="../data_set/element_list/elements.txt"
    with open(element_list_path, "r") as text_file:
        a=text_file.read()
    element_list=a.split()

    #load element vector(80*vector_length, 80*20 by default)
    vector_raw=np.loadtxt(element_vector_path)
    
    ##build output vector(5628*(4*vector_length) 5629*80 by default
    #build path
    input_file_path=test_output_path+input_file_name
    
    #ele_vector_length=20
    #compound_list_length=5628
    vector_input=np.zeros((len(data),vector_length*len(Composition(data["compound"][0]))))
    for i in range(len(data["compound"])):
        compound=Composition(data["compound"][i])
        ele_index=0
        for ele in compound:
            for target_ele_index in range(len(element_list)):
                if str(ele)==element_list[target_ele_index]:
                    vector_input[i,ele_index*vector_length:(ele_index+1)*vector_length]=vector_raw[target_ele_index,:]
                    ele_index+=1
    
    #whether do normlization of resulting vector
    if(result_norm):
        #normalization for LPE_vector should do after getting the purpose matrix
        vector_MAX=np.max(vector_input)
        vector_MIN=np.min(vector_input)

        vector_input=(vector_input-(vector_MAX+vector_MIN)/2)/(vector_MAX-vector_MIN)

    #save dataset to target folder/file
    if (save_result):
        np.savetxt(input_file_path,vector_input)
    
    #return result
    return vector_input
