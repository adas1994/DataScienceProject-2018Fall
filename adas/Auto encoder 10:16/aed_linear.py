import torch.nn  as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import argparse
import torch,sys
import pandas as pd
from sklearn import preprocessing
'''
parser = argparse.ArgumentParser()
parser.add_argument('inputfile',help='input data-matrix file')
parser.add_argument('num_elements',help='number of elements',type=int)
parser.add_argument('num_env',help='number of env',type=int)
parser.add_argument('latent_dim',help='no. of dimensions of the encoded latent vector',type=int)
#parser.add_argument('num_hiddenlayers',help='number of hidden layers',type=int)
#parser.add_argument('do_gridSearch',help='Whether to do grid search for hyper-parameter optim.',type=bool)
parser.add_argument('mode',help='training or test or eval mode;use 1,2 or3',type=int)
args = parser.parse_args()

input_file = args.inputfile
input_dim = args.num_env
num_ele = args.num_elements
#num_H = args.num_hiddenlayers
l_dim = args.latent_dim
#do_gridSearch = args.do_gridSearch
mode = args.mode
'''
input_file = "first_Matrix_09_13_2018.csv"
dataframe = pd.read_csv(input_file)
data_matrix = dataframe.as_matrix()

elements = np.asarray(data_matrix[:,0]).astype(str)
num_elements = len(elements)
data_matrix = data_matrix[:,1:].astype(float)
input_dim = data_matrix.shape[1]

'''if np.isnan(data_matrix).any():
    print("NaN exists")
elif not np.isfinite(data_matrix).all():
    print("Infinity exists")
count_infs = 0
for i in range(num_elements):
    for j in range(input_dim):
        if not np.isfinite(data_matrix[i,j]):
            print(i,j)
            count_infs = count_infs+1
print(count_infs)'''

data_matrix_unscaled = data_matrix.copy()
data_matrix = preprocessing.scale(data_matrix)
l_dim = 20
h1 = int(float(input_dim)/10.)
h2 = int(float(h1)/10.)
h3 = int(float(h2)/2.)
h4 = int(float(h2)/2.)
h5 = 200#int(float(h3)/10.)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('input_vec',nn.Linear(input_dim,h1))
        #self.encoder.add_module('relu1',nn.ReLU(True))
        self.encoder.add_module('1st_bn',nn.BatchNorm1d(h1))
        self.encoder.add_module('h1',nn.Linear(h1,h2))
        #self.encoder.add_module('relu2',nn.ReLU(True))
        self.encoder.add_module('2nd_bn',nn.BatchNorm1d(h2))
        self.encoder.add_module('h2',nn.Linear(h2,h3))
        #self.encoder.add_module('relu3',nn.ReLU(True))
        self.encoder.add_module('3rd_bn',nn.BatchNorm1d(h3))
        self.encoder.add_module('h3',nn.Linear(h3,h4))
        #self.encoder.add_module('relu4',nn.ReLU(True))
        self.encoder.add_module('4th_bn',nn.BatchNorm1d(h4))
        self.encoder.add_module('h4',nn.Linear(h4,h5))
        #self.encoder.add_module('relu5',nn.ReLU(True))
        self.encoder.add_module('5th_bn',nn.BatchNorm1d(h5))
        self.encoder.add_module('latent_vec',nn.Linear(h5,l_dim))
        #self.add_module('tanh',nn.Tanh())

    def forward(self,x):
        x_latent = self.encoder(x)
        return x_latent


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential()
        self.decoder.add_module('latent_vec',nn.Linear(l_dim,h5))
        #self.decoder.add_module('relu5',nn.ReLU(True))
        self.decoder.add_module('5th_bn',nn.BatchNorm1d(h5))
        self.decoder.add_module('h4',nn.Linear(h5,h4))
        #self.decoder.add_module('relu4',nn.ReLU(True))
        self.decoder.add_module('4th_bn',nn.BatchNorm1d(h4))
        self.decoder.add_module('h3',nn.Linear(h4,h3))
        #self.decoder.add_module('relu3',nn.ReLU(True))
        self.decoder.add_module('3rd_bn',nn.BatchNorm1d(h3))
        self.decoder.add_module('h2',nn.Linear(h3,h2))
        #self.decoder.add_module('relu2',nn.ReLU(True))
        self.decoder.add_module('2nd_bn',nn.BatchNorm1d(h2))
        self.decoder.add_module('h1',nn.Linear(h2,h1))
        #self.decoder.add_module('relu1',nn.ReLU(True))
        self.decoder.add_module('1st_bn',nn.BatchNorm1d(h1))
        self.decoder.add_module('reco',nn.Linear(h1,input_dim))
        self.decoder.add_module('norm',nn.BatchNorm1d(input_dim))

    def forward(self,x_latent):
            x_recon = self.decoder(x_latent)
            return x_recon

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        x_l = self.encoder.forward(x)
        x_r = self.decoder.forward(x_l)
        return x_r,x_l
num_epochs = 200
batch_size = num_elements
learning_rate = 2.5*1e-3

x_input = torch.from_numpy(data_matrix).type(torch.FloatTensor)
Encoder_model = Encoder()
Decoder_model = Decoder()
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
model.parameters(), lr=learning_rate, weight_decay=1e-5)


dataloader = DataLoader(x_input, batch_size=batch_size, shuffle=True)
epoch_loss = []
saved_encoded_rep, inter_rep = None,None
for epoch in range(num_epochs):
    for data in dataloader:
        data = Variable(data)
        # ===================forward=====================
        output,encoded_input = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch==num_epochs-1):
            saved_encoded_rep = encoded_input.detach().numpy().copy()
        
            
    # ===================log========================
    epoch_loss.append([epoch,loss.data[0].numpy()])
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './sim_autoencoderLin.pth')
print(type(epoch_loss))
epoch_loss = np.asarray(epoch_loss).astype(float)
np.savetxt("epoch_lossLin.txt",epoch_loss,delimiter='\t')
np.savetxt("encoded_repLin.txt",saved_encoded_rep,delimiter='\t')
saved_encoded_rep = saved_encoded_rep.reshape(num_elements,l_dim)
print(saved_encoded_rep.shape)
