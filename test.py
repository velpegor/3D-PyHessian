import numpy as np
import torch 
from torchvision import datasets, transforms, models
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model
import matplotlib.pyplot as plt
from VGG import VGG
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

# get the model 
model = VGG(activation_type='tanh', depth='16', cut_block=5, num_classes=100, oper_order='cab')
model = nn.DataParallel(model)
checkpoints = torch.load('/home/velpegor/workspace/3D-PyHessian/AGC5e-2_VGG11_tanh_cab_seed1_cifar100_0.0005_0.1_pth.tar')
model.load_state_dict(checkpoints['state_dict'])

# change the model to eval mode to disable running stats upate
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()
model = model.cuda()

input_set = []
# for illustrate, we only use one batch to do the tutorial
for inputs, targets in test_loader:
    # we use cuda to make the computation fast
    inputs, targets = inputs.cuda(), targets.cuda()
    
    # create the hessian computation module

hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=True)

# get the top1, top2 eigenvectors
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

# lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
lams1 = np.linspace(-3.0, 3.0, 100).astype(np.float32)
lams2 = np.linspace(-3.0, 3.0, 100).astype(np.float32)

loss_list = []

# create a copy of the model
model_perb1 = VGG(activation_type='tanh', depth='16', cut_block=5, num_classes=100, oper_order='cab')
model_perb1 = nn.DataParallel(model_perb1)
checkpoints_perb1 = torch.load('/home/velpegor/workspace/3D-PyHessian/AGC5e-2_VGG11_tanh_cab_seed1_cifar100_0.0005_0.1_pth.tar')
model_perb1.load_state_dict(checkpoints_perb1['state_dict'])
model_perb1.eval()
model_perb1 = model_perb1.cuda()

model_perb2 = VGG(activation_type='tanh', depth='16', cut_block=5, num_classes=100, oper_order='cab')
model_perb2 = nn.DataParallel(model_perb2)
checkpoints_perb2 = torch.load('/home/velpegor/workspace/3D-PyHessian/AGC5e-2_VGG11_tanh_cab_seed1_cifar100_0.0005_0.1_pth.tar')
model_perb2.load_state_dict(checkpoints_perb2['state_dict'])
model_perb2.eval()
model_perb2 = model_perb2.cuda()


for lam1 in lams1:
    for lam2 in lams2:
        model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
        model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
        loss_list.append((lam1, lam2, criterion(model_perb2(inputs), targets).item()))   

loss_list = np.array(loss_list)
                         
fig = plt.figure(figsize=(8,7))
fig.set_tight_layout(True)
landscape = fig.add_subplot(111, projection='3d')
landscape.plot_trisurf(loss_list[:,0], loss_list[:,1], loss_list[:,2],alpha=0.8, cmap='viridis')
landscape.set_zlim(0,30)

                       #cmap=cm.autumn, #cmamp = 'hot')


landscape.set_title('Loss Landscape : AGC5e-2_VGG11_tanh_cab')
landscape.set_xlabel('ε_1')
landscape.set_ylabel('ε_2')
landscape.set_zlabel('Loss')

#landscape.view_init(azim=90)
landscape.dist = 10
plt.savefig('cab_AGC(train, -3.0-3.0,100,final_real).png')