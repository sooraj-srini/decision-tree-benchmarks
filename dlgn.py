import numpy as np
from itertools import product as cartesian_prod

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import sys

from sklearn.svm import SVC

class DLGN_FC(nn.Module):
	def __init__(self, input_dim=None, output_dim=None, num_hidden_nodes=[], beta=30, mode='pwc'):		
		super(DLGN_FC, self).__init__()
		self.num_hidden_layers = len(num_hidden_nodes)
		self.beta=beta  # Soft gating parameter
		self.mode = mode
		self.num_nodes=[input_dim]+num_hidden_nodes+[output_dim]
		self.gating_layers=nn.ModuleList()
		self.value_layers=nn.ModuleList()
		
		for i in range(self.num_hidden_layers+1):
			if i!=self.num_hidden_layers:
				temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1])
				# a = temp.weight.detach() 
				# a /= a.norm(dim=1, keepdim=True)
				self.gating_layers.append(temp)
			temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1], bias=False)
			# a = temp.weight.detach()
			# a /= a.norm(dim=1, keepdim=True)
			self.value_layers.append(temp)


	def set_parameters_with_mask(self, to_copy, parameter_masks):
		# self and to_copy are DLGN_FC objects with same architecture
		# parameter_masks is compatible with dict(to_copy.named_parameters())
		for (name, copy_param) in to_copy.named_parameters():
			copy_param = copy_param.clone().detach()
			orig_param  = self.state_dict()[name]
			if name in parameter_masks:
				param_mask = parameter_masks[name]>0
				orig_param[param_mask] = copy_param[param_mask]
			else:
				orig_param = copy_param.data.detach()
	

								

	def return_gating_functions(self):
		effective_weights = []
		effective_biases =[]
		for i in range(self.num_hidden_layers):
			curr_weight = self.gating_layers[i].weight.detach()
			curr_bias = self.gating_layers[i].bias.detach()
			if i==0:
				effective_weights.append(curr_weight)
				effective_biases.append(curr_bias)
			else:
				effective_biases.append(torch.matmul(curr_weight,effective_biases[-1])+curr_bias)
				effective_weights.append(torch.matmul(curr_weight,effective_weights[-1]))
		return effective_weights, effective_biases
		# effective_weights (and effective biases) is a list of size num_hidden_layers
							

	def forward(self, x):
		gate_scores=[x]

		for el in self.parameters():
			if el.is_cuda:
				device = torch.device('cuda')
			else:
				device = torch.device('cpu')
		if self.mode=='pwc':
			values=[torch.ones(x.shape).to(device)]
		else:
			values=[x]
		
		for i in range(self.num_hidden_layers):
			gate_scores.append(self.gating_layers[i](gate_scores[-1]))
			curr_gate_on_off = torch.sigmoid(self.beta * gate_scores[-1])
			values.append(self.value_layers[i](values[-1])*curr_gate_on_off)
		values.append(self.value_layers[self.num_hidden_layers](values[-1]))
		# Values is a list of size 1+num_hidden_layers+1
		#gate_scores is a list of size 1+num_hidden_layers
		return values,gate_scores

#@title Train DLGN model
def train_dlgn (DLGN_obj, train_data_curr,vali_data_curr,test_data_curr,
				train_labels_curr,test_labels_curr,vali_labels_curr,num_epoch=1,
				parameter_mask=dict()):
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DLGN_obj.to(device)

	criterion = nn.CrossEntropyLoss()




	optimizer = optim.Adam(DLGN_obj.parameters(), lr=lr)



	train_data_torch = torch.Tensor(train_data_curr)
	vali_data_torch = torch.Tensor(vali_data_curr)
	test_data_torch = torch.Tensor(test_data_curr)

	train_labels_torch = torch.tensor(train_labels_curr, dtype=torch.int64)
	test_labels_torch = torch.tensor(test_labels_curr, dtype=torch.int64)
	vali_labels_torch = torch.tensor(vali_labels_curr, dtype=torch.int64)

	num_batches = no_of_batches
	batch_size = len(train_data_curr)//num_batches
	losses=[]
	DLGN_obj_store = []
	best_vali_error = len(vali_labels_curr)
	

	# print("H3")
	# print(DLGN_params)
	train_losses = []
	running_loss = 0.7*num_batches # initial random loss = 0.7 
	for epoch in tqdm(range(saved_epochs[-1])):  # loop over the dataset multiple times
		if epoch in saved_epochs:
			DLGN_obj_copy = deepcopy(DLGN_obj)
			DLGN_obj_copy.to(torch.device('cpu'))
			DLGN_obj_store.append(DLGN_obj_copy)
			train_losses.append(running_loss/num_batches)
			if running_loss/num_batches < 1e-5:
				break
		running_loss = 0.0
		for batch_start in range(0,len(train_data_curr),batch_size):
			if (batch_start+batch_size)>len(train_data_curr):
				break
			optimizer.zero_grad()
			inputs = train_data_torch[batch_start:batch_start+batch_size]
			targets = train_labels_torch[batch_start:batch_start+batch_size].reshape(batch_size)
			inputs = inputs.to(device)
			targets = targets.to(device)
			values,gate_scores = DLGN_obj(inputs)
			outputs = torch.cat((-1*values[-1], values[-1]), dim=1)
			loss = criterion(outputs, targets)			
			loss.backward()
			for name,param in DLGN_obj.named_parameters():
				parameter_mask[name] = parameter_mask[name].to(device)
				param.grad *= parameter_mask[name]   
			optimizer.step()
			running_loss += loss.item()    
		losses.append(running_loss/num_batches)
		inputs = vali_data_torch.to(device)
		targets = vali_labels_torch.to(device)
		values,gate_scores =DLGN_obj(inputs)
		vali_preds = torch.cat((-1*values[-1], values[-1]), dim=1)
		vali_preds = torch.argmax(vali_preds, dim=1)
		vali_error= torch.sum(targets!=vali_preds)
		if vali_error < best_vali_error:
			DLGN_obj_return = deepcopy(DLGN_obj)
			best_vali_error = vali_error
	plt.figure()
	plt.title("DLGN loss vs epoch")
	plt.plot(losses)
	if not os.path.exists('figures'):
		os.mkdir('figures')

	filename = 'figures/'+filename_suffix +'.pdf'
	plt.savefig(filename)
	DLGN_obj_return.to(torch.device('cpu'))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	return train_losses, DLGN_obj_return, DLGN_obj_store