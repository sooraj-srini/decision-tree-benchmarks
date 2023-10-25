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
from data_gen import set_torchseed



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
class trainDLGN:
	def __init__(self, args):
		self.lr = args.lr
		self.num_layer = args.numlayer
		self.num_neuron = args.numnodes
		self.beta = args.beta
		self.no_of_batches=10 
		self.weight_decay=0.0
		self.num_hidden_nodes=[self.num_neuron]*self.num_layer
		filename_suffix = str(self.num_layer)
		filename_suffix += "_"+str(self.num_neuron)
		filename_suffix += "_"+str(int(self.beta))
		filename_suffix += "_"+format(self.lr,".1e")
		self.filename_suffix = filename_suffix
		self.input_dim = args.input_dim


	def train_dlgn (self, DLGN_obj, train_data_curr,vali_data_curr,test_data_curr,
					train_labels_curr,test_labels_curr,vali_labels_curr,num_epoch=1,
					parameter_mask=dict()):
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		DLGN_obj.to(device)

		criterion = nn.CrossEntropyLoss()




		optimizer = optim.Adam(DLGN_obj.parameters(), lr=self.lr)



		train_data_torch = torch.Tensor(train_data_curr)
		vali_data_torch = torch.Tensor(vali_data_curr)
		test_data_torch = torch.Tensor(test_data_curr)

		train_labels_torch = torch.tensor(train_labels_curr, dtype=torch.int64)
		test_labels_torch = torch.tensor(test_labels_curr, dtype=torch.int64)
		vali_labels_torch = torch.tensor(vali_labels_curr, dtype=torch.int64)

		num_batches = self.no_of_batches
		batch_size = len(train_data_curr)//num_batches
		losses=[]
		DLGN_obj_store = []
		best_vali_error = len(vali_labels_curr)
		

		# print("H3")
		# print(DLGN_params)
		train_losses = []
		running_loss = 0.7*num_batches # initial random loss = 0.7 
		saved_epochs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,64,128,256,512,1024,2048, 
                4096, 8192, 16384, 32768]
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

		filename = 'figures/'+self.filename_suffix +'.pdf'
		plt.savefig(filename)
		DLGN_obj_return.to(torch.device('cpu'))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		return train_losses, DLGN_obj_return, DLGN_obj_store


	def train(self,train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
		set_torchseed(6675)
		# set_torchseed(5449)
		DLGN_init= DLGN_FC(input_dim=self.input_dim, output_dim=1, num_hidden_nodes=self.num_hidden_nodes, beta=self.beta)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		train_parameter_masks=dict()
		for name,parameter in DLGN_init.named_parameters():
			if name[:5]=="value_"[:5]:
				train_parameter_masks[name]=torch.ones_like(parameter) # Updating all value network layers
			if name[:5]=="gating_"[:5]:
				train_parameter_masks[name]=torch.ones_like(parameter)
			train_parameter_masks[name].to(device)

		set_torchseed(5000)
		train_losses, DLGN_obj_final, DLGN_obj_store = self.train_dlgn(train_data_curr=train_data,
													vali_data_curr=vali_data,
													test_data_curr=test_data,
													train_labels_curr=train_data_labels,
													vali_labels_curr=vali_data_labels,
													test_labels_curr=test_data_labels,
													DLGN_obj=deepcopy(DLGN_init),
													parameter_mask=train_parameter_masks)
		torch.cuda.empty_cache() 

		if not os.path.exists('outputs'):
			os.mkdir('outputs')
		# print(len(DLGN_obj_store))
		# print("Hi")
		device=torch.device('cpu')
		train_outputs_values, train_outputs_gate_scores =DLGN_obj_final(torch.Tensor(train_data).to(device))
		train_preds = train_outputs_values[-1]
		criterion = nn.CrossEntropyLoss()
		outputs = torch.cat((-1*train_preds,train_preds), dim=1)
		targets = torch.tensor(train_data_labels, dtype=torch.int64)
		train_loss = criterion(outputs, targets)
		train_preds = train_preds.detach().numpy()
		filename = 'outputs/'+self.filename_suffix+'.txt'
		original_stdout = sys.stdout
		with open(filename,'w') as f:
			sys.stdout = f
			print("Setup:")
			print("Num neurons : ", DLGN_obj_final.num_nodes)
			print(" Beta :", DLGN_obj_final.beta)
			print(" lr :", self.lr)
			print("=======================")
			print(train_losses)
			print("==========Best validated model=============")
			print("Train error=",np.sum(train_data_labels != (np.sign(train_preds[:,0])+1)//2 ))
			print("Train loss = ", train_loss)
			print("Num_train_data=",len(train_data_labels))
			sys.stdout = original_stdout


		test_outputs_values, test_outputs_gate_scores =DLGN_obj_final(torch.Tensor(test_data))
		test_preds = test_outputs_values[-1]
		test_preds = test_preds.detach().numpy()
		filename = 'outputs/'+self.filename_suffix+'.txt'
		original_stdout = sys.stdout
		with open(filename,'a') as f:
			sys.stdout = f
			print("Test error=",np.sum(test_data_labels != (np.sign(test_preds[:,0])+1)//2 ))
			print("Num_test_data=",len(test_data_labels))
			sys.stdout = original_stdout

		w_list = np.concatenate((w_list_old,-w_list_old),axis=0)

		effective_weights, effective_biases = DLGN_obj_store[0].return_gating_functions()
		wts_list_init=[]
		for layer in range(0,len(effective_weights)):
			wts =  np.array(effective_weights[layer].data.detach().numpy())
			wts /= np.linalg.norm(wts, axis=1)[:,None]
			wts_list_init.append(wts)
		wts_list_init = np.concatenate(wts_list_init)


		effective_weights, effective_biases = DLGN_obj_final.return_gating_functions()

		wts_list=[]
		for layer in range(len(effective_weights)):
			wts =  np.array(effective_weights[layer].data.detach().numpy())
			wts /= np.linalg.norm(wts, axis=1)[:,None]
			wts_list.append(wts)
		wts_list = np.concatenate(wts_list)

		pd0 =  pairwise_distances(w_list,wts_list_init)
		pd1 =  pairwise_distances(w_list,wts_list)


		filename = 'outputs/'+self.filename_suffix+'.txt'
		original_stdout = sys.stdout
		with open(filename,'a') as f:
			sys.stdout = f
			print("Shape of decision tree node hyperplanes ", w_list.shape)
			print("Shape of all halfspace directions of DLGN", wts_list.shape)
			print("Distance of closest init DLGN halfspace to each labelling func hyperplane \n", pd0.min(axis=1)[:len(w_list_old)])
			print(pd0.min(axis=1)[len(w_list_old):])
			print("Distance of closest lrnd DLGN halfspace to each labelling func hyperplane \n", pd1.min(axis=1)[:len(w_list_old)])
			print(pd1.min(axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.8 of the Dtree hyperplanes \n", np.sum(pd1<0.8, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.8, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.6 of the Dtree hyperplanes \n", np.sum(pd1<0.6, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.6, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.4 of the Dtree hyperplanes \n", np.sum(pd1<0.4, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.4, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.3 of the Dtree hyperplanes \n", np.sum(pd1<0.3, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.3, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.2 of the Dtree hyperplanes \n", np.sum(pd1<0.2, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.2, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.1 of the Dtree hyperplanes \n", np.sum(pd1<0.1, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.1, axis=1)[len(w_list_old):])
			print("=========================================")
			sys.stdout = original_stdout

		# Learned feature statistics of init model
		epoch_index=0
		effective_weights, effective_biases = DLGN_obj_store[epoch_index].return_gating_functions()
		wts_list=[]
		for layer in range(len(effective_weights)):
			wts =  np.array(effective_weights[layer].data.detach().numpy())
			wts /= np.linalg.norm(wts, axis=1)[:,None]
			wts_list.append(wts)
		wts_list = np.concatenate(wts_list)
		pd0 =  pairwise_distances(w_list,wts_list_init)
		pd1 =  pairwise_distances(w_list,wts_list)
		pd_w_list = pairwise_distances(w_list, w_list)


		filename = 'outputs/'+self.filename_suffix+'.txt'
		original_stdout = sys.stdout
		with open(filename,'a') as f:
			sys.stdout = f
			print("===================================")
			print("Initial epoch")
			print(epoch_index)
			print("===================================")
			train_outputs_values, train_outputs_gate_scores =DLGN_obj_store[epoch_index](torch.Tensor(train_data).to(device))
			train_preds = train_outputs_values[-1]
			criterion = nn.CrossEntropyLoss()
			outputs = torch.cat((-1*train_preds,train_preds), dim=1)
			targets = torch.tensor(train_data_labels, dtype=torch.int64)
			train_loss = criterion(outputs, targets)
			train_preds = train_preds.detach().numpy()
			test_outputs_values, test_outputs_gate_scores =DLGN_obj_store[epoch_index](torch.Tensor(test_data))
			test_preds = test_outputs_values[-1]
			test_preds = test_preds.detach().numpy()
			print("Train error=",np.sum(train_data_labels != (np.sign(train_preds[:,0])+1)//2 ))
			print("Num_train_data=",len(train_data_labels))
			print("Train loss=",train_loss.detach())
			print("Test error=",np.sum(test_data_labels != (np.sign(test_preds[:,0])+1)//2 ))
			print("Num_test_data=",len(test_data_labels))
			print("===================================")
			print("Shape of decision tree node hyperplanes ", w_list.shape)
			print("Shape of all halfspace directions of DLGN", wts_list.shape)
			print("Distance of closest init DLGN halfspace to each labelling func hyperplane \n", pd0.min(axis=1)[:len(w_list_old)])
			print(pd0.min(axis=1)[len(w_list_old):])
			print("Distance of closest lrnd DLGN halfspace to each labelling func hyperplane \n", pd1.min(axis=1)[:len(w_list_old)])
			print(pd1.min(axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.8 of the Dtree hyperplanes \n", np.sum(pd1<0.8, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.8, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.6 of the Dtree hyperplanes \n", np.sum(pd1<0.6, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.6, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.4 of the Dtree hyperplanes \n", np.sum(pd1<0.4, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.4, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.3 of the Dtree hyperplanes \n", np.sum(pd1<0.3, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.3, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.2 of the Dtree hyperplanes \n", np.sum(pd1<0.2, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.2, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.1 of the Dtree hyperplanes \n", np.sum(pd1<0.1, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.1, axis=1)[len(w_list_old):])
			print("=========================================")
			sys.stdout = original_stdout    

		# Learned feature statistics of last iteration model
		epoch_index=len(DLGN_obj_store)-1
		effective_weights, effective_biases = DLGN_obj_store[epoch_index].return_gating_functions()
		wts_list=[]
		for layer in range(len(effective_weights)):
			wts =  np.array(effective_weights[layer].data.detach().numpy())
			wts /= np.linalg.norm(wts, axis=1)[:,None]
			wts_list.append(wts)
		wts_list = np.concatenate(wts_list)

		pd0 =  pairwise_distances(w_list,wts_list_init)
		pd1 =  pairwise_distances(w_list,wts_list)
		pd_w_list = pairwise_distances(w_list, w_list)
		w_list_random = np.random.standard_normal(w_list.shape)
		w_list_random /= np.linalg.norm(w_list_random, axis=1)[:,None]

		pd4 = pairwise_distances(w_list_random,wts_list)


		filename = 'outputs/'+self.filename_suffix+'.txt'
		original_stdout = sys.stdout
		with open(filename,'a') as f:
			sys.stdout = f
			print("===================================")
			print("last epoch: Training loss = ", train_losses[-1])
			print(epoch_index)
			print(self.saved_epochs[epoch_index])
			print("===================================")
			train_outputs_values, train_outputs_gate_scores =DLGN_obj_store[epoch_index](torch.Tensor(train_data).to(device))
			train_preds = train_outputs_values[-1]
			criterion = nn.CrossEntropyLoss()
			outputs = torch.cat((-1*train_preds,train_preds), dim=1)
			targets = torch.tensor(train_data_labels, dtype=torch.int64)
			train_loss = criterion(outputs, targets)
			train_preds = train_preds.detach().numpy()
			test_outputs_values, test_outputs_gate_scores =DLGN_obj_store[epoch_index](torch.Tensor(test_data))
			test_preds = test_outputs_values[-1]
			test_preds = test_preds.detach().numpy()
			print("Train error=",np.sum(train_data_labels != (np.sign(train_preds[:,0])+1)//2 ))
			print("Num_train_data=",len(train_data_labels))
			print("Train loss=",train_loss.detach())
			print("Test error=",np.sum(test_data_labels != (np.sign(test_preds[:,0])+1)//2 ))
			print("Num_test_data=",len(test_data_labels))
			print("===================================")

			print("Shape of decision tree node hyperplanes ", w_list.shape)
			print("Shape of all halfspace directions of DLGN", wts_list.shape)
			print("Distance of closest init DLGN halfspace to each labelling func hyperplane \n", pd0.min(axis=1)[:len(w_list_old)])
			print(pd0.min(axis=1)[len(w_list_old):])
			print("Distance of closest lrnd DLGN halfspace to each labelling func hyperplane \n", pd1.min(axis=1)[:len(w_list_old)])
			print(pd1.min(axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.8 of the Dtree hyperplanes \n", np.sum(pd1<0.8, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.8, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.6 of the Dtree hyperplanes \n", np.sum(pd1<0.6, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.6, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.4 of the Dtree hyperplanes \n", np.sum(pd1<0.4, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.4, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.3 of the Dtree hyperplanes \n", np.sum(pd1<0.3, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.3, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.2 of the Dtree hyperplanes \n", np.sum(pd1<0.2, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.2, axis=1)[len(w_list_old):])
			print("Number of halfspaces within distance 0.1 of the Dtree hyperplanes \n", np.sum(pd1<0.1, axis=1)[:len(w_list_old)])
			print(np.sum(pd1<0.1, axis=1)[len(w_list_old):])


			print("=========================================")
			print("No. of halfspaces within distance 0.8 of a random Dtree hyperplanes \n", np.sum(pd4<0.8, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.8, axis=1)[len(w_list_old):])
			print("No. of halfspaces within distance 0.6 of a random Dtree hyperplanes \n", np.sum(pd4<0.6, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.6, axis=1)[len(w_list_old):])
			print("No. of halfspaces within distance 0.4 of a random Dtree hyperplanes \n", np.sum(pd4<0.4, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.4, axis=1)[len(w_list_old):])
			print("No. of halfspaces within distance 0.3 of a random Dtree hyperplanes \n", np.sum(pd4<0.3, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.3, axis=1)[len(w_list_old):])
			print("No. of halfspaces within distance 0.2 of a random Dtree hyperplanes \n", np.sum(pd4<0.2, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.2, axis=1)[len(w_list_old):])
			print("No. of halfspaces within distance 0.1 of a random Dtree hyperplanes \n", np.sum(pd4<0.1, axis=1)[:len(w_list_old)])
			print(np.sum(pd4<0.1, axis=1)[len(w_list_old):])
			sys.stdout = original_stdout    
