#-*- coding: utf-8 -*-


# Pedro Paulo Miranda de Freitas
#mlp lendo os arquivos do hdf5







import numpy as np
import os
import random as rd
import math
import time
import torch
import torch.nn as nn
from torchvision import models
import transforms
from PIL import Image
import re
import h5py

import pandas as pd
import matplotlib.pyplot as plt


#######################################################################################
##
##									FUNCOES
#######################################################################################



	###################################################################################
	##								LENDO O hdf5
	##                                         										# #
	###################################################################################


tpvec = np.array
tnvec = np.array
fpvec = np.array
fnvec = np.array
acvec = np.array



for camada in range(0,17):

	print('abrindo as features em hdf5...')
	f = h5py.File('features.h5','r')
	#f = h5py.File('rnd_aloi+aloi_batch_500_training_objs.hdf5','r')
	X_train = f['.'][str(camada) + '_X_TRAIN'].value
	y_train = f['.'][str(camada) + '_y_TRAIN'].value
	X_test = f['.'][str(camada) + '_X_TEST'].value
	y_test = f['.'][str(camada) + '_y_TEST'].value
	f.close()
	print('pronto!')


	###################################################################################
	##															 					# #
	##                  DEFININDO A FULLY CONNECTED E ITERANDO SOBRE A MESMA		# #
	##																				# #
	## M is the mini-batch size														# #
	## N is batch size;
	## D_in is input dimension;									# #
	## H is hidden dimension; 
	## D_out is output dimension.							# #
	###################################################################################

	#numero de imagens = X_train.shape[0]
	#tamanho de cada imagem = X_train.shape[1]
	M = 256
	N = X_train.shape[0]
	D_in = X_train.shape[1]
	#H = 100 
	D_out = 1

	 	
	epochs = 1000
	learning_rate = 1e-5

	mlp = nn.Sequential(
	    nn.Linear(D_in, D_out),
		nn.Sigmoid(),
	) #aqui pode entrar um dropout ou mais camadas

	mlp.cuda() #passando pra gpu
	###################################################################################
	##								LOSS FUNCTION    								# #
	##                                                                         		# #
	## na documentação diz que esse bce ja combinado com a sigmod é melhor que      # #
	## fazer separado por conta do 'log-sum-exp trick' seja la o que isso queira    # #
	## dizer entao vamos usar 													    # #
	###################################################################################
	loss_fn = nn.BCELoss()
	###################################################################################
	##								OPTIMIZATION                                    # #
	##                                              								# #
	## Usando a versao do optim do Adam, possui um atributo size_average que pode   # #
	## ser interessante                                                       		# #
	###################################################################################
	optimizer = torch.optim.Adam(mlp.parameters(), lr= learning_rate)
	iterations = epochs * math.ceil(N/M)
	counter = 0
	for t in range(int(iterations)):

		start_index = counter*M
		end_index = min(X_train.shape[0], (counter+1)*M )

		X_train_batch = X_train[start_index:end_index,:]
		# 1 foward pass: compute predicted y by passing x to the models

		X_train_batch = torch.from_numpy(X_train_batch).float()
		input = torch.autograd.Variable(X_train_batch.cuda(),requires_grad=True)
		# saidaMlp = mlp(input).squeeze().double()
		saidaMlp = mlp(input).squeeze().double()

		# 2 Compute and print loss
		y_train_batch = y_train[start_index:end_index]
		y_train_batch = torch.from_numpy(y_train_batch).cuda()

		loss = loss_fn(saidaMlp, torch.autograd.Variable(y_train_batch).double())
		# Before the backward pass, use the optimizer object to zero all of the
		# gradients for the variables it will update (which are the learnable weights
		# of the model)
		optimizer.zero_grad()

		print(t, loss.data[0])

		# 3 Backward pass: compute gradient of the loss with respect to the model
		# parameters
		loss.backward()

		# 4 Calling the step function on an Optimizer makes an update to its parameters
		optimizer.step()

		if end_index == X_train.shape[0]:
			counter = 0
		else:
			counter += 1

	#se necessario zerar featuresContainerRef, Alvo, e diff

		###################################################################################
		##				DANDO UM FOWARD NOS FRAMES DE TEST                              # #
		##                                              								# #
		## 			aproveitando para calcular tp, tn, fp, fn, e acurácia 				# #
		###################################################################################

	X_test = torch.from_numpy(X_test).float()
	test_output = mlp(torch.autograd.Variable(X_test.cuda(),requires_grad=True))
	test_output = test_output.data.squeeze().float().cpu()

	test_output = test_output.round().numpy()


	
	# #passando os 2 vetores pra numpy

	# calculando o melhor limiar (o que da melhor acuracia)

	# for limiar in np.arange(0.01,0.99,0.01):

	# 	test_output[test_output < limiar] = 0
	# 	test_output[test_output >= limiar] = 1
			

	# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
	TP = (np.sum((np.logical_and(test_output == 1, y_test == 1))/np.sum(y_test[y_test == 1])))*100
	 
	# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
	# TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))/(len(true_lables) - np.sum(true_labels[true_labels == 1]))
	TN = np.sum((np.logical_and(test_output == 0, y_test == 0))/(len(y_test) -np.sum(y_test[y_test == 1])))
	 
	# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
	FP = np.sum(np.logical_and(test_output == 1, y_test == 0))/(len(y_test) - np.sum(y_test[y_test == 1]))
	 
	# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
	FN = np.sum(np.logical_and(test_output == 0, y_test == 1))/np.sum(y_test[y_test == 1])
	
	#acuracia
	AC = (np.sum(np.logical_and(test_output , y_test))/np.sum(y_test))*100

	
	print(TP,TN)

	tpvec = np.append(tpvec,TP)
	tnvec = np.append(tnvec,TN)
	fpvec = np.append(fpvec,FP)
	fnvec = np.append(fnvec,FN)
	acvec = np.append(acvec,AC)

	#fim do loop das camadas

print(acvec,tpvec)

# np.savetxt('metricas.out', tnvec, newline=" ")
	###################################################################################
	##								REFERENCIAS                                     # #
    ##																				# #
    ## http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html   # #
	###################################################################################
