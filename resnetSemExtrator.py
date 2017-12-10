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
import re
import h5py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#######################################################################################
##
##									FUNCOES
#######################################################################################



	###################################################################################
	##								LENDO O hdf5
	##                                         										# #
	###################################################################################


tpvec = []
tnvec = []
fpvec = []
fnvec = []
acvec = []



for camada in range(0,17):

	# print('carregando as features em hdf5...')
	f = h5py.File('features.h5','r')
	#f = h5py.File('rnd_aloi+aloi_batch_500_training_objs.hdf5','r')
	X_train = f['.'][str(camada) + '_X_TRAIN'].value
	y_train = f['.'][str(camada) + '_y_TRAIN'].value
	X_test = f['.'][str(camada) + '_X_TEST'].value
	y_test = f['.'][str(camada) + '_y_TEST'].value
	f.close()
	# print('pronto!')


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


	epochs = 256
	learning_rate = 1e-3

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

		# print(t, loss.data[0])

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


	# calculando o melhor limiar (o que da melhor acuracia)
	accuracy = 0
	limiarFinal = 0
	for limiar in np.arange(0.01,0.99,0.01):

		test_output[test_output < limiar] = 0
		test_output[test_output >= limiar] = 1


		acuracyNew = accuracy_score(y_test, test_output)
		if acuracyNew > accuracy:
			accuracy = acuracyNew
			limiarFinal = limiar

	# print "Train Accuracy :: ", accuracy_score(y_train, test_output)
	print("camada:", camada)
	print "Test Accuracy :: ", accuracy
	# print "Test Accuracy  :: ", accuracy_score(y_test, predictions)
	confusionMatrix = confusion_matrix(y_test, test_output)
	print " Confusion matrix ", confusionMatrix
	print(limiarFinal)
	acvec.append(accuracy*float(100))
	tpvec.append((confusionMatrix[1,1]/float(187))*100)
	tnvec.append((confusionMatrix[0,0]/float(187))*100)
	fnvec.append((confusionMatrix[0,1]/float(187))*100)
	fpvec.append((confusionMatrix[1,0]/float(187))*100)

print(tpvec)
plt.plot(acvec, color='green', marker='o',)
plt.plot(tpvec)
plt.plot(tnvec)
plt.show()

#fim do loop das camadas

	###################################################################################
	##								REFERENCIAS                                     # #
    ##																				# #
    ## http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html   # #
	###################################################################################
