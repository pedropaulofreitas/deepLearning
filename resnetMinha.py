#-*- coding: utf-8 -*-

'''
Author: Pedro Paulo Miranda de Freitas
Data: 29/09/2017
primeiro teste: verificar para cada camada o quanto a
subtração das features influencia no resultado final
pipepline:
-> foward imagem sem objeto alvo
-> gerar histograma
-> foward respectiva referencia
-> gerar histograma
-> calcular a diferenca
-> gerar histograma
-> plotar os tres juntos
'''
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


global rnd_obj_lst
global feat_imp_vec
global pwd
global nb_frames_offset
global mv_res_dir
global countList
global camada
global nb_objs_test
global VDAO_DATABASE_LIGHTING_ENTRIES, VDAO_DATABASE_OBJECT_ENTRIES, VDAO_DATABASE_REFERENCE_ENTRIES, VDAO_DATABASE_OBJECT_POSITION_ENTRIES, FRAMES_SRC
global LAYER_NAME

pwd = os.getcwd()

LAYER_NAME = np.array(['res2a_branch2a', 'res2b_branch2a', 'res2c_branch2a',
					   'res3a_branch2a', 'res3b_branch2a', 'res3c_branch2a',
					   'res3d_branch2a', 'res4a_branch2a', 'res4b_branch2a',
					   'res4c_branch2a', 'res4d_branch2a', 'res4e_branch2a',
					   'res4f_branch2a', 'res5a_branch2a', 'res5b_branch2a',
					   'res5c_branch2a', 'avg_pool'])

	###################################################################################
	##								FUNCTIONS                                       # #
    ##																				# #
    ## 																				# #
	###################################################################################

## FUNÇÃO PARA DEFIMIR QUAIS SERAO OS OBJETOS DE TREINO DA RODADA
def setup_prog():

	global countList

	if (countList == 0):
		obj_vec = np.array([4,5,6,7,8,9,10,11])
	elif (countList == 1):
		obj_vec = np.array([0,1,2,3,8,9,10,11])
	elif (countList == 2):
		obj_vec = np.array([0,1,2,3,4,5,6,7])

	#obj_vec = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

	return obj_vec
#######################################################################################


countList = 1
pwd = os.getcwd()
feat_imp_vec = 0
obj_v_lst = setup_prog()

#HDF5_DST = mv_res_dir
nb_objs_train = 6
nb_objs_test = 9 - nb_objs_train
nb_train_vecs = 4
countList = 0

# from visualize import make_dot


# FRAMES_SRC = '/local/home/common/datasets/Reference_Object_frames_skip_17_full_aligned/'
FRAMES_SRC = '/media/freitas/common HD/dataset/Reference_Object_frames_skip_17_full_aligned/'


nb_frames_offset = 2

VDAO_DATABASE_LIGHTING_ENTRIES = np.array(['NORMAL-Light','EXTRA-Light'])

VDAO_DATABASE_OBJECT_ENTRIES = np.array(['Black_Backpack',
										 'Black_Coat',
										 'Brown_Box',
										 'Camera_Box',
										 'Dark-Blue_Box',
										 'Pink_Bottle',
										 'Shoe',
										 'Towel',
										 'White_Jar',
										 'Mult_Objs1',
										 'Mult_Objs2',
										 'Mult_Objs3'])

VDAO_DATABASE_OBJECT_POSITION_ENTRIES = np.array(['POS1','POS2','POS3'])






first_frame = True
nb_frames_test = 300
frames_per_ref = frames_per_obj = math.floor(nb_frames_test / nb_objs_test / 3)
if (1 == (frames_per_obj % 2)):
	frames_per_obj += 1
	frames_per_ref += 1
frames_per_obj_P = int(frames_per_obj/2)
frames_per_ref = frames_per_obj = int(frames_per_obj)
    ###################################################################################
    ##																				# #
    ##                         CARREGANDO OS FRAMES PARA TREINO            			# #
    ##																				# #
    ###################################################################################
print(frames_per_obj)
for obj_nb in range(VDAO_DATABASE_OBJECT_ENTRIES.shape[0]):
	if (obj_v_lst[obj_v_lst == obj_nb].size == 0 ):
		for pos_nb in VDAO_DATABASE_OBJECT_POSITION_ENTRIES:
			obj_src = VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/' + \
			VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_target'

			# first_frame = True
			temp = os.path.join(FRAMES_SRC,obj_src)
			if (os.path.exists(temp)):
				os.chdir(temp)
				frames_list = os.listdir(temp)
				nb_frames = len(frames_list) - nb_frames_offset

				if obj_nb in [9,10,11]:
					temp_rd_bg = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_target')
					temp_rd_bg_r = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_reference')
					frames_list = os.listdir(temp_rd_bg)
					nb_frames = len(frames_list)
					rnd_bg_obj_frames = np.array([])

				with open('object_frames.txt','r') as f:
					content = f.readlines()
					content = [x.strip() for x in content]
					if '' in content:
						content = content[0:-1]
					if (len(content) == 2):
						obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
						obj_frames = np.concatenate((obj_frames,np.array(list(map(int,re.findall('\d+',content[1]))))),axis=0)
						obj_frames = np.concatenate((np.arange(obj_frames[0],obj_frames[1]+1),np.arange(obj_frames[2],obj_frames[3]+1)), axis=0)
					else:
						obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
						obj_frames = np.arange(obj_frames[0],obj_frames[1]+1)


				if (first_frame == True):
					frames_test_lst = rd.sample(obj_frames.tolist(),frames_per_obj_P)
					frames_test_lst = np.asarray(frames_test_lst)
					for sel_fr in range(0,frames_per_obj_P):
						val_fr = False
						while (val_fr != True):
							nb_choice = rd.choice(range(0,nb_frames))
							if obj_nb in [9,10,11]:
								if not nb_choice in rnd_bg_obj_frames:
									frames_test_lst = np.concatenate((frames_test_lst,np.array([nb_choice,])),axis=0)
									rnd_bg_obj_frames = np.concatenate((rnd_bg_obj_frames,np.array([nb_choice,])),axis=0)
									val_fr = True
							else:
								if not nb_choice in obj_frames:
									frames_test_lst = np.concatenate((frames_test_lst,np.array([nb_choice,])),axis=0)
									obj_frames = np.concatenate((obj_frames,np.array([nb_choice,])),axis=0)
									val_fr = True
					first_frame = False
					frames_test_lst = frames_test_lst.astype(int)

				else:
					frames_test_lst = np.concatenate((frames_test_lst,rd.sample(obj_frames.tolist(),frames_per_obj_P)), axis = 0)
					for sel_fr in range(0,frames_per_obj_P):
						val_fr = False
						while (val_fr != True):
							nb_choice = rd.choice(range(0,nb_frames))
							if obj_nb in [9,10,11]:
								if not nb_choice in rnd_bg_obj_frames:
									frames_test_lst = np.concatenate((frames_test_lst,np.array([nb_choice,])),axis=0)
									rnd_bg_obj_frames = np.concatenate((rnd_bg_obj_frames,np.array([nb_choice,])),axis=0)
									val_fr = True
							else:
								if not nb_choice in obj_frames:
									frames_test_lst = np.concatenate((frames_test_lst,np.array([nb_choice,])),axis=0)
									obj_frames = np.concatenate((obj_frames,np.array([nb_choice,])),axis=0)
									val_fr = True
					frames_test_lst = frames_test_lst.astype(int)
str_objs = ''
for f in obj_v_lst:
	str_objs += str(int(f))

print(frames_test_lst.size)

# for camada =? 16:
for camada in range(0,17):
	###################################################################################
	##																				# #
	##                     ESCOHENDO A RESNET QUE VAI SER UTILIZADA					# #
	##																				# #
	###################################################################################

	# CARREGANDO O MODELO DA RESNET isso esta na cpu??? como passa pra gpu???
	resnet = models.resnet50(pretrained=True)


	if camada == 16:
		resnet = nn.Sequential(*list(resnet.children())[:-2]) #REMOVENDO As ULTIMAs CAMADA DE FC
		resnet.add_module("avrPool",nn.AvgPool2d(7,7))

	elif camada == 15:
		resnet = nn.Sequential(*list(resnet.children())[:-2]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-1]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(7,7))

	elif camada == 14:
		resnet = nn.Sequential(*list(resnet.children())[:-2]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-2]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(7,7))


	elif camada == 13:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))

	elif camada == 12:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-1]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))

	elif camada == 11:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-2]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))

	elif camada == 10:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-3]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))

	elif camada == 9:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-4]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))


	elif camada == 8:
		resnet = nn.Sequential(*list(resnet.children())[:-3]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-5]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(14,14))


	elif camada == 7:
		resnet = nn.Sequential(*list(resnet.children())[:-4]) #REMOVENDO As ULTIMAs CAMADA DE FC
		resnet.add_module("avrPool",nn.AvgPool2d(21,21))


	elif camada == 6:
		resnet = nn.Sequential(*list(resnet.children())[:-4]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-1]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(21,21))


	elif camada == 5:
		resnet = nn.Sequential(*list(resnet.children())[:-4]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-2]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(21,21))

	elif camada == 4:
		resnet = nn.Sequential(*list(resnet.children())[:-4]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-3]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(21,21))

	elif camada == 3:
                resnet = nn.Sequential(*list(resnet.children())[:-5]) #REMOVENDO As ULTIMAs CAMADA DE FC
                #cria uma lista com todos os filhos
                batata = list(resnet.children())
                #popa os filhos do ultimo filho e remove o ultimo deles
                fritas = list(batata.pop())[:-2]
                #concatena de volta
                batata = batata+fritas
                #transforma em sequencial de volta
                resnet = nn.Sequential(*batata)
                resnet.add_module("avrPool",nn.AvgPool2d(28,28))


	elif camada == 2:
		resnet = nn.Sequential(*list(resnet.children())[:-5]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-1]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(28,28))



	elif camada == 1:
		resnet = nn.Sequential(*list(resnet.children())[:-5]) #REMOVENDO As ULTIMAs CAMADA DE FC
		#cria uma lista com todos os filhos
		batata = list(resnet.children())
		#popa os filhos do ultimo filho e remove o ultimo deles
		fritas = list(batata.pop())[:-2]
		#concatena de volta
		batata = batata+fritas
		#transforma em sequencial de volta
		resnet = nn.Sequential(*batata)
		resnet.add_module("avrPool",nn.AvgPool2d(28,28))


	elif camada == 0:
		resnet = nn.Sequential(*list(resnet.children())[:-6]) #REMOVENDO As ULTIMAs CAMADA DE FC
		resnet.add_module("avrPool",nn.AvgPool2d(21,21))

	resnet = resnet.cuda()

	# switch to evaluate mode
	resnet.eval()
	###################################################################################
	##															 					# #
	##                 	     PARA AS FEATURES DE TREINO   							# #
    ##  																			# #
	###################################################################################

	X_train = np.array([]) # features ja extraidas e subtraidas são empilhadas aqui
	y_train = np.array([]) # respectivas lables de '0' sem obj '1' com obj

	for obj_nb in obj_v_lst:
		counterObjTrain = 0
		counterRefTrain = 0
		for pos_nb in VDAO_DATABASE_OBJECT_POSITION_ENTRIES:
			obj_src = VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/' + \
			VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_target'

			ref_src = VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/' + \
			VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_reference'

			temp = os.path.join(FRAMES_SRC,obj_src)
			tempr = os.path.join(FRAMES_SRC,ref_src)
			if (os.path.exists(temp)):
				os.chdir(temp)
				frames_list = os.listdir(temp)
				nb_frames = len(frames_list) - nb_frames_offset

				if obj_nb in [9,10,11]:
					temp_rd_bg = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_target')
					temp_rd_bg_r = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_reference')
					frames_list = os.listdir(temp_rd_bg)
					nb_frames = len(frames_list)
					rnd_bg_obj_frames = np.array([])


				with open('object_frames.txt','r') as f:
					content = f.readlines()
					content = [x.strip() for x in content]
					if '' in content:
						content = content[0:-1]
					if (len(content) == 2):
						obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
						obj_frames = np.concatenate((obj_frames,np.array(list(map(int,re.findall('\d+',content[1]))))),axis=0)
						obj_frames = np.concatenate((np.arange(obj_frames[0],obj_frames[1]+1),np.arange(obj_frames[2],obj_frames[3]+1)), axis=0)
					else:
						obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
						obj_frames = np.arange(obj_frames[0],obj_frames[1]+1)

				even_frame_vec = obj_frames
				if not obj_nb in [9,10,11]:
					for sel_fr in range(0,obj_frames.shape[0]):
						val_fr = False
						while (val_fr != True):
							nb_choice = rd.choice(range(0,nb_frames))
							if not nb_choice in even_frame_vec:
								even_frame_vec = np.concatenate((even_frame_vec,np.array([nb_choice,])),axis=0)
								val_fr = True
				else:
					for sel_fr in range(0,obj_frames.shape[0]):
						val_fr = False
						while (val_fr != True):
							nb_choice = rd.choice(range(0,nb_frames))
							if not nb_choice in rnd_bg_obj_frames:
								rnd_bg_obj_frames = np.concatenate((rnd_bg_obj_frames,np.array([nb_choice,])),axis=0)
								val_fr = True
					rnd_bg_obj_frames = rnd_bg_obj_frames.astype(int)

				for i in even_frame_vec:
					os.chdir(temp)
					# print('FOLD_NB: ' + str_objs  )
					# print(temp + '///TRAIN_BATCH')
					img_path = 'frame_' + str(i) + '.png'
					#carregando a imagem
					img = Image.open(img_path)
					#redimensionando as imagens
					img = img.resize((640,360))
					#transformando a imagem em cuda tensor
					img_np = transforms.to_tensor(img)
					#adicionando uma dimensao dummy, ja que entra 1 img por vez
					img_np = torch.unsqueeze(img_np, 0)
					input_var = torch.autograd.Variable(img_np, volatile=True)
					#foward do alvo
					X_train_temp = resnet(input_var.cuda())
					#transformando o output em um vetor 1D
					X_train_temp = X_train_temp.view(-1)
					# print(X_train_temp.size())
					# print ('########### FRAME {0} ###########'.format(i))
					counterObjTrain += 1

					if i in obj_frames:
						y_train = np.concatenate((y_train,np.ones((1,))),axis=0)
					else:
						y_train = np.concatenate((y_train,np.zeros((1,))),axis=0)


					if (X_train.size == 0):
						X_train = np.zeros(X_train_temp.size())

					os.chdir(tempr)
					# print('FOLD_NB: ' + str_objs)
					# print(tempr + '///TRAIN_BATCH')
					img_path = 'frame_' + str(i) + '.png'
					img = Image.open(img_path)
					img = img.resize((640,360))
					img_np = transforms.to_tensor(img)
					img_np = torch.unsqueeze(img_np, 0)
					input_var = torch.autograd.Variable(img_np, volatile=True)
					#calculando finalmente a direrenca das features alvo/referencia
					X_train_temp = X_train_temp - resnet(input_var.cuda()).view(-1)
					# print(X_train_temp.size())
					X_train_temp = X_train_temp.data.cpu().numpy()

					X_train = np.vstack((X_train,X_train_temp))
					# print ('########### FRAME {0} ###########'.format(i))
					counterRefTrain += 1

				if obj_nb in [9,10,11]:
					for i in rnd_bg_obj_frames:
						os.chdir(temp_rd_bg)
						# print('FOLD_NB: ' + str_objs)
						# print(temp_rd_bg + '///TRAIN_BATCH')
						img_path = 'frame_' + str(i) + '.png'
						img = Image.open(img_path)
						img = img.resize((640,360))
						img_np = transforms.to_tensor(img)
						img_np = torch.unsqueeze(img_np, 0)
						input_var = torch.autograd.Variable(img_np, volatile=True)
						X_train_temp = resnet(input_var.cuda())
						X_train_temp = X_train_temp.view(-1)
						# print ('########### FRAME {0} ###########'.format(i))
						counterObjTrain += 1

						y_train = np.concatenate((y_train,np.zeros((1,))),axis=0)

						os.chdir(temp_rd_bg_r)
						# print('FOLD_NB: ' + str_objs)
						# print(temp_rd_bg_r + '///TRAIN_BATCH')
						img_path = 'frame_' + str(i) + '.png'
						img = Image.open(img_path)
						img = img.resize((640,360))
						img_np = transforms.to_tensor(img)
						img_np = torch.unsqueeze(img_np, 0)
						input_var = torch.autograd.Variable(img_np, volatile=True)
						#calculando finalmente a direrenca das features alvo/referencia
						X_train_temp = X_train_temp - resnet(input_var.cuda()).view(-1)
						X_train_temp = X_train_temp.data.cpu().numpy()
						X_train = np.vstack((X_train,X_train_temp))
						# print ('########### FRAME {0} ###########'.format(i))
						counterRefTrain += 1


	X_train = X_train[1:]


	###################################################################################
	##															 					# #
	##               AGORA PARA AS FEATURES DE TESTE      							# #
    ##  																			# #
	###################################################################################

	mult_vec = 0
	last_pos = 0
	X_test = np.array([])
	y_test = np.array([])
	tp = np.array([])
	tn = np.array([])
	lst_test = ['FIRST LINE']



	for obj_nb in range(VDAO_DATABASE_OBJECT_ENTRIES.shape[0]):
		if (obj_v_lst[obj_v_lst == obj_nb].size == 0 ):
			counterObjTest = 0
			counterRefTest = 0
			for pos_nb in VDAO_DATABASE_OBJECT_POSITION_ENTRIES:
				obj_src = VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/' + \
				VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_target'

				ref_src = VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/' + \
				VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_reference'

				temp = os.path.join(FRAMES_SRC,obj_src)
				tempr = os.path.join(FRAMES_SRC,ref_src)
				if (os.path.exists(temp)):
					os.chdir(temp)
					frames_list = os.listdir(temp)
					nb_frames = len(frames_list) - nb_frames_offset

					if obj_nb in [9,10,11]:
						temp_rd_bg = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_target')
						temp_rd_bg_r = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_reference')
						frames_list = os.listdir(temp_rd_bg)
						nb_frames = len(frames_list)
						rnd_bg_obj_frames = np.array([])


					with open('object_frames.txt','r') as f:
						content = f.readlines()
						content = [x.strip() for x in content]
						if '' in content:
							content = content[0:-1]
						if (len(content) == 2):
							obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
							obj_frames = np.concatenate((obj_frames,np.array(list(map(int,re.findall('\d+',content[1]))))),axis=0)
							obj_frames = np.concatenate((np.arange(obj_frames[0],obj_frames[1]+1),np.arange(obj_frames[2],obj_frames[3]+1)), axis=0)
						else:
							obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
							obj_frames = np.arange(obj_frames[0],obj_frames[1]+1)

					counterBase = last_pos

					for i in frames_test_lst[(mult_vec * frames_per_obj):((mult_vec+1) * frames_per_obj)]:
						if obj_nb in [9,10,11]:
							if (counterBase == int(((mult_vec) * frames_per_obj) + frames_per_obj_P)):
								temp = temp_rd_bg
								tempr = temp_rd_bg_r

						os.chdir(temp)
						# print('FOLD_NB: ' + str_objs  )
						# print(temp + '///TEST_BATCH')
						img_path = 'frame_' + str(i) + '.png'
						#carregando a imagem
						img = Image.open(img_path)
						#redimensionando as imagens
						img = img.resize((640,360))
						#transformando a imagem em cuda tensor
						img_np = transforms.to_tensor(img)
						#adicionando uma dimensao dummy, ja que entra 1 img por vez
						img_np = torch.unsqueeze(img_np, 0)
						input_var = torch.autograd.Variable(img_np, volatile=True)
						#foward do alvo
						X_test_temp = resnet(input_var.cuda())
						#transformando o output em um vetor 1D
						X_test_temp = X_test_temp.view(-1)
						# print(X_train_temp.size())
						# print ('########### FRAME {0} ###########'.format(i))
						counterObjTest += 1

						app_str = obj_src + '/' + img_path
						lst_test.append(app_str)

						if obj_nb in [9,10,11]:
							if (counterBase >= ((mult_vec * frames_per_obj) + frames_per_obj_P)):
								y_test = np.concatenate((y_test,np.array([0,])),axis=0)
							else:
								y_test = np.concatenate((y_test,np.array([1,])),axis=0)
						else:
							if i in obj_frames:
								y_test = np.concatenate((y_test,np.array([1,])),axis=0)
							else:
								y_test = np.concatenate((y_test,np.array([0,])),axis=0)


						if (X_test.size == 0):
							X_test = np.zeros(X_test_temp.size())

						os.chdir(tempr)
						# print('FOLD_NB: ' + str_objs)
						# print(tempr + '///TEST_BATCH')
						img_path = 'frame_' + str(i) + '.png'
						img = Image.open(img_path)
						img = img.resize((640,360))
						img_np = transforms.to_tensor(img)
						img_np = torch.unsqueeze(img_np, 0)
						input_var = torch.autograd.Variable(img_np, volatile=True)
						#calculando finalmente a direrenca das features alvo/referencia
						X_test_temp = X_test_temp - resnet(input_var.cuda()).view(-1)
						# print(X_train_temp.size())
						X_test_temp = X_test_temp.data.cpu().numpy()

						X_test = np.vstack((X_test,X_test_temp))
						# print ('########### FRAME {0} ###########'.format(i))
						counterRefTest += 1

						counterBase += 1
					last_pos = (mult_vec+1) * frames_per_obj
					mult_vec += 1


	lst_test = lst_test[1:]
	X_test = X_test[1:]


	###################################################################################
	## Esse codigo todo pra gerarmos 4 vetores importantes:
	##	=> X_train :: features para treinar o mlp
	##	=> y_train :: respectivas labels
	##	=> X_test :: features para testar o mlp
	##  => y_test  :: respectivas labels
	###################################################################################


	##acho melhor salvar as features em hdf5 aqui, o que vai nos poupar um bom tempo no futuro

	os.chdir(pwd)
	print('salvando as features da camada'+ str(camada) +' em hdf5')
	f = h5py.File('features.h5','a')
	f.create_dataset(str(camada) + '_X_TRAIN', data=X_train, compression='gzip', compression_opts=9)
	f.create_dataset(str(camada) + '_y_TRAIN', data=y_train, compression='gzip', compression_opts=9)
	f.create_dataset(str(camada) + '_X_TEST', data=X_test, compression='gzip', compression_opts=9)
	f.create_dataset(str(camada) + '_y_TEST', data=y_test, compression='gzip', compression_opts=9)
	f.close()


	print('pronto!')



# 	###################################################################################
# 	##															 					# #
# 	##                  DEFININDO A FULLY CONNECTED E ITERANDO SOBRE A MESMA		# #
#     ##																				# #
#     ## M is the mini-batch size														# #
#     ## N is batch size;
#     ## D_in is input dimension;						                    			# #
#     ## H is hidden dimension;
#     ## D_out is output dimension.						                        	# #
# 	###################################################################################

# 	#numero de imagens = X_train.shape[0]
# 	#tamanho de cada imagem = X_train.shape[1]
# 	M = 256
# 	N = X_train.shape[0]
# 	D_in = X_train.shape[1]
# 	#H = 100
# 	D_out = 1


# 	epochs = 500
# 	learning_rate = 1e-3

# 	mlp = nn.Sequential(
# 	    nn.Linear(D_in, D_out),
# 		nn.Sigmoid(),
# 	) #aqui pode entrar um dropout ou mais camadas

# 	mlp.cuda() #passando pra gpu
# 	###################################################################################
# 	##								LOSS FUNCTION    								# #
# 	##                                                                         		# #
# 	## na documentação diz que esse bce ja combinado com a sigmod é melhor que      # #
# 	## fazer separado por conta do 'log-sum-exp trick' seja la o que isso queira    # #
# 	## dizer entao vamos usar 													    # #
# 	###################################################################################
# 	loss_fn = nn.BCELoss()
# 	###################################################################################
# 	##								OPTIMIZATION                                    # #
# 	##                                              								# #
# 	## Usando a versao do optim do Adam, possui um atributo size_average que pode   # #
# 	## ser interessante                                                       		# #
# 	###################################################################################
# 	optimizer = torch.optim.Adam(mlp.parameters(), lr= learning_rate)
# 	iterations = epochs * math.ceil(N/M)
# 	counter = 0
# 	for t in range(int(iterations)):

# 		start_index = counter*M
# 		end_index = min(X_train.shape[0], (counter+1)*M )

# 		X_train_batch = X_train[start_index:end_index,:]
# 		# 1 foward pass: compute predicted y by passing x to the models

# 		X_train_batch = torch.from_numpy(X_train_batch).float()
# 		input = torch.autograd.Variable(X_train_batch.cuda(),requires_grad=True)
# 		# saidaMlp = mlp(input).squeeze().double()
# 		saidaMlp = mlp(input).squeeze().double()

# 		# 2 Compute and print loss
# 		y_train_batch = y_train[start_index:end_index]
# 		y_train_batch = torch.from_numpy(y_train_batch).cuda()

# 		loss = loss_fn(saidaMlp, torch.autograd.Variable(y_train_batch).double())
# 		# Before the backward pass, use the optimizer object to zero all of the
# 		# gradients for the variables it will update (which are the learnable weights
# 		# of the model)
# 		optimizer.zero_grad()

# 		print(t, loss.data[0])

# 		# 3 Backward pass: compute gradient of the loss with respect to the model
# 		# parameters
# 		loss.backward()

# 		# 4 Calling the step function on an Optimizer makes an update to its parameters
# 		optimizer.step()

# 		if end_index == X_train.shape[0]:
# 			counter = 0
# 		else:
# 			counter += 1

#     #se necessario zerar featuresContainerRef, Alvo, e diff

# 		###################################################################################
# 		##				DANDO UM FOWARD NOS FRAMES DE TEST                              # #
# 		##                                              								# #
# 		## 			aproveitando para calcular tp, tn, fp, fn, e acurácia 				# #
# 		###################################################################################

# 	X_test = torch.from_numpy(X_test).float()
# 	test_output = mlp(torch.autograd.Variable(X_test.cuda(),requires_grad=True))
# 	test_output = test_output.data.squeeze().float().cpu()

# 	test_output = test_output.round().numpy()


# 	# print(test_output)
# 	#passando os 2 vetores pra numpy

# 	#calculando o melhor limiar (o que da melhor acuracia)

# 	# for limiar in np.arange(0.01,0.99,0.01):

# 	# 	test_output[test_output < limiar] = 0
# 	# 	test_output[test_output >= limiar] = 1

# 	TP, FP, TN, FN = perf_measure(test_output, y_test)
# 	print(type(TP))
# 	print(TP,TN)
# 	print(np.sum(y_test ==0 ), np.sum(y_test ==1))
# 	tp = np.concatenate((tp,np.array([TP,])),axis=0)
# 	tn = np.concatenate((tn,np.array([TN,])),axis=0)
# 	print(tp, tn)


# #fim do loop das camadas
# print(tp, tn)

	###################################################################################
	##								REFERENCIAS                                     # #
    ##																				# #
    ## http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html   # #
	###################################################################################
