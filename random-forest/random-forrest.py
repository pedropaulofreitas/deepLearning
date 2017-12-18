#-*- coding: utf-8 -*-

# Author: Pedro Paulo Miranda de Freitas
# Date: 09/12/17
#
# implementação de uma random forrest
# para classifiar as features obtidas
# através da resnet50

#packages

# import pandas as pd
import os
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import random as rd
import pickle
# hyperband stuff
from hyperband import Hyperband
from def_optim import DefOptim

#files path
FRAMES_SRC = '/media/freitas/common HD/dataset/Reference_Object_frames_skip_17_full_aligned/'
OUTPUT_PATH = 'resultados/resultados-random.csv'



def random_forest_classifier(features,target,numberOfTres,max_features, max_depth, min_samples_leaf):
    """
    To train the random forest classifier with features
    and target dataset
    returns: trained random forest classifier
    """
    classifier = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth =max_depth, max_features = max_features,n_estimators = numberOfTres,)
    classifier.fit(features,target)

    return classifier


def main():
    #trying diferent number of trees from 10 till 500 with step of 50

    tpvec = []
    tnvec = []
    fpvec = []
    fnvec = []
    acvec = []

    tpvecall = []
    tnvecall = []
    fpvecall = []
    fnvecall = []
    acvecall = []

    for camada in range(0,17):

        #carregando as features do hdf5
        print('carregando as features em hdf5...')
        f = h5py.File('../features.h5','r')
        #f = h5py.File('rnd_aloi+aloi_batch_500_training_objs.hdf5','r')
        X_train = f['.'][str(camada) + '_X_TRAIN'].value
        y_train = f['.'][str(camada) + '_y_TRAIN'].value
        X_test = f['.'][str(camada) + '_X_TEST'].value
        y_test = f['.'][str(camada) + '_y_TEST'].value
        f.close()
        print('pronto!')
        print X_train.shape
        print X_train[0].shape
        print "-----------------"
        scope = np.arange(3484)
        choice = rd.sample(scope, 484)
        notChoice = np.setdiff1d(scope, choice)

        X_val = X_train[choice]
        X_train = X_train[notChoice]
        y_val = y_train[choice]
        y_train = y_train[notChoice]
        print X_val.shape
        print y_val.shape
        print X_train.shape
        print y_train.shape
        thefile = open("camada" + str(camada)+ "_dados" + '.txt', 'w')
        thefile.write("camada"+str(camada)+"\n")

        def _tune(n_iterations, params):

            n_estimators = params['n_estimators']
            max_features = params['max_features']
            max_depth = params['max_depth']*n_iterations
            min_samples_leaf = params['min_samples_leaf']

            #criando a instancia de classificador random forest
            trained_model = random_forest_classifier(X_train, y_train, n_estimators,
                                                     max_features, max_depth, min_samples_leaf)
            # print ("Trained model :: ", trained_model)

            validation = trained_model.predict(X_val)
            score =  accuracy_score(y_val, validation)
            print '--------------', camada
            print "val Accuracy :: ", score
            testAcuracy = accuracy_score(y_test, trained_model.predict(X_test))
            print "Test Accuracy :: ", testAcuracy
            loss = 1 - score

            thefile.write("Test Accuracy :: "+ str(testAcuracy)+'\n')
            thefile.write("n_estimators :: "+ str(n_estimators)+'\n')
            thefile.write("max_features :: "+ str(max_features)+'\n')
            thefile.write("max_depth :: "+ str(max_depth)+'\n')
            thefile.write("min_samples_leaf :: "+ str(min_samples_leaf)+'\n')

            return {'loss':loss}

    	#<<<<<<<<<<<<<<<<<<<<<<<<<<using hyperband>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    	optim = DefOptim(_tune)

    	hb = Hyperband(optim.get_params, optim.try_params)
    	results = hb.run(skip_last=1)


        thefile.close()
    	print("{} total, best:\n".format( len( results )))

    	for r in sorted( results, key = lambda x: x['loss'] )[:5]:
    		print("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format(
    			r['loss'], r['seconds'], r['iterations'], r['counter'] ))
    		pprint( r['params'] )
    		print()

    	print("saving...")

    	with open( 'hyperband-sig.pkl'+str(camada), 'wb' ) as f:
    		pickle.dump( results, f)



        #passando features de teste pelo classificador
        predictions = trained_model.predict(X_test)
        print "Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train))
        accuracy = accuracy_score(y_test, predictions)
        print "Test Accuracy  :: ", accuracy
        confusionMatrix = confusion_matrix(y_test, trained_model.predict(X_test))
        print " Confusion matrix ", confusionMatrix
        acvec.append(accuracy*float(100))
        tpvec.append((confusionMatrix[1,1]/float(187))*100)
        tnvec.append((confusionMatrix[0,0]/float(187))*100)
        fnvec.append((confusionMatrix[0,1]/float(187))*100)
        fpvec.append((confusionMatrix[1,0]/float(187))*100)



    plt.figure(figsize=(14.195, 6.841))
    plt.plot(acvec, color='green', marker='o')
    plt.plot(tpvec)
    plt.plot(tnvec)
    plt.ylabel("performance")
    plt.xlabel("layer")
    plt.legend(['acurracy', 'true positive', 'true negative'])
    # plt.show()
    plt.savefig(str(numberOfTres) + "arvores" + "camada" + str(camada) + ".png")

    acvecall.append(acvec)
    tpvecall.append(tpvec)
    tnvecall.append(tnvec)
    fnvecall.append(fnvec)
    fpvecall.append(fpvec)


    # stack metrics
    # os.chdir(pwd)
    # print('salvando os resultados em hdf5')
    # f = h5py.File('random_forrest_hyperband.h5','a')
    # f.create_dataset(acvecall, data=acvecall, compression='gzip', compression_opts=9)
    # f.create_dataset(tpvecall, data=tpvecall, compression='gzip', compression_opts=9)
    # f.create_dataset(tnvecall, data=tnvecall, compression='gzip', compression_opts=9)
    # f.create_dataset(fnvecall, data=fnvecall, compression='gzip', compression_opts=9)
    # f.create_dataset(fpvecall, data=fpvecall, compression='gzip', compression_opts=9)
    # f.close()
    #
    # print('pronto!')



if __name__ == "__main__":
    main()
