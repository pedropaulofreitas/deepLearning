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


#files path
FRAMES_SRC = '/media/freitas/common HD/dataset/Reference_Object_frames_skip_17_full_aligned/'
OUTPUT_PATH = 'resultados/resultados-random.csv'

def random_forest_classifier(features,target):
    """
    To train the random forest classifier with features
    and target dataset
    returns: trained random forest classifier
    """
    classifier = RandomForestClassifier(500,)
    classifier.fit(features,target)

    return classifier

def main():


    tpvec = []
    tnvec = []
    fpvec = []
    fnvec = []
    acvec = []

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

        #criando a instancia de classificador random forest
        trained_model = random_forest_classifier(X_train, y_train)
        print ("Trained model :: ", trained_model)

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

    plt.plot(acvec, color='green', marker='o',)
    plt.plot(tpvec)
    plt.plot(tnvec)
    plt.show()

if __name__ == "__main__":
    main()
