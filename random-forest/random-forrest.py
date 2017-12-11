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



def random_forest_classifier(features,target,numberOfTres):
    """
    To train the random forest classifier with features
    and target dataset
    returns: trained random forest classifier
    """
    classifier = RandomForestClassifier(numberOfTres,)
    classifier.fit(features,target)

    return classifier


def main():
    #trying diferent number of trees from 10 till 500 with step of 50
    for numberOfTres in range(10,550,50):

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

            #criando a instancia de classificador random forest
            trained_model = random_forest_classifier(X_train, y_train, numberOfTres)
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
        thefile = open(str(numberOfTres) + "arvores" + "camada" + str(camada)+ "_dados" + '.txt', 'w')
        for item in acvec:
            print(item)
            thefile.write("%s\n" % str(item))
        trefile.close()
        
    # stack metrics
    os.chdir(pwd)
    print('salvando os resultados em hdf5')
    f = h5py.File('random_forrest.h5','a')
    f.create_dataset(acvecall, data=acvecall, compression='gzip', compression_opts=9)
    f.create_dataset(tpvecall, data=tpvecall, compression='gzip', compression_opts=9)
    f.create_dataset(tnvecall, data=tnvecall, compression='gzip', compression_opts=9)
    f.create_dataset(fnvecall, data=fnvecall, compression='gzip', compression_opts=9)
    f.create_dataset(fpvecall, data=fpvecall, compression='gzip', compression_opts=9)
    f.close()

    print('pronto!')



if __name__ == "__main__":
    main()
