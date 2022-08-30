# from __future__ import print_function
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
# pip install python-mnist
from mnist import MNIST
# pip install matplotlib
from matplotlib import pyplot as plt

emnist_data = MNIST(path='datas\\', return_type='numpy')
emnist_data.select_emnist('letters')
Images, Libelles = emnist_data.load_training()

print("Nombre d'images =",len(Images))
print("Nombre de libellés =",len(Libelles))

longueurImage = 28
largeurImage = 28

print("Transformation des tableaux d'images...")
Images = Images.reshape(124800, largeurImage, longueurImage)
Libelles = Libelles.reshape(124800, 1)

print("Affichage de l'image N°70000...")
from matplotlib import pyplot as plt

plt.imshow(Images[70000])
plt.show()

print(Libelles[70000])

print("En informatique, les index des listes doivent commencer à zero")
Libelles = Libelles-1

print("Libellé de l'image N°70000...")

print(Libelles[70000])



#Création des jeux d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(Images, Libelles, test_size=0.25, random_state= 111)

#Ajout d'une troisième valeur à nos tableaux d'images pour pouvoir être utilisés par le réseau de neurones, notemment le paramètre input_shape de la fonction Conv2D
X_train = X_train.reshape(X_train.shape[0], largeurImage, longueurImage, 1)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], largeurImage, longueurImage, 1)

#Création d'une variable servant de d'image de travail au réseau de neurone
imageTravail = (largeurImage, longueurImage, 1)


#Passons à présent à la phase de mise à l’échelle des valeurs de chaque pixel :
print("Valeur des pixels avant la mise à l'échelle :")
print(X_train[40000])

#Mise à l'echelle
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Creation des categories en One-Hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#------ créer le reseau de neurone------

reseauCNN = Sequential()

reseauCNN.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=imageTravail))

#Un seconde couche de 64 filtres de dimension 3x3
reseauCNN.add(Conv2D(64, (3, 3), activation='relu'))

#Une fonction de pooling
reseauCNN.add(MaxPooling2D(pool_size=(2, 2)))
reseauCNN.add(Dropout(0.25))

# Une mise à plat
reseauCNN.add(Flatten())

# Le réseau de neurone avec en entrée 128 neurones
# une fonction d'activation de type ReLU
reseauCNN.add(Dense(128, activation='relu'))
reseauCNN.add(Dropout(0.5))

# Une dernière couche de type softmax
reseauCNN.add(Dense(26, activation='softmax'))

#Nous compilons ensuite ce réseau de cette façon :
#Compilation du modèle
reseauCNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Apprentissage avec une phase de validation
# sur les jeux de test
batch_size = 128
epochs = 25

# le paramètre verbose=1, permet d'afficher les logs lors de l'apprentissage

historique_apprentissage = reseauCNN.fit(X_train, y_train,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         verbose=1,
                                         validation_data=(X_test, y_test))

# serialize weights to HDF5
reseauCNN.save("modele/modele_cas_pratiquev1.h5")


#Enfin, on vérifie la précision du réseau de neurones convolutifs sur les images de tests :
# Evaluation de la précision du modèle
score = reseauCNN.evaluate(X_test, y_test, verbose=0)
print('Precision sur les donnees de validation:', score[1])

