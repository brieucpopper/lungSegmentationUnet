import pdb 
import numpy as np
import cv2 
import h5py
import os
import matplotlib.pyplot as plt
from glob import glob
import struct
from tqdm import tqdm
import Dice_Loss

# ------------- for the model -------------- #
from  matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model



base_dir = 'light_database_lung_Caen.hdf5' #Emplacement de la base de données 

def ExtractXY(input_path, PatientsRead, image_size, interpolation=cv2.INTER_NEAREST, ShowProgress=False,shuffle=False,num_classes=2):
    hf = h5py.File(input_path, "r")
    X = []
    Y = []
    #Temp0 = hf['pixel_size_original'][:]
    if ShowProgress:
        LoopArray = tqdm(range(len(PatientsRead)))
    else:
        LoopArray = range(len(PatientsRead))

    for IndexPatientId in LoopArray:
        compteur = 20 
        PatientId = PatientsRead[IndexPatientId]
        try:
            Temp = hf[PatientId+'_dic_msk'][:]
        except:
            print('COULD NOT READ PATIENT ', PatientId)
            continue

        try:
            for SliceDicom, SliceMask in zip(Temp[0], Temp[1]):
                X.append(cv2.resize(SliceDicom, (image_size, image_size), interpolation=cv2.INTER_NEAREST))
                Y.append(cv2.resize(SliceMask, (image_size, image_size), interpolation=cv2.INTER_NEAREST))
        except:
            pdb.set_trace()
    Temp = None
    Temp1 = None
    hf.close()


    Y = np.array(Y)
    X = np.array(X)
    
    if num_classes == 2 :
        Y=np.where(Y==200,1,Y)
    elif num_classes ==3 :
        Y=np.where(Y==200,2,Y)

    Y=np.where(Y==0,0,Y)
    Y=np.where(Y==100,1,Y)
    
    print("X", type(X), X.shape)
    print("Y", type(Y), Y.shape)
    X = X/255.
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1).astype('float32')
    Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[2]).astype('float32')


    IndexShuffle = np.arange(Y.shape[0])
    if shuffle:
        np.random.shuffle(IndexShuffle)
        X = X[IndexShuffle]
        Y = Y[IndexShuffle]


    print("X expanded", type(X), X.shape)
    print("Y expanded", type(Y), Y.shape)
    return X, Y 

def SelectPatientsTrainVal(input_path, val_split, test_split, perte):
	hf = h5py.File(input_path, "r")
	PatientsId = hf['patient_id'][0]
	print("patientId shape ", PatientsId.shape)
	np.random.seed(42)
	np.random.shuffle(PatientsId)
	#Permet de supprimer une partie du dataset
	NPatients = PatientsId.shape[0] 
	PatientsIdgarde = PatientsId[:int((1 - perte) * NPatients + 0.5)]

	#Permet de séparer pour avoir des données de test 
	NPatients_true = PatientsIdgarde.shape[0]
	PatientsIdTest = PatientsIdgarde[int((1 - test_split) * NPatients_true + 0.5):]
	PatientsIdTraining = PatientsIdgarde[:int((1 - test_split) * NPatients_true + 0.5)]

	#Permet de séparer pour avoir des données de train et de validation
	NTraining = PatientsIdTraining.shape[0]
	PatientsIdTrain = PatientsIdTraining[:int((1 - val_split) * NTraining + 0.5)]
	PatientsIdVal = PatientsIdTraining[int((1 - val_split) * NTraining + 0.5):]
	
	hf.close()
	return np.array(PatientsIdTrain), np.array(PatientsIdVal), np.array(PatientsIdTest)


#Choix d'abord des patients qui seront dans notre données d'entrainement ou de validation 

PatientsIdTrain,PatientsIdVal, PatientsIdTest = SelectPatientsTrainVal(base_dir,.1,.3,.5)  #La partie validation correspond à 10% de notre dataset.

'''
# Si on a besoin de tester notre algo sur un petit jeu de données au début
PatientsIdTrain = np.array(['42NT_', '95MJ_', '55GR_', '103LJ_'],dtype='object')
PatientsIdVal = np.array(['8BM_', '71EA_',],dtype='object')
'''


#Création de X et Y train et validation à partir des patients choisis
X_val = []
Y_val = []
X_val, Y_val = ExtractXY(base_dir, PatientsRead=PatientsIdVal, image_size=256, num_classes=2, ShowProgress=True)

X_train = []
Y_train = []
X_train, Y_train = ExtractXY(base_dir, PatientsRead=PatientsIdTrain, image_size=256, num_classes=2, ShowProgress=True)

#Transformation en catégorie afin de ne pas avoir de mise en place de hiérarchie entre nos catégories (pourmons/pas poumons)
Y_val_cat = to_categorical(Y_val,num_classes=2)
Y_train_cat = to_categorical(Y_train,num_classes=2)

#Il faut reshape X car ainsi on a une image 256,256,1 channel (je pense que c'est pour pouvoir bien initialiser le premier layer)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)




#On remarque que l'architecture U-Net, est en fait une succession de bloc encoder (avec un copy paste) puis une succession de bloc decoder qui récupère le copy paste

def EncoderBlock(inputs, n_filters):
    conv = tf.keras.layers.Conv2D(n_filters, 
                  3,  
                  activation='relu',
                  padding='same')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters, 
                  3, 
                  activation='relu',
                  padding='same')(conv)
     
    next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)       
    return next_layer, conv


def DecoderBlock(prev_layer_input, copy_input, n_filters):
    up = tf.keras.layers.Conv2DTranspose(
                 n_filters,
                 (3,3), 
                 strides=(2,2),
                 padding='same')(prev_layer_input)
    merge = tf.keras.layers.concatenate([copy_input, up], axis=3)
    conv = tf.keras.layers.Conv2D(n_filters, 
                 3,  
                 activation='relu',
                 padding='same')(merge)
    conv = tf.keras.layers.Conv2D(n_filters,
                 3, 
                 activation='relu',
                 padding='same')(conv)
    return conv

#Définition du modèle, basé sur l'architecture U-Net
"""
input_img = tf.keras.Input(shape=(256, 256, 1))

block_1, copy_1 = EncoderBlock(input_img, 64)
block_2, copy_2 = EncoderBlock(block_1, 128)
block_3, copy_3 = EncoderBlock(block_2, 256)
block_4, copy_4 = EncoderBlock(block_3, 512)

conv = tf.keras.layers.Conv2D(1024, 
                 3,  
                 activation='relu',
                 padding='same')(block_4)
conv = tf.keras.layers.Conv2D(1024,
                 3, 
                 activation='relu',
                 padding='same')(conv)

up_1 = DecoderBlock(conv, copy_4, 512)
up_2 = DecoderBlock(up_1, copy_3, 256)
up_3 = DecoderBlock(up_2, copy_2, 128)
up_4 = DecoderBlock(up_3, copy_1, 64)



output_layer = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(up_4)
"""
def unet():
	inputs = Input((256, 256, 1))
	BN0 = BatchNormalization()(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	BN1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(BN1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	BN2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(BN2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	BN3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(BN3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	BN4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(BN4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	BN5 = BatchNormalization()(conv5)
	encode = [BN1, BN2, BN3, BN4, BN5]
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv5))
	BN6 = BatchNormalization()(up6)
	merge6 = concatenate([encode[-2], BN6], axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv6))
	BN7 = BatchNormalization()(up7)
	merge7 = concatenate([encode[-3], BN7], axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv7))
	BN8 = BatchNormalization()(up8)
	merge8 = concatenate([encode[-4], BN8], axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv8))
	BN9 = BatchNormalization()(up9)
	merge9 = concatenate([encode[-5], BN9], axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

	model = Model(inputs, conv10)
	return model 
autoencoder = unet()

#Création des fonctions de dice-loss
#smooth = 1.

"""
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
"""
"""
def dice_loss(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred) + 1e-6
    denominator = tf.reduce_sum(y_true + y_pred) + 1e-6
    return 1 - numerator/denominator
"""
def dice_loss(y_true, y_pred, smooth=1e-6, gama=2):
    y_true, y_pred = tf.cast(
         y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * \
        tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
    denominator = tf.reduce_sum(
        y_pred ** gama) + tf.reduce_sum(y_true ** gama) + smooth
    result = 1 - tf.divide(nominator, denominator)
    return result


def weighted_BCE_loss(y_true, y_pred, positive_weight=5):
    # y_true: (None,None,None,None)     y_pred: (None,512,512,1)
    y_pred = tf.keras.backend.clip(y_pred, min_value=1e-12, max_value=1 - 1e-12)
    weights = tf.keras.backend.ones_like(y_pred)  # (None,512,512,1)
    weights = tf.where(y_pred < 0.5, positive_weight * weights, weights)
    # weights[y_pred<0.5]=positive_weight
    out = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (None,512,512)
    out = tf.keras.backend.expand_dims(out, axis=-1) * weights  # (None,512,512,1)* (None,512,512,1)
    return tf.keras.backend.mean(out)

autoencoder.compile(optimizer='adam', loss=dice_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
#dice-loss 

print(autoencoder.summary())
print(PatientsIdTest)





filepath="Model27_weights_best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
callbacks_list = [checkpoint, es]


history = autoencoder.fit(X_train,Y_train,batch_size=16,epochs = 10,callbacks=callbacks_list, verbose=1,shuffle=True,validation_data=(X_val,Y_val))






plt.figure()
plt.plot(history.history['mean_io_u'])
plt.plot(history.history['val_mean_io_u'])
plt.title('model IoU')
plt.ylabel('Score IoU')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Results/Model27_IoU.png')
plt.show()


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Results/Model27_Loss.png')

autoencoder.save('./Results/Model27')



