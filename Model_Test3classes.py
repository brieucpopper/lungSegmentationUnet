import pdb 
import numpy as np
import cv2 
import h5py
import os
import matplotlib.pyplot as plt
from glob import glob
import struct
from tqdm import tqdm


# ------------- for the model -------------- #
from  matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import *






#Création des fonctions de dice-loss
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

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

def dice_loss(y_true, y_pred, smooth=1e-6, gama=2):
    y_true, y_pred = tf.cast(
         y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * \
        tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
    denominator = tf.reduce_sum(
        y_pred ** gama) + tf.reduce_sum(y_true ** gama) + smooth
    result = 1 - tf.divide(nominator, denominator)
    return result


base_dir = '/home/lung_segmentation/lung_segmentation_jupyter/Theo/light_database_lung_Caen.hdf5' #Emplacement de la base de données 
#model = tf.keras.models.load_model('Model9_weights_best.hdf5')




#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])





# print(model.summary())

#Patients Test Model 2
#PatientsIdTest = ['96DG_', '72SC_', '95MJ_', '46LM_', '86OD_', '68DA_', '22CM_', '93CM_', '101BH_', '35HN_', '30FD_']

#Patients Test Model 4 
#PatientsIdTest = ['78CP_', '91FB_', '59LL_', '33BJ_', '72SC_', '11RA_', '47KD_', '50BB_', '69MJ_', '49KC_']

#Patients test model 5 
#PatientsIdTest = ['90AY_', '67JC_', '51BR_', '46LM_', '103LJ_', '85BJ_', '70DC_', '65TJ_', '59LL_', '74BL_']

#Patients test model 6 
#PatientsIdTest = ['36TC_', '91FB_', '8BM_', '42NT_', '78CP_', '68DA_', '93CM_', '102TJ_', '81KA_', '85BJ_']

#Patients test model 7 
#PatientsIdTest = ['30FD_', '70DC_', '41TC_', '68DA_', '94AS_', '52CP_', '71EA_', '65TJ_', '35HN_', '37MA_']

#Patients test model 10
#PatientsIdTest = ['35HN_', '98RR_', '79DF_', '76NM_', '20JD_', '96DG_', '74BL_', '46LM_', '65TJ_']


#Patients test model 11
#PatientsIdTest = ['55GR_', '102TJ_', '81KA_', '70DC_', '22CM_', '56GJ_', '36TC_', '82MM_', '93CM_']

#Patients test model 16 
#PatientsIdTest = ['96DG_', '33BJ_', '87GF_', '47KD_', '55GR_', '48QL_', '91FB_', '42NT_','101BH_']

#Patients test model 17
#PatientsIdTest = ['22CM_', '60LD_', '95MJ_', '49KC_', '85BJ_', '11RA_', '86OD_', '63BM_', '65TJ_']

#Patients test model 19
#PatientsIdTest = ['52CP_', '42NT_', '49KC_', '22CM_', '47KD_', '55GR_', '87GF_', '78CP_']

#Patients test model 20
#PatientsIdTest =['74BL_', '52CP_', '103LJ_', '8BM_', '96DG_', '92CB_', '13WF_', '82MM_']

PatientsIdTest = ['76NM_', '47KD_', '70DC_', '8BM_', '71EA_', '94AS_', '41TC_', '30FD_']

X_test, Y_test = ExtractXY(base_dir, PatientsRead=PatientsIdTest, image_size=256, num_classes=3, ShowProgress=True)

model = tf.keras.models.load_model('Model25_weights_best.hdf5')

Y = model.predict(X_test)







#model = tf.keras.models.load_model('Model25_weights_best.hdf5', custom_objects={dice_loss:dice_loss}, compile=False)

#model.compile(optimizer='adam', loss=dice_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=3)])

def trois_images(PatientsIdTest): 
    PatientsIdTest = [PatientsIdTest[0]]
    X_test, Y_test = ExtractXY(base_dir, PatientsRead=PatientsIdTest, image_size=256, num_classes=3, ShowProgress=True)
    X = np.array([X_test[12], X_test[27], X_test[89]])
    Y_test = np.array([Y_test[12], Y_test[27], Y_test[89]])
    Y = np.argmax(model.predict(X), axis=-1)
    X = X *255
    Y_test = Y_test * 127
    Y = Y * 127
    X = X.reshape(3, X.shape[1], X.shape[2])
    Y = Y.reshape(3, Y.shape[1], Y.shape[2])
    Y_test = Y_test.reshape(3, Y_test.shape[1], Y_test.shape[2])
    vide = np.ones((X.shape[1],3)) * 255 
    Img1 = np.hstack((X[0], vide, Y_test[0], vide, Y[0]))
    Img2 = np.hstack((X[1], vide, Y_test[1], vide, Y[1]))
    Img3 = np.hstack((X[2], vide, Y_test[2], vide, Y[2]))
    plt.imsave('IMAGE_1.png', Img1, cmap ='gray', vmin=0, vmax=255)
    plt.imsave('IMAGE_2.png', Img2, cmap ='gray', vmin=0, vmax=255)
    plt.imsave('IMAGE_3.png', Img3, cmap ='gray', vmin=0, vmax=255)
    return 

trois_images(PatientsIdTest)


def Test_1Patient(PatientsIdTest):
    PatientId = PatientsIdTest[1]
    hf = h5py.File(base_dir, "r")
    X = []
    Y = []
    try:
        Temp = hf[PatientId+'_dic_msk'][:]
    except:
        print('COULD NOT READ PATIENT ', PatientId)
    
    try:
        for SliceDicom, SliceMask in zip(Temp[0], Temp[1]):
            X.append(cv2.resize(SliceDicom, (256, 256), interpolation=cv2.INTER_NEAREST))
            Y.append(cv2.resize(SliceMask, (256, 256), interpolation=cv2.INTER_NEAREST))
    except:
        pdb.set_trace()
    
    Temp = None
    hf.close()
    X = np.array(X)
    Y = np.array(Y)

    Y=np.where(Y==0,0,Y)
    Y=np.where(Y==100,1,Y)   
    Y=np.where(Y==200,2,Y)
    X = X/255.
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1).astype('float32')
    Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[2]).astype('float32')
    Y = Y[50:250:4]
    X = X[50:250:4]
    Y_train = np.argmax(model.predict(X), axis=-1)
    X = X *255
    Y= Y * 127
    Y_train = Y_train * 127
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    Y = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[2])
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], Y_train.shape[2])
    vide = np.ones((X.shape[1],3)) * 255
    for k in range(X.shape[0]):
        Img1 = np.hstack((X[k], vide, Y[k], vide, Y_train[k]))
        plt.imsave(f"Images/IMAGE_{k}.png", Img1, cmap ='gray', vmin=0, vmax=255)
    return



Test_1Patient(PatientsIdTest)
""" 

Y = np.argmax(model.predict(X_test), axis=-1)

print(Y[80][150])





X_1 = X_test[47] * 255
Y_1 = Y[47] *127
Y_test_1 = Y_test[47] *127




X_1 = X_1.reshape(X_1.shape[0], X_1.shape[1])
Y_1 = Y_1.reshape(Y_1.shape[0], Y_1.shape[1])
Y_test_1 = Y_test_1.reshape(Y_test_1.shape[0], Y_test_1.shape[1])


vide = np.ones((X_1.shape[0],3)) * 255 

Img = np.hstack((X_1, vide, Y_test_1, vide, Y_1))


plt.imsave('IMAGE.png', Img, cmap ='gray', vmin=0, vmax=255)

plt.imsave('X_1.png', X_1, cmap='gray', vmin=0, vmax=255)
plt.imsave('Y_predict_1.png', Y_1, vmin=0,cmap='gray', vmax=255)
plt.imsave('Y_true_1.png', Y_test_1, cmap='gray', vmin=0, vmax=255)


X_2 = X_test[80] * 255
Y_2 = Y[80] *127
Y_test_2 = Y_test[80] *127

X_2 = X_2.reshape(X_2.shape[0], X_2.shape[1])
Y_2 = Y_2.reshape(Y_2.shape[0], Y_2.shape[1])
Y_test_2 = Y_test_2.reshape(Y_test_2.shape[0], Y_test_2.shape[1])




plt.imsave('X_2.png', X_2, cmap='gray', vmin=0, vmax=255)
plt.imsave('Y_predict_2.png', Y_2, vmin=0,cmap='gray', vmax=255)
plt.imsave('Y_true_2.png', Y_test_2, cmap='gray', vmin=0, vmax=255)


"""