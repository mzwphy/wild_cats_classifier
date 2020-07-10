from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image
import glob
import os

dir_1 = os.getcwd()
path_to_leopard = dir_1 + '/data/leopard/'
path_to_cheetah = dir_1 + '/data/cheetah/'

#Augmentation Switch.
switch_on = False
#switch_on = True


def data_generator():
    if switch_on == True:
   
        datagen = ImageDataGenerator(
                rotation_range = 35,
                width_shift_range = 0.1,
                height_shift_range = 0.05,
                shear_range = 0.1,  
                zoom_range = 0.1,
                horizontal_flip = True,
                fill_mode = 'nearest')

        cheetah_list = []
        for j in glob.glob(path_to_cheetah + '*.jpg'):
            cheetah_list.append(j)
            print(cheetah_list)

        for filename in range(len(cheetah_list)):
            img_che = load_img(cheetah_list[filename])  # input images (cheetah)
            X_che = img_to_array(img_che)  
            X_che = X_che.reshape((1,) + X_che.shape)

            # save results to the `wildcats_che/` directory
            out_path_che = dir_1 + '/data/wildcats_che'
            i = 0
            for batch in datagen.flow(X_che, batch_size = 1, save_to_dir = out_path_che, save_prefix = 'che', save_format = 'jpeg'):
                i += 1
                if i > 25:
                   break  #do not remove


        leopard_list = []
        for k in glob.glob(path_to_leopard + '*.jpg'):
            leopard_list.append(k)
            print(leopard_list)

        for filename in range(len(leopard_list)):
            img_leo = load_img(leopard_list[filename])  # input images (leopard)
            X_leo = img_to_array(img_leo)  
            X_leo = X_leo.reshape((1,) + X_leo.shape) 

            # save results to the `wildcats_leo/` directory
            out_path_leo = dir_1 + '/data/wildcats_leo'
            i = 0
            for batch in datagen.flow(X_leo, batch_size = 1, save_to_dir = out_path_leo, save_prefix = 'leo', save_format = 'jpeg'):
                i += 1
                if i > 25:
                   break  #do not remove

    else:
        print('Augmentation not required')



