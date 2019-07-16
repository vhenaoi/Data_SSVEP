import preprocessing
import numpy as np
import os

# Folder with register

path = r'D:\UDEA-MAESTRIA\Proyecto\Resultados\Sujetos Tesis\Grupo 2'
path_save = r'D:\UDEA-MAESTRIA\Proyecto\Resultados\Sujetos Tesis/DataProcessing2'
folder_name = '2S'
register_number = 13
recording_name = 'Contrast_EEG'

# Set main value 
    
fs = 250
low_fre = 3
high_fre = 30
channels = 9 # Add channel for branchs
channel_marks = 8
number_acuity = 6

for i in range(1,register_number + 1):

    subject_name = folder_name + str(i)
    directory = path_save + '/' + subject_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path_load = path + '/' + subject_name + '/' + recording_name
    
    # General processing
    
    data, times = preprocessing.file_set( path_load , channels, low_fre, high_fre, fs)
    
    # Get data each mark 
    
    position = np.where(data[channel_marks,:])
    
    right = data[:, position[0][0]:position[0][number_acuity]]
    left = data[:, position[0][number_acuity*2]:position[0][number_acuity*3]]
    both = data[:, position[0][number_acuity*4]:position[0][number_acuity*5]]
    
    # Save data
    
    np.savetxt( directory + "contrast_right.csv", right, delimiter=",")
    np.savetxt( directory + "contrast_left.csv", left, delimiter=",")
    np.savetxt( directory + "contrast_both.csv", both, delimiter=",")
    

