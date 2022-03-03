import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os


# def load_R(parent_directory):

#     array_lst = []
    
        
#     glob_list = list(glob.glob(parent_directory + "/*.csv"))
#     glob_list.sort()

#     for file in glob_list:
#         array = np.loadtxt(file, delimiter = ',')
#         array_lst.append(array)
    

#     return np.hstack(tuple(array_lst))
#     #return array_lst

def load_R(file_path):
    array = np.loadtxt(file_path, delimiter = ",")

    return array


def load_I(file_path):
    
    df = pd.read_csv(file_path)
    
    return np.array(df['brightness']) 


def load_D(file_path):
    
    df = pd.read_csv(file_path)
    
    return np.array(df['flux']), np.array(df['wavelength'])
        
    

if __name__ == '__main__': 

    I_file_path = './I/I_vector.csv'
    R_file_path = './R/R_matrix.csv'
    D_file_path = './D/flux_vs_wavelength_data.csv'

    R = load_R(R_file_path)
    D, wavelengths = load_D(D_file_path)
    I = load_I(I_file_path)

    plt.imshow(R)
    plt.show()

    print(R.shape)
    print(D.shape)
    print(wavelengths.shape)
    print(I.shape)
    



