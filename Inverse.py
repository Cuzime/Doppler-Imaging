from Simulator import *
from Forward import *
from load_model import *
import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import mean_squared_error
import math

#import autograd.numpy as np
#from autograd import grad
from scipy.optimize import fsolve


#TODO: 
# add gaussian noise
# define accuracy measure
# make doppler shift more visible 
# implement maximum entropy method 

def R_transpose(R, D):

    return R.T @ D

def pseudo_inverse(R, D):

    #print(np.linalg.pinv(R).shape)
    #print(D.shape)

    return np.linalg.pinv(R).T @ D

def rmse(I,computed_I):

    return mean_squared_error(I, computed_I, squared = False)

def mse(I, computed_I):

    return mean_squared_error(I, computed_I, squared = True)

def maximum_entropy(X0, a, l = 1e4):
    #2.2e7
    computed_I = X0
    D = a
    # phi_list = list(range(0, 10))
    # R_guess_lst = []

    # for i in phi_list:
       
    #     I = s_R.rotate(np.pi * 2 / len(phi_list))
    #     stellar_disk_R = s_R.get_stellar_disk(I)
    #     max_wavelength = 5
    #     R_guess = get_R(s_R, 400, max_wavelength=max_wavelength)

    #     R_guess_lst.append(R_guess)
        

    # R = np.hstack(tuple(R_guess_lst))

    chi2 = mse(D, R.T @ computed_I)

    S = 0 
    p = computed_I/np.sum(computed_I)
    S = -np.sum(p*np.log2(p))

    Q = S - l*chi2
    
    return -Q


if __name__ == '__main__':

    s_R = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e1, 700)
    s_D = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e1, 700)
    I_file_path = './I/I_vector.csv'
    R_file_path = './R/R_matrix.csv'
    D_file_path = './D/flux_vs_wavelength_data.csv'

    R = load_R(R_file_path)
    print(R.shape)
    D, wavelengths = load_D(D_file_path)
    I = load_I(I_file_path)

    
    computed_stellar_disk = pseudo_inverse(R,D)

    #D = D + np.random.normal(0, 1e-14, D.shape)
    
    #true_stellar_disk = s.get_stellar_disk()
    s_R.plot_on_sphere(computed_stellar_disk, savefig = True, parent_directory='./stellar_disk_computed/')
    s_D.plot_on_sphere(I, savefig = True, parent_directory = './stellar_disk_ground_truths/')

    #print(computed_stellar_disk[len(computed_stellar_disk)-2])
    #print(rmse(true_stellar_disk, computed_stellar_disk))

    s_D.plot_on_sphere(computed_stellar_disk)

    '''optimize'''
    s_mem_guess = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e1, 700, guess = True)
    I0 = s_mem_guess.I
    I0 = np.full(I.shape, 4.5)
    result = minimize(maximum_entropy, x0 = I0, args = (D))
    optimized_I = result.x
    s_D.plot_on_sphere(I0)
    s_D.plot_on_sphere(optimized_I)
    
    
    line_spectra_guess = R.T @ optimized_I
    line_spectra_truth = R.T @ I
    print()
    plt.plot(line_spectra_guess)
    plt.show()
    plt.close()
    plt.plot(line_spectra_truth)
    plt.show()
    plt.close()
    
    
    #print(s_mem_guess.I)
    #print(optimized_I)




    

#MAKE D AND R HAVE DIFFERENT BUT CLOSE TEST STARS