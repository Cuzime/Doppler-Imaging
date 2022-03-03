import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import constants
from scipy.integrate import quad
from Simulator import * 
from PyAstronomy import pyasl
import pandas as pd

# setting these constants under SI unit will cause 32 bit float overflow
# h = constants.Planck
# c = constants.c
# k = constants.k

# T -> kilo Kelvin
# lambda (wavelength) -> 10e-7 -> 10e-4 m -> in units of anstrom

h = 1
c = 1
k = 1

inf = np.inf

def black_body(wavelength, temperature):
    """
    Input: wavelength, temperature
    Output: intensity calculated by B(l, T)
    """
    l, T = wavelength, temperature
    return 2*h*(c**2)/(l**5)*1/((np.exp(h*c/(l*k*T)))-1)

def integrate_black_body(init_wavelength, delta_wavelength, temperature):

    return quad(black_body, init_wavelength, init_wavelength+delta_wavelength, args=(temperature))[0]


def get_temperature(star):
    """
    Input: star
    Output: Temperature of each element(i) in I, T(i); type = 1D_array
    """
    return star.I


def black_body_matrix(temp, frac=[1, 1]):
    """
    Input: wavelength, temperature; type: np.array
    Output: matrix constructed using black body radiation
    """
    B = []
    for f in frac:
        Bi = []
        for T in temp:
            B_ij = np.array([quad(black_body, 0, inf, args=(T))[0]])
            Bi.append(B_ij*f)
        B.extend(Bi)
    B_mat = np.array(B)
    B_mat.reshape(len(temp), len(frac))
    return np.array(B_mat)

def get_v_radial(star, index):
    """
    Input: star, index of the patch in image vector I, converting into: w = angular velocity, theta & phi = angles of each patch, R = radius, i = inclination angle
    Output: radial speed
    """
    R = star.radius
    w = star.v_e / R
    lat, lon = star.get_lat_lon(star.I, index)
    #print(lat, lon)
    theta = lat
    phi = lon
    #print(theta, phi)
    i = star.inclination_angle

    v_ang = np.array([0, 0, w])
    position = np.array([R*np.sin(theta)*np.cos(phi), R*np.sin(theta)*np.sin(phi), R*np.cos(theta)])
    inclination = np.array([0, np.sin(i), np.cos(i)])

    v_radial = np.dot(np.cross(v_ang, position), inclination)
    # v_radial changed into km/s
    return -v_radial/1000

def doppler_shift(star):
    """
    Input: star, wavelength of the source (using Black Body radiation); converting into: rotational velocity of the star
    Computed the doppler shift fraction of wavelength using equation $l/l0 = sqrt((1-B)/(1+B))$ and used it to calculate 
    scaling factor using $np.sqrt(1 - (1 / v_e**2) * (l)**2 * (1 / np.sin(i)**2))$
    Output: scaling factor of intensity profile due to doppler shift 
    """
    v_r = np.array([get_v_radial(star, i) for i in range(len(star.I))])

    #l0 = np.array([quad(black_body, 0, inf, args=(T))[0] for T in temperature])
    B = v_r / 3e8
    l = B

    v_e = star.v_e / 3e8
    i = star.inclination_angle

    factor = np.sqrt(1 - (1 / v_e**2) * (l)**2 * (1 / np.sin(i)**2))
    

    return factor

def shift_spectrum(cur_spec, v_radial, wavelength_lst):
    """
    Input: current spectrum (list), \delta Lambda/Lambda (deci), wavelength (list)
    Output: shifted array
    """
    return pyasl.dopplerShift(wavelength_lst, cur_spec, v_radial, edgeHandling="firstlast")[0]



def get_projected_area(star, index):
    """
    Will compute projected area at different phases and will set zone equal to 0 if it is behind the disk 
    Input: star, index of the patch in image vector I; converting into: theta & phi = angles of each patch, i = inclination angle
    Output: factor for projected area using formula $sin(theta)*sin(phi)*sin(i)+cos(theta)*cos(i)$
    """

    lat, lon = star.get_lat_lon(star.I, index)
    theta = lat
    phi = lon + star.phase
    i = star.inclination_angle
    
    dot_product = np.sin(theta)*np.sin(phi)*np.sin(i) + np.cos(theta)*np.cos(i)
    
    if dot_product >= -1:
        
        return abs(dot_product)
    else:
        return 0.0



def get_R(star, num_wavelengths, max_wavelength = 15000):
    
    delta_wavelength = max_wavelength / num_wavelengths #meters

    
    #stellar_disk_vector = star.stellar_disk_vector
    
    stellar_disk_vector = star.I

    num_latitudes = star.num_latitudes
    inclination_angle = star.inclination_angle
    zones = star.zones 
    
    wavelength_lst = np.linspace(0.01, max_wavelength, num_wavelengths)
    temp_lst = np.power(stellar_disk_vector, 0.25) / sigma
    # plt.imshow(temp_lst.reshape((1, len(temp_lst))))
    # plt.show()

    doppler_shift_lst = doppler_shift(star)

    R = []

    for i in range(len(stellar_disk_vector)):
        row = []
        projected_area = get_projected_area(star, i)

        if stellar_disk_vector[i] != 0.0:
            for j in range(num_wavelengths):
                a = integrate_black_body(wavelength_lst[j], delta_wavelength, temp_lst[i])
                normalized_flux = a / stellar_disk_vector[i]
                row.append(normalized_flux)

        else: 
            for j in range(num_wavelengths):
                row.append(0.0)

        row = shift_spectrum(row, get_v_radial(star, i), wavelength_lst)
        row = row * get_projected_area(star, i)
        R.append(row)

    R = np.array(R)
    #print(doppler_shift_lst[0])
    #plt.plot(doppler_shift_lst)
    #plt.show()
    #plt.close()

    return R

    #def add_noise(snr):



if __name__ == '__main__':
    s_R = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e1, 700)
    s_D = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e1, 700)

    #s_R = Star(np.pi/4.2, 4.5, 3.4e6, 0.5e3, 100)
    #s = Star(np.pi/4, 5, 3e6, 4, 1000)
    

    dictionary = {'brightness': s_D.I}
    df = pd.DataFrame(dictionary)
    df.to_csv(f'./I/I_vector.csv')

    phi_list = list(range(0, 10))
    R_guess_lst = []
    R_truth_lst = []
    
    for i in phi_list:
       
        I_R = s_R.rotate(np.pi * 2 / len(phi_list))
        stellar_disk_R = s_R.get_stellar_disk(I_R)
        max_wavelength = 5
        R_guess = get_R(s_R, 400, max_wavelength=max_wavelength)

        I_D = s_D.rotate(np.pi * 2 / len(phi_list))
        stellar_disk_D = s_D.get_stellar_disk(I_D)
        max_wavelength = 5
        R_truth = get_R(s_D, 400, max_wavelength=max_wavelength)
        s_D.plot_on_sphere(s_D.stellar_disk_vector, savefig = True)



        # '''saving stellar disk vector to csv'''
        # index_lst = np.linspace(0, len(stellar_disk), len(stellar_disk))
        # dictionary = {'index': index_lst,'brightness': stellar_disk}
        # df = pd.DataFrame(dictionary)
        # df.to_csv(f'./stellar_disk_vector_{i}.csv')

        # '''saving R matrix to csv'''
        # np.savetxt(f'./R/R_matrix_{i}.csv', R, delimiter = ", ", fmt = '% s')
        R_guess_lst.append(R_guess)
        R_truth_lst.append(R_truth)
        

    R_guess_all_phases = np.hstack(tuple(R_guess_lst))
    R_truth_all_phases = np.hstack(tuple(R_truth_lst))

    np.savetxt(f'./R/R_matrix.csv', R_truth_all_phases, delimiter = ", ", fmt = '% s')

    line_spectra = R_truth_all_phases.T @ s_D.I
    

    wavelengths = np.linspace(0.01, max_wavelength, 400)
    phase_count = 10
    
    plt.xlabel('Wavelength')
    plt.ylabel('Normalized Flux')
    plt.grid()
    plt.plot(list(wavelengths) * phase_count, line_spectra, marker = '.', color = 'red', linewidth = 5, alpha = 0.3)
    plt.xscale('log')
    plt.savefig(f'./spectrum_deg.png')
    plt.close()


    dictionary = {'wavelength': list(wavelengths) * phase_count, 'flux': line_spectra}
    df = pd.DataFrame(dictionary)
    
    df.to_csv(f'./D/flux_vs_wavelength_data.csv')
            
    
    # plt.legend(phi_list)
    
    # plt.show()

   






    


    




