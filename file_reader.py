import numpy as np
import matplotlib.pyplot as plt
import glob


# file_name = "./HD199178_spectra/J_A+A_625_A79_sp_HD199178_1994_07-49552.340.dat.csv"

root_dir = "./HD199178_spectra/"
all_files = glob.glob(root_dir + "/*.csv")
period = 3.3175

def make_file_name(root_dir, year, month, HJD):
    baseline = "/J_A+A_625_A79_sp_HD199178_"
    file_name = baseline + str(year) + "_" + f"{month:02}" + "-" + "{0:.3f}".format(HJD)
    file_name = file_name + ".dat.csv"
    return root_dir + file_name

def read_single_file(file_name):
    wavelength = []
    flux = []
    with open(file_name) as f:
        lis = np.array([line.split() for line in f])
        for data in lis[4 : ]:
            wavelength.append(float(data[0]))
            flux.append(float(data[1]))
    return np.array(wavelength), np.array(flux)

def filtered_coord(wavelength, flux):
    """
    Input: unfilered lists of wavelength and flux; type == 2D array
    Output: new lists of wavelength and flux s.t. new_lst = [[old1], [old2]] if a gap exists between [old1] and
        [old2]; type == 2D array: new_wavelength, new_flux

    Precondition: lists of wavelength and flux have the same size
    """
    assert len(wavelength) == len(flux), 'Coordinates have the same size'

    new_wl = []
    new_fx = []
    i = 0
    while i < len(wavelength):
        indv_wl, indv_fx = [wavelength[i][0]], [flux[i][0]]
        j = 1
        while j < len(wavelength[i]):
            if is_gap(indv_wl, wavelength[i][j]):
                new_wl.append(indv_wl)
                new_fx.append(indv_fx)
                indv_wl, indv_fx = [wavelength[i][j]], [flux[i][j]]
            else:
                indv_wl.append(wavelength[i][j])
                indv_fx.append(flux[i][j])
            j += 1
        new_wl.append(indv_wl)
        new_fx.append(indv_fx)
        i += 1
    return new_wl, new_fx

def filtered_coord_single(wavelength, flux, chunk_num=0):
    """
    Input: unfilered lists of wavelength and flux; type == 2D array
    Output: new lists of wavelength and flux s.t. new_lst = [[old1], [old2]] if a gap exists between [old1] and
        [old2]; type == 2D array: new_wavelength, new_flux

    Precondition: lists of wavelength and flux have the same size
    """
    assert len(wavelength) == len(flux), 'Coordinates have the same size'

    new_wl = []
    new_fx = []
    has_gap = False
    i = 1
    indv_wl, indv_fx = [wavelength[0]], [flux[0]]
    while i < len(wavelength):
        has_gap = is_gap(indv_wl, wavelength[i])
        if has_gap:
            new_wl.append(indv_wl)
            new_fx.append(indv_fx)
            indv_wl, indv_fx = [wavelength[i]], [flux[i]]
        else:
            indv_wl.append(wavelength[i])
            indv_fx.append(flux[i])
        i += 1
    if not has_gap:
        new_wl.append(indv_wl)
        new_fx.append(indv_fx)
    print("Number of chunks: " + str(len(new_wl)))
    return new_wl[chunk_num], new_fx[chunk_num]

def is_gap(cur_lst, item):
    """ 
    Identify if a gap exists between the current list item and the next item based on the difference in mean.
    If the difference in mean between the new next element and current element in wavelength is 10x bigger 
    than the mean between the current element and the previous element, return True; False otherwise.

    Precondition: cur_lst has at least one element
    """
    assert len(cur_lst) > 0, 'empty list'
    
    if len(cur_lst) == 1:
        return False
    last_elem = cur_lst[len(cur_lst)-1]
    sec_last = cur_lst[len(cur_lst)-2]
    if (10 * abs(last_elem - sec_last)) < abs(item - last_elem):
        return True
    return False

def read_data(year, month):
    """
    Takes in the year and month of desired files, takes in an extra index CHUNK_NUM to select the chunk among
    the dataset with gaps.
    """
    wavelength_lst, flux_lst = [], []

    if month < 10:
        month = "0" + str(month)

    for file in all_files:
        if (str(year) + "_" + str(month)) in file:
            wavelength, flux = read_single_file(file)
            wavelength_lst.extend(wavelength)
            flux_lst.extend(flux)

    def select_chunk(chunk_num=0):
        print(len(wavelength_lst))
        return filtered_coord_single(wavelength_lst, flux_lst, chunk_num)
    print(len(wavelength_lst))
    return select_chunk

def make_matrix(x, y):
    """
    Input: x, y coordinates of the vector(s)
    Output: [[x1,y1],[x2,y2]]
    """
    mat = []
    for i in x:
        for j in y:
            mat.append([i,j])
    return np.array(mat)

def get_all_phases(T, initial=0):
    """
    Input: period, initial phase
    Output: all the phases for the 100 files
    """

    phases = [initial]
    i = 1
    while i < len(all_files):
        HJD_this = float(all_files[i][len(all_files[i])-17 : len(all_files[i])-8])
        HJD_last = float(all_files[i-1][len(all_files[i-1])-17 : len(all_files[i-1])-8])
        phase = (HJD_this - HJD_last) / T * 2 * np.pi + phases[i-1]
        phase = phase % (2*np.pi)
        phases.append(phase)
        i += 1
    return phases

def get_phases(year, month):
    """
    Input: Specific year and month desired in int
    Output: An array of phases for the given year and month
    """
    assert type(month) == int, 'month must be in int'
    
    if month < 10:
        month = "0" + str(month)

    index = []
    i = 0
    while i < len(all_files):
        if (str(year) + "_" + str(month)) in all_files[i]:
            index.append(i)
        i += 1
    all_phases = get_all_phases(period)
    phases = []
    for j in index:
        phases.append(all_phases[j])

    return np.array(phases)

print(get_phases(1994, 11))

"""
#new_wavelength, new_flux = filtered_coord(wavelength_lst, flux_lst)
wa, l = read_data(1998, 10)(2)
print(wa, l)
fig = plt.figure()
ax = fig.add_subplot(111)
for w, f in zip(wa, l):
    ax.clear()
    ax.plot(w, f)
    #plt.pause(0.1)

"""
#print(make_matrix(w, l))





