import numpy as np
import scipy as sp
import math
from math import radians, degrees, sin, cos, asin, acos, sqrt
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Forward import *

# sigma = 5.67e-8 #stefan's constant W / (m^2 K^4)
sigma = 2 * np.pi**5 / 15   # h = c = k = 1


class Star:

    def __init__(self, inclination_angle, temp, radius, v_e, num_of_patches, guess = False):
        spots_lat = np.array([0, np.pi/4, 0, 0, 0, 0])
        spots_long = np.array([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4])
        spots_radius = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
        spots_temp = np.array([8, 8, 8, 1, 1, 1])
        # spots_lat = spots_lat[0:1]
        # spots_long = spots_long[0:1]
        # spots_radius = spots_radius[0:1]
        # spots_temp = spots_temp[0:1]
        if guess == True:
            spots_lat = np.array([0, np.pi/4, 0, 0, 0, np.pi/4])
            spots_long = np.array([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4])
            spots_radius = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
            spots_temp = np.array([1, 2, 2.5, 6, 2.5, 5])
            # spots_lat = np.array([np.pi/4, np.pi/6, np.pi/8, 0, -np.pi/8, -np.pi/6])
            # spots_long = np.array([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 7*np.pi/4])
            # spots_radius = np.array([0.5e6, 0.5e6, 0.5e6, 0.5e6, 0.5e6, 0.5e6])
            # spots_temp = np.array([1, 2, 3, 2, 3, 1])
        

        self.inclination_angle = inclination_angle
        self.temp = temp
        self.radius = radius
        self.v_e = v_e

        self.spots_lat = spots_lat
        self.spots_long = spots_long
        self.spots_radius = spots_radius
        self.spots_temp = spots_temp
        # dtheta is the angle between nearby two longitudes
        self.num_latitudes, self.zones, self.dtheta = self.get_lats_and_zones(num_of_patches)
        self.I = self.make_image_vector(num_of_patches, spots_lat, spots_long, spots_radius, spots_temp)
        self.phase = 0

        self.stellar_disk_vector = self.get_stellar_disk(self.I)


    def add_sunspots(self, I, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes = int(self.num_latitudes)
        zones = self.zones

        angles = np.zeros(num_latitudes)
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        delta_angle = total_polar_angle / num_latitudes

        #define the latitudes we have in terms of angles
        for i in range(1, num_latitudes+1):
            polar_angle = delta_angle*i
            angles[i-1] = np.pi/2 - polar_angle

        for x in range(len(spots_lat)):
            init_latitude = spots_lat[x]
            init_longitude = spots_long[x]
            init_radius = spots_radius[x]

            def map_rad(self, I, spot_radius, zones):
                """
                Input: image vector I, zones, radius of the spots
                Output: new image space I
                """
                layered_I = []
                count = 0
                h = 0
                while h < len(zones):
                    layered_I.append(I[count:int(zones[h])+count])
                    count += int(zones[h])
                    h += 1

                def get_distance(x1, y1, sphere_radius=self.radius):
                    """
                    Calculate spherical distance of two points given their lon, lat and raduis of the sphere
                    """

                    """
                    converts xi, yi back to lon and lat using equations:
                        # i = math.floor((np.pi/2 - init_latitude) / delta_angle)
                        # delta_phi = 2*np.pi / zones[i]
                        # j = math.floor(init_longitude / delta_phi)
                    """
                    delta_phi_new = 2*np.pi / zones[y1]
                    lat1, lon1 = np.pi/2 - y1 * delta_angle, x1 * delta_phi_new
                    
                    dlon = lon1 - init_longitude
                    dlat = lat1 - init_latitude
                    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(init_latitude) * sin(dlon / 2) ** 2
                    return 2 * sphere_radius * asin(sqrt(a))
                
                #print(layered_I)
                h = 0
                while h < len(layered_I):
                    w = 0
                    while w < len(layered_I[h]):
                        if get_distance(w, h) <= spot_radius:
                            layered_I[h][w] = spots_temp[x]
                        #print(h,w)
                        w += 1
                    h += 1

                new_I = []
                for i in layered_I:
                    new_I.extend(i)

                return np.array(new_I)

            I = map_rad(self, I, init_radius, zones)
           
        return I

    def get_lats_and_zones(self, n):

        area_per_patch = self.calc_zone_area(self.radius, self.inclination_angle, n)
        length_zone = np.sqrt(area_per_patch)

        #calculate how many "latitudes" on sphere
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        num_latitudes = int(math.floor((total_polar_angle * self.radius) / length_zone))

        #calculate how many zones for each latitude
        delta_angle = total_polar_angle / num_latitudes

        zones = np.zeros(num_latitudes)

        for i in range(1, len(zones)+1):
            polar_angle = delta_angle*i
            num_zones = (2 * np.pi * self.radius * np.sin(polar_angle)) / length_zone

            zones[i-1] = int(math.floor(num_zones))
        # returns number of latitudes, an array of number of longitudes per lattitude,
        # and delta longitude
        return num_latitudes, zones, length_zone / self.radius

    def get_lat_lon(self, I, index):
        """
        Input: image vector I, index of the element in the original I
        Output: lat, lon of the element with given index
        """
        y = 0
        x = 0
        for idx, num in enumerate(self.zones):
            if index < num:
                y = idx
                x = index - 1
                break

            index = index - num
        num_latitudes = int(self.num_latitudes)
        angles = np.zeros(num_latitudes)
        total_polar_angle = np.pi - (np.pi/2 - self.inclination_angle)
        delta_angle = total_polar_angle / num_latitudes
        delta_phi_new = 2*np.pi / self.zones[y]
        lat, lon = (y + 1) * delta_angle, (x) * delta_phi_new
        #TODO: debug lon, val not correct
        return lat, lon


    def get_stellar_disk(self, lst):
        polar_angle = np.pi/2 - self.inclination_angle
        bins = self._sort_into_bins(lst)
        R = self.radius
        for idx, bin in enumerate(bins):
            theta = self.dtheta * (idx + 1)
            h = R*np.cos(theta) 
            delta_l = h*np.tan(polar_angle)
            if delta_l <= R*np.sin(theta):
                phi = 2*np.arccos((np.tan(polar_angle))/np.tan(theta))
                fraction_off = round(phi/(2*np.pi) * self.zones[idx])
                #bin[0:int(fraction_off//2 + 1)] = 0
                bin[len(bin)-int(fraction_off)-1:] = 0
        self.stellar_disk_vector = self._bins_to_I(bins)
        stellar_disk = self.stellar_disk_vector

        return stellar_disk

    # rotate to phase
    def rotate(self, delta_phase):
        self.spots_long = delta_phase + self.spots_long
        for idx, s in enumerate(self.spots_long):
            if s >= np.pi * 2:
                self.spots_long[idx] = s - np.pi * 2

        self.phase += delta_phase

        return self.make_image_vector(1, self.spots_lat, self.spots_long, self.spots_radius, self.spots_temp)
    



    def calc_zone_area(self, radius, inclination_angle, n):

        total_area = (4 * np.pi * radius**2) - (2 * np.pi * radius**2)*(1 - np.sin(inclination_angle))
        area_per_patch = total_area/n

        return area_per_patch

    def make_image_vector(self, n, spots_lat, spots_long, spots_radius, spots_temp):
        num_latitudes = self.num_latitudes
        num_zones = int(np.sum(self.zones))
        I = np.full(num_zones, self.temp)
        I = self.add_sunspots(I, spots_lat, spots_long, spots_radius, spots_temp)
        # change units of I to flux
        print(sigma * 5**4)
        I = sigma * (I**4)
        # print(I[0])
        return I

    def _sort_into_bins(self, I):
        arr = []
        start_idx = 0
        for z in self.zones:
            z = int(z)
            arr.append(copy.deepcopy(I[start_idx: start_idx + z]))
            start_idx = z + start_idx
        return arr

    def _bins_to_I(self, bins):
        I = []
        for bin in bins:
            I.extend(bin)
        return np.array(I)

    def plot_on_sphere(self, lst, savefig=False, parent_directory = './'):
        I = lst
        num_latitudes = self.num_latitudes
        zones = self.zones
        width = int(max(zones))
        height = int(num_latitudes)
        map = np.zeros((height, width)) - 1e7 
        bins = self._sort_into_bins(lst)
        for idx, bin in enumerate(bins):
            start_col = (width - len(bin)) // 2
            map[idx][start_col : start_col + len(bin)] = bin
        #colormap = plt.imshow(map, cmap='hot')
    
        colormap = plt.matshow(map, cmap = 'hot', norm = LogNorm(vmin=0.01, vmax=max(lst)*1000))
        plt.colorbar(colormap)
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        if savefig:
            plt.savefig(parent_directory + f'{self.phase * 180 / np.pi}_deg.png')
            plt.close()
        else:
            plt.show()

    # plots lst on sphere in 3d
    # def plot_on_sphere3d(self, lst):
        # bins = self._sort_into_bins(lst)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #
        # for idx, bin in enumerate(bins):
        #
        # # compute the x and y and z coordinate for each
        # xx, yy = np.linspace(-self.radius, self.radius, 1000), np.linspace(-self.radius, self.radius, 1000)
        # X, Y = np.meshgrid(xx, yy)
        # ax.plot_surface(X, Y, )




if __name__ == '__main__':
    s = Star(np.pi/4, 4, 3e6, 4, 10000, guess = False)
    # # s.make_image_vector(2000, np.array([0]), np.array([0]),1,np.array([2000]))
    # s_new = Star(np.pi/2, 5, 3e6, 4e6, 10000)
    # v_r = np.array([get_v_radial(s_new, i) for i in range(len(s_new.I))])
    # lon = np.array([s_new.get_lat_lon(s_new.I, i)[1] for i in range(len(s_new.I))])
    # #s.plot_on_sphere(lon)
    # #s.plot_on_sphere(s.I)
    # '''for i in range(15):
    #     dtheta = 360 / 50
    #     s.plot_on_sphere()
    #     s.rotate(dtheta * np.pi / 180)'''
    # #D = doppler_shift(s_new)
    # #s.plot_on_sphere(D)
    # # s.rotate(90* np.pi / 180)
    # # s.plot_on_sphere()
    # # print(s.I)

    # for i in range (1):
    #     s_new = Star((90 - i/50 * 90)/180*np.pi, 5, 3e6, 4e6, 10000)
    #     v_r = np.array([get_v_radial(s_new, i) for i in range(len(s_new.I))])
    #     #lon = np.array([s_new.get_lat_lon(s_new.I, i)[1] for i in range(len(s.I))])
    #     s_new.plot_on_sphere(v_r)






    # s.get_stellar_disk() 
    # # print(s.stellar_disk_vector == s.I)
    # print(s.stellar_disk_vector[0])
    # print(s.I[0])
    # #s.plot_on_sphere(s.I)
    # # s.plot_on_sphere(s.I - s.I)

    s.plot_on_sphere(s.I)

    #s.plot_on_sphere(s.stellar_disk_vector)