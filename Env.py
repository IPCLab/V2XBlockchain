from __future__ import division
import random, sys
import math
from datetime import timedelta, datetime
from math import radians, sin, cos, sqrt, atan2
from pyorbital.orbital import Orbital
import var
import numpy as np
import pandas as pd
from scipy.spatial import distance
# plt.use('TkAgg')
from pandas import DataFrame
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.fcm import fcm
import scipy.spatial as spatial
import itur
import re

# import blocksim api
from Models.Ethereum.Consensus import Consensus
from Models.Ethereum.Transaction import LightTransaction as LT, FullTransaction as FT
from Models.Ethereum.Block import Block
from Models.Ethereum.Node import Node
# 這裡會發生cicular import的問題，我先註解起來
# from Models.Ethereum.Incentives import Incentives
from Scheduler import Scheduler
from Event import Event, Queue
from Models.Ethereum.Consensus import Consensus as c
from Models.Ethereum.Distribution.DistFit import DistFit
# use for blocksim

ans = []
v = 0
data_n = pd.DataFrame()

tmp = pd.DataFrame()
alt =500


indexitreation = 0
count_index = -1
rankindex = 0
ans = []
dfcluster = pd.DataFrame()
dfcluster = pd.DataFrame()
new_df = pd.DataFrame()
new_df1 = pd.DataFrame()
new_df3 = pd.DataFrame()

new_df_S = pd.DataFrame()
new_df1_S = pd.DataFrame()
new_df3_S = pd.DataFrame()

veh_position = []
ST_position = []
id_vehicle = []
id_satellite = []

demend = 0
startwith = True
resource_use_reward = []
ST_resource_use_reward = []
seed_value = var.seed
random.seed(seed_value)
np.random.seed(seed_value)
#

step_count = 1
reward = 0

startwith = True
class V2Vchannels_LTE:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5  # hight
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_a, position_b):

        d1 = abs(position_a[0] - position_b[0])
        d2 = abs(position_a[1] - position_b[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        # 直線距離有沒有東西
        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0,
                                                                                                               3)  # standard dev is 3 db

class V2Ichannels_LTE:

    # Simulator of the V2I channels

    def __init__(self, bs):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50

        self.BS_position = []  # center of the grids
        self.shadow_std = 8
        for row in bs.itertuples(index=True, name='Pandas'):
            self.BS_position.append([getattr(row, "X"), getattr(row, "Y")])

    def determine_closest(self, BS, x, y):
        min_distance = float('inf')
        index_of_closest = -1
        for index, Base_station in enumerate(BS):
            x_coord, y_coord = Base_station
            current_distance = distance.euclidean((x, y), (x_coord, y_coord))
            if current_distance < min_distance and current_distance != 0:
                min_distance = current_distance
                index_of_closest = index
        return index_of_closest

    def get_path_d(self, position_a):

        index = self.determine_closest(self.BS_position, position_a[0], position_a[1])
        return index

    def get_path_loss(self, position_a, position_b):

        # index = self.determine_closest(self.BS_position, position_a[0], position_a[1])
        # print(self.BS_position[index])

        d1 = abs(position_a[0] - position_b[0])
        d2 = abs(position_a[1] - position_b[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8)

class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0 # use for datetime.utcnow() and it was used for Sattelite, but we did not use it anymore
        self.h_bs = 1.5 # heights of the base station
        self.h_ut = 1.5 # heights of the car station
        self.fc = 3.5 # 3.5GHz
        self.decorrelation_distance = 10  #3GPP TR 36.885 for Shadowing
        self.shadow_std = 3 #3GPP TR 36.885 for LOS Shadowing

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])

        d = math.hypot(d1, d2) + 0.001
        d3d = math.sqrt(d ** 2 + abs(self.h_bs - self.h_ut) ** 2)

        def PL_Los():
            return 38.77 + 16.7 * np.log10(d3d) + 18.2 * np.log10(self.fc)

        def PL_NLos():
            return 36.85 + 30.0 * np.log10(d3d) + 18.9 * np.log10(self.fc)

        rand = np.random.uniform() 
        blocker_height = 2.0
        PL_Los_p = min(1, 1.05 * np.exp(-0.0114 * d))
        PL_NLosv_p = min(0, 1 / (0.0312 * d) * np.exp(-(np.log(d) - 5.0063) ** 2 / 2.4544))
        PL_NLos_p = 1 - PL_Los_p - PL_NLosv_p

        if rand <= PL_Los_p:
            PL = PL_Los()
        elif rand <= PL_NLos_p:
            PL = PL_NLos()
            self.shadow_std = 4  # if Non line of sight, the std is 4
        elif rand <= (1 - PL_Los_p):
            if min( self.h_bs, self.h_ut ) > blocker_height:
                mu_a, sigma_a = 0, 0
            elif max (self.h_bs, self.h_ut )  < blocker_height:
                mu_a, sigma_a = 9.0 + max(0, 15.0 * np.log10(d) - 41.0), 4.5
            else:
                mu_a, sigma_a = 5.0 + max(0, 15.0 * np.log10(d) - 41.0), 4.0
            ANLOSv = max(0, np.random.normal(mu_a, sigma_a))
            PL = PL_Los() + ANLOSv
            self.shadow_std = 4  # if Non line of sight, the std is 4
        else:
            PL = PL_NLos()
            self.shadow_std = 4  # if Non line of sight, the std is 4

        # print("v2v pl",PL)
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return self.shadow_std # self.shadow_std = 3 #3GPP TR 36.885 for LOS Shadowing
    
class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self, bs):
        self.h_bs = 25
        self.h_ut = 1.5 
        self.fc = 3.5 #3GPP TR 37.885 Below 6GHz Parameter
        self.Decorrelation_distance = 50

        self.BS_position = []  # center of the grids
        self.shadow_std = 4
        for row in bs.itertuples(index=True, name='Pandas'):
            self.BS_position.append([getattr(row, "X"), getattr(row, "Y")])

    def determine_closest(self, BS, x, y):
        min_distance = float('inf')
        index_of_closest = -1
        for index, Base_station in enumerate(BS):
            x_coord, y_coord = Base_station
            current_distance = distance.euclidean((x, y), (x_coord, y_coord))
            if current_distance < min_distance and current_distance != 0:
                min_distance = current_distance
                index_of_closest = index
        return index_of_closest

    def get_path_d(self, position_A):

        index = self.determine_closest(self.BS_position, position_A[0], position_A[1])
        return index

    # + self.shadow_std * np.random.normal()
    def get_path_loss(self, position_A, position_B):

        # index = self.determine_closest(self.BS_position, position_A[0], position_A[1])
        # print(self.BS_position[index])

        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        # d3 = abs(self.h_bs - self.h_ut)  # h_ut  == rx_height and if rx_height is different, you should use rx_height
        d = math.hypot(d1, d2) + 0.001
        # d3D the Euclidean distance between TX and RX in 3D space in meters.
        d3d = math.sqrt(d ** 2 + abs(self.h_bs - self.h_ut) ** 2)
        c = 3 * 10 ** 8  # the propagation velocity in free space
        # from note 3GPP TR 38.901 V16.1.0 Table 7.4.1-1, 1 UMa hE = 1m with a probability equal to 1/(1+C(d2D, hUT))
        if d <= 18:
            g = 0
        elif d > 18:
            g = 5 / 4 * (d / 100) ** 3 * np.exp(-d / 150)
        h_e = 1
        if self.h_ut < 13:
            h_e = 1
        if self.h_ut >= 13 and self.h_ut <= 23:
            h_e = 1 / (((self.h_ut - 13) / 10) ** (1.5) * g)
        # Urban: TR 38.901 UMa LOS
        h_e_bs = self.h_bs - h_e  # The effective of the actual antenna heights
        h_e_ut = self.h_ut - h_e  # The effective environment height
        d_bp = 4 * h_e_bs * h_e_ut * self.fc * (10 ** 9) / c  # Breakpoint distance
        C_h_ut = 0

        # 3GPP TR 38.901 V16.1.0 Table 7.4.2-1 LOS probability
        def PL_Los_p():
            if d <= 18:
                return 1
            else:
                if self.h_ut <= 13:
                    C_h_ut = 0
                if self.h_ut > 13 and self.h_ut <= 23:
                    C_h_ut = (((self.h_ut - 13.0) / 10) ** (1.5))
                return (18.0 / d + np.exp(-d / 63.0) * (1 - 18.0 / d)) * (
                        1 + C_h_ut * (5.0 / 4.0) * ((d / 100.0) ** 3) * np.exp(-d / 150.0))

        # 3GPP TR 38.901 V16.1.0 Table 7.4.1-1: Pathloss models
        def PL_Los():
            if d >= 10 and d <= d_bp:
                return 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(self.fc)  # PL_1
            elif d >= d_bp and d <= 5000:
                return 28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(self.fc) - 9 * np.log10(
                    (d_bp) ** 2 + (self.h_bs - self.h_ut) ** 2)  # PL_2
            else:
                return 32.4 + 20.0 * np.log10(self.fc) + 30.0 * np.log10(d3d)  # Optional loss
                self.shadow_std = 7.8

        def PL_NLos():

            return 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(self.fc) - 0.6 * (self.h_ut - 1.5)

        rand = np.random.uniform(0, 1)

        if rand <= PL_Los_p():
            PL = PL_Los()
        else:
            if d >= 10 and d <= 5000:
                PL = max(PL_Los(), PL_NLos())
                self.shadow_std = 6  # if Non line of sight, the std is 4
            else:
                PL = PL_NLos()
                # Optional
                # PL = 32.4 + 20 * np.log10(self.fc) + 30 * np.log10(d3d)
                # self.shadow_std = 7.8
        # print("V2I pl",PL,'d3d',d3d)
        return PL

    # def get_path_loss1(self, position_A):
    #
    #     index = self.determine_closest(self.BS_position, position_A[0], position_A[1])
    #     # print(self.BS_position[index])
    #
    #     d1 = abs(position_A[0] - self.BS_position[index][0])
    #     d2 = abs(position_A[1] - self.BS_position[index][1])
    #     distance = math.hypot(d1, d2)
    #     return 128.1 + 37.6 * np.log10(
    #         math.sqrt(distance ** 2 + (self.h_bs - self.h_ut) ** 2) / 1000)  # + self.shadow_std * np.random.normal()
    def get_shadowing(self, delta_distance, shadowing):
        # nVeh = len(shadowing)
        # sha = np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
        #        + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, self.shadow_std)

        #
        #
        return self.shadow_std

class V2Schannels:

    # Simulator of the V2S channels

    def __init__(self):
        global alt
        self.h_bs = alt
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.sat_fre = 30    # 30GHz
        # self.Tx_EIRP = 23 # [dBm] hint: 0 dBw = 30 dBm
        # self.Rx_GT = 5  # [dB/T]
        # self.BW = 0.4 # Bandwidth [MHz]
        self.SF = 3 #Shadow fading margin [dB]
        self.Sc_Loss = 2.2 #Scintillation Loss [dB]
        self.R = 6371.0
        #self.ST_position = []  # center of the grids
        # self.shadow_std = 8 
        self.shadow_std = 3
        # 23/12/21 Ibrahim Update (Line556~557)

    def determine_closest_satellite(self, satellites, lat, lon):
        """
        Determines the closest satellite to a given point (lat, lon).

        satellites: list of tuples with (lat, lon, altitude) for each satellite.
        lat, lon: latitude and longitude of the point on Earth.

        Returns the index of the closest satellite.
        """
        min_distance = float('inf')
        index_of_closest = -1

        for index, satellite in enumerate(satellites):
            sat_lat, sat_lon  = satellite
            current_distance = self.DistanceCalculate(sat_lat,sat_lon,lat, lon)

            if current_distance < min_distance:
                min_distance = current_distance
                index_of_closest = index

        return index_of_closest

    # def get_path_d(self, position_a):

    #     index = self.determine_closest(self.ST_position, position_a[0], position_a[1])
    #     return  index

    # def DistanceCalculate(self,lat1, lon1, lat2, lon2):
    #         global  alt
    #         R = 6371  # Earth's radius in kilometers
    #
    #
    #
    #         # Convert latitude and longitude to radians
    #         lat1_rad = radians(lat1)
    #         lon1_rad = radians(lon1)
    #         lat2_rad = radians(lat2)
    #         lon2_rad = radians(lon2)
    #
    #         # Haversine formula
    #         dlon = lon2_rad - lon1_rad
    #         dlat = lat2_rad - lat1_rad
    #         a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    #         c = 2 * atan2(sqrt(a), sqrt(1 - a))
    #         surface_distance = R * c  # Surface distance in km
    #
    #         return sqrt(surface_distance**2 + alt**2)  # Distance to satellite in km
    def earth_to_cartesian(self,lat, lon, alt=0):
        """Convert Earth-centered lat, lon, alt to Cartesian x, y, z."""
        R = 6371  # Earth's radius in kilometers
        lat_rad = radians(lat)
        lon_rad = radians(lon)

        x = (R + alt) * cos(lat_rad) * cos(lon_rad)
        y = (R + alt) * cos(lat_rad) * sin(lon_rad)
        z = (R + alt) * sin(lat_rad)

        return x, y, z

    def DistanceCalculate(self,lat1, lon1,  lat2, lon2):
        """Compute the distance between two points given lat, lon, alt."""
        x1, y1, z1 = self.earth_to_cartesian(lat1, lon1, alt)
        x2, y2, z2 = self.earth_to_cartesian(lat2, lon2, 0)

        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    def get_path_loss(self, position_a,position_b):


        # index = self.determine_closest(self.BS_position, position_a[0], position_a[1])
        # print(self.BS_position[index])

        # distance = math.hypot(d1, d2)

        # lat_grid = position_a[0]
        # lon_grid = position_a[1]
        # lat_sat  = position_b[0]
        # lon_sat  = position_b[1]
        lat_grid, lon_grid = position_a
        lat_sat, lon_sat = position_b
        c =  2.99792458E8
        # lat = 24.966957 #Melting Layer height. Chungli's latitude is 24'N
        # lon  = 121.239038


        L_red = 0.3  # cumulus 0.3(g/m^3) cumulonimbus 1-3(g/m^3)

        # elevation_angle = math.atan( distance/ (self.h_bs - self.h_ms )+0.0001) #rad

        hs = itur.topographic_altitude(lat_grid, lon_grid)

        distance = self.DistanceCalculate(lat_sat, lon_sat, lat_grid, lon_grid)



        # elevation_angle = itur.utils.elevation_angle(self.h_bs, lat_sat, lon_sat, lat_grid, lon_grid)

        # rain_at = itur.models.itu618.rain_attenuation(lat_grid, lon_grid, self.sat_fre, elevation_angle,hs=hs,p=0.51)
        
        elevation_angle = itur.utils.elevation_angle(h=self.h_bs,lat_s= lat_sat, lon_s=lon_sat, lat_grid=lat_grid, lon_grid =lon_grid)

        rain_at = itur.models.itu618.rain_attenuation(lat=lat_grid,lon= lon_grid, f=self.sat_fre, el = elevation_angle, hs=hs, p=0.51)

        ################# Cloud attenuation ######################

        cloud_at = itur.models.itu840.cloud_attenuation(lat=lat_grid, lon=lon_grid,f= self.sat_fre, el = elevation_angle, p=0.51,
                                                        Lred=[L_red])
#        cloud_attenuation = np.append(cloud_attenuation, cloud_at)

        ################# water&oxygen attenuation #################

        # water_oxy_at = ( water_oxy_at_rate * altitude )/(math.sin(elevation_angle) )
#        water_oxy_attenuation = np.append( water_oxy_attenuation, water_oxy_at)

        ################# Total attenuation #######################

        sum = rain_at.item()+cloud_at.item()


        sum = float(np.asarray(re.findall("\d+\.\d+", str(sum.item()))))


        total_at =   self.Sc_Loss + self.SF +sum




        #
        # print("Pro_delay",92.4 + 20 * np.log10(self.sat_fre) + 20 * np.log10(
        #     math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000))
        # print("total_at",92.4 + 20 * np.log10(self.sat_fre) + 20 * np.log10(
        #     math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) + total_at)
        # print("D",math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)
        # loss =92.4 + 20 * np.log10(self.sat_fre) + 20 * np.log10(  distance  ) + total_at
        loss = 10 * np.log10(((4 * np.pi * distance * 1000 * self.sat_fre * 1e9) / c) ** 2) + total_at

        return loss + self.shadow_std
        # 23/12/21 Ibrahim Update (Line694)

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity, id, area, Geo_position):
        # self.time = time
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.id = id
        self.area = area
        self.Geo_position = Geo_position

        self.neighbors = []
        self.neighbors1 = []
        self.destinations = []
        self.neighbor_vehicle_destination = []
        self.penalty_switch = []

        self.neighborsBS = []
        self.neighborsBS1 = []
        self.destinationsBS = []
        self.neighbor_vehicle_destinationBS = []

        self.neighborsST = []
        self.neighborsST1 = []
        self.destinationsST = []
        self.neighbor_vehicle_destinationST = []

        self.success_fail = 0

class Environ:
    def __init__(self, method, test, Satellite):
        self.method = method 
        self.Satellite_ = Satellite
        self.test = test
        
        # 將環境中所有RSU(BS)設定為Blockchain Node
        # 這裡宣告一個空list來存所有RSU(BS) Node
        self.rsu_nodes_list = []
        
        if test == True:
            chunksize = 10 ** 6  # adjust this value depending on your system's memory
            chunks = []
            # for chunk in pd.read_csv('Data_0722_2_with_area_type.csv', chunksize=chunksize, on_bad_lines="skip"):
            for chunk in pd.read_csv(sys.argv[3], chunksize=chunksize, on_bad_lines="skip"):    
                # perform data preprocessing here if needed
                chunks.append(chunk)
            self.df = pd.concat(chunks, axis=0)

            # 我們的環境中，BS和RSU混用
            self.bs = pd.read_csv('data_config.csv', on_bad_lines ="skip")   
            num_rsu = len(self.bs)
            num_miner = 30
            num_general_node = num_rsu - num_miner

            # 分配RSU的Hashpower
            rsu_hashpowers = np.random.normal(loc = 0, scale = 1, size = num_rsu) # take N(0, 1) as initial hashpower setting
            # print(rsu_hashpowers)
            # print(sorted(rsu_hashpowers)[num_general_node-1])
            # print(rsu_hashpowers + sorted(rsu_hashpowers)[num_general_node])
            rsu_hashpowers = np.clip(rsu_hashpowers + (-sorted(rsu_hashpowers)[num_general_node-1]), 0, None) 
            # shift the distribution according to the num_general_node'th small value, and clip all value small than 0, 
            # ensure that there is num_general_node rsu node with 0 hash power 
            # print(rsu_hashpowers)

            # 為每一個RSU(BS)建立Node
            for idx, row in enumerate(self.bs.itertuples(index = True, name = 'Pandas')):
                rsu_node = Node(rsu_id = "rsu" + str(idx), rsu_hashpower = rsu_hashpowers[idx])
                Node.generate_gensis_block(rsu_node)
                self.rsu_nodes_list.append(rsu_node)      
                # print(f"RSU Node ID: {rsu_node.id}; Hashpower: {rsu_node.hashPower}")
                # print(f"Local Blockchain of {rsu_node.id}: {rsu_node.localBlockchain[0].id}") 
                    
            self.newsat = pd.read_csv('data_sat.csv', on_bad_lines ="skip")
            
            self.vehicle_time_id = var.train_sumo_step # var.train_sumo_step = 65
            self.agent = var.agent_test # agent_test = range(100,600,100)
            self.limted_cluster = var.limted_cluster # limted_cluster = 22
            self.limted_cluster_full = var.limted_cluster_full # limted_cluster_full = 22
            self.n_cluster_ = var.n_cluster_test # n_cluster_test = 20
            
            # payload size
            self.demand_size = int(var.V2V_payloadsize_test * 8) # V2V_payloadsize_test = V2V_payloadsize = 250
            self.demand_size_all = int(var.V2I_payloadsize_test * 8 * (demend + 1)) # V2I_payloadsize_test = V2I_payloadsize = 500 & demand = 0
            self.demand_s_size = int(var.V2S_payloadsize_test * 8) # V2S_payloadsize_test = V2S_payloadsize = 500

        # 這裡 else 做的事情和上面 if test == True 是一樣的    
        else:
            chunksize = 10 ** 6  # adjust this value depending on your system's memory
            chunks = []
            # for chunk in pd.read_csv('Data_0722_2_with_area_type.csv', chunksize=chunksize, on_bad_lines="skip"):
            for chunk in pd.read_csv('data_test_area_100.csv', chunksize=chunksize, on_bad_lines="skip"):
                # perform data preprocessing here if needed
                chunks.append(chunk)

            self.df = pd.concat(chunks, axis=0)
            
            self.bs = pd.read_csv('data_config.csv', on_bad_lines ="skip")
            
            self.newsat = pd.read_csv('data_sat.csv', on_bad_lines ="skip")
            
            self.vehicle_time_id = var.train_sumo_step
            self.agent = var.agent_tarin
            self.limted_cluster = var.limted_cluster
            self.limted_cluster_full = var.limted_cluster_full
            self.n_cluster_ = var.n_cluster_train
            
            self.demand_size = int(var.V2V_payloadsize * 8)
            self.demand_size_all = int(var.V2I_payloadsize * 8 * (demend + 1))
            self.demand_s_size = int(var.V2S_payloadsize * 8)

        # channel model 現在mode設定為NR
        if var.mode == 'LTE':
            self.V2Vchannels = V2Vchannels_LTE()
            self.V2Ichannels = V2Ichannels_LTE(self.bs)            
        else: # mode = NR
            self.V2Vchannels = V2Vchannels()
            # 將儲存Base Station dataset的dataframe bs作為class V2Ichannels的input參數
            self.V2Ichannels = V2Ichannels(self.bs)
        self.V2Schannels = V2Schannels()
        

        self.sumo_step = 0

        self.count_veh = 0
        self.vehicles = []
        self.veh_position = []
        self.veh_area = []
        self.neighbor_vehicles = []        
        self.ST_position = []
        self.new_array = []

        self.n_RB_max_r = 0
        self.n_SRB_max_r = 0
        self.n_RB_max = var.n_RB
        self.n_SRB_max = var.SRB_max
        self.SRB_max = var.SRB_max

        self.demand = []
        self.demand_urgent = []

        self.Switch_count = 0
        self.switch_ratio = 0        
        
        self.reward = 0
        self.restor = True        
        self.to_serv_from = []
        self.t = datetime.utcnow()

        self.V2I_Shadowing = []
        self.V2V_Shadowing = []
        self.distance = []
        self.delta_distance = []
        self.delta_distance1 = []

        self.change = True
        self.mode_state = []
        self.reward_succ = 1

        self.V2V_channels_abs = []
        self.V2I_channels_abs = []
        self.V2S_channels_abs = []

        self.check = False
        self.SINR_th = 0  # dB

        self.V2I_power_dB = 23  # dBm
        self.V2S_power_dB = 33.5  # dBm

        self.power_selection_v2i = []
        self.power_selection_v2i_n = []
        self.power_selection_v2s = []
        self.power_selection_v2s_n = []

        self.largev = 0

        self.V2V_power_dB_List = [23, 15, 10, 17]  # the power levels

        if var.mode == 'LTE':
             self.sig2_dB = -114.0
        else:
            self.sig2_dB = -174.0

        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.NoiseFigure = 1.2

        self.vehAntGain = 3
        self.vehNoiseFigure = 9

        self.sig2 = self.sig2_dB #10 ** ((self.sig2_dB )/ 10)
        # self.STBW =  40 * int(1e6)  # Bandwidth [MHz]
        self.STBW =  20 * int(1e6)  # Bandwidth [MHz]
        # self.ueEIRP = 23
        self.stGT = 1.1
        self.ueTXGain = 43.2  #dBi
        self.stRXGain = 30.5 #dBi
        self.n_RB = 0
        self.K = 1.38065e-23 #Boltzmann Constant
        self.T = 273.15
        
        self.n_Veh = 0
        self.n_neighbor = 0
        
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz if it is more or less than 1 MHz change in Data rate
        # self.bandwidth = 0.5 * int(1e6)
        
        self.v2v_selection = 0
        self.v2i_selection = 0
        self.v2s_selection = 0
        self.dump_selection = 0

        self.v2v_selection_all = 0
        self.v2i_selection_all = 0
        self.v2s_selection_all = 0
        self.dump_selection_all = 0

        self.SNR_SINR_list = []
        self.SNR_SINR_list_ep = 0

        self.SNR_list_ep_V2V = []
        self.SNR_list_ep_V2I = []
        self.SNR_list_ep_V2S = []

        self.SINR_list_ep_V2V = []
        self.SINR_list_ep_V2I = []
        self.SINR_list_ep_V2S = []
        
        self.urban_SNR_per_episode_V2V = []
        self.suburban_SNR_per_episode_V2V = []
        self.rural_SNR_per_episode_V2V = []

        self.urban_SNR_per_episode_V2I = []
        self.suburban_SNR_per_episode_V2I = []
        self.rural_SNR_per_episode_V2I = []

        self.urban_SNR_per_episode_V2S = []
        self.suburban_SNR_per_episode_V2S = []
        self.rural_SNR_per_episode_V2S = []

        self.step_count_V2V = np.ones(len(self.vehicles))
        self.step_count = np.ones(len(self.vehicles))        
        self.step_count_V2S = np.ones(len(self.vehicles))
        
        self.V2V_Rate_latency = np.zeros(len(self.vehicles))
        self.V2I_Rate_latency = np.zeros(len(self.vehicles))
        self.V2S_Rate_latency = np.zeros(len(self.vehicles))
        
        self.Data_rate_V2V_all = np.zeros(len(self.vehicles))
        self.Data_rate_V2I_all = np.zeros(len(self.vehicles))
        self.Data_rate_V2S_all = np.zeros(len(self.vehicles))
        
        self.Data_rate_V2V_all_check = np.zeros(len(self.vehicles))
        self.Data_rate_V2I_all_check = np.zeros(len(self.vehicles))
        self.Data_rate_V2S_all_check = np.zeros(len(self.vehicles))
        
        self.remain = np.zeros(len(self.vehicles))
        self.Data_rate_spacific = np.zeros(len(self.vehicles))
        
        self.SNR_rate_spacific_V2V = np.zeros(len(self.vehicles))
        self.SNR_rate_spacific_V2I = np.zeros(len(self.vehicles))
        self.SNR_rate_spacific_V2S = np.zeros(len(self.vehicles))

        self.demand_step = var.append
        self.switch = []
        self.demand_switch_V2V = []
        self.demand_switch_V2I = []
        self.demand_switch_V2S = []
        
        self.V2V_Rate = []
        self.V2I_Rate = []
        self.V2S_Rate = []
        
        self.V2V_SINR_all = []
        self.V2I_SINR_all = []
        self.V2S_SINR_all = []
        
        self.delay = []
        self.state_to_serv_from = []
        
        self.V2V_Interference_all = []
        self.V2I_Interference_all = []
        self.V2S_Interference_all = []
        
        # self.act_store = np.zeros(650, dtype='int32')  
        # the size is the maximum number of vehicles on the road
        self.df_act_store = pd.DataFrame(columns=['id', 'act', "step"])
        self.df_act_store_latency = pd.DataFrame(columns=['id', 'latency', "step"])
        
        self.truck_record = []
        self.veh_record = []
        self.act_store_index = 0


    def add_new_vehicles(self, start_position, start_direction, start_velocity, id, area, Geo_position):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity, id, area, Geo_position))
        # print(f"環境中的車子數量: {len(self.vehicles)}")
        # for vehicle in self.vehicles:
            # print(f"time of the vehicle {vehicle.id}: {vehicle.time}")
            # print 整個 Vehicle 物件的所有屬性及其值，可以使用 vars(vehicle) 或 vehicle.__dict__
            # print(f"Vehicle details: {vars(vehicle)}")
            # print(f"Start position of Vehicle {vehicle.id}: {vehicle.position}")

    def add_new_neighbor_vehicles(self, start_position, start_direction, start_velocity, id, area,Geo_position):

        self.neighbor_vehicles.append(Vehicle(start_position, start_direction, start_velocity, id, area,Geo_position))
    
    def add_new_satellite(self, start_position, start_direction, start_velocity, id, area,Geo_position):
        self.satellite.append(Vehicle(start_position, start_direction, start_velocity, id, area,Geo_position))

    def add_new_vehicles_by_number(self, n):
        self.renew_positions()


    def get_group(self, g, key):
        if key in g.groups: return g.get_group(key)
        return pd.DataFrame()

    def get_Geo_coordinate(self, time):

      
        o = Orbital(satellite=var.sat_name)



        self.t += timedelta(seconds=time)
        lon, lat, alt = o.get_lonlatalt(self.t)
        # lon, lat = np.rad2deg((lon, lat))
        # az, el = o.get_observer_look(self.t, obs_lon, obs_lat, obs_alt)




        return  lon, lat


    def averagex(self,x):
        if type(x) != list():
            return np.average(x)
        elif  x.count(0) ==0   :
            return np.average(x)
        x = sum(x) / (len(x) - x.count(0)) if (len(x) - x.count(0)) != 0 else 0
        return x
    
    
    def renew_positions(self):
        global alt
        global indexitreation
        global resource_use_reward
        global ST_resource_use_reward       
        global tmp
        global ans
        global dfcluster
        global rankindex
        global veh_position
        global ST_position
        global count_index
        global new_df
        global new_df1
        global new_df3
        global id_vehicle
        global id_satellite
        self.eps = False
        self.vehicles = []
        self.neighbor_vehicles = []
        self.satellite = []
        id_vehicle = []
        id_satellite = []
        self.veh_position = []
        self.veh_geo_position = []
        self.delta_distance = []
        self.ST_position = []
        self.veh_area = []

        # if self.test == True:
        #     # agent = var.agent_test # range(100,600,100)
        #     agent = range(102,600,100)
        # else:
        #     self.vehicle_time_id = var.train_sumo_step
        #     agent = var.agent_tarin

        while True:
            if self.method == 0:
                # self.vehicle_time_id = var.train_sumo_step = 65
                # self.vehicle_time_id = round(self.vehicle_time_id + 0.1, 1) # 取到小數點後一位
                self.vehicle_time_id = round(self.vehicle_time_id + 0.5, 1) # 取到小數點後一位
                lst = self.df.loc[round(self.df['time'], 1) == self.vehicle_time_id] # self.df -> vehicle dataset     
                # if (self.df['time'] == self.vehicle_time_id).any():
                #     print(f"vehicle time: {self.vehicle_time_id}")
                
                self.sumo_step = self.vehicle_time_id
                # agent = agent

                # if lst['vehicle_x'].count() > (max(agent) + 1):
                #     # 當 lst 裡面的車輛數 > 最大 agent (=500) + 1 時
                #     var.stop_step = self.vehicle_time_id                    
                #     # 這裡使得 test_episodic() 的 if env.sumo_step >= var.stop_step: break
                # elif agent.count(lst['vehicle_x'].count()) == 0:
                #     print(f"沒有符合range(100,600,100)的車輛數")
                #     print(f"sumo_step = {self.sumo_step}")
                #     continue
                
                # lsts = self.st.loc[self.st['time'] == self.vehicle_time_id]
                # new_st = lsts
                newcluster = lst
                count_index = count_index - 1
                i = 0
                # m = pd.DataFrame()
                
                print(f"# vehicles: {len(newcluster)}")
                print(f"sumo_step: {self.sumo_step}")

                if (len(newcluster) <= 1):
                    # newcluster 就是 lat 的數量
                    # self.vehicles.clear()
                    # continue
                    break
                else:
                    # print(f"len(newcluster): {len(newcluster)}")
                    # self.renew_positions()
                    for idx, row in newcluster.iterrows():                         
                        self.add_new_vehicles([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")],
                                              getattr(row, "vehicle_pos"), getattr(row, "vehicle_speed"), 
                                              getattr(row, "vehicle_id"), getattr(row, "area"),
                                              [getattr(row, "lat"), getattr(row, "lon")])
                        id_vehicle.append((getattr(row, "vehicle_id")))
                        self.veh_position.append([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")])
                        self.veh_geo_position.append([getattr(row, "lat"), getattr(row, "lon")])
                        self.veh_area.append((getattr(row, "area")))
                    # for idx, row in new_st.iterrows():
                    #     self.add_new_satellite([getattr(row, "satellite_x"), getattr(row, "satellite_y")],
                    #                            getattr(row, "satellite_pos"),
                    #                            getattr(row, "satellite_speed"), getattr(row, "satellite_id"),getattr(row, "area"),[getattr(row, "lat"), getattr(row, "lon")])
                    #     id_satellite.append((getattr(row, "satellite_id")))

                        # now var.sat_data is False
                        if var.sat_data == True:
                            lon, lat = self.get_Geo_coordinate(self.vehicle_time_id)
                        else:
                            newsat = self.newsat.loc[self.newsat['step'] == self.vehicle_time_id]
                            lon = newsat.iloc[0]['lon']
                            lat = newsat.iloc[0]['lat']
                            alt = newsat.iloc[0]['alt']
                            
                        self.ST_position.append([lat, lon])
                    break

            # 如果 method != 0
            else:
                if count_index < 0:
                    # self.df_act_store = pd.DataFrame(columns = ['id', 'act', 'step'])
                    self.act_store_index = 0
                    veh_position = []
                    # self.vehicle_time_id = round(self.vehicle_time_id + 0.1, 1)
                    self.vehicle_time_id = round(self.vehicle_time_id + 0.5, 1)
                    lst = self.df.loc[self.df['time'] == self.vehicle_time_id]
                    new_df = lst
                    new_df3 = lst
                    dfcluster = lst[['vehicle_x', 'vehicle_y']]
                    count_ = int(dfcluster['vehicle_x'].count() / var.n_cluster_train)
                    agent = agent
                    
                    if lst['vehicle_x'].count() > (max(agent) + 1):
                        var.stop_step = self.vehicle_time_id
                    elif agent.count(lst['vehicle_x'].count()) == 0:
                        continue

                    chagent = 0
                    
                    # agent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    # for i in range (len(agent)):
                    #     if agent[i] == dfcluster['vehicle_x'].count():
                    #         chagent = 1
                    #         break
                    # if  chagent == 0:
                    #     self.renew_positions()

                    # you can choose any cluster methods

                    # MinMaxKMeansMinCostFlow clustring 1
                    # lsts = self.st.loc[self.st['time'] == self.vehicle_time_id]
                    # new_st = lsts
                    # for idx, row in new_st.iterrows():
                    #     self.add_new_satellite([getattr(row, "satellite_x"), getattr(row, "satellite_y")],
                    #                            getattr(row, "satellite_pos"),
                    #                            getattr(row, "satellite_speed"), getattr(row, "satellite_id"),getattr(row, "area"),[getattr(row, "lat"), getattr(row, "lon")]
                    #                            )
                    #     id_satellite.append((getattr(row, "satellite_id")))
                        # self.ST_position.append([getattr(row, "satellite_x"), getattr(row, "satellite_y")])

                    if var.sat_data == True:
                        lon, lat = self.get_Geo_coordinate(self.vehicle_time_id)
                    else:
                        newsat = self.newsat.loc[self.newsat['step'] == self.vehicle_time_id]
                        lon = newsat.iloc[0]['lon']
                        lat = newsat.iloc[0]['lat']
                        alt = newsat.iloc[0]['alt']

                    self.ST_position.append([lat, lon])

                    if self.method == 1:
                        # model = KMeansConstrained(3, size_min = var.n_cluster_train-2,
                        #                           size_max = var.n_cluster_train+2).fit(dfcluster)
                        # model.fit(dfcluster)
                        # dfcluster['cluster'] = model.labels_
                        # model = KMeans(n_clusters = count_, init = 'k-means++', random_state = 0,max_iter = 500).fit(dfcluster)
                        # dfcluster = dfcluster.assign(cluster = model.labels_)

                        # FCM
                        dfcluster = dfcluster.assign(cluster=0)
                        dfcluster1 = dfcluster.to_numpy()
                        initial_centers = kmeans_plusplus_initializer(dfcluster1, count_, random_state=3425).initialize()
                        model = fcm(dfcluster1, initial_centers)
                        model.process()
                        # model.fit(dfcluster)
                        dfresult = model.get_clusters()
                        # centers = model.get_centers()

                        for indexcluster in range(len(dfresult)):
                            dfcluster.loc[dfcluster.index[dfresult[indexcluster]], 'cluster'] = indexcluster

                        # dfcluster = dfcluster.sort_values(["vehicle_x", "vehicle_y"], ascending = (False, True))
                        # dfcluster['dist'] = dfcluster[['vehicle_x', 'vehicle_y']].apply(
                        #     lambda row: np.linalg.norm((row.vehicle_x, row.vehicle_y)), axis = 1)
                        # dfcluster.sort_values('dist', ignore_index = True, inplace = True)
                        anslist = [pd.DataFrame(y) for x, y in dfcluster.groupby('cluster', as_index=False)]
                        # n= random.randint(var.n_cluster_train-2 , var.n_cluster_train+3)
                        #
                        # new_arrayi_index = 0
                        # new_arraycluster = []
                        # for count_index in range(len(anslist)):
                        #     l = anslist[count_index]
                        #     if len(l) > n:
                        #         final = [l[i:i + n] for i in range(0, len(l), n)]
                        #         for inde in range(len(final)):
                        #             new_arraycluster.append(final[inde])
                        #             new_arrayi_index +=  1
                        #     else:
                        #         new_arraycluster.append(l)
                        #     new_arrayi_index +=  1
                        new_arraycluster = anslist
                        # dfcluster = DataFrame(new_arraycluster, columns = ['vehicle_x', 'vehicle_y', 'cluster'])
                        count_ = len(new_arraycluster)
                    else:
                        centers = pd.DataFrame(self.V2Ichannels.BS_position, columns=['x', 'y'])
                        points_array = dfcluster.rename_axis('ID').values
                        point_tree = spatial.cKDTree(points_array)
                        cells_final = pd.DataFrame(columns=['vehicle_x', 'vehicle_y'])
                        alllist = []
                        for row1 in lst.itertuples(index=True, name='Pandas'):
                            alllist.append([getattr(row1, "vehicle_x"), getattr(row1, "vehicle_y"), 0])

                        for item in range(0, len(centers), 1):
                            zlist = point_tree.data[
                                point_tree.query_ball_point([centers.x.iloc[item], 
                                                             centers.y.iloc[item]], 500)]
                            # print(point_tree.data[point_tree.query_ball_point([m.vehicle_x.iloc[item], m.vehicle_y.iloc[item]], 500)])

                            if len(zlist) != 0:
                                # print(dflist.count())
                                dflist = DataFrame(zlist, columns=['vehicle_x', 'vehicle_y'])
                                dflist['cluster'] = (item)
                                cells_final = pd.concat([cells_final, dflist]).drop_duplicates(
                                    subset=['vehicle_x', 'vehicle_y'], keep='last')

                        dflist = DataFrame(alllist, columns=['vehicle_x', 'vehicle_y', 'cluster'])
                        cells_final = pd.concat([cells_final, dflist]).drop_duplicates(
                            subset=['vehicle_x', 'vehicle_y'], keep='first')
                        # cells_final = cells_final.sort_values(["cluster", "vehicle_x", "vehicle_y"], ascending = (True, True, True))
                        n = var.limted_cluster_train
                        # anslist = [pd.DataFrame(y) for x, y in cells_final.groupby('cluster', as_index = False)]
                        cells_final['dist'] = cells_final[['vehicle_x', 'vehicle_y']].apply(
                            lambda row: np.linalg.norm((row.vehicle_x, row.vehicle_y)), axis=1)
                        
                        cells_final.sort_values('dist', ignore_index=True, inplace=True)
                        
                        ans = [[y] if len(y) <= n else np.array_split(y, (len(y) + n) // n)
                               for x, y in cells_final.groupby('cluster', as_index=False)]

                        new_arraycluster = []

                        for group in ans:
                            for section in group:
                                new_arraycluster.append(section)

                        # for count_index in range(len(anslist)):
                        #     l = anslist[count_index]  
                        #     if len(l) > n:
                        #         df_shuffled = pd.DataFrame(l).sample(frac = 1)
                        #         df_shuffled['dist'] = df_shuffled[['vehicle_x', 'vehicle_y']].apply(
                        #             lambda row: np.linalg.norm((row.vehicle_x, row.vehicle_y)), axis = 1)
                        #         df_shuffled.sort_values('dist', ignore_index = True, inplace = True)
                        #         final = [df_shuffled[i:i+n] for i in range(0,df_shuffled.shape[0],n)]
                        #         for inde in range(len(final)):
                        #             new_arraycluster.append(final[inde])
                        #             new_arrayi_index +=  1
                        #     else:
                        #         new_arraycluster.append(l)
                        #         new_arrayi_index +=  1
                        # dfcluster = DataFrame(new_arraycluster, columns = ['vehicle_x', 'vehicle_y', 'cluster'])

                        count_ = len(new_arraycluster)

                    # KMeans clustering 2
                    # model = KMeans(n_clusters = count_, init = 'k-means++').fit(dfcluster)
                    # dfcluster['cluster'] = model.labels_
                    # clustring_with_tsne 3
                    # dfcluster['cluster'] = clustring(dfcluster, count_, 0.99)
                    # FCM 4
                    # model = fcm.FCM(count_, distance_func = haversine_distances)
                    # model.fit(dfcluster)
                    # dfcluster['cluster'] = model.labels_
                    # AgglomerativeClustering 5
                    # agglom = AgglomerativeClustering(n_clusters = count_, linkage = 'complete')
                    # agglom.fit(dfcluster)
                    # dfcluster['cluster'] = agglom.labels_
                    # # shrinkage.Shrinkage 6
                    # model = shrinkage.Shrinkage(count_, size_min = 2)
                    # model.fit(dfcluster)
                    # dfcluster['cluster'] = model.labels_
                    # if self.vehicle_time_id >50 and self.vehicle_time_id <200:
                    #     plt.scatter(dfcluster['vehicle_x'], dfcluster['vehicle_y'], c = kmeans.labels_.astype(float), s = 50, alpha = 0.5)
                    #     plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 50)
                    #     camera.snap()
                    # if self.vehicle_time_id == 200:
                    #     anim = camera.animate(blit = True)
                    #     anim.save('dots.gif', writer = 'imagemagick')
                    # plt.scatter(dfcluster['vehicle_x'], dfcluster['vehicle_y'], c = kmeans.labels_.astype(float), s = 50, alpha = 0.5)
                    # plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 50)
                    # plt.show()

                    count = count_
                    count_index = count - 1
                    self.sumo_step = self.vehicle_time_id
                    if self.method == 3 or self.method == 2 or self.method == 1:
                        ans = new_arraycluster
                    else:
                        ans = [pd.DataFrame(y) for x, y in dfcluster.groupby('cluster', as_index=False)]

                newcluster = ans[count_index]
                count_index = count_index - 1
                i = 0
                m = pd.DataFrame()

                if ((len(newcluster) not in range(var.n_cluster_train, var.n_cluster_train + 5)) or (
                        self.method == 2 and len(newcluster) != var.n_cluster_train_samsize)):
                    # self.vehicles.clear()
                    self.renew_positions()
                else:
                    print(len(newcluster), self.sumo_step)
                    # with open('clustersize.txt', 'a') as f:
                    #     f.write(str(len(newcluster))+" "+str(self.sumo_step)+'\n')
                    # self.renew_positions()
                    
                    if len(self.vehicles) == 0:
                        for idx1, row1 in newcluster.iterrows():
                            for idx, row in new_df.iterrows():
                                if getattr(row1, "vehicle_x") == getattr(row, "vehicle_x") and getattr(row1, "vehicle_y") == getattr(row, "vehicle_y"):
                                    self.add_new_vehicles([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")],
                                                          getattr(row, "vehicle_pos"), getattr(row, "vehicle_speed"), 
                                                          getattr(row, "vehicle_id"),getattr(row, "area"),
                                                          [getattr(row, "lat"), getattr(row, "lon")])
                                    id_vehicle.append((getattr(row, "vehicle_id")))
                                    self.veh_position.append([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")])
                                    self.veh_geo_position.append([getattr(row, "lat"), getattr(row, "lon")])
                                    self.veh_area.append((getattr(row, "area")))
                                    m = m.append(row, ignore_index=True)
                    else:
                        for idx1, row1 in newcluster.iterrows():
                            for idx, row in new_df.iterrows():
                                if getattr(row1, "vehicle_x") == getattr(row, "vehicle_x") and getattr(row1, "vehicle_y") == getattr(row, "vehicle_y"):
                                    if (i < len(self.vehicles)):
                                        # print("cluster", self.vehicle_time_id, getattr(row1, "cluster"))
                                        self.vehicles[i].position[0] = getattr(row, "vehicle_x")
                                        self.vehicles[i].position[1] = getattr(row, "vehicle_y")
                                        self.vehicles[i].direction = getattr(row, "vehicle_pos")
                                        self.vehicles[i].velocity = getattr(row, "vehicle_speed")
                                        self.vehicles[i].area = getattr(row, "area")
                                        self.vehicles[i].Geo_position[0] = getattr(row, "lat")
                                        self.vehicles[i].Geo_position[1] = getattr(row, "lon")
                                        i = i + 1
                                        id_vehicle.append((getattr(row, "vehicle_id")))
                                        m = m.append(row, ignore_index=True)
                                        self.veh_position.append([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")])
                                        self.veh_geo_position.append([getattr(row, "lat"), getattr(row, "lon")])
                                        self.veh_area.append((getattr(row, "area")))
                                    else:
                                        self.add_new_vehicles([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")],
                                                              getattr(row, "vehicle_pos"), getattr(row, "vehicle_speed"), 
                                                              getattr(row, "vehicle_id"),getattr(row, "area"),
                                                              [getattr(row, "lat"), getattr(row, "lon")])
                                        id_vehicle.append((getattr(row, "vehicle_id")))
                                        self.veh_position.append([getattr(row, "vehicle_x"), getattr(row, "vehicle_y")])
                                        self.veh_geo_position.append([getattr(row, "lat"), getattr(row, "lon")])
                                        self.veh_area.append((getattr(row, "area")))
                                        m = m.append(row, ignore_index=True)
                                    # print(getattr(row, "vehicle_id"))
                        # self.vehicles[i].id= getattr(row, "vehicle_id")
                    # for x in self.vehicles:
                    #     # g = df[df['vehicle_id'].isin([x.id])].groupby(['vehicle_id']).nth([self.vehicle_time_id]).reset_index()
                    # new_df = new_df[~new_df.vehicle_id.isin(id_vehicle)]
                    new_df1 = new_df3
                    new_df2 = new_df1[~new_df1.vehicle_id.isin(id_vehicle)]
                    # points = new_df2[['vehicle_x', 'vehicle_y']]
                    # points_array = points.rename_axis('ID').values
                    # point_tree = spatial.cKDTree(points_array)
                    # cells_final = pd.DataFrame(columns = ['vehicle_x', 'vehicle_y'])
                    # for item in range(0, len(m), 1):
                    #     zlist = point_tree.data[
                    #         point_tree.query_ball_point([m.vehicle_x.iloc[item], m.vehicle_y.iloc[item]], 2000)]
                    #     # print(point_tree.data[point_tree.query_ball_point([m.vehicle_x.iloc[item], m.vehicle_y.iloc[item]], 500)])
                    #     if len(zlist) !=  0:
                    #         # print(item, zlist)
                    #         dflist = DataFrame(zlist, columns = ['vehicle_x', 'vehicle_y'])
                    #         # print(dflist.count())
                    #         cells_final = cells_final.append(pd.DataFrame(
                    #             dflist), ignore_index = True)
                    #     # cells_final.append(pd.DataFrame(point_tree.data[point_tree.query_ball_point([m.vehicle_x.iloc[item], m.vehicle_y.iloc[item]], 500)],, ignore_index = True)
                    # # if len(new_df.index) > var.neighbor:

                    new_df2 = new_df2.drop_duplicates(['vehicle_id', 'vehicle_x', 'vehicle_y'], keep='last')

                    for idx1, row1 in new_df2.iterrows():
                        self.add_new_vehicles([getattr(row1, "vehicle_x"), getattr(row1, "vehicle_y")],
                                               getattr(row1, "vehicle_pos"), getattr(row1, "vehicle_speed"), 
                                               getattr(row1, "vehicle_id") , getattr(row1, "area"),
                                               [getattr(row1, "lat"), getattr(row1, "lon")] )
                        self.veh_position.append([getattr(row1, "vehicle_x"), getattr(row1, "vehicle_y")])
                        self.veh_geo_position.append([getattr(row, "lat"), getattr(row, "lon")])
                        self.veh_area.append((getattr(row, "area")))
                # print("vehicles", len(self.neighbor_vehicles))
                break

        self.V2V_Shadowing = np.random.normal(0, self.V2Vchannels.shadow_std,
                                              [len(self.vehicles) + len(self.neighbor_vehicles),
                                               len(self.vehicles) + len(self.neighbor_vehicles)])
        self.V2I_Shadowing = np.random.normal(0, self.V2Ichannels.shadow_std,
                                              [len(self.vehicles) + len(self.neighbor_vehicles),
                                               len(self.V2Ichannels.BS_position)])
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])
        self.delta_distance1 = np.asarray([c.velocity * self.time_slow for c in self.neighbor_vehicles])
        if len(self.delta_distance1) != 0:
            self.delta_distance = np.append(self.delta_distance, self.delta_distance1, 0)

        var.n_vehicle = len(self.vehicles)
        self.eps = True
        
        self.penalty = np.zeros(len(self.vehicles))
        self.penalty_per = np.zeros(len(self.vehicles))
        self.penalty_switch_p = np.ones(len(self.vehicles), dtype=bool)
        self.penaltya = np.zeros(len(self.vehicles))
        self.penalty_switch = np.zeros(len(self.vehicles))
        self.penaltyV2I = np.zeros(len(self.vehicles))
        
        self.power_selection_v2i = np.zeros(((len(self.vehicles)), 1))
        self.power_selection_v2i_n = np.zeros(((len(self.neighbor_vehicles)), 1))
        self.power_selection_v2s = np.zeros(((len(self.vehicles)), 1))
        self.power_selection_v2s_n = np.zeros(((len(self.neighbor_vehicles)), 1))

        self.n_RB = self.n_RB_max
        self.n_Veh = len(self.vehicles)
        self.n_neighbor = 1
        self.change = True
        
        self.V2V_Rate = np.zeros(len(self.vehicles))
        self.V2I_Rate = np.zeros(len(self.vehicles))
        self.V2S_Rate = np.zeros(len(self.vehicles))
        
        self.V2V_Rate_latency = np.zeros(len(self.vehicles))
        self.V2I_Rate_latency = np.zeros(len(self.vehicles))        
        self.V2S_Rate_latency = np.zeros(len(self.vehicles))
        
        self.V2V_Rate_latency_nz = np.zeros(len(self.vehicles))
        self.V2I_Rate_latency_nz = np.zeros(len(self.vehicles))
        self.V2S_Rate_latency_nz = np.zeros(len(self.vehicles))
        
        self.remain = np.zeros(len(self.vehicles))
        self.Data_rate_spacific = np.zeros(len(self.vehicles))
        
        self.SNR_rate_spacific_V2V = np.zeros(len(self.vehicles))
        self.SNR_rate_spacific_V2I = np.zeros(len(self.vehicles))
        self.SNR_rate_spacific_V2S = np.zeros(len(self.vehicles))
        
        self.V2V_selection = np.zeros(len(self.vehicles))
        
        self.V2V_Rateall = []
        self.V2I_Rateall = []
        
        self.switch_ratio = 0
        
        self.v2v_selection_all = 0
        self.v2i_selection_all = 0
        self.v2s_selection_all = 0        
        self.dump_selection_all = 0
        
        self.v2v_selection = 0
        self.v2i_selection = 0
        self.v2s_selection = 0        
        self.dump_selection = 0
        
        self.V2V_SINR_all = np.zeros((self.n_Veh, self.n_neighbor))
        self.V2I_SINR_all = np.zeros((self.n_Veh, self.n_neighbor))
        self.V2S_SINR_all = np.zeros((self.n_Veh, self.n_neighbor))
        
        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor))
        self.V2I_Interference_all = np.zeros((self.n_Veh, self.n_neighbor))
        self.V2S_Interference_all = np.zeros((self.n_Veh, self.n_neighbor))
        
        self.delay = np.zeros(len(self.vehicles))
        self.state_to_serv_from = np.zeros(len(self.vehicles))
        self.mode_state = np.zeros((self.n_Veh, self.n_neighbor))
        
        self.switch = np.zeros(len(self.vehicles))
        self.demand_switch_V2V = np.zeros((self.n_Veh, self.n_neighbor))
        self.demand_switch_V2I = np.zeros((self.n_Veh, self.n_neighbor))
        self.demand_switch_V2S = np.zeros((self.n_Veh, self.n_neighbor))
        
        self.latency_th = np.zeros(len(self.vehicles))
        self.n_RB_max_r = self.n_Veh
        self.new_array = [[]]
        self.new_array = [[var.n_RB, 0]] # [[10, 0]]
        self.n_RB_max_r = var.n_RB + 1

        if self.Satellite_ ==True:
            self.SRB_max = var.SRB_max # self.SRB_max = 20
            self.SRB_max_r = var.SRB_max + 1 # self.SRB_max_r = 21

            x = np.arange(0, self.n_RB_max) # self.n_RB_max = var.n_RB = 10
            y = [24, 23, 15, 10, 17]

            xs = np.arange(0, self.SRB_max) 
            ys = [25]

            arr = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            self.new_array = np.append(self.new_array, arr).T.reshape(-1, 2)

            arrSRB = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
            self.new_array = np.append(self.new_array, arrSRB).T.reshape(-1, 2)

            self.new_array = self.new_array[np.lexsort((self.new_array[:, 0], self.new_array[:, 0]))]
        else:
            self.SRB_max_r = 1
            x = np.arange(0, self.n_RB_max)
            y = [24, 23, 15, 10, 17]

            arr = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            self.new_array = np.append(self.new_array, arr).T.reshape(-1, 2)
            self.new_array = self.new_array[np.lexsort((self.new_array[:, 0], self.new_array[:, 0]))]

        resource_use_reward = np.zeros(self.n_RB_max_r)
        ST_resource_use_reward = np.zeros(self.SRB_max_r)
        
        self.n_Veh = len(self.vehicles)
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.change = True
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
        
        self.interference_all_v2v = np.zeros(self.n_RB_max_r)
        self.interference_all_v2i = np.zeros(self.n_RB_max_r)
        self.interference_all_v2s = np.zeros(self.SRB_max_r)
        
        self.V2V_Rate_avg = []
        self.V2S_Rate_avg = []

        # how many data transfer during the time
        # self.demand_veh = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        # self.demand_truck = self.demand_size * np.ones((self.n_Veh, self.n_neighbor)) / 2
        # self.demand = self.demand_veh + self.demand_truck
        # self.demand_all = self.demand_size_all_veh * np.ones((self.n_Veh, self.n_neighbor))
        # self.demand_s = self.demand_s_size_veh * np.ones((self.n_Veh, self.n_neighbor))
        
        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor)) # V2V
        self.demand_all = self.demand_size_all * np.ones((self.n_Veh, self.n_neighbor)) # V2I
        self.demand_s = self.demand_s_size * np.ones((self.n_Veh, self.n_neighbor)) # V2S
        
        self.individual_time_limit_all = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links_all = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
        self.Switch_count = 0
        
        self.penalty_switch_p = np.ones(len(self.vehicles), dtype=bool)
        self.penaltya = np.zeros(len(self.vehicles))
        self.penaltyV2I = np.zeros(len(self.vehicles))
        
        self.mode_state = np.zeros((self.n_Veh, self.n_neighbor))
        
        self.penalty_switchV2V = np.zeros(len(self.vehicles))
        self.penalty_switchV2I = np.zeros(len(self.vehicles))
        self.penalty_switchV2S = np.zeros(len(self.vehicles))
        
        self.Data_rate_V2V_all = np.zeros(len(self.vehicles))
        self.Data_rate_V2I_all = np.zeros(len(self.vehicles))
        self.Data_rate_V2S_all = np.zeros(len(self.vehicles))
        
        self.Data_rate_V2V_all_check = np.zeros(len(self.vehicles))
        self.Data_rate_V2I_all_check = np.zeros(len(self.vehicles))
        self.Data_rate_V2S_all_check = np.zeros(len(self.vehicles))
        
        self.step_count = np.ones(len(self.vehicles))
        self.step_count_V2V = np.ones(len(self.vehicles))
        self.step_count_V2S = np.ones(len(self.vehicles))
        
        self.RB_selected = np.zeros((self.n_Veh, self.n_neighbor))
        
        
        self.success_transmission_V2V = 0
        self.failed_transmission_V2V = 0
        self.failed_transmission_V2V_truck = 0
        
        self.success_transmission_V2I_veh = 0
        self.failed_transmission_V2I_truck = 0
        
        self.success_transmission_V2S = 0
        self.failed_transmission_V2S = 0
        self.failed_transmission_V2S_truck = 0
        
        self.success_transmission = 0
        self.failed_transmission = 0
        
        self.transmission_data = 0
        self.transmission_data_count = 0
        self.transmission_data_V2V = 0
        self.transmission_data_count_V2V = 0
        
        self.veh_failed = 0
        self.truck_failed = 0
        
        self.V2I_Rate_avg = []
        self.V2V_limit = 0.1  ## 100 ms V2V toleratable latency
        
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, self.n_neighbor))
        self.individual_time_interval_all = np.random.exponential(0.05, (self.n_Veh, self.n_neighbor))
        
        self.UnsuccessfulLink = np.zeros((self.n_Veh, self.n_neighbor))
        
        # channel model
        if var.mode == 'LTE':
            self.V2Vchannels = V2Vchannels_LTE()
            self.V2Ichannels = V2Ichannels_LTE(self.bs)
        else:
            self.V2Vchannels = V2Vchannels()
            self.V2Ichannels = V2Ichannels(self.bs)
        self.V2Schannels = V2Schannels()

    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """
        all_position = []
        # v2i
        for i in range(len(self.vehicles)):
            all_position.append(self.vehicles[i].position)
            destinationBS = self.V2Ichannels.determine_closest(self.V2Ichannels.BS_position,
                                                               self.vehicles[i].position[0],
                                                               self.vehicles[i].position[1])
            self.vehicles[i].destinationsBS = destinationBS
        
        for i in range(len(self.neighbor_vehicles)):
            all_position.append(self.neighbor_vehicles[i].position)
            destination1BS = self.V2Ichannels.determine_closest(self.V2Ichannels.BS_position,
                                                                self.neighbor_vehicles[i].position[0],
                                                                self.neighbor_vehicles[i].position[1])
            self.neighbor_vehicles[i].neighbor_vehicle_destinationBS = destination1BS

        # V2V
        for i in range(len(self.vehicles)):
            destination = self.V2Ichannels.determine_closest(all_position, self.vehicles[i].position[0],
                                                             self.vehicles[i].position[1])
            self.vehicles[i].destinations = destination

        for i in range(len(self.neighbor_vehicles)):
            destination = self.V2Ichannels.determine_closest(all_position, self.neighbor_vehicles[i].position[0],
                                                             self.neighbor_vehicles[i].position[1])
            self.neighbor_vehicles[i].neighbor_vehicle_destination = destination

        # V2S

        for i in range(len(self.vehicles)):
            destinationST = self.V2Schannels.determine_closest_satellite(self.ST_position, self.vehicles[i].position[0],
                                                               self.vehicles[i].position[1])
            self.vehicles[i].destinationsST = destinationST

        for i in range(len(self.neighbor_vehicles)):
            destination1ST = self.V2Schannels.determine_closest_satellite(self.ST_position, self.neighbor_vehicles[i].position[0],
                                                                self.neighbor_vehicles[i].position[1])
            self.neighbor_vehicles[i].neighbor_vehicle_destinationST = destination1ST

    def renew_channel(self):
        """ Renew slow fading channel """
        # print("these", self.delta_distance, len(self.vehicles))
        self.V2V_pathloss = np.zeros(
            (len(self.vehicles) + len(self.neighbor_vehicles),
             len(self.vehicles) + len(self.neighbor_vehicles))) + 15 * np.identity(
            len(self.vehicles) + len(self.neighbor_vehicles)) # 23/12/21 Ibrahim Update (Line1644 150->15)

        self.V2I_pathloss = np.zeros(
            (len(self.vehicles) + len(self.neighbor_vehicles), len(self.V2Ichannels.BS_position)))

        self.V2S_pathloss = np.zeros(
            (len(self.vehicles) + len(self.neighbor_vehicles), len(self.ST_position)))
        

        self.V2V_channels_abs = np.zeros(
            (len(self.vehicles) + len(self.neighbor_vehicles), len(self.vehicles) + len(self.neighbor_vehicles)))
        
        self.V2I_channels_abs = np.zeros(
            (max(len(self.vehicles) + len(self.neighbor_vehicles), len(self.V2Ichannels.BS_position)),
             max(len(self.vehicles) + len(self.neighbor_vehicles), len(self.V2Ichannels.BS_position))))
        
        self.V2S_channels_abs = np.zeros(
            (max(len(self.vehicles) + len(self.neighbor_vehicles), len(self.ST_position)),
             max(len(self.vehicles) + len(self.neighbor_vehicles), len(self.ST_position))))

        for i in range(len(self.vehicles) + len(self.neighbor_vehicles)):
            for j in range(i + 1, len(self.vehicles) + len(self.neighbor_vehicles)):
                # print(self.vehicles[j].id, self.vehicles[i].id)
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(
                    self.veh_position[i], self.veh_position[j]) + self.V2Vchannels.get_shadowing(
                    self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
        self.V2V_channels_abs = self.V2V_pathloss


        for i in range(len(self.vehicles) + len(self.neighbor_vehicles)):
            for j in range(len(self.V2Ichannels.BS_position)):
                # print(self.veh_position[i], self.V2Ichannels.BS_position[j])
                self.V2I_pathloss[i][j] = self.V2Ichannels.get_path_loss(
                    self.veh_position[i], self.V2Ichannels.BS_position[j]) + self.V2Ichannels.get_shadowing(
                    # distance.euclidean([self.veh_position[i]], [self.V2Ichannels.BS_position[j]]),
                    self.delta_distance[i], self.V2I_Shadowing[i][j])
        self.V2I_channels_abs = self.V2I_pathloss


        for i in range(len(self.vehicles) + len(self.neighbor_vehicles)):
            for j in range(len(self.ST_position)):
                self.V2S_pathloss[i][j] = self.V2Schannels.get_path_loss(
                    self.veh_geo_position[i], self.ST_position[j])
        self.V2S_channels_abs = self.V2S_pathloss

    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

        # V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB_max, axis=2)
        V2V_channels_with_fastfading = self.V2V_channels_abs[:, :, np.newaxis] * np.ones(self.n_RB_max)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        # V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, :, np.newaxis], self.n_RB_max, axis=2)
        V2I_channels_with_fastfading = self.V2I_channels_abs[:, :, np.newaxis] * np.ones(self.n_RB_max)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading  - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))

        # V2S_channels_with_fastfading = np.repeat(self.V2S_channels_abs[:, :, np.newaxis], self.SRB_max_r, axis=2)
        V2S_channels_with_fastfading = self.V2S_channels_abs[:, :, np.newaxis] * np.ones(self.SRB_max_r)
        self.V2S_channels_with_fastfading = V2S_channels_with_fastfading  - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2S_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2S_channels_with_fastfading.shape)) / math.sqrt(2))


    def convert_W_to_dB(self,W_value):
        """ Function that converts power values in Watts into dB values!"""
        _ = 10 * np.log10(W_value)
        return _
    
    def convert_dB_to_W(self,dB_value):
        """ Function that converts dB values into Watts!"""
        _ = 10 ** (dB_value / 10)
        return _

    
    def get_channel_quality(self, i):
        array_tolist = self.new_array.tolist()

        sinr = 0
        max = 0
        selectedj = 0
        selectedp = 0
        for j in range(self.n_RB_max):  # n_RB RB_subchannel
            # V2I
            signal = 10 ** (
                    (23 - self.V2I_channels_with_fastfading[
                        i, self.neighbor_vehicles[i].neighbor_vehicle_destinationBS, j] +
                     self.vehAntGain + self.bsAntGain) / 10)
            if self.interference_all_v2i[j] == 0.:
                sinr = signal
            else:
                sinr = 10 * np.log10((1 / (self.interference_all_v2i[j])) * signal + 0.00001)

            if sinr > max:
                max = sinr
                selectedj = j
                selectedp = 24

            # V2V
            for k in range(len(self.V2V_power_dB_List)):
                signal = 10 ** (
                        (self.V2V_power_dB_List[k] - self.V2V_channels_with_fastfading[
                            i, self.neighbor_vehicles[i].neighbor_vehicle_destination, j] +
                         2 * self.vehAntGain) / 10)
                if self.interference_all_v2v[j] == 0.:
                    sinr = signal
                else:
                    sinr = 10 * np.log10((1 / (self.interference_all_v2v[j])) * signal + 0.00001)

                if sinr > max:
                    max = sinr
                    selectedj = j
                    selectedp = self.V2V_power_dB_List[k]

        for j in range(self.n_SRB_max):  # n_SRB SRB_subchannel for V2S
            # V2S
            signal = 10 ** (
                    (33.5 - self.V2S_channels_with_fastfading[
                        i, self.neighbor_vehicles[i].neighbor_vehicle_destinationST, j] +
                     self.ueTXGain + self.stRXGain) / 10)

            if self.interference_all_v2s[j] == 0.:
                sinr = signal
            else:
                sinr = 10 * np.log10((1 / (self.interference_all_v2s[j])) * signal + 0.00001)

            if sinr > max:
                max = sinr
                selectedj = j
                selectedp = 25  # Assuming 25 is the power index for V2S, change it according to your model

        if selectedp + selectedj == 0:
            selectedp = 0
            selectedj = var.n_RB

        act = array_tolist.index([selectedj, selectedp])

        return act
    
    
    def Compute_Performance_Reward_Train(self, actions_power, step, global_clock, transmitted_clock, success_tx):
        global V2I_SINR
        global V2V_SINR
        global V2S_SINR
        global v
        global s_v
        global indexrsu
        global indexrsun
        global V2I_Interference1
        global V2V_Interference1
        global V2S_Interference1

        v = 0
        s_v = 0
        act = actions_power[:, :, 0]  # the channel_selection_part
        # print(f"act:{act}")

        if self.method != 0 and self.neighbor_vehicles:
            new_data = []

            for i in range(len(act)):
                vehicle_id = self.vehicles[i].id
                step_int = int(step)
                act_int = int(act[i][0])

                matching_rows = self.df_act_store[
                    (self.df_act_store["id"] == vehicle_id) & (self.df_act_store["step"] == step_int)]

                if not matching_rows.empty:
                    # Update existing row
                    self.df_act_store.loc[matching_rows.index[0], "act"] = act_int
                else:
                    # Append to the list for later DataFrame creation
                    new_data.append([vehicle_id, act_int, step_int])

            # Create a DataFrame from the new_data list and append it to self.df_act_store
            if new_data:
                new_df = pd.DataFrame(new_data, columns=["id", "act", "step"])
                self.df_act_store = pd.concat([self.df_act_store, new_df], ignore_index=True)

        resource_use = np.zeros(self.n_RB_max_r)
        sat_resource_use = np.zeros(self.SRB_max_r)
        resource_use_n = np.zeros(self.n_RB_max_r)
        power_selection = np.zeros((len(act), 1))
        power_selection_n = np.zeros((len(self.neighbor_vehicles), 1))

        self.count_truck_else = 0
        self.count_veh_else = 0
        self.count_truck_key = 0
        self.count_veh_key = 0

        actions = np.zeros((len(act), 1))
        actions_n = np.zeros((len(self.neighbor_vehicles), 1))
        actionsv21 = np.zeros((len(act), 1))

        checkv2i = np.zeros((len(act), 1))
        checkv2i_n = np.zeros((len(self.neighbor_vehicles), 1))

        if self.method != 0:
            if (len(self.neighbor_vehicles)) == 1:  # Check the vehicles on neighpores cluster
                j = 0
                self.power_selection_v2i_n[j][0] = 23
                checkv2i_n[j][0] = 1
            else:
                for j in range(len(self.neighbor_vehicles)):
                    indexc = 0
                    if self.df_act_store[(self.df_act_store["id"] == self.neighbor_vehicles[j].id) &
                                         (self.df_act_store["step"] == step)].empty == False:
                        stor_act = self.df_act_store.act.iloc[
                            self.df_act_store[(self.df_act_store["id"] == self.neighbor_vehicles[j].id) &
                                              (self.df_act_store["step"] == step)].index[0]]
                        indexc = 1
                        actions_n[j][0] = int(self.new_array[stor_act, 0])
                        actions_n[j][0] = int(round(actions_n[j][0]))
                        if np.int64(self.new_array[stor_act, 1]) != 25:
                            resource_use_n[np.int64(actions_n[j][0])] += 1
                            resource_use[np.int64(actions_n[j][0])] += 1
                        if np.int64(self.new_array[stor_act, 1]) == 24:  # V2I
                            self.power_selection_v2i_n[j][0] = 23
                            checkv2i_n[j][0] = 1
                        elif np.int64(self.new_array[stor_act, 1]) == 0:  # Dump channel
                            self.power_selection_v2i_n[j][0] = -100
                            power_selection_n[j][0] = -100
                            checkv2i_n[j][0] = -1
                        elif np.int64(self.new_array[stor_act, 1]) == 25:  # V2S
                            self.power_selection_v2s_n[j][0] = self.V2S_power_dB #dBm
                            sat_resource_use[np.int64(actions_n[j][0])] += 1
                            checkv2i_n[j][0] = 2
                        else:  # V2V
                            power_selection_n[j][0] = int(self.new_array[stor_act, 1])
                            power_selection_n[j][0] = int(round(power_selection_n[j][0]))
                            self.power_selection_v2i_n[j][0] = 23
                            checkv2i_n[j][0] = 0

                    if indexc == 0:
                        stor_act = self.get_channel_quality(j)
                        # stor_act = random.randint(0, len(self.new_array) - 1)
                        actions_n[j][0] = int(self.new_array[stor_act, 0])
                        actions_n[j][0] = int(round(actions_n[j][0]))
                        if np.int64(self.new_array[stor_act, 1]) != 25:
                            resource_use_n[np.int64(actions_n[j][0])] += 1
                            resource_use[np.int64(actions_n[j][0])] += 1
                        if np.int64(self.new_array[stor_act, 1]) == 24:  # V2I
                            self.power_selection_v2i_n[j][0] = 23
                            checkv2i_n[j][0] = 1
                        elif np.int64(self.new_array[stor_act, 1]) == 0:  # Dump channel
                            self.power_selection_v2i_n[j][0] = -100
                            power_selection_n[j][0] = -100
                            checkv2i_n[j][0] = -1
                        # print("ok")
                        elif np.int64(self.new_array[stor_act, 1]) == 25:  # V2S
                            self.power_selection_v2s_n[j][0] = self.V2S_power_dB # dBm
                            sat_resource_use[np.int64(actions_n[j][0])] += 1
                            checkv2i_n[j][0] = 2
                        else:  # V2V
                            power_selection_n[j][0] = int(self.new_array[stor_act, 1])
                            power_selection_n[j][0] = int(round(power_selection_n[j][0]))
                            self.power_selection_v2i_n[j][0] = 23
                            checkv2i_n[j][0] = 0


        if (len(self.vehicles)) == 1:  # no neighbor所以只能和v2i或v2s傳
            print(f"no neighbor所以只能和v2i或v2s傳")
            j = 0
            self.power_selection_v2i[j][0] = 23
            self.mode_state[j] = 1
            self.penalty[j] = 1
            checkv2i[j][0] = 1
            power_selection[j][0] = -100

            if self.demand_all[j] > 0:
                self.V2I_Rate_latency[j] += 1  # due to every step is 1ms
            self.v2i_selection += 1
            # reciver_rsu = "rsu" + str(self.vehicles[j].destinationsBS)
            # LT.create_transactions(step, self.vehicles[j].id, reciver_rsu, self.demand_all[j])

        else:  # have neighbor
            for j in range(len(act)):
                actions[j][0] = int(self.new_array[act[j][0], 0])
                actions[j][0] = int(round(actions[j][0]))
                actionsv21[j][0] = int(self.new_array[act[j][0], 0])
                actionsv21[j][0] = int(round(actionsv21[j][0]))
                # print(f"new_array: {self.new_array}")
                
                # V2I
                if np.int64(self.new_array[act[j][0], 1]) == 24:  
                    self.power_selection_v2i[j][0] = 23
                    self.penalty[j] = 1
                    checkv2i[j][0] = 1
                    self.mode_state[j] = 1
                    resource_use[np.int64(actions[j][0])] += 1
                    power_selection[j][0] = -100 

                    # if step == 4:
                    #     print(f"車子ID: {self.vehicles[j].id}")   
                    #     print(f"demand_all[j]: {self.demand_all[j]}")
                    #     print(f"demand[j]: {self.demand[j]}")
                    #     print(f"demand_s[j]: {self.demand_s[j]}")
                                              
                    if self.demand_all[j] > 0:
                        self.V2I_Rate_latency[j] += 1
                        # if step == 4:
                        #     print(f"self.V2I_Rate_latency[j] += 1")
                    
                    # 以下判斷是為了當agent在此次訊息傳輸尚未完成時，認為需要從v2i切換至v2v或v2s時所設置的傳輸延遲
                    if self.demand[j] > 0 and self.demand[j] < self.demand_size: # demand_size = 2000 bits
                        self.V2V_Rate_latency[j] += 1
                        # if step == 4:
                        #     print(f"self.V2V_Rate_latency[j] += 1")
                    elif self.demand_s[j] > 0 and self.demand_s[j] < self.demand_s_size: # demand_s_size = 4000 bits
                        self.V2S_Rate_latency[j] += 1
                        # if step == 4:
                        #     print(f"self.V2S_Rate_latency[j] += 1")
                    else:
                        self.v2i_selection += 1
                        # if step == 4:
                        #     print(f"self.v2i_selection += 1")

                        # receiver_rsu = "rsu" + str(self.vehicles[j].destinationsBS)                                                
                        # LT.create_transactions(step, self.vehicles[j].id, receiver_rsu, self.demand_all[j])
                                        
                # Dump channel     
                elif np.int64(self.new_array[act[j][0], 1]) == 0:
                    self.power_selection_v2i[j][0] = -100
                    self.penalty[j] = 2
                    checkv2i[j][0] = -1
                    resource_use[np.int64(actions[j][0])] += 1
                    power_selection[j][0] = -100
                    
                    if self.demand_all[j] > 0 and self.demand_all[j] < self.demand_size_all:
                        self.V2I_Rate_latency[j] += 1
                    elif self.demand[j] > 0 and self.demand[j] < self.demand_size:
                        self.V2V_Rate_latency[j] += 1
                    elif self.demand_s[j] > 0 and self.demand_s[j] < self.demand_s_size:
                        self.V2S_Rate_latency[j] += 1
                    else:
                        self.dump_selection += 1
                    
                # V2S
                elif np.int64(self.new_array[act[j][0], 1]) == 25:
                    self.power_selection_v2s[j][0] = self.V2S_power_dB
                    self.penalty[j] = 4
                    checkv2i[j][0] = 2 
                    sat_resource_use[np.int64(actions[j][0])] += 1
                    power_selection[j][0] = -100
                    
                    if self.demand_s[j] > 0:
                        self.V2S_Rate_latency[j] += 1
                    if self.demand[j] > 0 and self.demand[j] < self.demand_size:
                        self.V2V_Rate_latency[j] += 1
                    elif self.demand_all[j] > 0 and self.demand_all[j] < self.demand_size_all:
                        self.V2I_Rate_latency[j] += 1
                    else:
                        self.v2s_selection += 1
                    
                # V2V    
                else:  
                    self.power_selection_v2i[j][0] = 23
                    self.penalty[j] = 3
                    checkv2i[j][0] = 0
                    self.mode_state[j] = 0
                    resource_use[np.int64(actions[j][0])] += 1
                    power_selection[j][0] = int(self.new_array[act[j][0], 1])
                    power_selection[j][0] = int(round(power_selection[j][0]))
                       
                    if self.demand[j] > 0:
                        self.V2V_Rate_latency[j] += 1
                    if self.demand_all[j] > 0 and self.demand_all[j] < self.demand_size_all:
                        self.V2I_Rate_latency[j] += 1
                    elif self.demand_s[j] > 0 and self.demand_s[j] < self.demand_s_size:
                        self.V2S_Rate_latency[j] += 1
                    else :
                        self.v2v_selection += 1
        
        actions = np.int64(actions)
        actions_n = np.int64(actions_n)
        actionsv21 = np.int64(actionsv21)
        
        power_selection = np.int64(power_selection)
        power_selection_n = np.int64(power_selection_n)
        
        resource_use = np.int64(resource_use)
        sat_resource_use = np.int64(sat_resource_use)

        checkv2i = np.int64(checkv2i)
        checkv2i_n = np.int64(checkv2i_n)

        # ------------ Compute V2V rate -------------------------        
        
        # Noise calculations
        N_0_V2V =  self.convert_dB_to_W (self.sig2 + self.convert_W_to_dB(self.bandwidth)+ self.vehNoiseFigure )
        N_0_V2I = self.convert_dB_to_W (self.sig2 + self.convert_W_to_dB(self.bandwidth)+ self.bsNoiseFigure )
        N_I =  self.convert_dB_to_W (self.sig2 + self.convert_W_to_dB(self.STBW)+ self.NoiseFigure )

        V2I_Interference1 = np.zeros((len(self.vehicles), self.n_neighbor)) + N_0_V2I
        V2V_Interference1 = np.zeros((len(self.vehicles), self.n_neighbor)) + N_0_V2V
        V2S_Interference1 = np.zeros(len(self.vehicles)) + N_I        

        self.interference_all_v2i = np.zeros(self.n_RB_max_r) + N_0_V2I
        self.interference_all_v2v = np.zeros(self.n_RB_max_r) + N_0_V2V
        self.interference_all_v2s = np.zeros(self.SRB_max_r) + N_I
        
        V2V_Rate = np.zeros((len(self.vehicles), 1))
        V2V_Interference = np.zeros(((len(self.vehicles)), self.n_neighbor))
        V2V_SINR = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))

        V2I_Rate = np.zeros(len(self.vehicles))
        V2I_Interference = np.zeros(len(self.vehicles))   # V2I interference 
        V2I_SINR = np.zeros((len(self.vehicles), self.n_neighbor))
        V2I_Signals = np.zeros(len(self.vehicles))

        V2S_Rate = np.zeros(len(self.vehicles))
        V2S_Interference = np.zeros(len(self.vehicles))    # V2S interference         
        V2S_SINR = np.zeros((len(self.vehicles), self.n_neighbor))
        V2S_Signals = np.zeros(len(self.vehicles))

        finishV2V = 0
        finishV2I = 0

        self.switch = np.zeros(len(self.vehicles))

        # print(f"self.n_neighbor: {self.n_neighbor}")

        for l in range((self.n_neighbor)):
            for i in range(len(self.vehicles)):  # scanning all vehicles                
                if (self.demand[i] > 0 and self.demand[i] < self.demand_size and
                        self.penalty[i] != 3): # 表示這台車在上一個step選擇V2V傳輸並且data沒有傳完，而這次step並沒有選擇V2V傳輸 => switch發生
                    self.switch[i] = 0
                    self.penalty_per[i] = 0.5
                    self.Switch_count += 1
                    continue
                elif (self.demand_all[i] > 0 and self.demand_all[i] < self.demand_size_all and
                      self.penalty[i] != 1): # 表示這台車在上一個step選擇V2I傳輸並且data沒有傳完，而這次step並沒有選擇V2I傳輸 => switch發生
                    self.switch[i] = 1
                    self.penalty_per[i] = 0.5
                    self.Switch_count += 1
                    continue
                elif (self.demand_s[i] > 0 and self.demand_s[i] < self.demand_s_size and
                      self.penalty[i] != 4): # 表示這台車在上一個step選擇V2S傳輸並且data沒有傳完，而這次step並沒有選擇V2S傳輸 => switch發生
                    self.switch[i] = 2
                    self.penalty_per[i] = 0.5
                    self.Switch_count += 1
                    continue
                else: # 表示沒有switch發生
                    self.switch[i] = -1
                    self.penalty_per[i] = 1.5


                if checkv2i[i][0] == 0:  # V2V=0
                    if self.demand[i] <= 0:  # Check if the transmission has been finished
                        continue
                if checkv2i[i][0] == 1:  # V2I=1
                    if self.demand_all[i] <= 0:  # Check if the transmission has been finished
                        continue
                if checkv2i[i][0] == 2:  # V2S=2
                    if self.demand_s[i] <= 0:  # Check if the transmission has been finished
                        continue

                indexes = np.argwhere(actionsv21 == actionsv21[i][0])  # find the vehicles that share the same channel

                for j in range(len(indexes)):
                    if checkv2i[indexes[j, 0]][0] == 0:  
                        # check if the neighbors in the same cluster and share the channel  V2V=0
                        if self.demand[indexes[j, 0]] <= 0:  
                            # Check if the transmation has been finished of the neighbors vehicles or there is switching
                            continue
                        if checkv2i[indexes[j, 0]][0] == 1:  
                            # check if the neighbors in the same cluster and share the channel  V2V=0
                            if self.demand_all[indexes[j, 0]] <= 0:  
                                # Check if the transmation has been finished of the neighbors vehicles or there is switching
                                continue
                        if checkv2i[indexes[j, 0]][0] == 2:  
                            # check if the neighbors in the same cluster and share the channel  V2V=0
                            if self.demand_s[indexes[j, 0]] <= 0:  
                                # Check if the transmation has been finished of the neighbors vehicles or there is switching
                                continue

                    if self.vehicles[i] == self.vehicles[indexes[j, 0]]:  # Ignore the same vehicle
                        continue

                    if checkv2i[i][0] == 0:  # check the transmation mode is V2V = 0
                        if checkv2i[indexes[j, 0]][0] == 2:
                            continue

                        receiver_j = self.vehicles[i].destinations
                        if checkv2i[indexes[j, 0]][0] == 1:  
                            # check if the neighbors in the same cluster and share the channel  V2I = 1
                            V2V_Interference[i, l] += self.convert_dB_to_W  (
                                    (self.power_selection_v2i[indexes[j, 0]] 
                                     - self.V2V_channels_with_fastfading[
                                         indexes[j, 0], receiver_j, actionsv21[i][0]] 
                                         + 2 * self.vehAntGain))
                        elif checkv2i[indexes[j, 0]][0] == 0:  
                            # check if the neighbors in the same cluster and share the channel  V2V = 0
                            V2V_Interference[i, l] +=   self.convert_dB_to_W(
                                    (power_selection[indexes[j, 0]]
                                     - self.V2V_channels_with_fastfading[
                                         indexes[j, 0], receiver_j, actionsv21[i][0]] 
                                         + 2 * self.vehAntGain))

                    if checkv2i[i][0] == 1:  # check the transmation mode is V2I = 1
                        if checkv2i[indexes[j, 0]][0] == 2:
                            continue

                        receiver_j = self.vehicles[i].destinationsBS
                        if checkv2i[indexes[j, 0]][0] == 0:  
                            # check if the neighbors in the  smae cluster and share the channel  V2V = 0
                            V2I_Interference[i] += self.convert_dB_to_W(
                                    (power_selection[indexes[j, 0]] 
                                     - self.V2I_channels_with_fastfading[
                                         indexes[j, 0], receiver_j, actionsv21[i][0]]
                                         + self.vehAntGain + self.bsAntGain))
                            # print("1", V2I_Interference[i].astype(int))
                        elif checkv2i[indexes[j, 0]][0] == 1:  
                            # check if the neighbors in the  smae cluster and share the channel  V2I = 1
                            # If the neighboring vehicle's mode is V2I and they are  targeting the same base station                            
                            V2I_Interference[i] += self.convert_dB_to_W (
                                    (self.power_selection_v2i[indexes[j, 0]] 
                                     - self.V2I_channels_with_fastfading[
                                         indexes[j, 0], receiver_j, actionsv21[i][0]]
                                         + self.vehAntGain + self.bsAntGain))
                            # print("2", V2I_Interference[i].astype(int))

                    if checkv2i[i][0] == 2:  # check the transmation mode V2S=2, V2I=1 or V2V=0
                        if checkv2i[indexes[j, 0]][0] != 2:
                            continue

                        receiver_j = self.vehicles[i].destinationsST
                        # If the neighboring vehicle's mode is V2S and they are  targeting the same base station                        
                        V2S_Interference[i] += self.convert_dB_to_W(
                            self.power_selection_v2s[indexes[j, 0], indexes[j, 1]] -
                            self.V2S_channels_with_fastfading[
                                indexes[j, 0], receiver_j, actionsv21[i][0]]
                            + self.ueTXGain  + self.stRXGain)

                if self.method != 0:
                    indexes_n = np.argwhere(
                        actions_n == actionsv21[i][
                            0])  # # find the neighbors vehicles in the	 neighbors cluster that share the same channel

                    for j_n in range(len(indexes_n)):
                        if self.vehicles[i] == self.neighbor_vehicles[indexes_n[j_n, 0]]:  # Ignore the same vehicle
                            continue

                        # if self.df_act_store_latency[(self.df_act_store_latency["id"]  ==  self.neighbor_vehicles[indexes_n[j_n, 0]].id) &
                        #                              (self.df_act_store_latency["latency"]  ==  (step % var.max_latency)) ].empty  ==  False:
                        if self.df_act_store_latency[
                            (self.df_act_store_latency["id"] == self.neighbor_vehicles[indexes_n[j_n, 0]].id) &
                            (self.df_act_store_latency["latency"] == (step % var.max_latency))].empty == False:
                            continue
                        if checkv2i[i][0] == 0:  # check the transmation mode is V2V = 0
                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] == 2:
                                continue
                            # if finishV2V>0: # assuame that some of the  neighbor vehicles already finish
                            #     finishV2V -= 1
                            #     continue
                            if self.df_act_store_latency[
                                (self.df_act_store_latency["id"] == self.neighbor_vehicles[indexes_n[j_n, 0]].id) &
                                (self.df_act_store_latency["latency"] == int(self.V2V_Rate_latency[i]))].empty == False:
                                continue

                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] == 1:  # check if the neighbors in the  neighbors cluster and share the channel  V2I = 1
                                receiver_j = self.vehicles[i].destinations

                                V2V_Interference[i, l] += self.convert_dB_to_W (
                                        (self.power_selection_v2i_n[indexes_n[j_n, 0]] -
                                         self.V2V_channels_with_fastfading[len(self.vehicles) +
                                                                           indexes_n[j_n, 0], receiver_j, actionsv21[
                                                                               i][
                                                                               0]] + 2 * self.vehAntGain))

                            elif checkv2i_n[indexes_n[j_n, 0]][
                                0] == 0:  # check if the neighbors in the  neighbors cluster and share the channel  V2V = 0
                                receiver_j = self.vehicles[i].destinations
                                if len(indexes_n) <= 1:
                                    continue

                                V2V_Interference[i] += self.convert_dB_to_W (
                                        (power_selection_n[indexes_n[j_n, 0]]
                                         - self.V2V_channels_with_fastfading[
                                             indexes_n[j_n, 0] + len(self.vehicles), receiver_j,
                                             actionsv21[
                                                 i][0]] + 2 * self.vehAntGain ) )

                        if checkv2i[i][0] == 1:  # check the transmation mode is V2I = 1
                            # if not self.active_links[indexes[j, 0], indexes[j, 1]] and checkv2i[indexes[j, 0]][0]  ==  0:
                            #     continue
                            # if finishV2I > 0:  # assuame that some of the  neighbor vehicles already finish
                            #     finishV2I -=  1
                            #     continue
                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] == 2:
                                continue
                            if self.df_act_store_latency[
                                (self.df_act_store_latency["id"] == self.neighbor_vehicles[indexes_n[j_n, 0]].id) &
                                (self.df_act_store_latency["latency"] == int(self.V2I_Rate_latency[i]))].empty == False:
                                continue
                            receiver_j = self.vehicles[i].destinationsBS
                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] == 0:  # check if the neighbors in the  neighbors cluster and share the channel  V2V = 0

                                V2I_Interference[i] += self.convert_dB_to_W (
                                        (power_selection_n[indexes_n[j_n, 0]]
                                         - self.V2I_channels_with_fastfading[
                                             len(self.vehicles) + indexes_n[j_n, 0], receiver_j, actionsv21[
                                                 i][0]] + self.vehAntGain + self.bsAntGain))
                                # print("3", V2I_Interference[i].astype(int))

                            elif checkv2i_n[indexes_n[j_n, 0]][
                                0] == 1:  # check if the neighbors in the  neighbors cluster and share the channel  V2I = 1

                                V2I_Interference[i] += self.convert_dB_to_W (
                                        (self.power_selection_v2i_n[indexes_n[j_n, 0]] -
                                         self.V2I_channels_with_fastfading[
                                             len(self.vehicles) + indexes_n[j_n, 0], receiver_j, actionsv21[i][0]]
                                         + self.vehAntGain + self.bsAntGain))
                                # print("4", V2I_Interference[i].astype(int))
                        if checkv2i[i][0] == 2:  # check the transmation mode V2S=2, V2I=1 or V2V=0
                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] != 2:
                                continue
                            if self.df_act_store_latency[
                                (self.df_act_store_latency["id"] == self.neighbor_vehicles[indexes_n[j_n, 0]].id) &
                                (self.df_act_store_latency["latency"] == int(self.V2S_Rate_latency[i]))].empty == False:
                                continue


                            receiver_j = self.vehicles[i].destinationsST
                            if checkv2i_n[indexes_n[j_n, 0]][
                                0] == 2:


                                # V2S_Interference[i] += self.convert_dB_to_W(self.power_selection_v2s[indexes[j, 0], indexes[j, 1]] -
                                #                                self.V2S_channels_with_fastfading[
                                #                                    indexes[j, 0], receiver_j, actionsv21[i][0]]
                                #                                + self.ueTXGain + self.stRXGain )
                                V2S_Interference[i] += self.convert_dB_to_W(
                                    self.power_selection_v2s[indexes[j, 0], indexes[j, 1]] -
                                    self.V2S_channels_with_fastfading[
                                        indexes[j, 0], receiver_j, actionsv21[i][0]]
                                    + self.ueTXGain  + self.stRXGain)
                                # 23/12/12 Ibrahim Update (Line2322~2330)
                            # print("power:",self.power_selection_v2s[indexes[j, 0], indexes[j, 1]],"fd:",self.V2S_channels_with_fastfading[len(self.vehicles) + indexes[j, 0], receiver_j, actionsv21[i][0]])

        SINR = np.zeros((len(self.vehicles)))
        SNR = np.zeros((len(self.vehicles)))
        VDemen = np.zeros((len(self.vehicles), 1))
        
        self.demand_switch_V2V = np.zeros((self.n_Veh, self.n_neighbor))
        self.demand_switch_V2I = np.zeros((self.n_Veh, self.n_neighbor))
        self.demand_switch_V2S = np.zeros((self.n_Veh, self.n_neighbor))
        # N_0_V2V = self.convert_dB_to_W(self.sig2 + self.convert_W_to_dB(self.bandwidth))
        # N_0_V2I = self.convert_dB_to_W(self.sig2 + self.convert_W_to_dB(self.bandwidth))
        # print(self.convert_W_to_dB(N_0_V2V),"+++")

        V2V_Interference = V2V_Interference + N_0_V2V
        V2I_Interference = V2I_Interference + N_0_V2I
        V2S_Interference = V2S_Interference + N_I

        for i in range(len(act)):

            if self.switch[i] == 0:
                self.demand_switch_V2I[i] = 0 # it should be zero to reduce the switching
            if self.switch[i] == 1:
                self.demand_switch_V2V[i] =0
            if self.switch[i] == 2:
                self.demand_switch_V2S[i] =0
            if self.switch[i] != -1:
                continue

            self.remain_demand_all = np.ones((self.n_Veh, self.n_neighbor))
            self.remain_demand = np.ones((self.n_Veh, self.n_neighbor))
            self.remain_demand_s = np.ones((self.n_Veh, self.n_neighbor))

            if checkv2i[i][0] == 1:  # ------------ Compute V2I rate --------------------
                self.interference_all_v2i[actionsv21[i][0]] = V2I_Interference[i]
                if self.demand_all[i] <= 0:
                    self.demand_switch_V2V[i] = self.demand[i]  # show the demand
                    self.demand_switch_V2S[i] = self.demand_s[i]  # show the demand
                    continue

                receiver_j = self.vehicles[i].destinationsBS

                V2I_Signals[i] = self.convert_dB_to_W (
                        (self.power_selection_v2i[i][0] - self.V2I_channels_with_fastfading[
                            i, receiver_j, actionsv21[i][0]]
                         + self.vehAntGain + self.bsAntGain))

                SNR[i] = np.divide(V2I_Signals[i], N_0_V2I)
                if V2I_Interference[i] == 0 or math.isnan(V2I_Interference[i]):
                    V2I_Rate[i] = (self.bandwidth * np.log2(1 + V2I_Signals[i])) / 1e6 # V2I_Rate = np.zeros(len(self.vehicles))
                    SINR[i] = V2I_Signals[i]
                else:
                    V2I_Rate[i] = (self.bandwidth * np.log2(1 + np.divide(V2I_Signals[i], V2I_Interference[i]))) /1e6
                    SINR[i] = np.divide(V2I_Signals[i], V2I_Interference[i])

                V2I_SINR[i][0] = self.convert_W_to_dB(SINR[i])
                V2I_Interference1[i] = V2I_Interference[i]
                self.SINR_list_ep_V2I.append(self.convert_W_to_dB(SINR[i]))
                self.SNR_list_ep_V2I.append(self.convert_W_to_dB(SNR[i]))

                self.remain_demand_all[i] = self.demand_all[i]
                # self.demand_all[i] -= (V2I_Rate[i] * self.time_fast * self.bandwidth)
                 
                # bandwidth per RB, 1 MHz if it is more or less than 1 MHz change in Data rate
                expected_data = V2I_Rate[i] * self.time_fast * self.bandwidth # time_fast = 0.001
                # print(f"expected_data: {expected_data}")
                if expected_data >= self.demand_size:             
                    self.demand_all[i] -= expected_data

                # 當交易走V2I通道時，檢查是否有之前來自V2V的交易暫存於發送車輛上
                idx = 0            
                VehID = self.vehicles[i].id
                while VehID in LT.v2v_tx_pool.keys() and idx < len(LT.v2v_tx_pool[VehID]):
                    # 檢查交易發送車輛是否存在於v2v_tx_pool
                    tx_in_veh = LT.v2v_tx_pool[VehID][idx] 
                    if transmitted_clock > tx_in_veh.timestamp: # 現在的時刻需要大於暫存於v2v_tx_pool交易的時刻
                        LT.pool.append(tx_in_veh) # 將這筆交易放進pool
                        del LT.v2v_tx_pool[VehID][idx] # 將這筆交易從v2v_tx_pool刪除
                        # print(f"成功將V2V2I交易 {tx_in_veh.id} 加進RSU的pool")
                        # print(f"發送車輛: {tx_in_veh.sender}, 接收RSU: {tx_in_veh.to}")
                    else: 
                        idx += 1
                
                # Create V2I Transactions!!!
                
                if self.demand_all[i] <= 0:  # eliminate negative demands  
                    if self.demand[i] == self.demand_size and self.demand_s[i] == self.demand_s_size:                  
                        receiver_rsu = "rsu" + str(receiver_j)         
                        transmitted_data = self.demand_size_all
                        LT.create_transactions(global_clock, transmitted_clock, self.vehicles[i].id, receiver_rsu, transmitted_data)                    
                        # print(f"車子ID: {self.vehicles[i].id}")
                        # print(f"接收RSU ID:{receiver_rsu}")
                        # print(f"傳送的訊息量: {transmitted_data}")
                        # print(f"車子ID: {self.vehicles[i].id}, 接收RSU ID:{receiver_rsu}, 傳送的訊息量: {transmitted_data}")
                        self.demand_all[i] = 0
                        success_tx += 1
                        self.vehicles[i].success_fail += 1
                else:
                    self.vehicles[i].success_fail -= 1
                    # receiver_rsu = "rsu" + str(receiver_j)         
                    # transmitted_data = V2I_Rate[i] * self.time_fast * self.bandwidth                                      
                    # LT.create_transactions(global_clock, self.vehicles[i].id, receiver_rsu, transmitted_data)                    
                    # print(f"車子ID: {self.vehicles[i].id}")
                    # print(f"接收RSU ID:{receiver_rsu}")
                    # print(f"傳送的訊息量: {transmitted_data}")
                
                # Create V2I Transactions!!!

                if (V2I_Rate[i] * self.time_fast * self.bandwidth) >=  (
                        self.demand_size_all / (var.max_latency * var.TTI)) or \
                        self.demand_all[i] <=  0: # max_latency = 3；TTI = 1
                    self.success_transmission +=  1
                else:
                    self.failed_transmission +=  1

                self.demand_switch_V2I[i] = self.demand_all[i]
                self.Data_rate_V2I_all[i] += V2I_Rate[i]
                self.remain[i] = self.demand_all[i]
                self.Data_rate_spacific[i] = V2I_Rate[i]
                self.SNR_rate_spacific_V2I[i] = self.convert_W_to_dB(SNR[i])

                if self.demand_all[i] <= 0:
                    self.demand_switch_V2V[i] = self.demand[i]  # show the demand
                    self.demand_switch_V2S[i] = self.demand_s[i]  # show the demand

            if checkv2i[i][0] == 0:  # ------------ Compute V2V rate --------------------
                self.interference_all_v2v[actionsv21[i][0]] = V2V_Interference[i]
                if self.demand[i] <= 0:
                    self.demand_switch_V2I[i] = self.demand_all[i]  # show the demand
                    self.demand_switch_V2S[i] = self.demand_s[i]  # show the deman
                    continue

                receiver_j = self.vehicles[i].destinations

                V2V_Signal[i, 0] = self.convert_dB_to_W (
                        (power_selection[i]
                         - self.V2V_channels_with_fastfading[
                             i, receiver_j, actionsv21[i][0]] + 2 * self.vehAntGain))

                SNR[i] = np.divide(V2V_Signal[i][0], N_0_V2V)
                # print("Transmiter postion ",self.veh_position[i],"reciver postion", self.veh_position[j], "channels_with_fastfading",self.V2V_channels_with_fastfading[
                #              i, receiver_j, actionsv21[i][0]] ,"SNR in dB",self.convert_W_to_dB(SNR[i]),"N_0_V2V",self.convert_W_to_dB(N_0_V2V))

                if V2V_Interference[i] == 0 or math.isnan(V2V_Interference[i]):
                    # V2V_Rate[i][0] = np.log2(1 + V2V_Signal[i][0])
                    V2V_Rate[i][0] = (self.bandwidth * np.log2(1 + V2V_Signal[i][0])) / 1e6
                    SINR[i] = V2V_Signal[i][0]
                else:
                    V2V_Rate[i][0] = (self.bandwidth * np.log2(1 + np.divide(V2V_Signal[i][0], V2V_Interference[i][0]))) / 1e6
                    SINR[i] = np.divide(V2V_Signal[i][0], V2V_Interference[i][0])

                V2V_SINR[i][0] = self.convert_W_to_dB(SINR[i])
                V2V_Interference1[i] = V2V_Interference[i]
                self.SINR_list_ep_V2V.append(self.convert_W_to_dB(SINR[i]))
                self.SNR_list_ep_V2V.append( self.convert_W_to_dB(SNR[i]))
                
                self.remain_demand[i] = self.demand[i]       
                # self.demand[i] -= (V2V_Rate[i][0] * self.time_fast * self.bandwidth)         

                # 計算預計要傳出的data size
                expected_data = V2V_Rate[i][0] * self.time_fast * self.bandwidth
                if expected_data >= self.demand_size:
                    self.demand[i] -= expected_data

                # Create V2V Transactions!!!
                
                if self.demand[i] <= 0:  # eliminate negative demands
                    if self.demand_all[i] == self.demand_size_all and self.demand_s[i] == self.demand_s_size:
                        receiver_veh = "veh" + str(receiver_j)
                        transmitted_data = self.demand_size
                        LT.create_transactions(global_clock, transmitted_clock, self.vehicles[i].id, receiver_veh, transmitted_data)
                        # print(f"車子ID: {self.vehicles[i].id}")
                        # print(f"接收車子 ID:{receiver_veh}")
                        # print(f"傳送的訊息量: {transmitted_data}")
                        # print(f"車子ID: {self.vehicles[i].id}, 接收車子 ID:{receiver_veh}, 傳送的訊息量: {transmitted_data}")
                        self.demand[i] = 0
                        success_tx += 1
                        self.vehicles[i].success_fail += 1
                else:
                    self.vehicles[i].success_fail -= 1
                    # receiver_veh = "veh" + str(receiver_j)
                    # transmitted_data = V2V_Rate[i][0] * self.time_fast * self.bandwidth
                    # LT.create_transactions(global_clock, self.vehicles[i].id, receiver_veh, transmitted_data)
                    # print(f"車子ID: {self.vehicles[i].id}")
                    # print(f"接收車子 ID:{receiver_veh}")
                    # print(f"傳送的訊息量: {transmitted_data}")
                
                # Create V2V Transactions!!!

                if (V2V_Rate[i][0] * self.time_fast * self.bandwidth) >= (
                        self.demand_size / (var.max_latency * var.TTI)) or \
                        self.demand[i] <= 0:
                    self.success_transmission_V2V += 1
                else:
                    self.failed_transmission_V2V += 1

                self.demand_switch_V2V[i] = self.demand[i]                
                self.Data_rate_V2V_all[i] += V2V_Rate[i]
                self.remain[i] = self.demand[i]
                self.Data_rate_spacific[i] = V2V_Rate[i]
                self.SNR_rate_spacific_V2V[i] = self.convert_W_to_dB(SNR[i])

                if self.demand[i] <= 0:
                    self.demand_switch_V2I[i] = self.demand_all[i]  # show the demand
                    self.demand_switch_V2S[i] = self.demand_s[i]  # show the deman

            if checkv2i[i][0] == 2:  # ------------ Compute V2S rate --------------------
                # print("checkv2i", checkv2i[i][0],self.demand_s[i])
                self.interference_all_v2s[actionsv21[i][0]] = V2S_Interference[i]
                if self.demand_s[i] <= 0:
                    self.demand_switch_V2V[i] = self.demand[i]  # show the demand
                    self.demand_switch_V2I[i] = self.demand_all[i]  # show the deman
                    continue

                receiver_j = self.vehicles[i].destinationsST

                V2S_Interference[i] = V2S_Interference[i]
                self.interference_all_v2s[actionsv21[i][0]] = V2S_Interference[i]

                V2S_Signals[i] = self.convert_dB_to_W(
                    self.power_selection_v2s[i][0] - self.V2S_channels_with_fastfading[
                        i, receiver_j, actionsv21[i][0]]
                    + self.ueTXGain + self.stRXGain)

                SNR[i] = np.divide((V2S_Signals[i]), N_I)
                # print(  "SNR in dB",self.convert_W_to_dB(SNR[i]),"N_0_V2S", self.convert_W_to_dB( N_I))
                
                if V2S_Interference[i] == 0 or math.isnan(V2S_Interference[i]):
                    V2S_Rate[i] =( self.STBW*np.log2(1 + (V2S_Signals[i])))/  1e6
                    SINR[i] = V2S_Signals[i]
                else:
                    V2S_Rate[i] =( self.STBW* np.log2(1 + np.divide(V2S_Signals[i], V2S_Interference[i])))/  1e6
                    SINR[i] = np.divide((V2S_Signals[i]), V2S_Interference[i])

                V2S_SINR[i][0] = self.convert_W_to_dB(SINR[i])
                V2S_Interference1[i] = V2S_Interference[i]
                self.SINR_list_ep_V2S.append(self.convert_W_to_dB(SINR[i]))
                self.SNR_list_ep_V2S.append(self.convert_W_to_dB(SNR[i]))

                self.remain_demand_s[i] = self.demand_s[i]
                # self.demand_s[i] -= (V2S_Rate[i] * self.time_fast *  1e6)

                expected_data = V2S_Rate[i] * self.time_fast *  1e6
                if expected_data >= self.demand_s_size:
                    self.demand_s[i] -= expected_data
                
                # self.demand_s[i] = max(self.demand_s[i], 0)

                # Create V2S Transactions!!!

                if self.demand_s[i] <= 0:
                    # print(f"self.demand[i]: {self.demand[i]}, self.demand_size: {self.demand_size}")
                    if self.demand[i] == self.demand_size and self.demand_all[i] == self.demand_size_all:
                        receiver_sat = "sat" + str(receiver_j)
                        transmitted_data = self.demand_s_size
                        LT.create_transactions(global_clock, transmitted_clock, self.vehicles[i].id, receiver_sat, transmitted_data)
                        # print(f"車子ID: {self.vehicles[i].id}")
                        # print(f"接收衛星 ID:{receiver_sat}")
                        # print(f"傳送的訊息量: {transmitted_data}")
                        # print(f"車子ID: {self.vehicles[i].id}, 接收衛星 ID:{receiver_sat}, 傳送的訊息量: {transmitted_data}")
                        self.demand_s[i] = 0
                        success_tx += 1
                        self.vehicles[i].success_fail += 1
                else:
                    self.vehicles[i].success_fail -= 1
                    # receiver_sat = "sat" + str(receiver_j)
                    # transmitted_data = (V2S_Rate[i] * self.time_fast *  1e6)
                    # LT.create_transactions(global_clock, self.vehicles[i].id, receiver_sat, transmitted_data)
                    # print(f"車子ID: {self.vehicles[i].id}")
                    # print(f"接收衛星 ID:{receiver_sat}")
                    # print(f"傳送的訊息量: {transmitted_data}")
                
                # Create V2S Transactions!!!

                if (V2S_Rate[i] * self.time_fast  * 1e6 ) >=  (
                        self.demand_s_size / (var.max_latency * var.TTI)) or \
                        self.demand_s[i] <=  0:
                    self.success_transmission_V2S +=  1
                else:
                    self.failed_transmission_V2S +=  1

                self.demand_switch_V2S[i] = self.demand_s[i]
                self.Data_rate_V2S_all[i] += V2S_Rate[i]
                self.remain[i] = self.demand_s[i]
                self.Data_rate_spacific[i] = V2S_Rate[i]
                self.SNR_rate_spacific_V2S[i] = self.convert_W_to_dB(SNR[i])

                if self.demand_s[i] <= 0:
                    self.demand_switch_V2V[i] = self.demand[i]  # show the demand
                    self.demand_switch_V2I[i] = self.demand_all[i]  # show the deman

        if np.sum(V2I_Rate) != 0:
            self.V2I_Rate_avg.append(np.sum(V2I_Rate))
        if np.sum(V2V_Rate) != 0:
            self.V2V_Rate_avg.append(np.sum(V2V_Rate))
        if np.sum(V2S_Rate) != 0:
            self.V2S_Rate_avg.append(np.sum(V2S_Rate))

        for j in range(len(resource_use)):
            if resource_use[j] > 0:
                v = v + 1
        v = (v / (self.n_RB_max_r))
        u = 0
        for j in range((self.n_RB_max)):
            if resource_use[j] > 0:
                u = u + 1
        for j in range(len(sat_resource_use)):
            if sat_resource_use[j] > 0:
                s_v = s_v + 1
        s_v = (s_v / (self.SRB_max_r))
        s_u = 0
        sat_Utilization=0
        if self.Satellite_ == True:
            for j in range((self.n_SRB_max)):
                if sat_resource_use[j] > 0:
                    s_u = s_u + 1

        sat_Utilization = s_v
        Utilization = v

        # self.success_transmission = len(self.demand_all[self.demand_all <= 0])
        # self.failed_transmission = len(
        #     self.demand_all[(self.demand_all > 0) & (self.demand_all != self.demand_size_all)])
        
        # if self.test == True:
        self.success_transmission_V2V = len(self.demand[self.demand <= 0])
        self.failed_transmission_V2V = len(
            self.demand[(self.demand > 0) & (self.demand != self.demand_size)])
        
        self.success_transmission_V2S = len(self.demand_s[self.demand_s <= 0])
        self.failed_transmission_V2S = len(
            self.demand_s[(self.demand_s > 0) & (self.demand_s != self.demand_s_size)])

        self.success_transmission = len(self.demand_all[self.demand_all <= 0])
        self.failed_transmission = len(
            self.demand_all[(self.demand_all > 0) & (self.demand_all != self.demand_size_all)])
        # 23/12/12 Ibrahim Update (Line2563~2569)

        if (self.failed_transmission_V2V + self.success_transmission_V2V) == 0:
            failed_percentage_v2v = 0
        else:
            failed_percentage_v2v = self.failed_transmission_V2V / (
                    self.failed_transmission_V2V + self.success_transmission_V2V)
        if (self.failed_transmission + self.success_transmission) == 0:
            failed_percentage = 0
        else:
            failed_percentage = self.failed_transmission / (self.failed_transmission + self.success_transmission)

        if (self.failed_transmission_V2S + self.success_transmission_V2S) == 0:
            failed_percentage_V2S = 0
        else:
            failed_percentage_V2S = self.failed_transmission_V2S / (
                        self.failed_transmission_V2S + self.success_transmission_V2S)

        success_by_area = {'suburban': 0, 'urban': 0, 'rural': 0}
        failed_by_area = {'suburban': 0, 'urban': 0, 'rural': 0}

        # Function to update success and failure counts
        def update_counts(demands, demand_size):
            for i in range(len(demands)):
                if demands[i] <= 0:
                    success_by_area[self.vehicles[i].area] += 1
                elif demands[i] > 0 and demands[i] != demand_size:
                    failed_by_area[self.vehicles[i].area] += 1

        # Assuming you have lists of demands and areas for each type of demand
        # For example: v2v_demands, v2s_demands, v2i_demands and their respective areas and sizes
        # Update counts for each type of demand
        update_counts(self.demand, self.demand_size)

        update_counts(self.demand_all, self.demand_size_all)

        if self.Satellite_ == True:
            update_counts(self.demand_s, self.demand_s_size)

        # Now calculate the success rates for each area
        self.success_rates = {}
        for area in success_by_area:
            total_transmissions = success_by_area[area] + failed_by_area[area]
            self.success_rates[area] = (success_by_area[area] / total_transmissions) if total_transmissions > 0 else 0
        # 23/12/12 Ibrahim Update (Line2585~2623)

        # print("check",len(self.new_array),sum(V2S_Rate),self.Satellite_,self.success_transmission_V2S,self.failed_transmission_V2S)
        return V2I_Rate, V2V_Rate, V2S_Rate, failed_percentage_v2v, Utilization, sat_Utilization, \
            failed_percentage, failed_percentage_V2S, success_tx

    def act_for_testing(self, actions, step, global_clock, transmitted_clock, success_tx):
        global delaychange
        global V2V_d
        global V2I_d
        global V2S_d
        
        global V2S_pd
        # V2S_pd = np.zeros(size_of_agent)
        # V2V_pd = np.zeros(size_of_agent)
        # V2I_pd = np.zeros(size_of_agent)
        #
        # V2I_d = np.zeros(size_of_agent)
        # V2S_d = np.zeros(size_of_agent)
        Tr_d = np.zeros(len(self.V2I_Rate))
        Pr_d = np.zeros(len(self.V2I_Rate))
        Sum_d = np.zeros(len(self.V2I_Rate))
        action_temp = actions.copy()

        # 少 veh, truck

        V2I_Rate, V2V_Rate, V2S_Rate, V2V_fail_veh, u, sat_u, fail, V2S_fail, success_tx = self.Compute_Performance_Reward_Train(
            action_temp, step, global_clock, transmitted_clock, success_tx)

        V2I_success = 1 - fail
        V2V_success_veh = 1 - V2V_fail_veh
        V2S_success = 1 - V2S_fail

        for i in range(len(self.vehicles)):

            if V2I_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

            # for i in range(len(V2I_Rate)):
            if V2V_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

            if V2S_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])
                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])
                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])
            # if self.demand_s[i] > 0 and V2V_d[i] == 0 and V2S_Rate[i] > 0:
            # 	V2S_pd[i] = step + 12
            #
            # if self.demand[i] <= 0 and V2V_d[i] == 0 and V2S_Rate[i] <= 0:
            # 	V2V_d[i] = max(step, V2S_pd[i])
            #
            # if self.demand_all[i] <= 0 and V2V_d[i] == 0 and V2S_Rate[i] > 0:
            # 	V2V_d[i] = step + 12
            # res[i] = V2V_d[i]

            # V2I_d[i] += V2I_Rate[i]
            # V2S_d[i] += V2S_Rate[i]

            # if V2I_Rate[i] + V2V_Rate[i][0] + V2S_Rate[i] > 0:
            #     Tr_d[i] = 1
            # else:
            #     Tr_d[i] = 0
            # if V2S_Rate[i] > 0:
            #     Pr_d[i] = 12
            # elif V2I_Rate[i] + V2V_Rate[i][0] > 0:
            #     Pr_d[i] = 0.001
            # else:
            #     Pr_d[i] = 0
            # Sum_d[i] = Pr_d[i] + Tr_d[i]
        self.SNR_SINR_list.append(
            [self.SNR_SINR_list_ep, np.average(self.SNR_list_ep_V2V), np.average(self.SINR_list_ep_V2V),
             np.average(self.SNR_list_ep_V2I), np.average(self.SINR_list_ep_V2I),
             np.average(self.SNR_list_ep_V2S), np.average(self.SINR_list_ep_V2S),
             np.average(self.urban_SNR_per_episode_V2V),
             np.average(self.suburban_SNR_per_episode_V2V), np.average(self.rural_SNR_per_episode_V2V),
             np.average(self.urban_SNR_per_episode_V2I), np.average(self.suburban_SNR_per_episode_V2I),
             np.average(self.rural_SNR_per_episode_V2I), np.average(self.urban_SNR_per_episode_V2S),
             np.average(self.suburban_SNR_per_episode_V2S), np.average(self.rural_SNR_per_episode_V2S)])
        self.SNR_list_ep_V2V = []
        self.SNR_list_ep_V2I = []
        self.SNR_list_ep_V2S = []
        self.SINR_list_ep_V2V = []
        self.SINR_list_ep_V2I = []
        self.SINR_list_ep_V2S = []
        self.urban_SNR_per_episode_V2V = []
        self.suburban_SNR_per_episode_V2V = []
        self.rural_SNR_per_episode_V2V = []
        self.urban_SNR_per_episode_V2I = []
        self.suburban_SNR_per_episode_V2I = []
        self.rural_SNR_per_episode_V2I = []
        self.urban_SNR_per_episode_V2S = []
        self.suburban_SNR_per_episode_V2S = []
        self.rural_SNR_per_episode_V2S = []
        self.SNR_SINR_list_ep = round(self.SNR_SINR_list_ep + 0.1, 1)
        
        # 下面這個if的loop有demand
        if step % var.max_latency == 0:            
            avarage_data = self.averagex(self.V2V_Rate_avg)
            avarage_datav2i = self.averagex(self.V2I_Rate_avg)
            avarage_datav2s = self.averagex(self.V2I_Rate_avg) # 為什麼這裡不是用self.V2S_Rate_avg???

            self.switch_ratio = (self.Switch_count / (len(self.vehicles) * var.max_latency)) * 100
            
            self.v2v_selection_all = (self.v2v_selection / (len(self.vehicles) * var.max_latency)) * 100
            self.v2i_selection_all = (self.v2i_selection / (len(self.vehicles) * var.max_latency)) * 100
            self.v2s_selection_all = (self.v2s_selection / (len(self.vehicles) * var.max_latency)) * 100
            
            self.dump_selection_all = (self.dump_selection / (len(self.vehicles) * var.max_latency)) * 100
            
            # V2I self.demand_all
            if not (all(i <= 0 for i in self.demand_all)):
                for j in range(len(V2I_Rate)):
                    if round(avarage_datav2i, 2) <= 0 or math.isnan(avarage_datav2i):
                        avarage_datav2i = 10
                    if self.demand_all[j] > 0 and self.V2I_Rate_latency[j] > 0 and self.demand_all[
                        j] != self.demand_size_all:
                        self.V2I_Rate_latency[j] += (
                                (self.demand_all[j] + self.sig2) / ((avarage_datav2i) * (1e6)))
            
            # V2V self.demand
            if not (all(i <= 0 for i in self.demand)):
                for j in range(len(V2I_Rate)):
                    if round(avarage_data, 2) <= 0 or math.isnan(avarage_data):
                        avarage_data = 10
                    if self.demand[j] > 0 and self.V2V_Rate_latency[j] > 0 and self.demand[
                        j] != self.demand_size:
                        self.V2V_Rate_latency[j] += (
                                (self.demand[j] + self.sig2) / ((avarage_data) * (1e6)))
            
            # V2S self.demand_s
            if not (all(i <= 0 for i in self.demand_s)):
                for j in range(len(V2I_Rate)):
                    if round(avarage_datav2s, 2) <= 0 or math.isnan(avarage_datav2s):
                        avarage_data = 10
                    if self.demand_s[j] > 0 and self.V2S_Rate_latency[j] > 0 and self.demand_s[
                        j] != self.demand_s_size:
                        self.V2V_Rate_latency[j] += (
                                (self.demand_s[j] + self.sig2) / ((avarage_datav2s) * (1e6)))

            for i in range(len(V2I_Rate)):
                if self.method != 0:
                    if self.demand_all[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                            self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[0]
                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2I_Rate_latency[i])
                        else:
                            self.df_act_store_latency.loc[
                                len(self.df_act_store_latency)] = [self.vehicles[i].id, 
                                                                   int(self.V2I_Rate_latency[ i]), int(step)]
                    if self.demand[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                            self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[
                                0]
                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2V_Rate_latency[i])
                        else:
                            self.df_act_store_latency.loc[len(self.df_act_store_latency)] = [self.vehicles[i].id,
                                                                                             int(self.V2V_Rate_latency[
                                                                                                     i]),
                                                                                             int(step)]
                    if self.demand_s[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                            self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[
                                0]
                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2S_Rate_latency[i])
                        else:
                            self.df_act_store_latency.loc[len(self.df_act_store_latency)] = [self.vehicles[i].id,
                                                                                             int(self.V2S_Rate_latency[
                                                                                                     i]),
                                                                                             int(step)]
                if self.demand[i] < self.demand_size and self.demand[i] != 0:
                    V2V_Rate[i] = 0

            self.V2V_Rate_avg = []
            self.V2I_Rate_avg = []
            self.success_transmission_V2V = 0
            self.failed_transmission_V2V = 0
            self.success_transmission_V2S = 0
            self.failed_transmission_V2S = 0
            self.success_transmission = 0
            self.failed_transmission = 0
            self.Switch_count = 0
            self.v2v_selection = 0
            self.v2i_selection = 0
            self.v2s_selection = 0
            self.dump_selection = 0

        self.V2V_Rate_latency_nz = [v for v in self.V2V_Rate_latency if v != 0]
        self.V2I_Rate_latency_nz = [v for v in self.V2I_Rate_latency if v != 0]
        self.V2S_Rate_latency_nz = [v for v in self.V2S_Rate_latency if v != 0]

        return V2I_Rate, V2V_Rate, V2S_Rate, u, sat_u,  Sum_d, V2V_success_veh, V2S_success, V2I_success, success_tx

    def act_for_training(self, actions, step):
        global v
        action_temp = actions.copy()
        reward_early_finishV2V = np.zeros(len(self.vehicles))
        reward_early_finishV2I = np.zeros(len(self.vehicles))
        reward_early_finishV2S = np.zeros(len(self.vehicles))

        V2I_Rate, V2V_Rate, V2S_Rate, V2V_fail, u, sat_u, fail, V2S_fail = self.Compute_Performance_Reward_Train(
            action_temp, step)
        for i in range(len(V2I_Rate)):

           if self.demand_all[i] <= 0 :
               reward_early_finishV2I[i] = (var.max_latency - self.V2I_Rate_latency[i])
           if self.demand[i] <= 0:
               reward_early_finishV2V[i] = (var.max_latency - self.V2V_Rate_latency[i])
           if self.demand_s[i] <= 0:
               reward_early_finishV2S[i] = (var.max_latency - self.V2S_Rate_latency[i])

                # self.penalty_switchV2I[i] = (self.Data_rate_V2I_all[
                #                                  i] * self.time_fast * self.bandwidth * var.TTI) - (
                #                                 self.demand_size_all)





                # print(step,i,reward_early_finishV2V[i])


        self.V2I_Rate = V2I_Rate
        self.V2V_Rate = V2V_Rate
        self.V2S_Rate = V2S_Rate
        w1 = var.V2I_w
        w2 = var.V2V_w
        w3 = var.Early_finish_w
        Sum_d = np.zeros(len(self.V2I_Rate))
        V2I_success = 1 - fail
        V2V_success = 1 - V2V_fail
        V2S_success = 1 - V2S_fail

        bounes = 0
        for i in range(len(self.vehicles)):

            if V2I_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2I.append(self.SNR_rate_spacific_V2I[i])

            # for i in range(len(V2I_Rate)):
            if V2V_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2V.append(self.SNR_rate_spacific_V2V[i])

            if V2S_Rate[i] != 0:

                if self.vehicles[i].area == 'urban':
                    self.urban_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])
                if self.vehicles[i].area == 'suburban':
                    self.suburban_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])
                if self.vehicles[i].area == 'rural':
                    self.rural_SNR_per_episode_V2S.append(self.SNR_rate_spacific_V2S[i])

            # if  self.demand[i] <= 0:
            #     bounes += 1.1
            # if  self.demand_all[i] <= 0:
            #     bounes += 1.8
            # if  self.demand_s[i] <= 0:
            #     bounes += 1.8
            # if self.demand_s[i] > 0 and V2V_d[i] == 0 and V2S_Rate[i] > 0:
            # 	V2S_pd[i] = step + 12
            #
            # if self.demand[i] <= 0 and V2V_d[i] == 0 and V2S_Rate[i] <= 0:
            # 	V2V_d[i] = max(step, V2S_pd[i])
            #
            # if self.demand_all[i] <= 0 and V2V_d[i] == 0 and V2S_Rate[i] > 0:
            # 	V2V_d[i] = step + 12
            # res[i] = V2V_d[i]

            # V2I_d[i] += V2I_Rate[i]
            # V2S_d[i] += V2S_Rate[i]

            # if V2I_Rate[i] + V2V_Rate[i][0] + V2S_Rate[i] > 0:
            #     Tr_d[i] = 1
            # else:
            #     Tr_d[i] = 0
            # if V2S_Rate[i] > 0:
            #     Pr_d[i] = 12
            # elif V2I_Rate[i] + V2V_Rate[i][0] > 0:
            #     Pr_d[i] = 0.001
            # else:
            #     Pr_d[i] = 0
            # Sum_d[i] = Pr_d[i] + Tr_d[i]

        if step % var.max_latency == 0:
            print("V2I_success", V2I_success, "V2I sum data rate", sum(V2I_Rate))
            print("V2V_success", V2V_success, "V2V sum data rate", sum(V2V_Rate))
            print("V2S_success", V2S_success, "V2S sum data rate", sum(V2S_Rate))
            self.SNR_SINR_list.append(
                [self.SNR_SINR_list_ep, np.average(self.SNR_list_ep_V2V), np.average(self.SINR_list_ep_V2V),
                 np.average(self.SNR_list_ep_V2I), np.average(self.SINR_list_ep_V2I),
                 np.average(self.SNR_list_ep_V2S), np.average(self.SINR_list_ep_V2S),
                 np.average(self.urban_SNR_per_episode_V2V),
                 np.average(self.suburban_SNR_per_episode_V2V), np.average(self.rural_SNR_per_episode_V2V),
                 np.average(self.urban_SNR_per_episode_V2I), np.average(self.suburban_SNR_per_episode_V2I),
                 np.average(self.rural_SNR_per_episode_V2I), np.average(self.urban_SNR_per_episode_V2S),
                 np.average(self.suburban_SNR_per_episode_V2S), np.average(self.rural_SNR_per_episode_V2S)])
            self.SNR_list_ep_V2V = []
            self.SNR_list_ep_V2I = []
            self.SNR_list_ep_V2S = []
            self.SINR_list_ep_V2V = []
            self.SINR_list_ep_V2I = []
            self.SINR_list_ep_V2S = []
            self.urban_SNR_per_episode_V2V = []
            self.suburban_SNR_per_episode_V2V = []
            self.rural_SNR_per_episode_V2V = []
            self.urban_SNR_per_episode_V2I = []
            self.suburban_SNR_per_episode_V2I = []
            self.rural_SNR_per_episode_V2I = []
            self.urban_SNR_per_episode_V2S = []
            self.suburban_SNR_per_episode_V2S = []
            self.rural_SNR_per_episode_V2S = []
            self.SNR_SINR_list_ep = round(self.SNR_SINR_list_ep + 0.1, 1)
            for i in range(len(V2I_Rate)):
                if self.method != 0:
                    if self.demand_all[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                                self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[
                                    0]

                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2I_Rate_latency[i])

                        else:
                            self.df_act_store_latency.loc[len(self.df_act_store_latency)] = [self.vehicles[i].id,
                                                                                             int(self.V2I_Rate_latency[
                                                                                                     i]),
                                                                                             int(step)]
                    if self.demand[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                                self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[
                                    0]
                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2V_Rate_latency[i])
                        else:
                            self.df_act_store_latency.loc[len(self.df_act_store_latency)] = [self.vehicles[i].id,
                                                                                             int(self.V2V_Rate_latency[
                                                                                                     i]),
                                                                                             int(step)]
                    if self.demand_s[i] <= 0:
                        if (self.df_act_store_latency["id"] == self.vehicles[i].id).any():
                            indexdfl = \
                                self.df_act_store_latency[self.df_act_store_latency["id"] == self.vehicles[i].id].index[
                                    0]
                            self.df_act_store_latency.loc[indexdfl, 'latency'] = int(self.V2S_Rate_latency[i])
                        else:
                            self.df_act_store_latency.loc[len(self.df_act_store_latency)] = [self.vehicles[i].id,
                                                                                             int(self.V2S_Rate_latency[
                                                                                                     i]),
                                                                                             int(step)]
            self.penaltyV2I = np.zeros(len(self.vehicles))
            self.penalty = np.zeros(len(self.vehicles))
            self.penalty_per = np.zeros(len(self.vehicles))

            self.V2I_Rate_latency = np.zeros(len(self.vehicles))
            self.V2V_Rate_latency = np.zeros(len(self.vehicles))
            self.V2S_Rate_latency = np.zeros(len(self.vehicles))

            self.success_transmission_V2V = 0
            self.failed_transmission_V2V = 0
            self.success_transmission = 0
            self.failed_transmission = 0
            self.success_transmission_V2S = 0
            self.failed_transmission_V2S = 0
            if self.demand[i] < self.demand_size and self.demand[i] != 0:
                V2V_Rate[i] = 0

        # reward = (V2V_success +V2I_success + V2S_success - 1) * (1 + (0.5 * (1 - (
        #			  (np.sum(V2I_Rate) + np.sum(V2V_Rate) + np.sum(V2S_Rate)) / ((12 * 12000 * 8) / (100 * 1000))))))
        # reward = (V2V_success - 1 + V2I_success - 1 + V2S_success - 1) * (
        #             1 + (0.5 * (1 - ((np.sum(V2I_Rate) + np.sum(V2V_Rate) + np.sum(V2S_Rate)) / (
        #             (12000 * 8 * len(self.V2I_Rate)) / (100 *

        remain =(
                (np.sum(self.demand) + np.sum(self.demand_all) + np.sum(self.demand_s)) / (
                    ((self.demand_size) + (self.demand_s_size) + (self.demand_size_all)) * len(self.vehicles)))
        sucess =0
        if self.Satellite_ ==True:
            sucess = np.average ([V2V_success, V2I_success ,V2S_success])
        else :
            sucess =np.average ([V2V_success, V2I_success ])
        result_count_V2V = len([x for x in self.demand if x == 0])
        result_count_V2I = len([x for x in self.demand_all if x == 0])
        result_count_V2S = len([x for x in self.demand_s if x == 0])
        # reward =self.reward_c(sucess,remain) *(result_count_V2V + result_count_V2I +result_count_V2S+ 1) / len(self.vehicles)
        # reward = 2*((V2V_success) + (V2I_success) + (V2S_success)) -(1 *  (
        #         (np.sum(self.demand) + np.sum(self.demand_all) + np.sum(self.demand_s)) / (
        #             ((self.demand_size) + (self.demand_s_size) + (self.demand_size_all)) * len(self.vehicles))))
        # if len(self.demand[self.demand <= 0]) +len(self.demand_s[self.demand_s <= 0]) +len(self.demand_all[self.demand_all <= 0]) == len(self.vehicles):
        #     reward = reward + 1

        # reward = ((V2V_success ) + (V2I_success ) + (V2S_success ))  * (
        #         (np.sum(self.demand) + np.sum(self.demand_size) + np.sum(self.demand_s)) / ((( self.demand_size  )+(self.demand_s_size )+(self.demand_size_all) )*len(self.vehicles)))
        # if ((np.sum(reward_early_finishV2I)) + (
        #         np.sum(reward_early_finishV2V)+np.sum(reward_early_finishV2S[i]))) == 0:
        #     reward = V2V_success + V2I_success
        # else:
        #     reward = V2V_success + V2I_success + (1 / ((np.sum(reward_early_finishV2I)) + (
        #         np.sum(reward_early_finishV2V)+np.sum(reward_early_finishV2S[i]))))

        reward_loss_V2V, reward_loss_V2S, reward_loss_V2I = 0, 0, 0
        for i in range(len(self.vehicles)):
            if self.demand[i] < self.demand_size:
                reward_loss_V2V += (self.demand[i] / self.demand_size)
            if self.demand_s[i] < self.demand_s_size:
                reward_loss_V2S += (self.demand_s[i] / self.demand_s_size)
            if self.demand_all[i] < self.demand_size_all:
                reward_loss_V2I += (self.demand_all[i] / self.demand_size_all)

        total_loss = reward_loss_V2I + reward_loss_V2S + reward_loss_V2V
        average_loss = total_loss / len(self.vehicles)

        # The optimal values for success rates were very high, close to 1.
        # It means success rates are crucial.
        # Given that, let's put a significant weight on success rates.
        alpha = 2.0

        # Loss should be minimized. Given the optimal values for loss were small,
        # they play a vital role in the reward function.
        # Therefore, let's assign a higher weight to it.
        beta = 2.5

        # Calculate switch penalty
        # switch_penalty = 0.5 * self.switch_ratio
        switch_penalty = 0.8 * self.switch_ratio

        if self.Satellite_:
            success_component = np.average([V2I_success, V2V_success, V2S_success])
            reward = alpha * success_component - beta * average_loss - switch_penalty
        else:
            success_component = np.average([V2I_success, V2V_success])
            reward = alpha * success_component - beta * average_loss - switch_penalty


        v = 0

        done = False
        if np.sum(V2V_Rate) + np.sum(V2I_Rate)+ np.sum(V2S_Rate) == 0:
            done = True
        return reward ,done

    def reward_c(self,x, y):
        k1 = 1.0  # Constant factor for x, y, and z
        k2 = 1.0  # Constant factor for n_i, m_i, and L_i


        # Reward for x, y, and z above threshold
        xyz_reward = k1 * ((x) )


        # Penalty for n_i, m_i, and L_i
        nml_penalty = k2 * y

        # Final reward
        reward = xyz_reward * nml_penalty

        return reward

    def Compute_SINR(self, actions):
        global V2I_SINR
        global V2V_SINR
        global V2S_SINR

        self.V2I_SINR_all = V2I_SINR
        self.V2V_SINR_all = V2V_SINR
        self.V2S_SINR_all = V2S_SINR
        self.V2S_SINR_all = V2S_SINR

    def reset(self):
        self.vehicle_time_id = var.tarin_sumo_step
    
    def Compute_Interference(self, actions):
        global V2I_Interference1
        global V2V_Interference1


        self.V2I_Interference_all = self.convert_W_to_dB(V2I_Interference1)
        self.V2V_Interference_all = self.convert_W_to_dB(V2V_Interference1)
    
    
    def new_random_game(self, n_Veh=0):
        # make a new game
        self.neighbor_vehicles = []
        self.vehicles = []
        self.satellite = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.renew_positions()
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()