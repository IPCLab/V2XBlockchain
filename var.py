n_vehicle = 8
mode = 'NR' # mode NR or LTE
number_eps = 300   # 1200
n_neighbor = 1
n_RB = 5
# n_RB = 10
SRB_max = 10
# SRB_max = 20
# use in reward
V2V_max_rate = -1
V2I_max_rate = -1
to_serve_index = 0
to_serve_value = []
n_cluster = n_vehicle
n_cluster_test = 20   # same cluster update Kmeans
limted_cluster = 22    # RSU cluster
limted_cluster_train = 19   # Train  cluster
limted_cluster_full = 22   # Full cluster
limted_cluster_Kmeans = 6   # Kmeans cluster
min_c = 5   # minimum_size of cluster
size_cluster = 6
n_cluster_train = 16
n_cluster_train_samsize = 17
n_episode_test = 10000
# where V2I_w and V2V_w the weight of the transmission rate
V2I_w = 1
V2V_w = 1
Early_finish_w = 1
neighbor = 10
size = False
opt = 1   # 0 'GradientDescentOptimizer', 1 'AdamOptimizer', 2'RMSPropOptimizer'
learning_ratechange = False   # False means the original learning rate
# stop_step = 1327.4  # 345.9 144.5
stop_step = 49.9  # 345.9 144.5
# stop_step = 99.9
# stop_step = 0.5
test_sumo_step = 65
# train_sumo_step = 65
train_sumo_step = -1
# train_sumo_step = -0.2
append = 1   # 1 append to the current result 0 create a new results file
# V2V_payloadsize = 250
V2V_payloadsize = 650
# V2I_payloadsize = 500
V2I_payloadsize = 650
# V2S_payloadsize = 500
V2S_payloadsize = 650
V2V_payloadsize_test = V2V_payloadsize
V2I_payloadsize_test = V2I_payloadsize
V2S_payloadsize_test = V2S_payloadsize
agent_tarin = range(100,300,1)
# agent_test = range(80,301,20)
agent_test = range(100,600,100)
max_latency = 3
# max_latency = 100
T_max = 100
propagation_delay=12.44
TTI = 1
# Yujing District
lat= 23.1061
lon =120.4703
sat_data = False # False =use the   satellite dataset. True you can genearte online data set
sat_name ="DTUSAT-2" # The Satellite name SPACEBEE-122  DTUSAT-2 TEN-KOH XW-2B
change_cluster_period =10
n_input = 0
n_output = 0
load  =True
import sys
seed = int(sys.argv[1])