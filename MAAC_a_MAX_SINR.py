from __future__ import division, print_function
import numpy as np
import Env
import var
import os
os.environ['OMP_NUM_THREADS'] = '16'
import pandas as pd

import datetime
from Models.Ethereum.BlockCommit import BlockCommit
from Event import Queue
from Statistics import Statistics
from Models.Ethereum.Transaction import LightTransaction as LT

import warnings
warnings.filterwarnings("ignore")

method = 0
# 0	No cluster, 1 K-means Clustering, 2	Same size K-means Clustering, 4	Spectral Clustering , 5	KMedoids Clustering , 6	Fuzzy C-Means Clustering ,
# 7 Agglomerative Clustering, 8	HDBSCAN	, 9	Mini Batch Kmeans, 10 RSU HDBSCAN Clustering, 11 Updated K-means Clustering

env = Env.Environ(method, test=True, Satellite=True)

env.new_random_game()  # initialize parameters in env

BATCH_SIZE = 1

n_RB_max = env.n_RB_max

V2I_SINR = None
V2V_SINR = None
V2S_SINR = None

def main():
    all_resultsnew = [] # 用來儲存最終要輸出的各項指標結果
    all_resultsnew_type = [] # 用來儲存最終要輸出的各項指標結果(有分不同區域型態)

    def get_channel_quality():
        array_tolist = env.new_array.tolist()
        act = np.zeros([len(env.vehicles), n_neighbor], dtype='int32')
        for i in range(len(env.vehicles)):
            sinr = 0
            max = 0
            selectedj = 0 # RB_subchannel
            selectedp = 0 # power

            for j in range(env.n_RB_max):  # n_RB -> RB_subchannel
                # V2I
                signal = 10 ** (
                        (23 - env.V2I_channels_with_fastfading[
                            i, env.vehicles[i].destinationsBS, j] +
                         env.vehAntGain + env.bsAntGain) / 10)
                # self.vehAntGain = 3, self.bsAntGain = 8
                if env.interference_all_v2i[j] == 0:
                    sinr = signal
                else:
                    sinr = 10 * np.log10((1 / (env.interference_all_v2i[j])) * signal + 0.00001)
                if sinr > max:
                    max = sinr
                    selectedj = j # RB_subchannel
                    selectedp = 24 # power

                # V2V
                for k in range(len(env.V2V_power_dB_List)):
                    signal = 10 ** (
                            (env.V2V_power_dB_List[k] - env.V2V_channels_with_fastfading[
                                i, env.vehicles[i].destinations, j] +
                             2 * env.vehAntGain) / 10)
                    if env.interference_all_v2v[j] == 0:
                        sinr = signal
                    else:
                        sinr = 10 * np.log10((1 / (env.interference_all_v2v[j])) * signal + 0.00001)
                    if sinr > max:
                        max = sinr
                        selectedj = j # RB_subchannel
                        selectedp = env.V2V_power_dB_List[k] # power
            # V2S
            for j in range(env.n_SRB_max):  # n_SRB SRB_subchannel for V2S
                signal = 10 ** (
                        (33.5 - env.V2S_channels_with_fastfading[
                            i, env.vehicles[i].destinationsST, j] +
                         env.ueTXGain + env.stRXGain) / 10)
                if env.interference_all_v2s[j] == 0:
                    sinr = signal
                else:
                    sinr = 10 * np.log10((1 / (env.interference_all_v2s[j])) * signal + 0.00001)
                if sinr > max:
                    max = sinr
                    selectedj = j # SRB_subchannel
                    selectedp = 25  # Assuming 25 is the power index for V2S, change it according to your model

            if selectedp + selectedj == 0: # dump channel
                selectedj = var.n_RB # RB_subchannel
                selectedp = 0 # power

            act[i, 0] = array_tolist.index([selectedj, selectedp])
        return act

    label = 'MAX_SINR'

    n_neighbor = var.n_neighbor # 1
    n_episode_test = var.n_episode_test  # test episodes = 10000
    # n_episode_test = 2
    # n_step_per_episode = int(env.time_slow / env.time_fast) # test step = 100
    n_step_per_episode = 3
    # n_step_per_episode = 1

    epsi_final = 0.02

    def averagex(x):
        # If x is not a list or numpy array, return its average (assuming it's a number or can be converted to numpy array)
        if not isinstance(x, list):
            return np.average(x)
        else:
        # Convert list to numpy array if necessary for faster operations
            x = np.array(x)

        # Use boolean indexing to filter out zero values
        non_zero_values = x[x != 0]

        # Return average of non-zero values or 0 if the array is empty
        return np.average(non_zero_values) if non_zero_values.size > 0 else 0

    # 將每個 episode 結束後的各項結果以 CSV 資料格式輸出
    def resultsappend(all_resultsnew, all_resultsnew_type):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(os.path.join(current_dir, "{0}CSV".format(label))):
            os.makedirs(os.path.join(current_dir, "{0}CSV".format(label)))
            print(os.path.join(current_dir, "{0}CSV".format(label)))

        if env.demand_step == 0:
            all_results = pd.DataFrame(all_resultsnew,
                                       columns=['step', 'n_vehicles', 'Demand',
                                                'V2I_data_rate',
                                                'V2V_data_rate',
                                                'V2S_data_rate',
                                                'SUM',
                                                'Utilization',
                                                'V2S_Utilization',
                                                'V2I_success_rate', 'LossV2I',
                                                'V2V_success_rate', 'V2I_Latency',
                                                'V2S_success_rate', 'V2S_Latency', 'LossV2S',
                                                'V2V_Latency', 'LossV2V',
                                                'Switch_ratio',
                                                "V2V_selection", "V2I_selection", "V2S_selection", "dump_selection"])
            all_results.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}.csv".format(label)))
        else:
            all_results = pd.DataFrame(all_resultsnew,
                                       columns=['step', 'n_vehicles', 'Demand',
                                                'V2I_data_rate',
                                                'V2V_data_rate',
                                                'V2S_data_rate',
                                                'SUM',
                                                'Utilization',
                                                'V2S_Utilization',
                                                'success_rate', 'Loss',
                                                'V2V_success_rate', 'V2I_Latency',
                                                'V2S_success_rate', 'V2S_Latency', 'LossV2S',
                                                'V2V_Latency', 'LossV2V',
                                                'Switch_ratio',
                                                "V2V_selection", "V2I_selection", "V2S_selection", "dump_selection"])
            if not os.path.isfile(os.path.join(current_dir, "{0}CSV/Allresults{0}.csv".format(label))):
                all_results.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}.csv".format(label)))
            else:
                all_results.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}.csv".format(label)),
                                   mode='a',
                                   header=False)

        if env.demand_step == 0:
            all_results_type = pd.DataFrame(all_resultsnew_type,
                                            columns=['step', 'n_vehicles', 'Urban_success',
                                                     'suburban_success',
                                                     'rural_success',
                                                     'urban_rate',
                                                     'suburban_rate',
                                                     'rural_rate',
                                                     'urban_SNR_V2V',
                                                     'suburban_SNR_V2V', 'rural_SNR_V2V',
                                                     'urban_SNR_V2I',
                                                     'suburban_SNR_V2I', 'rural_SNR_V2I',
                                                     'urban_SNR_V2S',
                                                     'suburban_SNR_V2S', 'rural_SNR_V2S'])
            all_results_type.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}_type.csv".format(label)))
            SINR_SNRdf = pd.DataFrame(env.SNR_SINR_list,
                                      columns=['step', 'V2V_SNR', 'V2V_SINR', 'V2I_SNR', 'V2I_SINR', 'V2S_SNR',
                                               'V2S_SINR', 'urban_SNR_V2V',
                                               'suburban_SNR_V2V', 'rural_SNR_V2V',
                                               'urban_SNR_V2I',
                                               'suburban_SNR_V2I', 'rural_SNR_V2I',
                                               'urban_SNR_V2S',
                                               'suburban_SNR_V2S', 'rural_SNR_V2S'])
            SINR_SNRdf.to_csv(os.path.join(current_dir, "{0}CSV/SINR_SNR_test.csv".format(label)))
        else:
            all_results_type = pd.DataFrame(all_resultsnew_type,
                                            columns=['step', 'n_vehicles', 'Urban_success',
                                                     'suburban_success',
                                                     'rural_success',
                                                     'urban_rate',
                                                     'suburban_rate',
                                                     'rural_rate',
                                                     'urban_SNR_V2V',
                                                     'suburban_SNR_V2V', 'rural_SNR_V2V',
                                                     'urban_SNR_V2I',
                                                     'suburban_SNR_V2I', 'rural_SNR_V2I',
                                                     'urban_SNR_V2S',
                                                     'suburban_SNR_V2S', 'rural_SNR_V2S'])
            if not os.path.isfile(os.path.join(current_dir, "{0}CSV/Allresults{0}_type.csv".format(label))):
                all_results_type.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}_type.csv".format(label)))
            else:
                all_results_type.to_csv(os.path.join(current_dir, "{0}CSV/Allresults{0}_type.csv".format(label)),
                                        mode='a',
                                        header=False)
            SINR_SNRdf = pd.DataFrame(env.SNR_SINR_list,
                                      columns=['step', 'V2V_SNR', 'V2V_SINR', 'V2I_SNR', 'V2I_SINR', 'V2S_SNR',
                                               'V2S_SINR', 'urban_SNR_V2V',
                                               'suburban_SNR_V2V', 'rural_SNR_V2V',
                                               'urban_SNR_V2I',
                                               'suburban_SNR_V2I', 'rural_SNR_V2I',
                                               'urban_SNR_V2S',
                                               'suburban_SNR_V2S', 'rural_SNR_V2S'])
            SINR_SNRdf.to_csv(os.path.join(current_dir, "{0}CSV/SINR_SNR_test.csv".format(label)))

    def test_episodic():
        Switch_ratio_list = []

        current_time = datetime.datetime.now() # 獲取現在的時間
        time_str = current_time.strftime("%Y%m%d_%H%M%S") # 將時間轉換成字串，以便放入檔名中

        os.makedirs(os.path.join(os.getcwd(), "Performance_Result/Random_Seed"), exist_ok=True)

        fname_trans_perform = "Performance_Result/Random_Seed/{0}_TX_Perform_His_{1}.xlsx".format(time_str, label)
        writer_trans_perform = pd.ExcelWriter(fname_trans_perform, engine = 'xlsxwriter')

        fname_system_perform = "Performance_Result/Random_Seed/{0}_BC_Sys_Perform_{1}.xlsx".format(time_str, label)
        writer_system_perform = pd.ExcelWriter(fname_system_perform, engine = 'xlsxwriter')

        fname_tx_number = "Performance_Result/Random_Seed/{0}_TX_Num_His_{1}.xlsx".format(time_str, label)
        writer_tx_number = pd.ExcelWriter(fname_tx_number, engine = 'xlsxwriter')

        fname_veh_selection = "Performance_Result/Random_Seed/{0}_Veh_Selection_{1}.xlsx".format(time_str, label)
        writer_veh_selection = pd.ExcelWriter(fname_veh_selection, engine = 'xlsxwriter')

        # for blockchain
        BlockCommit.generate_initial_events(env.rsu_nodes_list, 0)
        global_clock = 0
        transmitted_clock = 0
        v2s2b_delay = 0.024
        mgp = 0.5 # message generation period
        # mgp = 0.1
        LT.clean_pool()

        trans_perform_his = {} # 存一個episode中100個step所有的transaction performance
        tx_number_his = {}

        global veh_selection

        for idx_episode in range(n_episode_test):
            print(f"=========Episode: {idx_episode}=========")

            # reset all saving data (list, array, int)
            if 1:
                env.renew_positions() # update vehicle position
                env.renew_neighbor() # update neighbor(?) vehicle position
                env.renew_channel() # update channel slow fading
                env.renew_channels_fastfading() # update channel fast fading

                env.count_veh += len(env.vehicles)
                action_all_testing = np.zeros([len(env.vehicles), n_neighbor, 1], dtype='int32')

                env.demand = env.demand_size * np.ones((len(env.vehicles), env.n_neighbor))
                env.demand_all = env.demand_size_all * np.ones((len(env.vehicles), env.n_neighbor))
                env.demand_s = env.demand_s_size * np.ones((len(env.vehicles), env.n_neighbor))

                env.individual_time_limit = env.time_slow * np.ones((len(env.vehicles), env.n_neighbor))
                env.individual_time_limit_all = env.time_slow * np.ones((len(env.vehicles), env.n_neighbor))
                env.active_links = np.ones((len(env.vehicles), env.n_neighbor), dtype='bool')
                env.active_links_all = np.ones((len(env.vehicles), env.n_neighbor), dtype='bool')
                V2I_rate_per_episode = np.zeros((len(env.vehicles)))
                V2V_rate_per_episode = np.zeros((len(env.vehicles)))
                V2S_rate_per_episode = np.zeros((len(env.vehicles)))
                urban_rate_per_episode = np.zeros((len(env.vehicles)))
                suburban_rate_per_episode = np.zeros((len(env.vehicles)))
                rural_rate_per_episode = np.zeros((len(env.vehicles)))
                urban_SNR_per_episode_V2V = []
                suburban_SNR_per_episode_V2V = []
                rural_SNR_per_episode_V2V = []
                urban_SNR_per_episode_V2I = []
                suburban_SNR_per_episode_V2I = []
                rural_SNR_per_episode_V2I = []
                urban_SNR_per_episode_V2S = []
                suburban_SNR_per_episode_V2S = []
                rural_SNR_per_episode_V2S = []
                Urban_success_list = 0
                suburban_success_list = 0
                rural_success_list = 0
                Urban_success_list_per_episode = []
                suburban_success_list_per_episode = []
                rural_success_list_per_episode = []
                U_per_episode = []
                V2S_U_per_episode = []
                S_per_episode = 0
                D_per_episode = 0
                L_per_episode = 0
                S_per_episodeV2V = 0
                D_per_episodeV2V = 0
                L_per_episodeV2V = 0
                S_per_episodeV2S = 0
                D_per_episodeV2S = 0
                L_per_episodeV2S = 0
                V2I_rate_per_episode_i = []
                V2V_rate_per_episode_i = []
                V2S_rate_per_episode_i = []
                urban_rate_per_episode_i = []
                suburban_rate_per_episode_i = []
                rural_rate_per_episode_i = []
                urban_SNR_per_episode_V2V_i = []
                suburban_SNR_per_episode_V2V_i = []
                rural_SNR_per_episode_V2V_i = []
                urban_SNR_per_episode_V2I_i = []
                suburban_SNR_per_episode_V2I_i = []
                rural_SNR_per_episode_V2I_i = []
                urban_SNR_per_episode_V2S_i = []
                suburban_SNR_per_episode_V2S_i = []
                rural_SNR_per_episode_V2S_i = []
                V2S_latency_list =[]
                V2V_latency_list = []
                V2I_latency_list = []
                U_per_episode_i = []
                V2S_U_per_episode_i = []
                S_per_episode_v2i = []
                L_per_episode_v2i = []
                D_per_episode_v2i = []
                S_per_episode_V2S = []
                L_per_episode_V2S = []
                D_per_episode_V2S = []
                S_per_episode_v2v = []
                Switch_per_episode = []
                L_per_episode_v2v = []
                D_per_episode_v2v = []
                v2v_selection_per_episode = []
                v2i_selection_per_episode = []
                V2S_selection_per_episode = []
                dump_selection_per_episode = []
            # Take steps in the environment untill terminal state of epsiode

            if env.sumo_step >= var.stop_step:
                print(f"##sumo step = stop step -> sim done##")
                break

            transmitted_clock = global_clock

            success_tx = 0

            tx_number_his[f"Episode{idx_episode}"] = [None, None, None]

            veh_selection = {}

            for test_step in range(1, n_step_per_episode + 1):
                print(f"=========Step: {test_step}=========")

                action_all_testing[:, :, 0] = get_channel_quality() # for MAX SINR channel selection method

                action_temp = action_all_testing.copy()

                transmitted_clock += 0.001
                transmitted_clock = round(transmitted_clock, 3)

                # print(f"Global Clock: {global_clock}")
                # print(f"Transmitted Clock: {transmitted_clock}")

                # Create V2X Transactions
                V2I_rate, V2V_rate, V2S_rate, u, V2S_u,  Sum_d, V2V_success, V2S_success, V2I_success, \
                    success_tx = env.act_for_testing(action_temp, test_step, global_clock, transmitted_clock, success_tx)

                new_array_list = env.new_array.tolist()
                veh_selection = {}
                for i in range(len(env.vehicles)):
                    index = action_all_testing[i, :, 0]
                    selectedj, selectedp = new_array_list[index[0]]
                    vehicle_id = env.vehicles[i].id
                    veh_selection[vehicle_id] = [selectedj, selectedp, env.vehicles[i].area, None, env.vehicles[i].success_fail]
                    if veh_selection[vehicle_id][1] == 24:
                        veh_selection[vehicle_id][3] = 'V2I'
                        # veh_selection[vehicle_id][1] = 23
                    elif veh_selection[vehicle_id][1] == 25:
                        veh_selection[vehicle_id][3] = 'V2S'
                        # veh_selection[vehicle_id][1] = 33.5
                    else:
                        veh_selection[vehicle_id][3] = 'V2V'
                # print(veh_selection)

                column_name_veh_select = ['Sub-channel', 'Mode Code Name', 'Area', 'Mode', 'SorF']
                df_veh_selection = pd.DataFrame.from_dict(veh_selection, orient = "index", columns = column_name_veh_select)
                df_veh_selection.to_excel(writer_veh_selection, sheet_name = f"Ep{idx_episode}_St{test_step}")

                del_key = []
                # 如果有經過24ms的交易存於v2s_tx_pool中
                for key in LT.v2s_tx_pool.keys():
                    # 因為V2S和S2I的傳輸延遲各為12ms
                    if (key+v2s2b_delay) == transmitted_clock:
                        for tx_in_sat in LT.v2s_tx_pool[key]:
                            LT.pool.append(tx_in_sat) # 將交易一個個放進pool
                            # print(f"成功將V2S2I交易 {tx_in_sat.id} 加進RSU的pool")
                        del_key.append(key)
                    if transmitted_clock == (global_clock + round(n_step_per_episode*0.001, 3)):
                        if (key+v2s2b_delay) > transmitted_clock and (key+v2s2b_delay) <= global_clock+mgp:
                            for tx_in_sat in LT.v2s_tx_pool[key]:
                                LT.pool.append(tx_in_sat) # 將交易一個個放進pool
                                # print(f"成功將V2S2I交易 {tx_in_sat.id} 加進RSU的pool")
                            del_key.append(key)
                for key in del_key:
                    del LT.v2s_tx_pool[key]

                # print(f"Number of Transactions in v2v_tx_pool: {sum(len(tx_list) for tx_list in LT.v2v_tx_pool.values())}")

                # Utility平均
                U_per_episode.append(averagex(u))
                V2S_U_per_episode.append(averagex(V2S_u))

                # 各種V2X的Data Rate總和
                for i in range(len(env.vehicles)):
                    V2I_rate_per_episode[i] += V2I_rate[i]  # sum V2I rate in bps
                    V2V_rate_per_episode[i] += V2V_rate[i][0]  # sum V2V rate in bps
                    V2S_rate_per_episode[i] += V2S_rate[i]  # sum V2V rate in bps

                # 記錄不同area下各種V2X的SNR
                for i in range(len(env.vehicles)):
                    if (V2I_rate[i] + V2V_rate[i] + V2S_rate[i]) == 0:
                        env.Data_rate_spacific[i] = 0
                        env.SNR_rate_spacific_V2V[i] = 0
                        env.SNR_rate_spacific_V2I[i] = 0
                        env.SNR_rate_spacific_V2S[i] = 0

                    if env.vehicles[i].area == 'urban':
                        urban_rate_per_episode[i] += env.Data_rate_spacific[i]
                        urban_SNR_per_episode_V2V.append(env.SNR_rate_spacific_V2V[i])
                        urban_SNR_per_episode_V2I.append(env.SNR_rate_spacific_V2I[i])
                        urban_SNR_per_episode_V2S.append(env.SNR_rate_spacific_V2S[i])
                    if env.vehicles[i].area == 'suburban':
                        suburban_rate_per_episode[i] += env.Data_rate_spacific[i]
                        suburban_SNR_per_episode_V2V.append(env.SNR_rate_spacific_V2V[i])
                        suburban_SNR_per_episode_V2I.append(env.SNR_rate_spacific_V2I[i])
                        suburban_SNR_per_episode_V2S.append(env.SNR_rate_spacific_V2S[i])
                    if env.vehicles[i].area == 'rural':
                        rural_rate_per_episode[i] += env.Data_rate_spacific[i]
                        rural_SNR_per_episode_V2V.append(env.SNR_rate_spacific_V2V[i])
                        rural_SNR_per_episode_V2I.append(env.SNR_rate_spacific_V2I[i])
                        rural_SNR_per_episode_V2S.append(env.SNR_rate_spacific_V2S[i])

                env.renew_channels_fastfading()
                env.Compute_Interference(action_temp)
                env.Compute_SINR(action_temp)

                # max_latency = 3
                # 每3個step執行下面兩個if迴圈
                if test_step % var.max_latency == 0:

                    Switch_ratio_list.append(env.switch_ratio)

                    S_per_episode = V2I_success
                    S_per_episodeV2V = V2V_success
                    S_per_episodeV2S = V2S_success

                    u_at = u_st = s_at = s_st = r_at = r_st = u_sr = s_sr = r_sr = 0

                    for i in range(len(env.vehicles)):
                        if env.vehicles[i].area == 'urban':
                            u_at += 1
                            if env.remain[i] <= 0:
                                u_st += 1
                        if env.vehicles[i].area == 'suburban':
                            s_at += 1
                            if env.remain[i] <= 0:
                                s_st += 1
                        if env.vehicles[i].area == 'rural':
                            r_at += 1
                            if env.remain[i] <= 0:
                                r_st += 1

                    if u_at != 0:
                        u_sr = u_st / u_at  # success rate
                    if s_at != 0:
                        s_sr = s_st / s_at
                    if r_at != 0:
                        r_sr = r_st / r_at

                    Urban_success_list = u_sr
                    suburban_success_list = s_sr
                    rural_success_list = r_sr

                    demandsum = [v for v in env.demand_all if v != env.demand_size_all]
                    D_per_episode = np.sum(demandsum)
                    demandsum = [v for v in env.demand if v != env.demand_size]
                    D_per_episodeV2V = np.sum(demandsum)
                    demandsum = [v for v in env.demand_s if v != env.demand_s_size]
                    D_per_episodeV2S = np.sum(demandsum)

                    env.remain = np.zeros(len(env.vehicles))
                    if len(env.V2V_Rate_latency_nz) == 0:
                        V2V_latency_list.append(0)
                        L_per_episodeV2V = 0
                    else:
                        V2V_latency_list.append(averagex(env.V2V_Rate_latency_nz))
                        L_per_episodeV2V = averagex(env.V2V_Rate_latency_nz)

                    if len(env.V2I_Rate_latency_nz) == 0:
                        V2I_latency_list.append(0)
                        L_per_episode = 0
                    else:
                        V2I_latency_list.append(averagex(env.V2I_Rate_latency_nz))
                        L_per_episode = averagex(env.V2I_Rate_latency_nz)

                    if len(env.V2S_Rate_latency_nz) == 0:
                        V2S_latency_list.append(0)
                        L_per_episodeV2S = 0
                    else:
                        V2S_latency_list.append(averagex(env.V2S_Rate_latency_nz))
                        L_per_episodeV2S = averagex(env.V2S_Rate_latency_nz)

                if test_step % var.max_latency == 0:
                    k = 0
                    for i in range(len(V2V_rate_per_episode)):
                        if env.demand[i] < env.demand_size and env.demand[i] != 0:
                            V2V_rate_per_episode[i] = 0

                    V2V_D_Rate = np.sum(V2V_rate_per_episode) / var.max_latency
                    V2I_D_Rate = np.sum(V2I_rate_per_episode) / var.max_latency
                    V2S_D_Rate = np.sum(V2S_rate_per_episode) / var.max_latency

                    urban_D_Rate = np.sum(urban_rate_per_episode) / var.max_latency
                    suburban_D_Rate = np.sum(suburban_rate_per_episode) / var.max_latency
                    rural_D_Rate = np.sum(rural_rate_per_episode) / var.max_latency

                    urban_SNR_per_episode_V2V = averagex(urban_SNR_per_episode_V2V)
                    suburban_SNR_per_episode_V2V = averagex(suburban_SNR_per_episode_V2V)
                    rural_SNR_per_episode_V2V = averagex(rural_SNR_per_episode_V2V)
                    urban_SNR_per_episode_V2I = averagex(urban_SNR_per_episode_V2I)
                    suburban_SNR_per_episode_V2I = averagex(suburban_SNR_per_episode_V2I)
                    rural_SNR_per_episode_V2I = averagex(rural_SNR_per_episode_V2I)
                    urban_SNR_per_episode_V2S = averagex(urban_SNR_per_episode_V2S)
                    suburban_SNR_per_episode_V2S = averagex(suburban_SNR_per_episode_V2S)
                    rural_SNR_per_episode_V2S = averagex(rural_SNR_per_episode_V2S)

                    V2I_rate_per_episode_i.append(averagex(V2I_D_Rate))
                    V2V_rate_per_episode_i.append(averagex(V2V_D_Rate))
                    V2S_rate_per_episode_i.append(averagex(V2S_D_Rate))
                    urban_rate_per_episode_i.append(averagex(urban_D_Rate))
                    suburban_rate_per_episode_i.append(averagex(suburban_D_Rate))
                    rural_rate_per_episode_i.append(averagex(rural_D_Rate))
                    urban_SNR_per_episode_V2V_i.append(np.average(urban_SNR_per_episode_V2V))
                    suburban_SNR_per_episode_V2V_i.append(np.average(suburban_SNR_per_episode_V2V))
                    rural_SNR_per_episode_V2V_i.append(np.average(rural_SNR_per_episode_V2V))
                    urban_SNR_per_episode_V2I_i.append(np.average(urban_SNR_per_episode_V2I))
                    suburban_SNR_per_episode_V2I_i.append(np.average(suburban_SNR_per_episode_V2I))
                    rural_SNR_per_episode_V2I_i.append(np.average(rural_SNR_per_episode_V2I))
                    urban_SNR_per_episode_V2S_i.append(np.average(urban_SNR_per_episode_V2S))
                    suburban_SNR_per_episode_V2S_i.append(np.average(suburban_SNR_per_episode_V2S))
                    rural_SNR_per_episode_V2S_i.append(np.average(rural_SNR_per_episode_V2S))
                    Urban_success_list_per_episode.append(Urban_success_list)
                    suburban_success_list_per_episode.append(suburban_success_list)
                    rural_success_list_per_episode.append(rural_success_list)
                    U_per_episode_i.append(averagex(U_per_episode))
                    V2S_U_per_episode_i.append(averagex(V2S_U_per_episode))
                    S_per_episode_v2i.append(S_per_episode)

                    Switch_per_episode.append(env.switch_ratio)
                    v2v_selection_per_episode.append(env.v2v_selection_all)
                    v2i_selection_per_episode.append(env.v2i_selection_all)
                    V2S_selection_per_episode.append(env.v2s_selection_all)
                    dump_selection_per_episode.append(env.dump_selection_all)
                    L_per_episode_v2i.append(L_per_episode)
                    D_per_episode_v2i.append(D_per_episode)
                    S_per_episode_v2v.append(S_per_episodeV2V)
                    L_per_episode_v2v.append(L_per_episodeV2V)
                    D_per_episode_v2v.append(D_per_episodeV2V)
                    S_per_episode_V2S.append(S_per_episodeV2S)
                    L_per_episode_V2S.append(L_per_episodeV2S)
                    D_per_episode_V2S.append(D_per_episodeV2S)
                    V2I_rate_per_episode = np.zeros((len(env.vehicles)))
                    V2V_rate_per_episode = np.zeros((len(env.vehicles)))
                    V2S_rate_per_episode = np.zeros((len(env.vehicles)))
                    urban_rate_per_episode = np.zeros((len(env.vehicles)))
                    suburban_rate_per_episode = np.zeros((len(env.vehicles)))
                    rural_rate_per_episode = np.zeros((len(env.vehicles)))

                    urban_SNR_per_episode_V2V = []
                    suburban_SNR_per_episode_V2V = []
                    rural_SNR_per_episode_V2V = []
                    urban_SNR_per_episode_V2I = []
                    suburban_SNR_per_episode_V2I = []
                    rural_SNR_per_episode_V2I = []
                    urban_SNR_per_episode_V2S = []
                    suburban_SNR_per_episode_V2S = []
                    rural_SNR_per_episode_V2S = []

                    Urban_success_list = 0
                    suburban_success_list = 0
                    rural_success_list = 0
                    S_per_episode = 0
                    D_per_episode = 0
                    L_per_episode = 0
                    U_per_episode = []
                    V2S_U_per_episode = []
                    env.switch_ratio = 0
                    S_per_episodeV2V = 0
                    D_per_episodeV2V = 0
                    L_per_episodeV2V = 0
                    S_per_episodeV2S = 0
                    D_per_episodeV2S = 0
                    L_per_episodeV2S = 0
                    env.v2v_selection_all = 0
                    env.v2i_selection_all = 0
                    env.v2s_selection_all = 0
                    env.dump_selection_all = 0

                    env.demand = env.demand_size * np.ones((len(env.vehicles), env.n_neighbor))
                    env.demand_all = env.demand_size_all * np.ones((len(env.vehicles), env.n_neighbor))
                    env.demand_s = env.demand_s_size * np.ones((len(env.vehicles), env.n_neighbor))

                    env.individual_time_limit = env.time_slow * np.ones((len(env.vehicles), env.n_neighbor))
                    env.individual_time_limit_all = env.time_slow * np.ones((len(env.vehicles), env.n_neighbor))
                    env.active_links = np.ones((len(env.vehicles), env.n_neighbor), dtype='bool')

                    env.active_links_all = np.ones((len(env.vehicles), env.n_neighbor), dtype='bool')
                    env.V2V_Rate_latency = np.zeros(len(env.vehicles))
                    env.V2I_Rate_latency = np.zeros(len(env.vehicles))
                    env.V2S_Rate_latency = np.zeros(len(env.vehicles))
                    env.V2I_Rate_latency_nz = np.zeros(len(env.vehicles))
                    env.V2V_Rate_latency_nz = np.zeros(len(env.vehicles))
                    env.V2S_Rate_latency_nz = np.zeros(len(env.vehicles))

            # 到這裡一個 episode 的所有 step 執行完畢

            Statis = Statistics(test_step, env.rsu_nodes_list)

            while not Queue.isEmpty() :
                block_event = Queue.get_block_event()

                if block_event.type == "create_block":
                    if block_event.time >= (global_clock) and block_event.time < (global_clock+mgp):
                        if len(LT.pool) > 82:
                            LT.calculateGas()
                        if len(LT.pool_with_gas) > 0: # 如果交易池內還有交易可以包才包區塊
                            remain_trans = BlockCommit.generate_block(block_event, env.rsu_nodes_list, Statis)
                            print(f"Block ID: {block_event.block.id}, TX: {len(block_event.block.transactions)}")
                            print(f"remain_trans: {remain_trans}")
                            if remain_trans is not None: # 如果該區塊事件非有效區塊，就不會有回傳值(remain_trans)
                                trans_perform_his[block_event.block.id] = [None, None, None, None, None, None]
                                trans_perform_his[block_event.block.id][0] = len(block_event.block.transactions)
                                trans_perform_his[block_event.block.id][1] = block_event.time - block_event.block.start_time
                                trans_perform_his[block_event.block.id][2] = block_event.time - block_event.block.start_time
                                trans_perform_his[block_event.block.id][3] = \
                                    trans_perform_his[block_event.block.id][0] / trans_perform_his[block_event.block.id][2]
                                trans_perform_his[block_event.block.id][4] = block_event.time
                                trans_perform_his[block_event.block.id][5] = len(env.vehicles)
                                print(f"create_block")
                                print(f"Block ID: {block_event.block.id}, block_event.time: {block_event.time}")
                        Queue.remove_event(block_event)
                    else: # global_clock 還沒走到 block_event.time
                        if env.sumo_step == var.stop_step-mgp: # 但已經是最後一個step了(block_event.time超過設定的模擬時間)
                            Queue.remove_event(block_event) # 就刪除這個 block_event (create_block)
                            continue # enter again the while loop
                        else: # 還沒走到最後一個step，該block_event保留於event_list
                            break # leave the while loop

                elif block_event.type == "receive_block":
                    if block_event.time >= (global_clock) and block_event.time < (global_clock+mgp):
                        BlockCommit.receive_block(block_event, env.rsu_nodes_list)
                        trans_perform_his[block_event.block.id][2] = block_event.time - block_event.block.start_time
                        trans_perform_his[block_event.block.id][3] = \
                            trans_perform_his[block_event.block.id][0] / trans_perform_his[block_event.block.id][2]
                        trans_perform_his[block_event.block.id][4] = block_event.time
                        # print(f"receive_block")
                        # print(f"Block ID: {block_event.block.id}, block_event.time: {block_event.time}")
                        # trans_perform_his[block_event.block.id][4] = len(env.vehicles)
                        Queue.remove_event(block_event)
                    else: # global_clock 還沒走到 block_event.time
                        if env.sumo_step == var.stop_step-mgp: # 但已經是最後一個step了(block_event.time超過設定的模擬時間)
                            Queue.remove_event(block_event) # 就刪除這個 block_event (create_block)
                            continue # enter again the while loop
                        else: # 還沒走到最後一個step，該block_event保留於event_list
                            break # leave the while loop

            '''
            Consens = Consensus()
            Consens.fork_resolution(env.rsu_nodes_list)
            Statis.calculate(test_step, env.rsu_nodes_list, Consens)
            '''

            global_clock += mgp
            global_clock = round(global_clock, 1)


            print(f"***************************************************")
            print(f"global_clock: {global_clock}")

            print(f"Number of Transactions in pool: {len(LT.pool)}")
            print(f"Number of Transactions in pool_with_gas: {len(LT.pool_with_gas)}")
            print(f"Number of Transactions in v2v_tx_pool: {sum(len(tx_list) for tx_list in LT.v2v_tx_pool.values())}")
            print(f"Number of Transactions in v2s_tx_pool: {sum(len(tx_list) for tx_list in LT.v2s_tx_pool.values())}")
            print(f"Number of Vehicles: {len(env.vehicles)}")

            print(f"TX Success Rate: {(success_tx/len(env.vehicles)*100)}%")

            tx_number_his[f"Episode{idx_episode}"][0] = success_tx
            tx_number_his[f"Episode{idx_episode}"][1] = len(env.vehicles)
            tx_number_his[f"Episode{idx_episode}"][2] = success_tx/len(env.vehicles)*100

            # compute all statistics information, print them, and add them into result listresult
            if 1:
                urban_rate_per_episode_i = averagex(urban_rate_per_episode_i)
                suburban_rate_per_episode_i = averagex(suburban_rate_per_episode_i)
                rural_rate_per_episode_i = averagex(rural_rate_per_episode_i)
                urban_SNR_per_episode_V2V_i = averagex(urban_SNR_per_episode_V2V_i)
                suburban_SNR_per_episode_V2V_i = averagex(suburban_SNR_per_episode_V2V_i)
                rural_SNR_per_episode_V2V_i = averagex(rural_SNR_per_episode_V2V_i)
                urban_SNR_per_episode_V2I_i = averagex(urban_SNR_per_episode_V2I_i)
                suburban_SNR_per_episode_V2I_i = averagex(suburban_SNR_per_episode_V2I_i)
                rural_SNR_per_episode_V2I_i = averagex(rural_SNR_per_episode_V2I_i)
                urban_SNR_per_episode_V2S_i = averagex(urban_SNR_per_episode_V2S_i)
                suburban_SNR_per_episode_V2S_i = averagex(suburban_SNR_per_episode_V2S_i)
                rural_SNR_per_episode_V2S_i = averagex(rural_SNR_per_episode_V2S_i)

                V2I_D_Rate = averagex(V2I_rate_per_episode_i)
                V2S_D_Rate = averagex(V2S_rate_per_episode_i)
                V2V_D_Rate = averagex(V2V_rate_per_episode_i)

                print("eps", idx_episode, "Veh", len(env.vehicles), "From",
                    len(env.vehicles) + len(env.neighbor_vehicles),
                    "neighbors vehicles", len(env.neighbor_vehicles), "Tested", env.largev,
                    '----- Time idx_episode', idx_episode, '-----')
                print('----------------------')
                print('Data_rate_V2I', round(averagex(V2I_D_Rate), 2),"Mbps")
                print('Data_rate_V2V', round(averagex(V2V_D_Rate), 2),"Mbps")
                print('Data_rate_V2S', round(averagex(V2S_D_Rate), 2),"Mbps")
                print('Utilization', round(averagex(U_per_episode_i), 2),"%")
                print('Utilization_V2S', round(averagex(V2S_U_per_episode_i), 2),"%")
                print('Success Rate V2I', round(np.average(S_per_episode_v2i), 3))
                print('Success Rate V2V', round(np.average(S_per_episode_v2v), 3))
                print('Success Rate V2S', round(np.average(S_per_episode_V2S), 3))
                print('LatencyV2I', round(sum(L_per_episode_v2i), 3),"ms")
                print('LatencyV2V', round(sum(L_per_episode_v2v), 3),"ms")
                print('LatencyV2S', round(sum(L_per_episode_V2S) + 12, 3),"ms")
                print('Loss:', round(averagex(D_per_episode_v2i), 3) * 0.0002,"Mbps")
                print('Loss V2V:', round(averagex(D_per_episode_v2v), 3) * 0.0002,"Mbps")
                print('Loss V2S:', round(averagex(D_per_episode_V2S), 3) * 0.0002,"Mbps")
                print('Switch Ratio:', round(averagex(Switch_per_episode), 3),"%")
                print('V2V selection Ratio:', round(averagex(v2v_selection_per_episode), 3),"%")
                print('V2I selection Ratio:', round(averagex(v2i_selection_per_episode), 3),"%")
                print('V2S selection Ratio:', round(averagex(V2S_selection_per_episode), 3),"%")
                print('Dump selection Ratio:', round(averagex(dump_selection_per_episode), 3),"%")
                print('----------------------')

                listresult_type = [env.sumo_step, len(env.vehicles), round(np.average(Urban_success_list_per_episode), 3),
                                round(np.average(suburban_success_list_per_episode), 3),
                                round(np.average(rural_success_list_per_episode), 3),
                                round(averagex(urban_rate_per_episode_i), 3),
                                round(averagex(suburban_rate_per_episode_i), 3),
                                round(averagex(rural_rate_per_episode_i), 3),
                                round(averagex(urban_SNR_per_episode_V2V_i), 3),
                                round(averagex(suburban_SNR_per_episode_V2V_i), 3),
                                round(averagex(rural_SNR_per_episode_V2V_i), 3),
                                round(averagex(urban_SNR_per_episode_V2I_i), 3),
                                round(averagex(suburban_SNR_per_episode_V2I_i), 3),
                                round(averagex(rural_SNR_per_episode_V2I_i), 3),
                                round(averagex(urban_SNR_per_episode_V2S_i), 3),
                                round(averagex(suburban_SNR_per_episode_V2S_i), 3),
                                round(averagex(rural_SNR_per_episode_V2S_i), 3)]

                listresult = [env.sumo_step, len(env.vehicles), env.demand_s_size,
                            round(V2I_D_Rate, 3),
                            round(V2V_D_Rate, 3),
                            round(V2S_D_Rate, 3),
                            (round(V2I_D_Rate, 3) + round(V2S_D_Rate, 3) + round(V2V_D_Rate,
                                                                                3)),
                            round(averagex(U_per_episode_i), 3),
                            round(averagex(V2S_U_per_episode_i), 3),
                            round(np.average(S_per_episode_v2i), 3), # V2I_success_rate
                            round(averagex(D_per_episode_v2i), 3), # LossV2I
                            round(np.average(S_per_episode_v2v), 3), # V2V_success_rate
                            round(sum(L_per_episode_v2i), 3), # V2I_Latency
                            round(np.average(S_per_episode_V2S), 3), # V2S_success_rate
                            round(sum(L_per_episode_V2S), 3) + var.propagation_delay, # V2S_Latency
                            round(averagex(D_per_episode_V2S), 3), # LossV2S
                            round(sum(L_per_episode_v2v), 3), # V2V_Latency
                            round(averagex(D_per_episode_v2v), 3), # LossV2V
                            round(averagex(Switch_per_episode), 3), # Switch_ratio
                            round(averagex(v2v_selection_per_episode), 3),
                            round(averagex(v2i_selection_per_episode), 3),
                            round(averagex(V2S_selection_per_episode), 3),
                            round(averagex(dump_selection_per_episode), 3)]

                all_resultsnew.append(listresult)
                all_resultsnew_type.append(listresult_type)
        # 到這裡所有 episode 執行完畢

        writer_veh_selection.close()

        column_name_tx_number = ['# Transactions', '# Vehicles', 'TX Success Rate']
        df_tx_number_his = pd.DataFrame.from_dict(tx_number_his, orient = "index", columns = column_name_tx_number)
        df_tx_number_his.to_excel(writer_tx_number, sheet_name = "TX_Num_His")
        writer_tx_number.close()

        print(f"**********{label}**********")

        tx_in_pool = len(LT.pool)
        tx_in_pool_with_gas = len(LT.pool_with_gas)
        tx_in_v2v_pool = sum(len(tx_list) for tx_list in LT.v2v_tx_pool.values())
        tx_in_v2s_pool = sum(len(tx_list) for tx_list in LT.v2s_tx_pool.values())
        print(f"Number of Transactions in pool: {len(LT.pool)}")
        print(f"Number of Transactions in pool_with_gas: {len(LT.pool_with_gas)}")
        print(f"Number of Transactions in v2v_tx_pool: {sum(len(tx_list) for tx_list in LT.v2v_tx_pool.values())}")
        print(f"Number of Transactions in v2s_tx_pool: {sum(len(tx_list) for tx_list in LT.v2s_tx_pool.values())}")


        # 統計每個區塊的performance
        column_name_trans_perform = ['# Transactions', 'PoW Time', 'Block Latency', 'Block Throughput', 'Finish Timestamp', '# Vehicles']
        df_trans_perform_his = pd.DataFrame.from_dict(trans_perform_his, orient = "index", columns = column_name_trans_perform)
        df_trans_perform_his.to_excel(writer_trans_perform, sheet_name = "Transactions(Block)_Performance")
        writer_trans_perform.close()

        # 計算上鏈的交易總量
        num_tx_sys = 0
        for block_id in trans_perform_his.keys():
            num_tx_sys += trans_perform_his[block_id][0]
        print(f"# Transactions on the Blockchain: {num_tx_sys}")

        m_to_tx = tx_in_pool + tx_in_pool_with_gas + tx_in_v2v_pool + tx_in_v2s_pool + num_tx_sys
        print(f"# Message to TX: {m_to_tx}")

        # 計算最長鏈的交易總量
        max_localchain = 0
        max_localchain_tx = 0
        for node in env.rsu_nodes_list:
            if len(node.localBlockchain) > max_localchain:
                max_localchain = len(node.localBlockchain)
                max_localchain_tx = sum([len(block.transactions) for block in node.localBlockchain])

        # 統計模擬結束時，系統的performance
        system_perform_ep = {} # 存系統總交易數量、總交易時間、交易Throughput
        system_perform_ep[f"System Performance"] = [None, None, None, None]
        system_perform_ep[f"System Performance"][0] = max_localchain_tx # 紀錄最長鏈的交易總數
        system_perform_ep[f"System Performance"][1] = global_clock
        system_perform_ep[f"System Performance"][2] = \
            round(system_perform_ep[f"System Performance"][0] / system_perform_ep[f"System Performance"][1], 3)
        system_perform_ep[f"System Performance"][3] = len(env.vehicles)

        episode_key = f"System Performance"
        print(f"Sum of Transactions on Longest-Blockchain: {system_perform_ep[episode_key][0]}")
        print(f"Simulation Time: {system_perform_ep[episode_key][1]}")
        print(f"System Transactions Throughput: {system_perform_ep[episode_key][2]}")

        print(f"Retransmission Success Rate: {np.mean([tx_number_his[key][2] for key in tx_number_his.keys()])}")
        # print(f"Number of Vehicles: {system_perform_ep[episode_key][3]}")

        column_name_system_perform = ['# Transactions', 'Sys_Latency', 'Sys_Throughput', '# Vehicles']
        df_system_perform_ep = pd.DataFrame.from_dict(system_perform_ep, orient = "index", columns = column_name_system_perform)
        df_system_perform_ep.to_excel(writer_system_perform, sheet_name = 'System_Performance')
        writer_system_perform.close()

        resultsappend(all_resultsnew, all_resultsnew_type) # 呼叫 resultsappend 將每個 item 計算結果加入.CSV

    try:
        env.restor == False
        # print("len",len(sesses))
        test_episodic()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    import random
    import sys
    print(f"set random seed {sys.argv[1]}\nSet method {sys.argv[2]}\nSet dataset {sys.argv[3]}")
    seed = int(sys.argv[1])
    random.seed(seed)
    np.random.seed(seed)
    main()
