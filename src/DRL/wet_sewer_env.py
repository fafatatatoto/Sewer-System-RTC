import pandas as pd
import numpy as np
from datetime import date, time, datetime
from pyswmm import Simulation, Nodes, Links, SystemStats
from swmm_api.input_file import section_labels as sections
from swmm_api import SwmmInput
import gymnasium as gym
from gymnasium import spaces
from dateutil.relativedelta import relativedelta
import os
from collections import defaultdict
from scipy.spatial import KDTree
from inp_generator import inp_generate
import random


class Sewer_Env(gym.Env):
    def __init__(self, 
                 train_fct_s_times,
                 train_fct_s_times_eval, 
                 val_fct_s_times,
                 inp_file = r"C:\Users\xul51\cleanrl\envs\history_h6f18_20230630_1200_0.55_modA_D_loquan_20240820.inp",
                 real_file = r"C:\Users\xul51\cleanrl\envs\Realtime_data_202301_202408.csv",
                 io_file = r"C:\Users\xul51\cleanrl\envs\IOtable.xlsx",
                 base_Hlv = 3, 
                 base_Llv = 2.5,
                 ):
        super(Sewer_Env, self).__init__() 
        # read files
        self.input_file = inp_file
        self.real_file = real_file
        self.io_file = io_file
        self.train_fct_s_times = train_fct_s_times
        self.fct_s_times = train_fct_s_times + val_fct_s_times
        self.total_events_len = len(self.fct_s_times)
        self.evaluate_events = train_fct_s_times_eval + val_fct_s_times
        self.evaluate_events_len = len(self.evaluate_events)
        self.train_sample_len = int(len(train_fct_s_times) * 1)
        self.events_to_run_len = self.train_sample_len
        # self.len_sampled_events = len(train_fct_s_times) * 2/3
        # self.num_of_event = len(train_fct_s_times)
        # self.total_train_events = len(train_fct_s_times)
        # self.total_events = len(fct_s_times)
        self.which_event = 0
        generator = inp_generate(self.input_file, self.real_file, self.io_file)
        self.save_path = r'C:\Users\xul51\cleanrl\envs'
        self.real_data = pd.read_csv(self.real_file, encoding = 'big5', index_col=0) 
        self.real_data.index = pd.to_datetime(self.real_data.index) 
        target_df  = pd.read_excel(self.io_file, sheet_name='list')
        self.INTERCEPTOR_ORIFICE_list = target_df['INTERCEPTOR_ORIFICE'].dropna().astype(str).tolist() 
        self.log_df_all_structure = {f: None for f  in self.fct_s_times}
        self.wlv_log_history = {f:[] for f in self.fct_s_times} 
        self.basic_flow = {f: None for f  in self.fct_s_times}
        self.evaluate = False
        self.control_time_step = 600  # control time step in seconds
        self.low_ratio = base_Llv / base_Hlv


        try:
            self.R17 = ['北投國小','陽明高中','太平國小','雙園','博嘉國小','中正國中','市政中心','留公國中',             
                '桃源國中','奇岩','建國','民生國中','長安國小','台灣大學(新)','玉成','內湖','東湖國小']
            self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','A0A010','T15','C0A9F0','T014']
            self.R17_df = self.real_data.loc[:, self.R17]
            self.R17_mean = self.real_data.loc[:, self.R17].mean(axis=1)

        except:
            self.R17 = ['北投國小','陽明高中','太平國小','雙園','博嘉國小','中正國中','市政中心','瑠公國中',             
                '桃源國中','奇岩','建國','民生國中','長安國小','台灣大學(新)','玉成','內湖','東湖國小']
        
            self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','T15','T014']
            self.R17_df = self.real_data.loc[:, self.R17]
            self.R17_mean = self.real_data.loc[:, self.R17].mean(axis=1)
            
        for fct_s_time in self.fct_s_times:
            # inp_file_path = os.path.join(self.save_path, f"rain_h6f25_{fct_s_time.strftime('%Y%m%d_%H%M')}.inp")
            # hsf_file_path = os.path.join(self.save_path, f"hsf_{fct_s_time.strftime('%m%d%H%M')}.HSF")
            # if os.path.exists(inp_file_path) and os.path.exists(hsf_file_path):
            #     pass
            # else:
            generator.out_inp(fct_s_time, self.save_path, base_Hlv)
            inp_path = os.path.join(self.save_path, f"rain_h6f25_{fct_s_time.strftime('%Y%m%d_%H%M')}.inp")
            self.sim = Simulation(inp_path)  # read input file
            self.inp = SwmmInput.read_file(inp_path ,encoding='big5')
            
            self.basic_time_set(fct_s_time)

            # rainfall_df setup
            rainfall_stat = ["北投國小", "陽明高中","太平國小","雙園","博嘉國小","中正國中","市政中心","桃源國中","奇岩","建國",\
                            "民生國中","長安國小","玉成","內湖","東湖國小"]
            set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index)*2, freq='10min')
            self.rainfall_df = self.real_data.loc[set_rain_index, rainfall_stat]
            # self.log_df_structure = pd.DataFrame(columns = ['DH_tank_head', 'diffuser_head', 'current_pump_open_num', 'ori_open', 'total_flow', 'indicator',\
            #                                     'action1',  'remark', 'pump_target_open', 'p1_t','p2_t','p3_t','p4_t','p5_t',\
            #                                         'p6_t','p7_t','p8_t','p9_t', 'o1_t', 'o2_t','cd',\
            #                                         'reward_level', 'reward_pump', 'reward_ori', 'total_reward', 'penalty',  'state' ],\
            #                         index = self.hsf_fct_time_index)
            self.log_df_all_structure[fct_s_time] = pd.DataFrame(columns = ['DH_tank_head', 'diffuser_head', 'current_pump_open_num', 'ori_open', 'total_flow', 'indicator',\
                                                'action1', 'action2', 'action3', 'probs_1', 'remark', 'pump_target_open', 'p1_t','p2_t','p3_t','p4_t','p5_t',\
                                                    'p6_t','p7_t','p8_t','p9_t', 'o1_t', 'o2_t','cd', 'open_cd', 'close_cd', 'overflow', 'basic_flow', 'rain_mode',\
                                                    'reward_level', 'reward_pump', 'reward_ori', 'total_reward', 'penalty',  'state' ],\
                                    index = self.hsf_fct_time_index)
            self.log_df_all = self.log_df_all_structure.copy()
            #calculate basic flow
            self.basic_flow[fct_s_time] = self.basic_flow_cal()
            
        
            # basic init
            self.sim.step_advance(self.control_time_step)  # set control time step
            # self.sys_stats = SystemStats(self.sim)

            # self.indicator_log_df = pd.DataFrame(columns=['indicator'], index=self.hsf_fct_time_index)
        

            # init node object 
            self.node_object = Nodes(self.sim)  
            self.DH_TANK = self.node_object["DIHWA"]
            self.node3850_0313S = self.node_object['3850-0313S']
            # self.EMG_IN = node_object["EMG_IN"]
            # self.EMG_OUT = node_object["EMG_OUT"]
            # self.GINMEI_TANK = node_object["GINMEI_TANK"]
            # self.SONSHAN_TANK = node_object["SONSHAN_TANK"]
            # self.SULIN_TANK = node_object["SULIN_Outlet"]
            # self.SONSHIN_TANK = node_object["SONSHIN_TANK"]
            # self.SINJAN_TANK = node_object["SINJAN_TANK"]
            # self.ZUNSHAO_TANK = node_object["ZUNSHAO_TANK"]
            

            # init link object
            self.link_object = Links(self.sim)  
            self.PUMP_DH1 = self.link_object["PUMP_DH1"]
            self.PUMP_DH2 = self.link_object["PUMP_DH2"]
            self.PUMP_DH3 = self.link_object["PUMP_DH3"]
            self.PUMP_DH4 = self.link_object["PUMP_DH4"]
            self.PUMP_DH5 = self.link_object["PUMP_DH5"]
            self.PUMP_DH6 = self.link_object["PUMP_DH6"]
            self.PUMP_DH7 = self.link_object["PUMP_DH7"]
            self.PUMP_DH8 = self.link_object["PUMP_DH8"]
            self.PUMP_DH9 = self.link_object["PUMP_DH9"]
            # self.PUMP_EMG_IN = link_object["PUMP_EMG_IN"]
            # self.PUMP_EMG_OUT = link_object["PUMP_EMG_OUT"]
            self.PUMP_GINMEI = self.link_object["PUMP_GINMEI"]
            self.PUMP_SONSHAN = self.link_object["PUMP_SONSHAN"]
            self.PUMP_QUENYAN = self.link_object["PUMP_QUENYAN"]
            self.PUMP_SULIN = self.link_object["PUMP_SULIN"]
            self.PUMP_SONSHIN = self.link_object["PUMP_SONSHIN"]
            self.PUMP_SINJAN = self.link_object["PUMP_SINJAN"]
            self.PUMP_ZUNSHAO = self.link_object["PUMP_ZUNSHAO"]
            self.orifice_3031 = self.link_object["DIHWA_IN_3031"]
            self.orifice_3041 = self.link_object["DIHWA_IN_3041"]
            self.ori_LOQUAN = self.link_object['LOQUAN-Outlet']
            self.ori_HURU = self.link_object['HURU-Outlet']
            self.PUMP_B43_1 = self.link_object['PUMP-B43_OUT1']
            self.PUMP_B43_2 = self.link_object['PUMP-B43_OUT2']
            self.PUMP_B43_3 = self.link_object['PUMP-B43_OUT3']
            self.PUMP_B43_4 = self.link_object['PUMP-B43_OUT4']


            self.pumps = [self.PUMP_DH1, self.PUMP_DH2, self.PUMP_DH3, self.PUMP_DH4, self.PUMP_DH5, self.PUMP_DH6,\
                        self.PUMP_DH7, self.PUMP_DH8, self.PUMP_DH9]
            self.orifices = [self.orifice_3031, self.orifice_3041]
            
        
            # simulation start
            self.sim.start()
            if self.sim.current_time == self.sim.start_time:
                self.PUMP_DH1.target_setting = 0
                self.PUMP_DH2.target_setting = 0
                self.PUMP_DH3.target_setting = 0
                self.PUMP_DH4.target_setting = 0
                self.PUMP_DH5.target_setting = 0
                self.PUMP_DH6.target_setting = 0
                self.PUMP_DH7.target_setting = 0
                self.PUMP_DH8.target_setting = 0
                self.PUMP_DH9.target_setting = 0
                # self.PUMP_EMG_IN.target_setting = 0
                # self.PUMP_EMG_OUT.target_setting = 0
                self.PUMP_GINMEI.target_setting = 0
                self.PUMP_SONSHAN.target_setting = 0
                self.PUMP_QUENYAN.target_setting = 0
                self.PUMP_SULIN.target_setting = 0
                self.PUMP_SONSHIN.target_setting = 0
                self.PUMP_SINJAN.target_setting = 0
                self.PUMP_ZUNSHAO.target_setting = 0
            sim_len = self.sim.end_time - self.sim.start_time
            self.T = int(sim_len.total_seconds()/self.control_time_step)
            self.t = 1
            
            history_weight = 0
            start_modify = False
            first_level = self.real_data.loc[self.time_dict['fct_s_time'], '迪化LT-1濕井液位高度'] - 12.89
            print('first_level', first_level) 

            
            while(True):
                if self.sim.current_time < self.time_dict['fct_s_time']:
                    open_num = (self.real_data.loc[self.sim.current_time + pd.to_timedelta('00:10:00'), [f'迪化抽水機{idx+1}' \
                                                                                for idx in np.arange(9)]] != 0).sum()
                    # lv_interval = np.array([1.09, 1.4, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])
                    lv_interval = np.array([1.09, 1.6, 2.4, 3.0, 3.6, 4.0, 4.4, 4.8, 5.2])
                    for idx in range(open_num):
                        self.pumps[idx].target_setting = self.low_ratio * history_weight
                    index = np.searchsorted(lv_interval, self.DH_TANK.depth) if np.searchsorted(lv_interval, self.DH_TANK.depth) < 9 else 8
                    if index > 0:
                        self.pumps[index].target_setting = (1 - ((lv_interval[index] - self.DH_TANK.depth)/(lv_interval[index] - lv_interval[index-1]) * (1 - self.low_ratio))) * history_weight
                        # self.pump_setting[self.pumps[index]] = self.base_Llv / self.base_Hlv
                        for i in np.arange(0, index):
                            self.pumps[i].target_setting = 1 * history_weight

                    for idx in range(open_num, 9):
                        self.pumps[idx].target_setting = 0 # history close
                    
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p1_t'] = self.PUMP_DH1.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p2_t'] = self.PUMP_DH2.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p3_t'] = self.PUMP_DH3.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p4_t'] = self.PUMP_DH4.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p5_t'] = self.PUMP_DH5.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p6_t'] = self.PUMP_DH6.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p7_t'] = self.PUMP_DH7.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p8_t'] = self.PUMP_DH8.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'p9_t'] = self.PUMP_DH9.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'o1_t'] = self.orifice_3031.target_setting
                    self.log_df_all[fct_s_time].loc[self.sim.current_time, 'o2_t'] = self.orifice_3041.target_setting
                    
                    if self.sim.current_time == self.time_dict['fct_s_time'] - pd.to_timedelta('00:10:00') :
                        history_weight = 1

                    if self.DH_TANK.head >= first_level and start_modify == False:
                        start_modify = True

                    if start_modify:
                        if self.DH_TANK.head >= first_level + 0.2:
                            history_weight = 1.5
                        elif self.DH_TANK.head >= first_level + 0.1:
                            history_weight = 1.2
                        elif self.DH_TANK.head >= first_level:
                            history_weight = 1
                        elif self.DH_TANK.head < first_level - 1:
                            history_weight = 0.5
                        elif self.DH_TANK.head < first_level:
                            history_weight = 0.6
                        
                    

                    # if self.DH_TANK.head > -9.5:
                    #     history_weight = min(history_weight + 0.015, 0.8)
                    # elif -9.5 > self.DH_TANK.head > -10:
                    #     history_weight = min(history_weight + 0.01, 0.8)
                    # elif -10 > self.DH_TANK.head > -11:
                    #     history_weight = min(history_weight + 0.008, 0.8)
                    # elif -11 > self.DH_TANK.head > -11.8:
                    #     history_weight = min(history_weight + 0.005, 0.8)
                    # else:
                    #     pass
                    
                    print('DH tank',self.DH_TANK.head)
                    if self.sim.current_time == (self.time_dict['fct_s_time'] - pd.to_timedelta('00:10:00')):
                        self.sim.save_hotstart(os.path.join(self.save_path, f"hsf_{self.time_dict['fct_s_time'].strftime('%m%d%H%M')}.HSF"))
                            
                    self.sim.__next__()
                    self.wlv_log_history[fct_s_time].append(self.DH_TANK.head)
                else:
                    break
            self.sim.close()

        self.train_sample = random.sample(self.train_fct_s_times, self.train_sample_len)
        self.run_hsf(os.path.join(self.save_path, f"rain_h6f25_{self.train_sample[0].strftime('%Y%m%d_%H%M')}.inp"), self.train_sample[0])
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(len(self.previous_state),), dtype=np.float32)

    def run_hsf(self, inp, fct_s_time):
        if hasattr(self, 'sim'):
            self.sim.close()
        self.basic_time_set(fct_s_time)
        # rainfall_df setup
        rainfall_stat = ["北投國小", "陽明高中","太平國小","雙園","博嘉國小","中正國中","市政中心","桃源國中","奇岩","建國",\
                        "民生國中","長安國小","玉成","內湖","東湖國小"]
        set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index)*2, freq='10min')
        self.rainfall_df = self.real_data.loc[set_rain_index, rainfall_stat]
        self.sim = Simulation(inp)
        self.sim.step_advance(self.control_time_step)
        self.sim.use_hotstart(os.path.join(self.save_path, f"hsf_{fct_s_time.strftime('%m%d%H%M')}.HSF"))
        self.sim.start_time = fct_s_time - pd.to_timedelta('00:10:00')
        # self.sys_stats = SystemStats(self.sim)
        self.log_df_all = self.log_df_all_structure.copy()


        # init node object 
        self.node_object = Nodes(self.sim)  
        self.DH_TANK = self.node_object["DIHWA"]
        self.node3850_0313S = self.node_object['3850-0313S']
        # self.EMG_IN = node_object["EMG_IN"]
        # self.EMG_OUT = node_object["EMG_OUT"]
        # self.GINMEI_TANK = node_object["GINMEI_TANK"]
        # self.SONSHAN_TANK = node_object["SONSHAN_TANK"]
        # self.SULIN_TANK = node_object["SULIN_Outlet"]
        # self.SONSHIN_TANK = node_object["SONSHIN_TANK"]
        # self.SINJAN_TANK = node_object["SINJAN_TANK"]
        # self.ZUNSHAO_TANK = node_object["ZUNSHAO_TANK"]
        

        # init link object
        self.link_object = Links(self.sim)  
        self.PUMP_DH1 = self.link_object["PUMP_DH1"]
        self.PUMP_DH2 = self.link_object["PUMP_DH2"]
        self.PUMP_DH3 = self.link_object["PUMP_DH3"]
        self.PUMP_DH4 = self.link_object["PUMP_DH4"]
        self.PUMP_DH5 = self.link_object["PUMP_DH5"]
        self.PUMP_DH6 = self.link_object["PUMP_DH6"]
        self.PUMP_DH7 = self.link_object["PUMP_DH7"]
        self.PUMP_DH8 = self.link_object["PUMP_DH8"]
        self.PUMP_DH9 = self.link_object["PUMP_DH9"]
        # self.PUMP_EMG_IN = link_object["PUMP_EMG_IN"]
        # self.PUMP_EMG_OUT = link_object["PUMP_EMG_OUT"]
        self.PUMP_GINMEI = self.link_object["PUMP_GINMEI"]
        self.PUMP_SONSHAN = self.link_object["PUMP_SONSHAN"]
        self.PUMP_QUENYAN = self.link_object["PUMP_QUENYAN"]
        self.PUMP_SULIN = self.link_object["PUMP_SULIN"]
        self.PUMP_SONSHIN = self.link_object["PUMP_SONSHIN"]
        self.PUMP_SINJAN = self.link_object["PUMP_SINJAN"]
        self.PUMP_ZUNSHAO = self.link_object["PUMP_ZUNSHAO"]
        self.orifice_3031 = self.link_object["DIHWA_IN_3031"]
        self.orifice_3041 = self.link_object["DIHWA_IN_3041"]
        self.ori_LOQUAN = self.link_object['LOQUAN-Outlet']
        self.ori_HURU = self.link_object['HURU-Outlet']
        self.PUMP_B43_1 = self.link_object['PUMP-B43_OUT1']
        self.PUMP_B43_2 = self.link_object['PUMP-B43_OUT2']
        self.PUMP_B43_3 = self.link_object['PUMP-B43_OUT3']
        self.PUMP_B43_4 = self.link_object['PUMP-B43_OUT4']

        self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','T15','T014']
        self.R17_df = self.real_data.loc[:, self.R17]
        self.R17_mean = self.real_data.loc[:, self.R17].mean(axis=1)

        # self.pump_cd = 0
        self.pump_open_cd = 0
        self.pump_close_cd = 0
        self.pumps = [self.PUMP_DH1, self.PUMP_DH2, self.PUMP_DH3, self.PUMP_DH4, self.PUMP_DH5, self.PUMP_DH6,\
                      self.PUMP_DH7, self.PUMP_DH8, self.PUMP_DH9]
        self.orifices = [self.orifice_3031, self.orifice_3041]

        self.target_open_num = 0
        self.target_open_num_unrevised = 0
        self.open_num_diff = 0
        self.previous_wlv = 0
        self.too_low = False
        self.too_high = False
        self.too_low_ori = False
        self.too_high_ori = False
        self.truncated_log = []
        self.last_pump_ratio = self.low_ratio
        self.wlv_log = self.wlv_log_history[fct_s_time].copy()   
        # grid_shape = [(0.3, 0.5, 0.1), (0.3, 0.8, 0.1), (0.3, 1, 0.1), (0, 1, 0.1), (0, 1, 0.1), (0, 5, 1), (0, 1, 0.2), (0, 1, 0.05)]
        # self.action_penalty = ContinuousActionPenalty(grid_shape, 0.15, 1)

        # simulation start
        self.sim.start()
        # print(self.sim.current_time, self.DH_TANK.head)

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)
        self.t = 1
            

        self.previous_open_num = self.open_num_cal(1)
        self.previous_ori_open_volumn = self.orifice_3031.current_setting + self.orifice_3041.current_setting
        self.previous_indicator = self.indicator_identify()
        self.indicator_trans = self.indicator_transform(self.previous_indicator)
        if self.indicator_trans in (1, 2):
            self.target_level = -10.5
        else:
            self.target_level = -9.5
        self.wlv_log.append(self.DH_TANK.head)
        self.previous_wlv = self.DH_TANK.head
        self.current_basic_flow = self.basic_flow[fct_s_time].loc[self.sim.current_time]
        self.previous_state = self.state_transform()

    def basic_time_set(self, fct_s_time):
        hsf_time_len = 6
        hsf_e_time_len = 1/6
        fct_lead_time_len = 25

        t_time = fct_s_time.replace(minute= int(fct_s_time.minute/10)*10, second=0,microsecond=0)    
        hsf_s_time = fct_s_time - relativedelta(hours=hsf_time_len) - relativedelta(minutes=10) 
        hsf_e_time = fct_s_time - relativedelta(hours=hsf_e_time_len)
        fct_e_time = fct_s_time + relativedelta(hours=fct_lead_time_len)  #- relativedelta(minutes=10) 

        self.time_dict = {'t_time':t_time,
                        'hsf_time_len': hsf_time_len,
                        'fct_lead_time_len': fct_lead_time_len,
                        'hsf_s_time': hsf_s_time,
                        'hsf_e_time': hsf_e_time,
                        'fct_s_time': fct_s_time,
                        'fct_e_time': fct_e_time,
                        'hsf_time_len': hsf_time_len,
                        'hsf_e_time_len': hsf_e_time_len,
                        'fct_lead_time_len': fct_lead_time_len,                 
                        }
        self.hsf_fct_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['fct_e_time'],freq='10min')
        self.hsf_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['hsf_e_time'],freq='10min')
        self.fct_time_index = pd.date_range(self.time_dict['fct_s_time'],self.time_dict['fct_e_time'],freq='10min')


    def step(self, action):
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'action1'] = action[0]
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'action2'] = action[1]
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'action3'] = action[2]
        # if isinstance(action, list):
        # self.log_df_all[self.fct_s_times[self.which_event]].loc[self.sim.current_time, 'probs_1'] = np.round(action[-1], 2)
        self.action1_unregularized = action[0]
        self.action1_regularized = min(action[0] // 0.2, 4)
        action1 = self.action1_regularized
        action2 = action[1] * (1 - self.low_ratio) + self.low_ratio
        action3 = action[2] * 2
        # self.current_state_action = np.concatenate([self.previous_state[[1,2,3,4,7]], np.array([action1, action[1], action[2]])])
        # self.action_penalty.add_action(self.current_state_action)
        self.orifice_3031.target_setting = 1
        self.orifice_3041.target_setting = 1
        self.too_high = False
        self.too_low = False
        self.too_high_ori = False
        self.too_low_ori = False
        open_num = int(max(min(np.ceil(self.previous_open_num), 9), 1))
        if action1 == 0:
            if open_num - 2 < 1:
                self.too_low = True
            self.target_open_num = max(open_num - 2, 1)
            self.target_open_num_unrevised = open_num - 2

        elif action1 == 1:
            if open_num - 1 < 1:
                self.too_low = True
            self.target_open_num = max(open_num - 1, 1)
            self.target_open_num_unrevised = open_num - 1

        elif action1 == 2:
            self.target_open_num = open_num
            self.target_open_num_unrevised = open_num

        elif action1 == 3:
            if open_num + 1 > 9:
                self.too_high = True
            self.target_open_num = min(open_num + 1, 9)
            self.target_open_num_unrevised = open_num + 1

        elif action1 == 4:
            if open_num + 2 > 9:
                self.too_high = True
            self.target_open_num = min(open_num + 2, 9)
            self.target_open_num_unrevised = open_num + 2
            
        self.last_pump_ratio = action2
        pump_weight = self.flow_rate_mod(self.target_open_num)

        for i in range(self.target_open_num):
            self.pumps[i].target_setting = 1 * pump_weight
        for i in range(self.target_open_num, 9):
            self.pumps[i].target_setting = 0
        if 0 < self.target_open_num <= 7: 
            self.pumps[self.target_open_num - 1].target_setting = self.last_pump_ratio * pump_weight
        
        ori_delta = [-1, -0.7, -0.5, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.05, -0.02, 0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1]
        action3_id = int(min(action[2] // (1/len(ori_delta)), len(ori_delta) - 1))
        self.ori_open_volumn = max(min(self.previous_ori_open_volumn + ori_delta[action3_id], 2), 0)
        self.ori_open_volumn_unrevised = self.previous_ori_open_volumn + ori_delta[action3_id]
        if self.ori_open_volumn_unrevised > 2:
            self.too_high_ori = True
        elif self.ori_open_volumn_unrevised < 0:
            self.too_low_ori = True
        self.orifice_3031.target_setting = max(self.ori_open_volumn - 1, 0)
        self.orifice_3041.target_setting = min(self.ori_open_volumn, 1)
        # ori_open_volumn = action3
        # self.orifice_3031.target_setting = max(ori_open_volumn - 1, 0)
        # self.orifice_3041.target_setting = min(ori_open_volumn, 1)


        
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p1_t'] = self.PUMP_DH1.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p2_t'] = self.PUMP_DH2.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p3_t'] = self.PUMP_DH3.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p4_t'] = self.PUMP_DH4.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p5_t'] = self.PUMP_DH5.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p6_t'] = self.PUMP_DH6.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p7_t'] = self.PUMP_DH7.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p8_t'] = self.PUMP_DH8.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'p9_t'] = self.PUMP_DH9.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'o1_t'] = self.orifice_3031.target_setting
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time, 'o2_t'] = self.orifice_3041.target_setting
        
        def interceptor_operation(state):
            if state == 'on':  #截流站關閉
                for inv_name in self.INTERCEPTOR_ORIFICE_list:
                    self.link_object[inv_name].target_setting = 0  #0代表關閉
            elif state == 'off':
                for inv_name in self.INTERCEPTOR_ORIFICE_list:
                    self.link_object[inv_name].target_setting = 1


        def indi_3_measures(state):
            if state == 'on':
                self.PUMP_GINMEI.target_setting = 0
                self.PUMP_SONSHAN.target_setting = 0
                self.PUMP_QUENYAN.target_setting = 0
                
                self.PUMP_SULIN.target_setting = 0 # 士林通常沒開 0代表沒紓流
                self.PUMP_SONSHIN.target_setting = 0 # 松信通常沒開 0代表沒紓流
                self.PUMP_SINJAN.target_setting = 0   # 新建通常沒開 0代表沒紓流
                self.PUMP_ZUNSHAO.target_setting = 0 # 忠孝通常沒開 0代表沒紓流
            elif state == 'off':

                self.PUMP_GINMEI.target_setting = 1
                self.PUMP_SONSHAN.target_setting = 1
                self.PUMP_QUENYAN.target_setting = 1
                self.PUMP_SULIN.target_setting = 0
                self.PUMP_SONSHIN.target_setting = 0
                self.PUMP_SINJAN.target_setting = 0
                self.PUMP_ZUNSHAO.target_setting = 0        

        def indi_4_measures(state):
            if state == 'on':
                self.ori_LOQUAN.target_setting = 0 #六館通常沒開 0代表沒紓流
                self.ori_HURU.target_setting = 0 #葫蘆通常沒開 0代表沒紓流
                self.PUMP_B43_1.target_setting = 1
                self.PUMP_B43_2.target_setting = 1
                self.PUMP_B43_3.target_setting = 1
                self.PUMP_B43_4.target_setting = 1
            elif state == 'off':
                self.ori_LOQUAN.target_setting = 0
                self.ori_HURU.target_setting = 0
                self.PUMP_B43_1.target_setting = 0
                self.PUMP_B43_2.target_setting = 0
                self.PUMP_B43_3.target_setting = 0
                self.PUMP_B43_4.target_setting = 0


        def is_in_2pm_3am(dt):
            weekday = dt.weekday()
            if weekday == 1 and dt.time() >= time(14, 0):
                return True
            elif weekday == 2 and dt.time() <= time(8, 0):
                return True
            else:
                return False
            
        if is_in_2pm_3am(self.sim.current_time) or (self.indicator_df.loc[self.sim.current_time:(self.sim.current_time + relativedelta(minutes=1*60)),'mode'] == '大雨特報紓流模式').any()\
            or (self.previous_indicator in ('2-1', '2-2', '3', '4', '5')):
            interceptor_operation('on')
        else:
            interceptor_operation('off')
        if self.previous_indicator in ('3', '4', '5'):
            indi_3_measures('on')
        else:
            indi_3_measures('off')
        if self.previous_indicator in ('4', '5'):
            indi_4_measures('on')
        else:
            indi_4_measures('off')

        self.sim.__next__()        
        self.open_num_diff = self.open_num_cal(0) - open_num
        # self.diff_combo.append(self.open_num_diff)

        # if self.pump_cd != 0:
        #     self.pump_cd -= 1  
        if self.pump_close_cd != 0:
            self.pump_close_cd -= 1
        if self.pump_open_cd != 0:
            self.pump_open_cd -= 1    
        
        self.current_basic_flow = self.basic_flow[self.train_sample[self.which_event]].loc[self.sim.current_time]
        indicator = self.indicator_identify()
        self.indicator_trans = self.indicator_transform(indicator)
        if self.indicator_trans in (1, 2):
            self.target_level = -10.5
        elif self.indicator_trans == 0:
            self.target_level = -8.8
        else:
            self.target_level = -9.5
        # self.previous_action1 = action

        # self.cal_var()  

        # if self.DH_TANK.head > -8 or self.DH_TANK.head < -11.8:
        #     self.truncated = True
        # else:
        #     self.truncated = False
        # reward calculation
        reward = self.reward_indi_cal(self.previous_indicator)
        # pump cd init
        if self.open_num_diff in (1, 2):
            self.pump_close_cd = 6
        elif self.open_num_diff in (-1, -2):
            self.pump_open_cd = 6
        # if self.open_num_diff != 0:
        #     self.pump_cd = 6
        
        self.state = self.state_transform()

        self.previous_indicator = indicator
        self.previous_wlv = self.DH_TANK.head
        self.previous_open_num = self.open_num_cal(0)
        self.previous_ori_open_volumn = self.ori_open_volumn
        self.previous_state = self.state.copy()
        
        self.wlv_log.append(self.DH_TANK.head)
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'DH_tank_head'] = self.DH_TANK.head
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'diffuser_head'] = self.node3850_0313S.head
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'current_pump_open_num'] = self.open_num_cal(0)
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'ori_open'] = sum(ori.current_setting for ori in self.orifices)
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'total_flow'] = sum(pump.flow for pump in self.pumps)
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'indicator'] = self.indicator_transform(indicator)
        # self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'cd'] = self.pump_cd
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'open_cd'] = self.pump_open_cd
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'close_cd'] = self.pump_close_cd
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'overflow'] = self.overflow + self.DH_TANK.flooding
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'basic_flow'] = self.current_basic_flow
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'rain_mode'] = self.indicator_df.loc[self.sim.current_time - relativedelta(minutes=10),'mode']

        

        if self.t < self.T-1:
            done = False
        else:
            self.which_event += 1 
            done = True
        self.t += 1
        self.done = False

        if done and self.which_event < self.events_to_run_len:
            self.run_hsf(os.path.join(self.save_path, f"rain_h6f25_{self.train_sample[self.which_event].strftime('%Y%m%d_%H%M')}.inp"), self.train_sample[self.which_event])
        elif done and self.which_event == self.events_to_run_len:
            self.done = True

        info = {}
        self.truncated = False
        
        return self.state, reward, self.done, self.truncated, info
        
    def flow_rate_mod(self, current_number):
        Hd = self.DH_TANK.head
        if current_number != 1:
            pump_weight = 1
            if current_number == 2:
                y = 651.07*Hd**3 + 16871*Hd**2 + 147166*Hd + 455791 #R² = 0.5783
                ymax, ymin = 26728/2+200,  15272/2-200
                yone = y /current_number 
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 3:
                y = 199.62*Hd**3 + 4995*Hd**2 + 45596*Hd + 185604   #R² = 0.5948  #3
                ymax, ymin = 40140/3+200,  22727/3-500   
                yone = y /current_number 
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 4 :
                y = -2.6306*Hd**4 + 452.61*Hd**3 + 12527*Hd**2 + 111122*Hd + 377371 #R² = 0.3714
                ymax, ymin = 54923.5/4+200,  31000.9/4-200
                yone = y /current_number   
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 5 :
                y = -2387.8*Hd**2 - 36617*Hd - 78639  #R² = 0.8997 #5
                ymax, ymin = 63869/5+300,  36749/5-300 
                yone = y /current_number  
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 6 :
                y = 674.57*Hd**3 + 16595*Hd**2 + 138328*Hd + 454156 # R² = 0.6979 #6
                ymax, ymin = 70920/6+400,  46442/6 -400   
                yone = y /current_number     
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)               
            elif current_number == 7 :                            
                y = -1920.4*Hd**2 + -28867*Hd + -32398  #R² = 0.6218  #7
                ymax, ymin = 83700/7+500,  59274/7-500 
                yone = y /current_number    
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 8 :
                y = 5852.1*Hd + 132043 +800 #R² = 0.8154 #8
                ymax, ymin = 12000,  8500
                yone = y /current_number
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            elif current_number == 9:
                y = 10426*Hd + 190377  #R² = 0.7931 #9   
                ymax, ymin = 12000,  10141
                yone = y /current_number
                yone = min (ymax,yone)
                yone = max (ymin,yone)
                pump_weight = yone/  (2.8* 3600)   
            
        else:
            pump_weight = 1
        return pump_weight
    
    
    def reset(self, seed=None, options=None):
        if self.evaluate:
            self.which_event = 0
            self.train_sample = self.evaluate_events
            self.run_hsf(os.path.join(self.save_path, f"rain_h6f25_{self.train_sample[0].strftime('%Y%m%d_%H%M')}.inp"), self.train_sample[0])
        else:
            self.which_event = 0
            self.train_sample = random.sample(self.train_fct_s_times, self.train_sample_len)
            self.run_hsf(os.path.join(self.save_path, f"rain_h6f25_{self.train_sample[0].strftime('%Y%m%d_%H%M')}.inp"), self.train_sample[0])

        return self.previous_state, {}
    
    def render(self):
        pass

    def close(self):
        self.sim.report()
        self.sim.close()

    def evaluate_mode(self, switch):
        if switch == 'on':
            self.events_to_run_len = self.evaluate_events_len
            self.evaluate = True

        elif switch == 'off':
            self.events_to_run_len = self.train_sample_len
            self.evaluate = False

    def basic_flow_cal(self):
        inflow_nodes = [name[0] for name in self.inp[sections.INFLOWS].keys() if name[0] not in 
                        ['INTWL_CHUNSAN', 'INTWL_DALON', 'INTWL_LOUQUAN', 'INTWL_SINJAN', 
                        'INTWL_ZUNSHAO', 'INTWL_HUANHE', 'INTWL_YANPING']]

        def DH_distance_cal(name): 
            DH_x, DH_y = self.inp[sections.COORDINATES]['DIHWA'].x, self.inp[sections.COORDINATES]['DIHWA'].y
            target_x, target_y = self.inp[sections.COORDINATES][name].x, self.inp[sections.COORDINATES][name].y
            return np.sqrt((DH_x - target_x)**2 + (DH_y - target_y)**2)

        distance_dict = {name: DH_distance_cal(name) for name in inflow_nodes}
        
        d1_name = [name for name, dist in distance_dict.items() if dist < 4000]
        d2_name = [name for name, dist in distance_dict.items() if 4000 <= dist < 9000]
        d3_name = [name for name, dist in distance_dict.items() if dist >= 9000]

        def get_time_series_sum(names):
            inflow_names = [self.inp[sections.INFLOWS][(name, 'FLOW')]['time_series'] for name in names]
            return sum(np.array(self.inp[sections.TIMESERIES][name]['data'])[:, 1] for name in inflow_names)

        d1_sum = get_time_series_sum(d1_name)
        d2_sum = get_time_series_sum(d2_name)
        d3_sum = get_time_series_sum(d3_name)
        d1 = pd.Series(d1_sum[:-1], index=self.hsf_fct_time_index)
        d2 = pd.Series(d2_sum[:-1], index=self.hsf_fct_time_index).shift(periods=30, freq='T')
        d3 = pd.Series(d3_sum[:-1], index=self.hsf_fct_time_index).shift(periods=60, freq='T')

        basic_flow = d1.add(d2, fill_value=0).add(d3, fill_value=0)
        return basic_flow

    def plot(self, save_path, save_name, probs):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.dates as mdates
        import matplotlib.ticker as ticker
        from matplotlib.ticker import MaxNLocator
        from matplotlib.font_manager import FontProperties
        # print(len(probs), probs)
        probs_list = [probs[i:i + 151] for i in range(0, len(probs), 151)]
        # print(len(probs_list))

        self.save_path_plot = save_path
        self.save_name = save_name
        cm = 1/2.54
        font_title = {'fontname': 'Times New Roman', 'fontsize': 15}
        font_label = {'fontname': 'Times New Roman', 'fontsize': 12}
        font_prop = FontProperties(family='Times New Roman')
        font_prop_lg = FontProperties(family='Times New Roman', size=13)
        hoursLoc = mpl.dates.HourLocator(byhour=[0,12], interval=1, tz=None )
        hoursLoc2 = mpl.dates.HourLocator(byhour=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], interval=1, tz=None )

        for i, fct_s_time in enumerate(self.train_sample):
            fct_e_time = fct_s_time + relativedelta(hours=25)
            fct_time_index = pd.date_range(fct_s_time, fct_e_time,freq='10min')
            fig, axes = plt.subplots(4, 1, figsize=(5*4*1.4*cm, 5*4*1.1*cm), sharex=False,gridspec_kw={'height_ratios': [1,1, 1, 1]})
            # mpl.rc('axes', labelsize=14, titlesize=16)
            # axes_0 = axes[0].twinx()
            # sewer = axes_0.plot(fct_time_index, self.basic_flow[fct_s_time][fct_time_index], color='blue', linewidth=2)
            # axes_0.set_ylabel('inflow rate (CMS)', fontdict=font_label)
            # axes_0.set_ylim(0, 21)
            # axes_0.set_yticks([0, 5, 10, 15, 20])
            # axes_0.set_yticklabels([0, 5, 10, 15, 20], fontproperties=FontProperties(family='Times New Roman'))
            locations_dict = {
                    '桃源國中': 'Taoyuan Junior High School',
                    '北投國小': 'Beitou Elementary School',
                    '陽明高中': 'Yangming High School',
                    '太平國小': 'Taiping Elementary School',
                    '民生國中': 'Minsheng Junior High School',
                    '中正國中': 'Zhongzheng Junior High School',
                    '三興國小': 'Sanxing Elementary School',
                    '格致國中': 'Gezhi Junior High School',
                    '東湖國小': 'Donghu Elementary School',
                    '留公國中': 'Liugong Junior High School',
                    '舊莊國小': 'Jiuzhuang Elementary School',
                    '市政中心': 'Municipal Center',
                    '博嘉國小': 'Bojia Elementary School',
                    '北政國中': 'Beizheng Junior High School',
                    '雙園': 'Shuangyuan',
                    '玉成': 'Yucheng',
                    '建國': 'Jianguo',
                    '福德': 'Fude',
                    '奇岩': 'Qiyan',
                    '中洲': 'Zhongzhou',
                    '磺溪橋': 'Huangxi Bridge',
                    '中和橋': 'Zhonghe Bridge',
                    '白馬山莊': 'Baima Mountain Villa',
                    '望星橋': 'Wangxing Bridge',
                    '宜興橋': 'Yixing Bridge',
                    '長安國小': 'Chang’an Elementary School',
                    '萬華國中': 'Wanhua Junior High School',
                    '永建國小': 'Yongjian Elementary School',
                    '五常國小': 'Wuchang Elementary School',
                    '仁愛國小': 'Ren’ai Elementary School',
                    '興華國小': 'Xinghua Elementary School',
                    '南港高工': 'Nangang Vocational High School',
                    '台灣大學(新)': 'National Taiwan University (New)',
                    '臺北': 'Taipei',
                    '科教館': 'National Taiwan Science Education Center',
                    '天母': 'Tianmu',
                    '汐止': 'Xizhi',
                    '松山': 'Songshan',
                    '石牌': 'Shipai',
                    '關渡': 'Guandu'
                }
            # locations = ['桃源國中', '北投國小', '陽明高中', '太平國小', '民生國中', '中正國中', 
            #     '三興國小', '格致國中', '東湖國小', '留公國中', '舊莊國小', '市政中心', 
            #     '博嘉國小', '北政國中', '雙園', '玉成', '建國', '福德', '奇岩', '中洲', 
            #     '磺溪橋', '中和橋', '白馬山莊', '望星橋', '宜興橋', '長安國小', '萬華國中', 
            #     '永建國小', '五常國小', '仁愛國小', '興華國小', '南港高工', '台灣大學(新)', 
            #     '臺北', '科教館', '天母', '汐止', '松山', '石牌', '關渡']
            max_rain_st = self.real_data.loc[fct_time_index, locations_dict.keys()].sum(axis=0).idxmax()
            rain = axes[0].bar(fct_time_index, self.real_data.loc[fct_time_index, max_rain_st], width=0.005, color='skyblue')
            axes[0].set_title(f'Rainfall ({locations_dict[max_rain_st]})',pad=23,loc='center', fontdict=font_title)
            axes[0].set_ylabel('Rainfall (mm)', fontdict=font_label)
            axes[0].set_ylim(0, 41)
            axes[0].set_yticks([0, 10, 20, 30, 40])
            axes[0].set_yticklabels([0, 10, 20, 30, 40], fontproperties=FontProperties(family='Times New Roman'))
            
            axes[1].axhspan(ymax = -8, ymin=-9,xmin=0 , xmax=1, color='blue', alpha=0.3, linewidth=0)
            axes[1].axhspan(-10.1, -9, color='green', alpha=0.3, linewidth=0)
            axes[1].axhspan(-11.8, -10.1, color='orange', alpha=0.3, linewidth=0)
            Lobs_diff = axes[1].plot(fct_time_index, self.real_data.loc[fct_time_index, '緊急繞流井雷達波液位計'] - 12.39, linewidth=2, label='緊繞液位觀測',color='#0000FF')   #觀測
            Lobs_tkwl = axes[1].plot(fct_time_index, self.real_data.loc[fct_time_index,'迪化LT-1濕井液位高度'] - 12.89, linewidth=2, label='濕井液位觀測',color='#444444') #####000000   #觀測
            Lfct_tkwl = axes[1].plot(fct_time_index, self.log_df_all[fct_s_time].loc[fct_time_index, 'DH_tank_head'], linewidth=2, label='DH tank',color='#FF3333') #color_list[self.strategy]
            Lfct_diff =  axes[1].plot(fct_time_index, self.log_df_all[fct_s_time].loc[fct_time_index, 'diffuser_head'], linewidth=2, label='diffuser',color='#FF8800')
            axes[1].set_title('Water level of wet well and diffuser',pad=23,loc='center', fontdict=font_title)
            axes[1].set_ylabel('water level (m)', fontdict=font_label)
            axes[1].set_ylim(-12.5,0)
            axes[1].set_yticks([-12, -10, -8, -6, -4, -2, 0])
            axes[1].set_yticklabels([-12, -10, -8, -6, -4, -2, 0], fontproperties=FontProperties(family='Times New Roman'))
            
            Lobs_q = axes[2].plot(fct_time_index, self.real_data.loc[fct_time_index, '迪化抽水站總瞬間流量(cmh)']/3600, color='#444444', label='observation', linewidth=2) #觀測
            Lfct_q = axes[2].plot(fct_time_index, self.log_df_all[fct_s_time].loc[fct_time_index, 'total_flow'], label='simulation',color='#FF3333', linewidth=2)
            axes[2].set_title('Pumping rate of Dihwa pumping station', pad=23, loc='center', fontdict=font_title)
            axes[2].set_ylabel('Pumping rate (m³/s)', fontdict=font_label)
            axes[2].set_yticks([0, 5, 10, 15, 20, 25, 30 ,35])
            axes[2].set_yticklabels([0, 5, 10, 15, 20, 25, 30 ,35], fontproperties=FontProperties(family='Times New Roman'))
            # axes[1].set_ylim(0,120000)
            
            Lobs_pump =  axes[3].step(fct_time_index, self.real_data.loc[fct_time_index, ['迪化抽水機1', '迪化抽水機2', '迪化抽水機3', '迪化抽水機4',\
                                              '迪化抽水機5', '迪化抽水機6', '迪化抽水機7', '迪化抽水機8', '迪化抽水機9']].sum(axis=1), where='mid', color='#444444', label='歷史運作台數', linewidth=2)   #觀測
            Lfct_pump = axes[3].step(fct_time_index, self.log_df_all[fct_s_time].loc[fct_time_index, 'current_pump_open_num'], where='mid', color='#FF3333', label='預報運作台數', linewidth=2)
            axes[3].set_title('Number of working pumps',pad=23,loc='center', fontdict=font_title)
            axes[3].set_ylabel('Number', fontdict=font_label)
            axes[3].set_ylim(0,10)
            axes[3].set_yticks([0,3,6,9])
            axes[3].set_yticklabels([0,3,6,9], fontproperties=FontProperties(family='Times New Roman'))
            # date_labels = [date.strftime('%Y-%m-%d') for date in fct_time_index]
            # time_labels = [date.strftime('%H:%M') for date in fct_time_index]
            axes[3].set_xticks(fct_time_index)
            axes[3].xaxis.set_major_locator(hoursLoc)
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H\n%Y/%m/%d'))

            axes[0].xaxis.set_major_locator(hoursLoc)
            axes[0].xaxis.set_minor_locator(hoursLoc2)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H'))        
            axes[0].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            axes[0].tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 12, length=10,width=.4)  
            axes[0].tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 12, length=4,width=.4)   
            for label in axes[0].get_xticklabels(which='both'):
                label.set_fontproperties(font_prop)

            axes[1].xaxis.set_major_locator(hoursLoc)
            axes[1].xaxis.set_minor_locator(hoursLoc2)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))        
            axes[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            axes[1].tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 12, length=10,width=.4)  
            axes[1].tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 12, length=4,width=.4)   
            for label in axes[1].get_xticklabels(which='both'):
                label.set_fontproperties(font_prop)

            axes[2].xaxis.set_major_locator(hoursLoc)
            axes[2].xaxis.set_minor_locator(hoursLoc2)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H'))        
            axes[2].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            axes[2].tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 12, length=10,width=.4)  
            axes[2].tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 12, length=4,width=.4)   
            for label in axes[2].get_xticklabels(which='both'):
                label.set_fontproperties(font_prop)

            axes[3].xaxis.set_minor_locator(hoursLoc2)
            axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))

            axes[3].tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 12, length=10,width=.4)  
            axes[3].tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 12, length=4,width=.4)   
            for label in axes[3].get_xticklabels(which='both'):
                label.set_fontproperties(font_prop)

            L1 = plt.legend(Lobs_tkwl, [ 'Wet well (observation)' ],loc='upper center',  bbox_to_anchor=(.1, 4.75-5.15),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L1); 
            L3 = plt.legend(Lobs_diff, [ 'Diffuser (observation)' ],loc='upper center',  bbox_to_anchor=(.1+0.267, 4.75-5.15),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L3); 
            L2 = plt.legend(Lfct_tkwl, [ 'Wet well (simulation)' ],loc='upper center',  bbox_to_anchor=(.1+0.267*2, 4.75-5.15),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L2); 
            L4 = plt.legend(Lfct_diff, [ 'Diffuser (simulation)' ],loc='upper center',  bbox_to_anchor=(.1+0.267*3, 4.75-5.15),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L4); 
            L5 = plt.legend(Lobs_q, [ 'observation' ],loc='upper center',  bbox_to_anchor=(.3, 3.-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L5); 
            L6 = plt.legend(Lfct_q, [ 'simulation' ],loc='upper center',  bbox_to_anchor=(.7, 3.-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L6); 
            L7 = plt.legend(Lobs_pump, [ 'observation' ],loc='upper center',  bbox_to_anchor=(0.3, 1.29-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L7); 
            L8 = plt.legend(Lfct_pump, [ 'simulation' ],loc='upper center',  bbox_to_anchor=(0.7, 1.29-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            plt.gca().add_artist(L8);  
            # L9 = plt.legend(sewer, [ 'sewer' ],loc='upper center',  bbox_to_anchor=(0.7, 6.43-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            # plt.gca().add_artist(L9); 
            # L10 = plt.legend(rain, [ 'rainfall' ],loc='upper center',  bbox_to_anchor=(0.3, 6.43-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
            # plt.gca().add_artist(L10); 
            
            
            
            plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=.8)
            fig.savefig(os.path.join(self.save_path_plot,f'{self.save_name}_{fct_s_time.strftime("%Y%m%d_%H%M")}_simulation.png'),dpi=300)
            probs_str = [str(text) for text in np.round(probs_list[i], 2).tolist()]
            self.log_df_all[fct_s_time].loc[(fct_s_time - relativedelta(minutes=10)):fct_time_index[-2], 'probs_1'] = probs_str
            self.log_df_all[fct_s_time].to_csv(os.path.join(self.save_path_plot,f'{self.save_name}_{fct_s_time.strftime("%Y%m%d_%H%M")}_violation.csv'))
            plt.close()

            
            pass



    def metric_calculation(self):
        from metric import metric_cal
        Metric = []
        for fct_s_time in self.train_sample:
            fct_e_time = fct_s_time + relativedelta(hours=25)
            fct_time_index = pd.date_range(fct_s_time, fct_e_time,freq='10min')
            DH_water_level = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], 'DH_tank_head']
            DF_water_level = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], 'diffuser_head']
            pump = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], 'current_pump_open_num']
            indicator = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], 'indicator']
            orifice = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], ['o1_t', 'o2_t']].sum(axis=1)
            overflow = self.log_df_all[fct_s_time].loc[fct_time_index[:-1], 'overflow']           
            run_log = {'DH_water_level':DH_water_level, 'DF_water_level':DF_water_level, 'pump':pump, 'indicator':indicator, 'orifice':orifice, 'overflow':overflow, 'time':fct_s_time}
            level_metric, pump_metric, orifice_metric, overflow_metric, electricity_metric, metric = metric_cal(run_log)
            Metric.append([level_metric, pump_metric, orifice_metric, overflow_metric, electricity_metric, metric])
        return Metric
    
    
    def state_transform(self):
        self.rain_log = self.rainfall_data(30)
        def scale_value(value, min_range, max_range):
             return (value - min_range) / (max_range - min_range)

        indi_range = [0, 5]
        DH_tank_range = [-14, -3]
        diffuser_range = [-14, -3]
        p_open_num_range = [0, 9]
        # lp_open_ratio_range = [self.low_ratio, 1]
        # ori_open_range = [0, 1]
        fct_rain_range = [0, 100]
        p_cd_range = [0, 6]
        p_diff_range = [-2, 2]
        R17sum_range = [0, 20]
        basic_flow_range = [0, 20]
        ori_range = [0, 2]
        scaled_indi_value = scale_value(self.indicator_trans, *indi_range)
        scaled_DH_target_value = scale_value(self.target_level, *DH_tank_range)
        scaled_DH_tank_value = scale_value(self.DH_TANK.head, *DH_tank_range)
        scaled_diffuser_value = scale_value(self.node3850_0313S.head, *diffuser_range)
        # scaled_p_open_num_value = scale_value(self.pump_open_num, *p_open_num_range)
        scale_previous_p_open_num_value = scale_value(self.previous_open_num, *p_open_num_range)
        scale_target_p_open_num_value = scale_value(self.target_open_num, *p_open_num_range)
        scale_target_p_open_num_unrevised_value = scale_value(self.target_open_num_unrevised, *p_open_num_range)
        # scaled_lp_open_ratio_value = scale_value(self.last_pump_ratio, *lp_open_ratio_range)
        # scaled_ori_open_value = scale_value(self.ori_open_var, *ori_open_range)
        scaled_R17sum_value = scale_value(self.rain_log, *R17sum_range)
        # scaled_fct_rain_value = scale_value(self.fct_rainfall, *fct_rain_range)
        # scaled_p_cd_value = scale_value(self.pump_cd, *p_cd_range)
        scaled_p_open_cd_value = scale_value(self.pump_open_cd, *p_cd_range)
        scaled_p_close_cd_value = scale_value(self.pump_close_cd, *p_cd_range)
        scaled_p_diff_value = scale_value(self.open_num_diff, *p_diff_range)
        scaled_basic_flow = scale_value(self.current_basic_flow, *basic_flow_range)
        scaled_orifice = scale_value(self.previous_ori_open_volumn, *ori_range)
         
        
        state = np.array([scaled_indi_value,
                          scaled_DH_target_value,
                               scaled_DH_tank_value,
                               scaled_diffuser_value,
                            #    self.grad_wlv,
                            #    self.pump_open_var,
                            #    scaled_p_open_num_value,
                            #    self.previous_action1, 
                               scale_previous_p_open_num_value,
                               scale_target_p_open_num_value,
                               scale_target_p_open_num_unrevised_value,
                            #    scaled_lp_open_ratio_value,
                            #    scaled_ori_open_value,
                            #    scaled_fct_rain_value,
                            #    scaled_p_cd_value,
                               scaled_p_open_cd_value,
                               scaled_p_close_cd_value, 
                               scaled_p_diff_value,
                               scaled_orifice,
                               scaled_basic_flow
                               ] + list(scaled_R17sum_value)
                            ).astype(np.float32)
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'state'] = str(np.round(state, 2))
        
        return state
    
    def rainfall_data(self, minute):
        length = 360 // minute
        rain_log = []
        for i in range(1, length+1):
            rain_log.append(self.R17_mean.loc[(self.sim.current_time - relativedelta(minutes=minute * i)):(self.sim.current_time - relativedelta(minutes=minute * (i - 1)))].sum())

        return np.array(rain_log)
    
                

    def indicator_identify(self):
        def cal_rain(rainfall_df, two_hour_threshold=1.5):
            Rtmpall = rainfall_df.rolling(6).sum()
            Rtmp = Rtmpall.loc[self.hsf_fct_time_index] 
            Ridx = Rtmp.max(axis=1).idxmax()
            Rname = Rtmp.idxmax(axis=1)
            Rnamemax = Rname.loc[Ridx]
            tf_24h80 = (Rtmpall[Rnamemax].rolling(24*6).sum() / 6 >= 80).tolist()
            tf_1h40 = (Rtmpall[Rnamemax] >= 40).tolist()
            tmp3h5 = Rtmpall[Rnamemax] >= 5
            tmp3h5_0 = np.array(tmp3h5)
            tmp3h5_1 = np.array(tmp3h5.shift(1))
            tmp3h5_2 = np.array(tmp3h5.shift(1))
            tf_3h5 = np.logical_and(tmp3h5_0, tmp3h5_1, tmp3h5_2).tolist()
            tf_24h200 = (Rtmpall[Rnamemax].rolling(24*6).sum() / 6 >= 200).tolist()
            tf_3h100 = (Rtmpall[Rnamemax].rolling(3*6).sum() / 6 >= 100).tolist()
            indicator_df = pd.DataFrame(columns=['mode'], index=self.hsf_fct_time_index)
            indicator_df.loc[:, :] = 'sunny'
            tmp_list2 = np.logical_or.reduce([tf_24h80, tf_1h40, tf_3h5], axis=0).tolist()

            if True in tmp_list2:
                True_first2 = tmp_list2.index(True)
                True_last2 = len(tmp_list2) - tmp_list2[::-1].index(True) - 1
                indicator_df.loc[self.hsf_fct_time_index[True_first2:True_last2], 'mode'] = 'heavy rain'

                for i in range(True_first2, True_last2+1):
                    if len(self.hsf_fct_time_index) > i >= 18:
                        two_hour_rain = Rtmpall[Rnamemax][i-18:i].sum()
                        if two_hour_rain < two_hour_threshold:
                            indicator_df.loc[self.hsf_fct_time_index[i], 'mode'] = 'sunny'

            tmp_list = np.logical_or.reduce([tf_24h200, tf_3h100], axis=0).tolist()
            
            if True in tmp_list:
                True_first = tmp_list.index(True)
                True_last = len(tmp_list) - tmp_list[::-1].index(True) - 1
                indicator_df.loc[self.hsf_fct_time_index[True_first:True_last], 'mode'] = 'torrential rain'

            return indicator_df

        def current_pump_open_num():
            return sum(pump.current_setting != 0 for pump in self.pumps)
        
        self.indicator_df = cal_rain(self.rainfall_df)
        current_open_num = current_pump_open_num()
        self.grad_wlv = self.DH_TANK.head - self.previous_wlv
        if len(self.wlv_log) >= 5:
            slope_wlv, _ = np.polyfit(np.arange(5), self.wlv_log[-5:], 1)
            # print(slope_wlv)
        else:
            slope_wlv = 0
        # self.grad_wlv = self.node3850_0313S.head - self.previous_diffuser_wlv
        self.indicator = {indi:None for indi in ['0', '1', '2-1', '2-2', '3', '4', '5']}
        self.indicator['0'] = True
        # print(current_open_num, round(self.DH_TANK.head,2), round(self.grad_wlv,2))
        try:
            if (self.indicator_df.loc[self.sim.current_time:(self.sim.current_time + relativedelta(minutes=6*60)),'mode'] == 'heavy rain').any():
                self.indicator['1'] = True
        except KeyError:
            pass
        if current_open_num >= 5 and self.DH_TANK.head >= -10.1 and slope_wlv > -0.2:
            self.indicator['2-1'] = True
        if self.indicator_df.loc[self.sim.current_time,'mode'] == 'torrential rain':
            self.indicator['2-1'] = True
            self.indicator['2-2'] = True
        if current_open_num >= 6 and self.DH_TANK.head >= -10.1 and slope_wlv > -0.3:
            self.indicator['3'] = True
        if current_open_num >= 7 and self.DH_TANK.head >= -10.1 and slope_wlv > -0.3 and self.indicator['3'] == True:
            self.indicator['4'] = True
        if current_open_num >= 7 and self.DH_TANK.head >= -10.0 and slope_wlv > 0.3 and self.indicator['4'] == True:
            self.indicator['5'] = True
        if self.node3850_0313S.head > -8 and current_open_num >= 6:
            self.indicator['3'] = True
        for indi, tf in reversed(self.indicator.items()):
            if tf == True:
                self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time,'indicator'] = indi
                return indi

    def reward_indi_cal(self, indicator):
        def basic_level_reward(target_level):
            reward_in = 0
            # basic level taboo
            # self.truncated_log.append(self.truncated)
            # if len(self.truncated_log) % 10 == 0:
            #     reward_in += 50000
            
            # if np.abs(self.DH_TANK.head - target_level) < 1:
            #     reward_in += 10000 - np.abs(self.DH_TANK.head - target_level) * 1000
            # elif 1.5 > np.abs(self.DH_TANK.head - target_level) > 1:
            #     reward_in += 10000 - np.abs(self.DH_TANK.head - target_level) * 2000
            # elif 2 > np.abs(self.DH_TANK.head - target_level) > 1.5:
            #     reward_in += 10000 - np.abs(self.DH_TANK.head - target_level) * 4000
            if self.DH_TANK.head < -11.8:
                reward_in -= 30000 * (-11.8 - self.DH_TANK.head) + 5000
                self.remark.append(f'DH_TANK lv too low {(-11.8 - self.DH_TANK.head):.2f} m')
            elif self.DH_TANK.head > -8:
                reward_in -= 20000 * (self.DH_TANK.head - (-8)) + 5000
                self.remark.append(f'DH_TANK lv too high {(self.DH_TANK.head - (-8)):.2f} m')

            if self.node3850_0313S.head < -11.8:
                reward_in -= 30000 * (-11.8 - self.node3850_0313S.head) + 2000
                self.remark.append(f'diffuser lv too low {(-11.8 - self.node3850_0313S.head):.2f} m')
            elif self.node3850_0313S.head > -8 and self.node3850_0313S.head < -6:
                reward_in -= 2000 * (self.node3850_0313S.head - (-8)) + 2000
                self.remark.append(f'diffuser lv too high {(self.node3850_0313S.head - (-8)):.2f} m')
            elif self.node3850_0313S.head > -6:
                reward_in -= 10000 * (self.node3850_0313S.head - (-8)) + 2000
                self.remark.append(f'diffuser lv too high {(self.node3850_0313S.head - (-8)):.2f} m')

            nodes = ['3846-0232', '4149-0041', '3848-0003', '4054-0953', '4047-0079', '3953-0485', '4048-0075', 
                    '4055-0143', '4150-0005', '4250-0008', '3847-0734', '3955-0009', '4250-0106', '3955-0006', 
                    '3954-0232', '4047-0082', '3850-0061', 'North_well', '3954-0229', '3847-0733', '4250-0011', 
                    '4150-0006', '3948-0076', '3955-0145', '3954-0001', '4055-0041', '3755-0003', '4150-0011', 
                    '3954-0234', '3850-0313S', '3953-0412', '3947-0100', '3850-0002', '3955-0003', '3951-0023', 
                    '4149-0865', '3950-0166', '3955-0010', '3850-0309', 'ShiTszTo_well', '3954-0306', '3949-0080', 
                    '4055-0062', '3953-0417', '3954-0231', '3850-0310', '4047-0807', '4148-0029', '3852-0010', 
                    '4054-0014', '3954-0013', '3955-0004', '3848-0475', '3946-0033', '4148-0921', '4046-0856', 
                    '4250-0107', '3848-0476', '3846-0001', '3954-0226', '4147-0723', '4054-0216', '3851-0157', 
                    '3954-0308', '3946-0341', '3851-0076', '3948-0077', '4055-0046', '3850-0312', '3846-0231', 
                    '3947-0099', '3954-0007', '4047-0081', '3951-0474', '3948-0914', '3853-0056', '3850-0001', 
                    '3954-0317', '3849-0002', '3754-0380', '3954-0018', 'B42', '3955-0011', '3754-0382', 
                    '4054-0039', '4050-0042', '4049-0837', '3852-0002', '3847-0001', '3953-0413', '3955-0021', 
                    'South_well', '3947-1082', '3947-0101', '3954-0230', '3954-0028', '3954-0310', '3954-0029', 
                    '3848-0002', '3849-0066', '3852-0001', '3848-0001', '4054-0015', '3846-0003', '4055-0056', 
                    '4148-0027', '3954-0037', '4149-0866', '4150-0008', '4149-0042', '3851-0001', 'TP_SZT', 
                    '3954-0150', '3850-0003', '3954-0009', '3955-0007', '3850-0313', '3849-0001', '3955-0002', 
                    '3954-0227', '3954-0307', '3754-0381', '3849-0003', '4148-0028', '4150-0071', 
                    '3851-0158', '4046-0854', '4149-1162', '3755-0001', '3949-0078', '3954-0309', 'B01', 
                    '4055-0051', '3950-0111', '4054-0016', '4250-0010', '3853-0550', '4149-1131', '3846-0233', 
                    '3955-0152', '3953-0405', '3846-0004', '3846-0046', '4250-0009', '3852-0532', '3955-0001', 
                    '3850-0318', '3953-0053', '4049-0018', '3951-0001', '3755-0002', '3850-0311', '3954-0382', 
                    '3954-0017', '3850-0004', '4250-0032', '3946-0342', '3853-0551', '3853-0054', '3846-0462', 
                    '3949-0079', '3955-0005', '3954-0233', '3955-0151', '3955-0020', '4149-1130', '4047-0080', 
                    '4150-0009', '4055-0042', '4149-0040', '4150-0017', '3954-0228', '3850-0243', '4048-0762', 
                    '3947-1054', '4149-1129', '3850-0247', '3948-0075', '3850-0248', '4054-0445', '4150-0072', 
                    '4049-0019', '4147-0006', '4147-0722', '4150-0003', '4054-0001', '3955-0008', '4150-0004', 
                    'B39', '4046-0855', '4150-0010']
            self.overflow = sum(self.node_object[n].flooding for n in nodes)
            reward_in -= self.overflow * 15000
            if self.DH_TANK.flooding > 0:
                # print('flood',self.DH_TANK.flooding)
                self.remark.append(f'DH flood: {self.DH_TANK.flooding:.2f}')
                reward_in -= self.DH_TANK.flooding * 20000


            # target diff
            weight = 2000
            if self.DH_TANK.head - target_level > 0.4:
                level_diff = self.DH_TANK.head - target_level
                reward_in -= weight * level_diff
                self.remark.append(f'DH_TANK above target {level_diff:.2f} m')    
            elif self.DH_TANK.head - target_level < -0.4:
                level_diff = target_level - self.DH_TANK.head
                reward_in -= weight * level_diff
                self.remark.append(f'DH_TANK below target {level_diff:.2f} m')
            else:
                pass
            if self.node3850_0313S.head - target_level > 0.5:
                level_diff = self.node3850_0313S.head - target_level
                reward_in -= weight * level_diff
                self.remark.append(f'diffuser above target {level_diff:.2f} m')    
            elif self.node3850_0313S.head - target_level < -0.5:
                level_diff = target_level - self.node3850_0313S.head
                reward_in -= weight * level_diff
                self.remark.append(f'diffuser below target {level_diff:.2f} m')
            else:
                pass

            if np.abs(self.grad_wlv) > 0.5:
                reward_in -= np.abs(self.grad_wlv) * 50000
            reward_out = reward_in

            return reward_out

        def pump_reward(indicator):
            reward_in = 0

            if indicator in ('0', '1', '2-1', '2-2'):    
                if np.abs(self.open_num_diff) > 1:
                    reward_in -= 1000 * np.abs(self.open_num_diff)
                    self.remark.append(f'open num violation {self.open_num_diff}')

                if self.open_num_diff != 0:
                    if max(self.pump_close_cd, self.pump_open_cd) == 5:
                        reward_in -= 4000 * 2
                        self.remark.append(f'pump cd violation cd=5')
                    elif max(self.pump_close_cd, self.pump_open_cd) == 4:
                        reward_in -= 3700 * 2
                        self.remark.append(f'pump cd violation cd=4')
                    elif max(self.pump_close_cd, self.pump_open_cd) == 3:
                        reward_in -= 3500 * 2
                        self.remark.append(f'pump cd violation cd=3')
                    elif max(self.pump_close_cd, self.pump_open_cd) == 2:
                        reward_in -= 3200 * 2
                        self.remark.append(f'pump cd violation cd=2')
                    elif max(self.pump_close_cd, self.pump_open_cd) == 1:
                        reward_in -= 3000 * 2
                        self.remark.append(f'pump cd violation cd=1')
                    # if self.pump_cd != 0:
                    #     reward_in -= 15000
                    #     self.remark.append(f'pump cd violation cd:{self.pump_cd}')
                    else:
                        reward_in -= 2000
                        self.remark.append(f'pump open')

                # if self.open_num_diff != 0:
                #     if self.pump_cd == 5:
                #         reward_in -= 4000 * 2
                #         self.remark.append(f'pump cd violation cd=5')
                #     elif self.pump_cd == 4:
                #         reward_in -= 3700 * 2
                #         self.remark.append(f'pump cd violation cd=4')
                #     elif self.pump_cd == 3:
                #         reward_in -= 3500 * 2
                #         self.remark.append(f'pump cd violation cd=3')
                #     elif self.pump_cd == 2:
                #         reward_in -= 3200 * 2
                #         self.remark.append(f'pump cd violation cd=2')
                #     elif self.pump_cd == 1:
                #         reward_in -= 3000 * 2
                #         self.remark.append(f'pump cd violation cd=1')
                #     # if self.pump_cd != 0:
                #     #     reward_in -= 15000
                #     #     self.remark.append(f'pump cd violation cd:{self.pump_cd}')
                #     else:
                #         reward_in -= 2000
                #         self.remark.append(f'pump open')
            else:
                if np.abs(self.open_num_diff) > 2:
                    reward_in -= 1000 * np.abs(self.open_num_diff)
                    self.remark.append(f'open num violation {self.open_num_diff}')
                if self.open_num_diff in (1, 2):
                    if self.pump_open_cd == 5:
                        reward_in -= 3000 * 2
                    elif self.pump_open_cd == 4:
                        reward_in -= 2500 * 2
                    elif self.pump_open_cd == 3:
                        reward_in -= 2300 * 2
                    elif self.pump_open_cd == 2:
                        reward_in -= 2000 * 2
                    elif self.pump_open_cd == 1:
                        reward_in -= 1700 * 2
                    else:
                        reward_in -= 1000
                        self.remark.append(f'pump open')
                elif self.open_num_diff in (-1, -2):
                    if self.pump_close_cd == 5:
                        reward_in -= 3000 * 2
                    elif self.pump_close_cd == 4:
                        reward_in -= 2500 * 2
                    elif self.pump_close_cd == 3:
                        reward_in -= 2300 * 2
                    elif self.pump_close_cd == 2:
                        reward_in -= 2000 * 2
                    elif self.pump_close_cd == 1:
                        reward_in -= 1700 * 2
                    else:
                        reward_in -= 1000
                        self.remark.append(f'pump close')

            # if self.open_num_diff == 0 and self.open_num_cal() > 1 and -11.8 < self.DH_TANK.head < -8:
            #     combo_weight = 50000 
            # else: 
            #     combo_weight = 0
            # for diff in self.diff_combo[::-1]:
            #     if diff != 0:
            #         break
            #     combo_weight = min(combo_weight * 1.1, 100000)  
            # reward_in += combo_weight

            if self.too_high or self.too_low:
                reward_in -= 5000

            reward_out  = reward_in

            return reward_out
        
        def orifice_reward():
            reward_in = 0
            # version 1
            # if self.DH_TANK.head < -9:
            #     reward_in -= (2 - (self.orifice_3031.current_setting + self.orifice_3041.current_setting)) * 10000
            #     self.remark.append(f'orifice open when lower -9, 3031:{self.orifice_3031.current_setting:.2f}, 3041:{self.orifice_3041.current_setting:.2f}')
            
            if self.orifice_3031.current_setting + self.orifice_3041.current_setting < 1.95:
                # if self.node3850_0313S.head > -8 and self.target_open_num == 9 and sum(self.rain_log) > 8:
                if self.target_open_num == 9 and sum(self.rain_log) > 8:
                    pass
                else:
                    reward_in -= 2000
                    reward_in -= (2 - (self.orifice_3031.current_setting + self.orifice_3041.current_setting)) * 5000
                    self.remark.append(f'orifice open when lower -9.5, 3031:{self.orifice_3031.current_setting:.2f}, 3041:{self.orifice_3041.current_setting:.2f}')
            else:
                # if self.node3850_0313S.head < -8 or self.target_open_num != 9:
                if self.target_open_num != 9 or sum(self.rain_log) < 8:
                    reward_in += (self.orifice_3031.current_setting + self.orifice_3041.current_setting - 1.95) * 100000


            if self.too_high_ori:
                reward_in -= (self.ori_open_volumn_unrevised - 2) * 1000 + 5000
            elif self.too_low_ori:
                reward_in -= (0 - self.ori_open_volumn_unrevised) * 1000 + 5000

            # if (self.DH_TANK.head < -9.2 or self.node3850_0313S.head < -8.5) and self.orifice_3031.current_setting + self.orifice_3041.current_setting < 2:
            #     reward_in -= 2000
            #     reward_in -= (2 - (self.orifice_3031.current_setting + self.orifice_3041.current_setting)) * 5000
            #     # reward_in -= (-9.2 - self.DH_TANK.head) * 5000
            #     self.remark.append(f'orifice open when lower -9.5, 3031:{self.orifice_3031.current_setting:.2f}, 3041:{self.orifice_3041.current_setting:.2f}')
            
            # elif self.DH_TANK.head >= -9.5 and self.DH_TANK.head < -8 and self.node3850_0313S.head > -8 and self.orifice_3031.current_setting + self.orifice_3041.current_setting < 2:
            #     reward_in += (2 - (self.orifice_3031.current_setting + self.orifice_3041.current_setting)) * 5000 
            #     reward_in += (1 - np.abs(-9 - self.DH_TANK.head)) * 2000
            #     self.remark.append(f'orifice open perfect, DH: {self.DH_TANK.head}, 31:{self.orifice_3031.current_setting:.2f}, 41:{self.orifice_3041.current_setting:.2f}')
            
            # elif self.DH_TANK.head >= -9.5 and self.node3850_0313S.head >= -8 and self.orifice_3031.current_setting + self.orifice_3041.current_setting < 2:
            #     reward_in += (2 - (self.orifice_3031.current_setting + self.orifice_3041.current_setting)) * 3000
            #     reward_in += (1 - np.abs(-9 - self.DH_TANK.head)) * 1000
            #     self.remark.append(f'orifice open good, DH: {self.DH_TANK.head}, 31:{self.orifice_3031.current_setting:.2f}, 41:{self.orifice_3041.current_setting:.2f}')
            

            reward_out = reward_in
            return reward_out
        
        def action1_centered_reward():
            reward_in = 0
            action1_range = [0.1, 0.3, 0.5, 0.7, 0.9] 
            reward_in -= (action1_range[int(self.action1_regularized)] - self.action1_unregularized)**2 * 300000
            reward_out = reward_in
            return reward_out


        self.remark = []
        if indicator == '0':
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()
            
            
        elif indicator == '1':
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()

        elif indicator in ('2-1', '2-2'):
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()
            pass

        elif indicator == '3':
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()
            pass

        elif indicator == '4':
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()
            pass
        
        elif indicator == '5':
            # reward_action1 = action1_centered_reward()
            reward_level = basic_level_reward(self.target_level)
            reward_pump = pump_reward(indicator)
            reward_ori = orifice_reward()
            pass
        
        # action_count = self.arb.get_count(self.action_log_temp)
        # if action_count > 80:
        #     reward = reward_level + reward_pump + reward_ori - min(action_count * 10, 7000)
        # else:    
        #     reward = reward_level + reward_pump + reward_ori


        # if action_count > 40:
        #     high_value = 8000
        #     penalty = action_count * 10
        #     if action_count * 10 <= high_value:
        #         pass
        #     else:
        #         penalty = high_value + (penalty - high_value) * 0.1
        #     reward = reward_level + reward_pump + reward_ori - penalty
        # else:    
        #     reward = reward_level + reward_pump + reward_ori

        # penalty = 0
        # if action_count > 40:
        #     penalty = min((action_count - 40) * 10, 10000)
        #     if reward_level + reward_pump + reward_ori > 8000:
        #         penalty /= 2
        #     reward = (reward_level + reward_pump + reward_ori) - penalty
        # else:
        #     reward = reward_level + reward_pump + reward_ori

        # penalty = self.action_penalty.calculate_penalty(self.current_state_action)
        # if reward_level + reward_pump + reward_ori > -4000:
        #     penalty /= 2
        # reward = reward_level + reward_pump + reward_ori - penalty
        reward = reward_level + reward_pump + reward_ori
        penalty = 0

        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'reward_level'] = reward_level
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'reward_pump'] = reward_pump
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'reward_ori'] = reward_ori
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'total_reward'] = reward
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'penalty'] = penalty
        self.log_df_all[self.train_sample[self.which_event]].loc[self.sim.current_time - relativedelta(minutes=10), 'remark'] = "\n".join(self.remark)
        return reward 
    
        
        
    def indicator_transform(self, indicator):
        if indicator == '0':
            indi_trans = 0
        elif indicator == '1':
            indi_trans = 1
        elif indicator in ('2-1', '2-2'):
            indi_trans = 2
        elif indicator == '3':
            indi_trans = 3
        elif indicator == '4':
            indi_trans = 4
        elif indicator == '5':
            indi_trans = 5
        return indi_trans
    
    def open_num_cal(self, Type):
        if Type == 0:
            return sum(pump.current_setting != 0 for pump in self.pumps)
        elif Type == 1:
            return sum(pump.current_setting for pump in self.pumps)
                       
class ContinuousActionPenalty:
    def __init__(self, grid_shape, radius, beta):
        self.radius = radius
        self.beta = beta
        self.grid_points = self._initialize_grid(grid_shape) 
        self.visit_counts = defaultdict(int) 
        self.kdtree = KDTree(self.grid_points)

    def _initialize_grid(self, grid_shape):
        grids = [np.arange(start, stop, grid_resolution) for start, stop, grid_resolution in grid_shape]
        grid_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(grid_shape))
        return grid_points

    def _calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def add_action(self, action):
        neighbors = self.kdtree.query_ball_point(action, r=self.radius)
        # print(len(neighbors))
        for idx in neighbors:
            grid_point = tuple(self.kdtree.data[idx])
            self.visit_counts[grid_point] += 1
    
    def get_visit_count(self, action):
        nearest_grid_point, dist = self._get_nearest_grid_point(action)
        return self.visit_counts[tuple(nearest_grid_point)], dist

    def _get_nearest_grid_point(self, action):
        distances = np.linalg.norm(self.grid_points - np.array(action), axis=1)
        nearest_index = np.argmin(distances)
        nearest_distance = np.min(distances)
        return self.grid_points[nearest_index], nearest_distance
    
    def calculate_penalty(self, current_action):
        penalty = 0
        visit_count, dist = self.get_visit_count(current_action)
        if visit_count > 8:
            penalty = min(self.beta * visit_count / (dist + 1e-5), 5000)
        # print(penalty)
        return penalty

# class ContinuousActionPenalty:
#     def __init__(self, penalty_radius, beta):
#         self.visited_actions = defaultdict(int) 
#         self.penalty_radius = penalty_radius 
#         self.beta = beta 
#         # state = np.array([scaled_indi_value,
#         #                   scaled_DH_target_value,
#         #                        scaled_DH_tank_value,
#         #                        scaled_diffuser_value,
#         #                     #    self.grad_wlv,
#         #                     #    self.pump_open_var,
#         #                     #    scaled_p_open_num_value,
#         #                     #    self.previous_action1, 
#         #                        scale_previous_p_open_num_value,
#         #                        scale_target_p_open_num_value,
#         #                        scale_target_p_open_num_unrevised_value,
#         #                     #    scaled_lp_open_ratio_value,
#         #                     #    scaled_ori_open_value,
#         #                     #    scaled_fct_rain_value,
#         #                        scaled_p_cd_value,
#         #                        scaled_p_diff_value,
#         #                        scaled_basic_flow
#         #                        ] + list(scaled_R17sum_value)
#         #                     ).astype(np.float32)
#         # weights = [1, 7, 7, 7, 7, 2, 2, 6, 2, 4] + [1 for _ in range(12)] + [10, 6, 10]
#         weights = [1, 1, 1, 1, 1, 1, 1, 1]
#         self.weights = np.array(weights) / np.sum(weights)

#     def add_action(self, action):
#         def custom_round(action):
#             action_rounded = np.empty_like(action) 
#             action_rounded[:-3] = np.round(action[:-3], decimals=1)  
#             action_rounded[-3:] = np.round(action[-3:], decimals=2) 
#             return action_rounded
#         action_tuple = tuple(custom_round(action))  
#         self.visited_actions[action_tuple] += 1
#         for visited_action in self.visited_actions:
#             distance = np.linalg.norm(np.array(action_tuple) - np.array(visited_action))
#             if distance < self.penalty_radius:
#                 self.visited_actions[visited_action] += 1

#     def calculate_penalty(self, current_action):
#         # def calculate_distance(point1, point2):
#         #     return np.linalg.norm(np.array(point1) - np.array(point2))
#         def calculate_distance(point1, point2):
#             point1 = np.array(point1)
#             point2 = np.array(point2)
#             return np.sqrt(np.sum(self.weights * (point1 - point2) ** 2))
#         penalty = 0
#         current_action_tuple = tuple(current_action)
#         nearby_actions = []
#         for action, visit_count in self.visited_actions.items():
#             dist = calculate_distance(current_action_tuple, action)
#             # print('dist', dist)
#             if dist < self.penalty_radius:
#                 nearby_actions.append((dist, visit_count)) 

#         nearby_actions.sort(key=lambda x: x[0])
#         nearby_actions = nearby_actions[:4]

#         for dist, visit_count in nearby_actions:
#             # if visit_count > 1:
#             penalty += self.beta * visit_count / (dist + 1e-5)

#         # print(penalty)

#         return penalty



if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.callbacks import EvalCallback
    import os
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.monitor import Monitor
    inp_file = r'C:\Users\309\YH\00Code\123\pyswmm\5.Dihwa_R1_INT0_Difup1_EMG1_Difdn1_h6f48_20230630_0000.inp'
    rain_file = r'C:\Users\309\YH\00Code\01RunSWMM\rainfall.csv'
    env = Sewer_Env(inp_file, rain_file)
    # check_env(env, warn=True)
    # env.close()
    # vec_env = DummyVecEnv(env)
    log_dir = r"C:\Users\309\YH\00Code\01RunSWMM\rein\tmp\gym"
    eval_log_dir = r"C:\Users\309\YH\00Code\01RunSWMM\rein\eval_logs"
    env = Monitor(env, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    n_training_envs = 1
    n_eval_envs = 1
    train_env = make_vec_env(lambda: env, n_envs=n_training_envs)
    eval_env = make_vec_env(lambda: env, n_envs=n_eval_envs)


    # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
    # When using multiple training environments, agent will be evaluated every
    # eval_freq calls to train_env.step(), thus it will be evaluated every
    # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=max(10 // n_training_envs, 1),
                                n_eval_episodes=5, deterministic=True,
                                render=False)

    model = SAC("MlpPolicy", train_env)
    model.learn(50, callback=eval_callback)
