# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:18:00 2024

@author: 309
"""
from datetime import datetime, time as TIME, timedelta
import pandas as pd
import numpy as np
from pyswmm import Simulation,Output,Nodes,Link,Links, RainGages
import swmm,os
from swmm_api import SwmmInput
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
from swmm_api.input_file.sections import Outfall
from swmm_api.input_file.sections import Pump, Control
from swmm_api.input_file.sections import TimeseriesData, Timeseries
import platform
# 
# from swmm_api.input_file.sections import _convert_lines
import swmm_api
# import datetime
from dateutil.relativedelta import relativedelta 
from swmm.toolkit.shared_enum import LinkAttribute, NodeAttribute
from swmm_api.input_file.sections import Pump, Control
# 引入 time 模組
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpltw
from sqlalchemy import create_engine #, MetaData
   
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D      
from  matplotlib.ticker import FormatStrFormatter 
from run_sewer_pyswmm0906 import run_sewer



#%% setting
# Evnow =1
Evnow = 0
import os
import time, sys
import platform
import pandas as pd
from datetime import datetime, time as TIME, timedelta
oridir_loc = os.getcwd()  
# pasttime = 24

# Nowtime = Now.replace(minute= int(Now.minute/10)*10, second=0,microsecond=0)  
# now_fct_s_time = Nowtime
#先跑SPM========================================================================= 
# conda install numpy=1.19.5 -c conda-forge
# global arg_str 
if platform.system().lower() == 'windows':
    # self.sys = 'windows'
    oripath = os.getcwd()
    # oripath =  r'/home/sewer/SWMM/00Code/01RunSWMM'
    #pardir = os.path.abspath(os.path.join(oripath, os.path.pardir))
    # pardir = r'/home/sewer/SWMM/00Code'
    # path_inp = os.path.join('01RunSWMM','00_Dihwa_xizhi_RR_BaseFlow_Sunday_INTWL_NTU_20240308.inp')
    # path_inp = os.path.join('00_Dihwa_xizhi_RR_BaseFlow_Sunday_INTWL_NTU_20240308.inp')

    # path_inp = r'C:\Users\309\YH\00Code\01RunSWMM\00_Dihwa_xizhi_RR_BaseFlow_Sunday_INTWL_NTU_20240605 - 複製.inp'
    # path_realdata = os.path.join('Realtime_data.csv') 
    # path_iotable = r'C:\Users\309\YH\00Code\01RunSWMM\IOtable.xlsx'
    # save_path = os.path.join(oripath,'00Result/final/rain/pid1_grid')
    # save_path = os.path.join(oripath,'00Result/final/rain/RBC_imperfect')
    save_path = os.path.join(oripath,'00Result/final/rain')
    os.makedirs(save_path, exist_ok=True)
    

elif platform.system().lower() == 'linux':
    # self.sys = 'linux'
    if __name__ == '__main__':
        oripath = r"/home/sewer/GitRepos/sewer-platform/backend/docker_compose/apscheduler/scripts"
        # save_path = r"/home/sewer/GitRepos/sewer-platform/backend/docker_compose/fastapi/app/data"
        save_path = r"/home/sewer/NTUcode/00Result"
        Now = pd.Timestamp.now()
    else:
        oripath = r"/scripts"
        save_path = r"/to_fastapi_app_data"
        Now = pd.Timestamp.now()  #+ pd.Timedelta(hours=8)
# path_inp = os.path.join('history_h6f18_20230630_1200_0.55_modA_D_20240625.inp')
path_inp = os.path.join('history_h6f18_20230630_1200_0.55_modA_D_loquan_20240820.inp')    

#linux 系統直接讀取資料庫
path_realdata =  os.path.join(oripath,'Realtime_data.csv') 
path_iotable = os.path.join(oripath,'IOtable.xlsx') 
level_path = os.path.join(oripath,'液位計人孔位置.xlsx') 
# save_path = os.path.join(oripath,'00Result')
# os.makedirs(save_path, exist_ok=True)
pasttime = 24 #資料抓的時間
strategy = 1
hsf_time_len = 6
hsf_e_time_len = 1/6
# fct_lead_time_len = 25
fct_lead_time_len = 48
# 100000/9

if Evnow == 1:
    t_time = Now.replace(minute= int(Now.minute/10)*10, second=0,microsecond=0)   
    fct_s_time=t_time + relativedelta(minutes=10)  
    Rfct_type = 1
    
    hsf_s_time = fct_s_time - relativedelta(hours=hsf_time_len) - relativedelta(minutes=10) 
    hsf_e_time = fct_s_time -  relativedelta(hours=hsf_e_time_len)
    fct_e_time = fct_s_time + relativedelta(hours=fct_lead_time_len)  #- relativedelta(minutes=10) 

    time_dict = {'t_time':t_time,
                    'hsf_time_len': hsf_time_len,
                    'fct_lead_time_len': fct_lead_time_len,
                    'hsf_s_time': hsf_s_time,
                    'hsf_e_time': hsf_e_time,
                    'fct_s_time': fct_s_time,
                    'fct_e_time': fct_e_time,
                    'hsf_time_len': hsf_time_len,
                    'hsf_e_time_len': hsf_e_time_len,
                    'fct_lead_time_len': fct_lead_time_len, 
                    'load_data_s_time': hsf_s_time,
                    'load_data_e_time': hsf_e_time
                    }
    list_fct_s_time = [fct_s_time]
else:
    Now = pd.Timestamp.now()
    fct_s_time1 = datetime(2020, 5, 16, 0, 0)
    fct_s_time2 = datetime(2020, 5, 21, 0, 0)
    fct_s_time3 = datetime(2022, 5, 26, 0, 0)
    fct_s_time4 = datetime(2022, 9, 29, 0, 0) #完全沒下雨
    fct_s_time5 = datetime(2023, 6, 30, 6, 0)
    fct_s_time6 = datetime(2023, 11, 20, 0, 0) #完全沒下雨
    fct_s_time7 = datetime(2024, 1, 16, 0, 0) #小雨
    fct_s_time8 = datetime(2024, 2, 20, 0, 0) #完全沒下雨
    fct_s_time10 = datetime(2024, 7, 8, 13, 10) #完全沒下雨
    fct_s_time11 = datetime(2024, 3, 21, 0, 0) #沒下雨的  3/21 3/22 周四周五 都是沒下雨 且前24小時沒下雨
    fct_s_time12 = datetime(2024, 4, 13, 0, 0) #沒下雨的 周六周日 4/13 -4/14
    fct_s_time13 = datetime(2024, 5, 12, 6, 0)    
    fct_s_time14 = datetime(2024, 6, 21, 6, 0) 
    
    fct_s_time15 = datetime(2024, 7, 8, 11, 0)
    fct_s_time16 = datetime(2024, 7, 10, 6, 0)
    fct_s_time17 = datetime(2024, 7, 24, 0, 0) # typhoon       
    fct_s_time18 = datetime(2024, 7, 25, 0, 0) # typhoon     

    fct_s_time81 = datetime(2024, 8, 15, 0, 0)     
    fct_s_time82 = datetime(2024, 8, 19, 0, 0)    #9        
    fct_s_time83 = datetime(2024, 8, 20, 0, 0)    #9        
    fct_s_time84 = datetime(2024, 8, 23, 0, 0)    #9           
    fct_s_time85 = datetime(2024, 8, 29, 0, 0)    #9
    
    fct_s_time19 = datetime(2024, 5, 9, 0, 0)    # dry 1
    fct_s_time20 = datetime(2024, 5, 29, 0, 0)    # dry 2
    fct_s_time21 = datetime(2024, 7, 19, 0, 0)    # dry 3
    fct_s_time22 = datetime(2024, 6, 26, 0, 0)
    fct_s_time23 = datetime(2024, 4, 28, 0, 0)

    fct_s_time31 = datetime(2024, 6, 24, 6, 0)
    fct_s_time32 = datetime(2024, 4, 18, 0, 0)
    fct_s_time33 = datetime(2023, 8, 10, 6, 0)

    # train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), 
    #             datetime(2024, 4, 24, 0, 0), datetime(2024, 6, 24, 0, 0), datetime(2024, 7, 8, 11, 0), datetime(2024, 7, 24, 0, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0)]
    # train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), 
    #             datetime(2023, 8, 22, 6, 0), datetime(2024, 4, 24, 0, 0), datetime(2024, 6, 24, 0, 0), datetime(2024, 7, 8, 11, 0),
    #             datetime(2024, 7, 24, 0, 0), datetime(2024, 8, 19, 6, 0), datetime(2024, 8, 23, 6, 0), datetime(2024, 7, 1, 6, 0), 
    #             datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 3, 31, 3, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0),
    #             datetime(2024, 6, 28, 6, 0), datetime(2023, 5, 22, 12, 0), datetime(2023, 8, 16, 0, 0),]
    # train_events = [datetime(2023, 1, 11, 12, 0), datetime(2023, 1, 25, 12, 0), datetime(2023, 2, 26, 12, 0), datetime(2023, 3, 13, 12, 0), datetime(2023, 4, 13, 12, 0), datetime(2023, 4, 17, 12, 0), 
    #                     datetime(2023, 5, 24, 12, 0), datetime(2023, 6, 10, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 27, 12, 0), datetime(2023, 9, 5, 12, 0),
    #                     datetime(2023, 10, 6, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 1, 15, 12, 0), datetime(2024, 1, 29, 12, 0), datetime(2024, 2, 5, 12, 0), 
    #                     datetime(2024, 2, 29, 12, 0), datetime(2024, 3, 6, 12, 0), datetime(2024, 3, 26, 12, 0), datetime(2024, 4, 14, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 5, 14, 12, 0), 
    #                     datetime(2024, 5, 26, 12, 0), datetime(2024, 6, 1, 12, 0), datetime(2024, 6, 26, 12, 0), datetime(2024, 7, 7, 12, 0), datetime(2024, 7, 29, 12, 0), datetime(2024, 8, 11, 12, 0)]
    # val_events = [datetime(2023, 2, 24, 12, 0), datetime(2023, 3, 24, 12, 0), datetime(2023, 5, 1, 12, 0), datetime(2023, 6, 17, 12, 0), datetime(2023, 8, 12, 12, 0), datetime(2023, 9, 7, 12, 0), 
    #                 datetime(2023, 11, 30, 12, 0), datetime(2024, 1, 27, 12, 0), datetime(2024, 2, 10, 12, 0), datetime(2024, 3, 12, 12, 0), datetime(2024, 4, 28, 12, 0), datetime(2024, 5, 22, 12, 0), 
    #                 datetime(2024, 6, 14, 12, 0), datetime(2024, 7, 19, 12, 0), datetime(2024, 8, 21, 12, 0), datetime(2024, 8, 27, 12, 0)]

    # train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), datetime(2024, 3, 31, 3, 0),
    #                 datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 7, 1, 6, 0), datetime(2024, 7, 8, 11, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0),
    #             datetime(2024, 6, 28, 6, 0), datetime(2023, 5, 22, 12, 0), datetime(2023, 8, 16, 0, 0),]
    # test_events = [datetime(2023, 6, 10, 9, 0), datetime(2023, 5, 30, 22), datetime(2023, 6, 8, 18), datetime(2023, 9, 2, 0), datetime(2024, 5, 12, 12, 0), 
    #            datetime(2024, 6, 16, 6, 0), datetime(2024, 7, 2, 3, 0)]
    train_events = [datetime(2023, 1, 11, 12, 0), datetime(2023, 1, 25, 12, 0), datetime(2023, 2, 26, 12, 0), datetime(2023, 3, 13, 12, 0), datetime(2023, 4, 13, 12, 0), datetime(2023, 4, 17, 12, 0), 
                datetime(2023, 5, 24, 12, 0), datetime(2023, 6, 3, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 29, 12, 0), datetime(2023, 9, 15, 12, 0),
                datetime(2023, 10, 6, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 1, 16, 12, 0), datetime(2024, 1, 29, 12, 0), datetime(2024, 2, 5, 12, 0), 
                datetime(2024, 2, 29, 12, 0), datetime(2024, 3, 9, 12, 0), datetime(2024, 3, 26, 12, 0), datetime(2024, 4, 14, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 5, 14, 12, 0), 
                datetime(2024, 5, 26, 12, 0), datetime(2024, 6, 3, 12, 0), datetime(2024, 6, 26, 12, 0), datetime(2024, 7, 6, 12, 0), datetime(2024, 7, 29, 12, 0), datetime(2024, 8, 11, 12, 0)]
    # train_eval_events = [datetime(2023, 1, 25, 12, 0), datetime(2023, 5, 24, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 3, 9, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 6, 3, 12, 0)]
    val_events = [datetime(2023, 2, 24, 12, 0), datetime(2023, 3, 24, 12, 0), datetime(2023, 5, 1, 12, 0), datetime(2023, 6, 17, 12, 0), datetime(2023, 8, 12, 12, 0), datetime(2023, 9, 7, 12, 0), 
                    datetime(2023, 11, 29, 12, 0), datetime(2024, 1, 27, 12, 0), datetime(2024, 2, 10, 12, 0), datetime(2024, 3, 12, 12, 0), datetime(2024, 4, 28, 12, 0), datetime(2024, 5, 22, 12, 0), 
                    datetime(2024, 6, 14, 12, 0), datetime(2024, 7, 19, 12, 0), datetime(2024, 8, 27, 12, 0)]
    test_events = [datetime(2023, 1, 3, 12, 0), datetime(2023, 1, 27, 12, 0), datetime(2023, 2, 25, 12, 0), datetime(2023, 5, 6, 12, 0), datetime(2023, 7, 28, 12, 0), datetime(2023, 10, 31, 12, 0), datetime(2023, 12, 21, 12, 0),
            datetime(2023, 11, 15, 12, 0), datetime(2024, 1, 8, 12, 0), datetime(2024, 3, 10, 12, 0), datetime(2024, 4, 16, 12, 0), datetime(2024, 5, 20, 12, 0), datetime(2024, 6, 30, 12, 0), datetime(2024, 7, 27, 12, 0), datetime(2024, 8, 17, 12, 0)]
    # train_events = []
    # train_events = [datetime(2023, 6, 23, 6, 0), datetime(2023, 8, 20, 6, 0), datetime(2024, 3, 31, 3, 0),
    #                 datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 7, 1, 6, 0), datetime(2024, 7, 8, 11, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0),
    #             datetime(2024, 6, 28, 6, 0), datetime(2023, 5, 22, s12, 0), datetime(2023, 8, 16, 0, 0),]
    # train_events = [datetime(2023, 1, 25, 12, 0)]
    # train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 10, 6, 0), datetime(2023, 8, 16, 0, 0), datetime(2024, 7, 10, 0, 0),datetime(2024, 6, 16, 6, 0), datetime(2024, 7, 2, 3, 0)]
    # val_events = []
    # test_events = []
    time_dict = {event: {} for event in train_events + val_events + test_events}
    # 8月15、19、20、23、29日
    
    # runcell(0, 'E:/E_Program/SSO_swmm/00Code/Sewer_simulation3/run_sewer_pyswmm0906.py')
    # runcell(0, 'E:/E_Program/SSO_swmm/00Code/Sewer_simulation3/run_sewer_pyswmm0815.py')
    # fct_s_time = fct_s_time83
    # fct_s_time = fct_s_time31
    for fct_s_time in train_events + val_events + test_events:
        t_time =  fct_s_time - relativedelta(minutes=10)  
        
        hsf_s_time = fct_s_time - relativedelta(hours=hsf_time_len) - relativedelta(minutes=10) 
        hsf_e_time = fct_s_time -  relativedelta(hours=hsf_e_time_len)
        fct_e_time = fct_s_time + relativedelta(hours=fct_lead_time_len)  #- relativedelta(minutes=10) 

        time_dict[fct_s_time] = {'t_time':t_time,
                        'hsf_time_len': hsf_time_len,
                        'fct_lead_time_len': fct_lead_time_len,
                        'hsf_s_time': hsf_s_time,
                        'hsf_e_time': hsf_e_time,
                        'fct_s_time': fct_s_time,
                        'fct_e_time': fct_e_time,
                        'hsf_time_len': hsf_time_len,
                        'hsf_e_time_len': hsf_e_time_len,
                        'fct_lead_time_len': fct_lead_time_len, 
                        'load_data_s_time': hsf_s_time,
                        'load_data_e_time': t_time+pd.Timedelta(hours=fct_lead_time_len)
                        }
# t_time =  datetime(2024, 7, 5, 3, 10)

    Rfct_type = 0
    
    list_fct_s_time = train_events + val_events  + test_events# [fct_s_time]
# for kkk in range(12):
# list_fct_s_time = pd.date_range(fct_s_time,fct_s_time+ \
#                                 relativedelta(hours=16),freq='h' )
# fct_s_time = list_fct_s_time[0]
# list_fct_s_time = [fct_s_time10]
# fct_s_time = list_fct_s_time[0]
for fct_s_time in list_fct_s_time: 
    print('1.Run start time: %s' %Now)
    print('1.Run swmm now time: %s' %t_time)


    T1 = time.time() 

    # from Load_RTdata import loadingRTData
    print('2.Loading Realtime data')
    #linux 系統直接讀取資料庫
    #Realtime_data_df , rain_bqpf_df  = loadingRTData(time_dict,Rfct_type,Evnow)
    # Realtime_data_df['六館電動閘門_全關'] #SGS_LG_G01_C
    # bbb = Realtime_data_df
    # Realtime_data_df = pd.read_csv('Realtime_data_20190101_20240822.csv', encoding = 'big5', index_col=0)
    # Realtime_data_df = pd.read_csv('Realtime_data_20240701_20240823.csv', encoding = 'big5', index_col=0)
    # Realtime_data_df = pd.read_csv('Realtime_data_20240701_20240902.csv', encoding = 'big5', index_col=0)
    #修正雨量後
    Realtime_data_df = pd.read_csv('Realtime_data_202301_202408.csv', encoding = 'big5', index_col=0)
    # Realtime_data_df = pd.read_csv('Realtime_data_201901_202408.csv', encoding = 'big5', index_col=0)
    Realtime_data_df.index = pd.to_datetime(Realtime_data_df.index)
    # Realtime_data_df.to_csv('123.csv',encoding='big5')
    # aaa =         pd.read_csv('123.csv', index_col=0,encoding='big5')
    # aaa.index
    # 水利處的 資料延遲 往前10分鐘
    # Realtime_data_df.iloc[:,:41] = Realtime_data_df.iloc[:,:41].shift(-3).fillna(0)
    
    Realtime_data_df.iloc[:,:50] = Realtime_data_df.iloc[:,:50].shift(-1).fillna(0)  
    # Realtime_data_df.iloc[:,:50] = Realtime_data_df.iloc[:,:50].shift(-3).fillna(0)       
    col_RT = list(Realtime_data_df.columns )
    list_old = ['液位_迪化_1480', '液位_迪化_1466', '液位_迪化_1467',
    '液位_迪化_1456', '液位_迪化_1286', '液位_迪化_1465', '液位_迪化_1460',
    '液位_迪化_0643', '液位_迪化_1587', '液位_迪化_1461', '液位_迪化_1473',
    '液位_迪化_1453', '液位_迪化_1454', '液位_迪化_1468', '液位_迪化_0640' ,'液位_迪化_FC01']
    list_new = ['液位_迪化_AA80', '液位_迪化_AB66', '液位_迪化_AC67',
    '液位_迪化_AF56', '液位_迪化_BA86', '液位_迪化_BK65', '液位_迪化_DC60',
    '液位_迪化_Dca43', '液位_迪化_DD87', '液位_迪化_E61', '液位_迪化_EA73',
    '液位_迪化_F53', '液位_迪化_F54', '液位_迪化_FB68', '液位_迪化_FC40','液位_迪化_EA01']
    
    for ooo, nnn in zip(list_old, list_new):
        Realtime_data_df[nnn] = Realtime_data_df[ooo]
        
    # rain_bqpf_df = pd.read_csv('rain_bqpf_df.csv', encoding = 'big5', index_col=0)
    rain_bqpf_df = pd.DataFrame()
    a = run_sewer(path_inp, Realtime_data_df, path_iotable)

    # from 
    # savename_dict = {        
    #                 0: '0.Dihwa_R0_INT0_Difup0_EMG0_Difdn0',
    #                 1: '1.Dihwa_R1_INT1_Difup0_EMG0_Difdn0',
    #                 2: '2.Dihwa_R1_INT0_Difup0_EMG0_Difdn0',
    #                 3: '3.Dihwa_R1_INT0_Difup1_EMG0_Difdn0',
    #                 4: '4.Dihwa_R1_INT0_Difup1_EMG1_Difdn0',
    #                 5: '5.Dihwa_R1_INT0_Difup1_EMG1_Difdn1',
    #                 99: 'history'
    #                                 }
    
    savename_dict = {        
                    1: 'Dihwa',
                    99: 'history'
                                    }
    # strategy_list = [1,2,3,4,5]
    # save_path = r'C:\Users\309\YH\00Code\123\08Result'
    # save_path = r'C:\Users\309\YH\00Code\123\Big'
    # save_path = r'C:\Users\309\YH\00Code\123\pyswmm\try'
    # strategy_list = [1,2,3,4,5,99]
    strategy_list = [1]
    # pid_combination = []
    # for A in range(0, 10):
    #     for B in range(0, 10 - A):
    #         pid_combination.append([A+1, B+1, 9-A-B+1])
    
    # dc = 0.45  # 閘門入流細數
    dc = 0.5
    # pid_type = 1
    pc = [7, 3, 2]
    self=a
    pid_type = 2
    imperfect_input = [0.]
    random_seed = [118, 39, 641, 325, 715, 864, 31, 930, 645, 575,
                    452, 367, 289, 723, 101, 876, 54, 298, 493, 732,
                    159, 804, 215, 678, 842, 371, 907, 394, 520, 802,
                    490, 612, 250, 388, 963, 178, 839, 415, 660, 245]
    random_seed = [118, 39, 641, 325, 715,]
    # imperfect_input = [0]
    random_seed = [0]
    strategy = 1
    for ipi in imperfect_input:
        for rs in random_seed:
        # for pc in pid_combination[19:]:
            # for dc in np.arange(0.2, 0.9, 0.05):
            # save_name =  '%s_h%01df%02d_%s_%s_%s_%s' %(savename_dict[strategy],
            #                                     hsf_time_len,fct_lead_time_len,\
            #                                         time_dict[fct_s_time]['t_time'].strftime("%Y%m%d_%H%M"), f'pid{pid_type}', f'rs{rs}', f'DF')
            save_name =  '%s_h%01df%02d_%s_%s' %(savename_dict[strategy],
                                    hsf_time_len,fct_lead_time_len,\
                                        time_dict[fct_s_time]['t_time'].strftime("%Y%m%d_%H%M"), f'pid{pid_type}')
            # savename =  '%s_h%01df%02d_%s_%s' %(savename_dict[strategy],hsf_time_len,fct_lead_time_len,\
            #                                     fct_s_time.strftime("%Y%m%d_%H%M"),dc)
            a.out_inp(time_dict[fct_s_time], strategy, oripath, save_name,save_path, dc,Rfct_type,rain_bqpf_df)
            log_write_not_done = True
            if log_write_not_done:
                a.action_log_set()
                log_write_not_done = False
            # T1 = time.time()
            a.run(pid_type, pc[0], pc[1], pc[2], ipi, rs)
            a.action_history_log.to_csv('00Result\log_%s.csv' %save_name)
            a.load()
            # print(a.metric())
            # a.load(save_name, save_path, time_dict, strategy)
            # if platform.system().lower() == 'windows':
            # a.summarySuggest()
            if __name__ == '__main__':
                pass

                # a.plot_b43_emg()
                # a.plot_wl(level_path)
                a.plot()
            a.time_dict['fct_e_time'] = a.time_dict['fct_e_time'] + relativedelta(hours=1/6)
            # a.summarySTData()
    # a.plot_all(save_path, time_dict)
    # self.sim.close()
    
    T2 = time.time()
    self=a
    print('Done! Run time: %.3f min' %( (T2-T1)/60))