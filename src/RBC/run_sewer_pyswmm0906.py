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

class run_sewer():
    def __init__(self,path_inp, Realtime_data_df, path_iotable):
        self.inp_path = path_inp
        self.original_inp = SwmmInput.read_file(path_inp,encoding='big5') # inp file
        self.inp = self.original_inp.copy()
        self.iotable_path = path_iotable
        self.realtime_raw_df = Realtime_data_df
        self.oripath = os.getcwd()
        # ===========        
        if platform.system().lower() == 'windows':
            self.sys = 'windows'
        elif platform.system().lower() == 'linux':
            self.sys = 'linux'
        # ===========
        wl_name = pd.read_excel(self.iotable_path, sheet_name='SEW_USLV')
        wl_cols = ['液位_%s_%s' %(kkk,iii)  for iii,kkk in zip(wl_name['ID'].tolist(), wl_name['範圍'].tolist()) ]
        wl_cols_eng = wl_name['Tag'].tolist()
            
        sel_Var_df = pd.read_excel(self.iotable_path, sheet_name='sel_Var')
        var_col = sel_Var_df.columns 
        col_type = ['Rainfall','WL','INTWL_FLOW',  
                    'Interceptor', 'ContactBedTreatment',
                    'DiffuserFacility', 'ReliftStation',
                    'SewageTreatmentPlant',  'PumpingStation','PUMP_INST']
        cols = []
        cols_en = []
        for ccc in col_type:
            if ccc  == 'WL':
                cols = cols + wl_cols
                cols_en = cols_en + wl_cols_eng
            else:
                cols = cols + sel_Var_df.loc[sel_Var_df['%s_index' %ccc] == 1,ccc].tolist()
                cols_en = cols_en + sel_Var_df.loc[sel_Var_df['%s_index' %ccc] == 1,'%s_eng'%ccc].tolist()
        cols_en = list(map(str,cols_en))
        self.col_df = pd.DataFrame(index =cols_en, columns=['cols'])
        self.col_df['cols'] = cols


        target_df  = pd.read_excel(self.iotable_path, sheet_name='list')
        self.Interceptor_list = target_df['INTERCEPTOR'].dropna().astype(str).tolist()
        self.OUTLET_list = target_df['OUTLET'].dropna().astype(str).tolist()
        self.INTERCEPTOR_ORIFICE_list = target_df['INTERCEPTOR_ORIFICE'].dropna().astype(str).tolist() 
        self.INTERCEPTOR_ORIFICE_ch_list = target_df['INTERCEPTOR_ORIFICE_ch '].dropna().astype(str).tolist() 

        # self.realtime_raw_df = pd.read_csv(self.realdata_path, encoding = 'big5', index_col=0)
        # self.realtime_raw_df = Realtime_data_df
        # self.realtime_raw_df = pd.read_csv(self.realdata_path, index_col=0)
        self.realtime_raw_df.index = pd.to_datetime(self.realtime_raw_df.index)
# Realtime_data_df['迪化SLG-3031主閘門全關']
        #==================================    
        #晴天基礎入流量 列表
        self.Sunday_inflow_list = []
        for jjj in self.inp[sections.INFLOWS].keys():
            self.Sunday_inflow_list.append(self.inp[sections.INFLOWS][jjj]['time_series'])
        #雨量站  列表        
        self.R_timeseries_list = []
        for kkk in self.inp[sections.RAINGAGES].keys():
            if kkk in ['Rain_1','Rain_2']: continue
            self.R_timeseries_list.append(self.inp[sections.RAINGAGES][kkk]['timeseries'])   
# self=a
        # inp setiing
        self.report_setting()  #設定 是否輸出 rpt control 內容
        # self.inp[sections.CONTROLS] = Control.create_section()
        out_df  = pd.read_excel(self.iotable_path,sheet_name='out')
        self.pump_name = out_df.loc[out_df.Type2=='pumps','Object Name'].to_list()
        self.action_history_log = pd.DataFrame(columns=['pump_open_num',\
                                                        'pump_GINMEI', 'pump_SONSHAN', 'pump_QUENYAN', 'pump_SULIN', 'pump_SONSHIN', 'pump_SINJAN',\
                                                        'pump_ZUNSHAO', 'pump_B43_1', 'pump_B43_2', 'pump_B43_3', 'pump_B43_4', 'ori_3031', 'ori_3041',\
                                                        'ori_LOQUAN', 'ori_HURU' ] + self.INTERCEPTOR_ORIFICE_list)



    def report_setting(self, open=False):
        if open==False:
            self.inp[sections.REPORT]['CONTROLS'] = 'NO'
        else: 
            self.inp[sections.REPORT]['CONTROLS'] = 'YES'
 
    def option_setting(self, time_dict, sum_type='hsf_fct'):
        # 熱啟動加模擬一起跑
        if sum_type == 'hsf_fct':
            # time_dict['fct_e_time'] = time_dict['fct_e_time'] + relativedelta(hours=1/6)
            self.inp[sections.OPTIONS]['START_DATE'] = datetime.date(time_dict['hsf_s_time'])
            self.inp[sections.OPTIONS]['START_TIME'] = datetime.time(time_dict['hsf_s_time'])
            
            fct_e_time_add10 = time_dict['fct_e_time'] #+  relativedelta(minutes=10) 
            self.inp[sections.OPTIONS]['END_DATE'] = datetime.date(fct_e_time_add10)
            self.inp[sections.OPTIONS]['END_TIME'] = datetime.time(fct_e_time_add10)
            self.inp[sections.OPTIONS]['REPORT_START_DATE'] = datetime.date(time_dict['hsf_s_time'])
            self.inp[sections.OPTIONS]['REPORT_START_TIME'] = datetime.time(time_dict['hsf_s_time'])
            # datetime.time 會出錯 所以虛擬一個時間
            self.inp[sections.OPTIONS]['WET_STEP'] = datetime.time(datetime(2020,10,10,0,10))
            self.inp[sections.OPTIONS]['DRY_STEP'] = datetime.time(datetime(2020,10,10,0,10))
        for kkk in self.inp[sections.RAINGAGES].keys():
            self.inp[sections.RAINGAGES][kkk]['interval'] = self.inp[sections.OPTIONS]['WET_STEP'] #虛擬時間 取10分鐘

    def storage_init_set(self):
        self.inp[sections.STORAGE]['DIHWA']['depth_init'] = self.realtime_raw_df.loc[self.time_dict['hsf_s_time'], '迪化LT-1濕井液位高度'] - 11.8 + 12.89

    def R_setting(self, type_sunwet):
        # set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index)+3, freq='10min') 
        # self.rainfall_df = pd.DataFrame(index=set_rain_index)
        # for iii in self.R_timeseries_list:
        #     self.inp[sections.RAINGAGES][iii]['form'] = 'VOLUME'
        #     tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
        #     f_df = pd.DataFrame(index=set_rain_index, columns=['Time','Value'])
        #     col_ch = self.col_df.loc[iii,'cols']
        #     f_df['Time'] = set_rain_index.strftime('%m/%d/%Y %H:%M:%S') 
        #     if type_sunwet == 0:
        #         f_df.loc[set_rain_index,'Value'] = 0
        #     elif type_sunwet == 1:
        #         f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]
        #         if col_ch in ['雙園', '萬華國中', '臺北', '中正國中', '台灣大學(新)', '博嘉國小', '興華國小', '永建國小', '北政國中']:
        #         # if col_ch in ['雙園', '萬華國中', '博嘉國小', '興華國小', '永建國小', '北政國中']:
        #         # if col_ch in ['萬華國中']:
        #             f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1.3
        #         # elif col_ch in ['民生國中']:
        #         #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*0.5
        #         # elif col_ch in ['建國']:
        #         #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*0.7
        #         # elif col_ch in ['五常國小']:
        #         #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1.3
        #         # # if col_ch in ['博嘉國小', '興華國小', '永建國小', '北政國中']: 
        #         #     # f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1
        #         # else:
        #         #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]
        #     self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
        #     self.rainfall_df[iii] = f_df.Value
        # self.rainfall_df.replace(np.nan, 0, inplace=True)
        self.R_list =  pd.read_excel('ID_tables.xlsx',sheet_name='Rainfall')['編號'].dropna().to_list()#.toslit()
        if self.Rfct_type ==0:
            set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index), freq='10min') 
            self.rainfall_df = pd.DataFrame(index=set_rain_index)
            for iii in self.R_list:
                if iii in  self.R_timeseries_list:
                    
                    self.inp[sections.RAINGAGES][iii]['form'] = 'VOLUME'
                    tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
                    f_df = pd.DataFrame(index=set_rain_index, columns=['Time','Value'])
                    col_ch = self.col_df.loc[iii,'cols']
                    f_df['Time'] = set_rain_index.strftime('%m/%d/%Y %H:%M:%S') 
                    # col_ch = '萬華國中'
                    # if type_sunwet == 0:
                        # f_df.loc[set_rain_index,'Value'] = 0
                    # elif type_sunwet == 1:
                    f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]
                    if col_ch in [ '臺北', '中正國中', '台灣大學(新)', '博嘉國小',
                                  '興華國小', '永建國小', '北政國中']: #A區
                        f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1
                    elif col_ch in ['雙園', '萬華國中']: #BK區
                        f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1.8
                    elif col_ch in [ '科教館', '陽明高中', '望興橋']:  #C區
                        f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1                            
                    elif col_ch in ['關渡', '桃園國中', '北投國小', '奇岩', '中和橋', '磺溪橋'
                                  , '天母', '福德', ]: #D區
                        f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1.5                       
                                         
                        # self.realtime_raw_df.loc[set_rain_index, '雙園'].sum() *1.8  
                        # *1.3
                        # print()
                        # elif col_ch in ['民生國中']:
                        #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*0.5
                        # elif col_ch in ['建國']:
                        #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*0.7
                        # elif col_ch in ['五常國小']:
                        #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1.3
                        # # if col_ch in ['博嘉國小', '興華國小', '永建國小', '北政國中']: 
                        #     # f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]*1
                        # else:
                        #     f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]
                    self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
                self.rainfall_df.loc[self.hsf_time_index,iii] = self.realtime_raw_df.loc[self.hsf_time_index, col_ch]
                self.rainfall_df.loc[self.fct_time_index,iii] = self.realtime_fct_df.loc[self.fct_time_index, col_ch]                    
                     
                    # self.rainfall_df[iii] = f_df.Value
            self.rainfall_df.replace(np.nan, 0, inplace=True)
        elif self.Rfct_type ==1: #用預報雨量     
            # self.realtime_fct_df 
            
            set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index), freq='10min') 
            self.rainfall_df = pd.DataFrame(index=set_rain_index)
            # for iii in self.R_timeseries_list
            for iii in self.R_list: #40
                if iii in  self.R_timeseries_list:
                    self.inp[sections.RAINGAGES][iii]['form'] = 'VOLUME'
                    tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
                    f_df = pd.DataFrame(index=set_rain_index, columns=['Time','Value'])
                    col_ch = self.col_df.loc[iii,'cols']
                    f_df['Time'] = set_rain_index.strftime('%m/%d/%Y %H:%M:%S') 
                    if type_sunwet == 0:
                        f_df.loc[set_rain_index,'Value'] = 0 
                    elif type_sunwet == 1:
                        f_df.loc[self.hsf_time_index,'Value'] = self.realtime_raw_df.loc[self.hsf_time_index, col_ch]
                        f_df.loc[self.fct_time_index,'Value'] =  self.realtime_fct_df.loc[self.fct_time_index, col_ch]                    
                        # self.hsf_time_index
                        # self.fct_time_index
                        # f_df.loc[self.hsf_fct_time_index,'Value'] = self.realtime_df.loc[self.hsf_fct_time_index, col_ch]*0.5
                    self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
                self.rainfall_df.loc[self.hsf_time_index,iii] = self.realtime_raw_df.loc[self.hsf_time_index, col_ch]
                self.rainfall_df.loc[self.fct_time_index,iii] = self.realtime_fct_df.loc[self.fct_time_index, col_ch]                    
                # self.rainfall_df[iii] = f_df.Value
            self.rainfall_df.replace(np.nan, 0, inplace=True)
        # R17 = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','A0A01','T15','C0A9F','T014']
        # self.current_rain_intensity = self.rainfall_df.loc[self.time_dict['t_time'],R17].mean()
        R17 = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','A0A010','T15','C0A9F0','T014']
        self.current_rain_intensity = self.rainfall_df.loc[self.time_dict['t_time'],R17].mean()            
            # set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index), freq='10min') 
            # self.rainfall_df = pd.DataFrame(index=set_rain_index)
            # for iii in self.R_timeseries_list:
            #     self.inp[sections.RAINGAGES][iii]['form'] = 'VOLUME'
            #     tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
            #     f_df = pd.DataFrame(index=set_rain_index, columns=['Time','Value'])
            #     col_ch = self.col_df.loc[iii,'cols']
            #     f_df['Time'] = set_rain_index.strftime('%m/%d/%Y %H:%M:%S') 
            #     if type_sunwet == 0:
            #         f_df.loc[set_rain_index,'Value'] = 0
            #     elif type_sunwet == 1:
               
            #         f_df.loc[self.hsf_time_index,'Value'] = self.realtime_raw_df.loc[self.hsf_time_index, col_ch]
            #         f_df.loc[self.fct_time_index,'Value'] =  self.realtime_fct_df.loc[self.fct_time_index, col_ch]                    
            #         # self.hsf_time_index
            #         # self.fct_time_index
            #         # f_df.loc[self.hsf_fct_time_index,'Value'] = self.realtime_df.loc[self.hsf_fct_time_index, col_ch]*0.5
            #     self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
            #     self.rainfall_df[iii] = f_df.Value
            # self.rainfall_df.replace(np.nan, 0, inplace=True)
            
    def Diffuser_t0_setting(self):
        hsf_first_time = self.hsf_fct_time_index[0].strftime('%m/%d/%Y %H:%M:%S') 
        for rrr in self.OUTLET_list:
            self.inp[sections.CONTROLS][rrr]= Control(name= rrr, 
                                conditions=[Control._Condition(logic='IF', object_kind='SIMULATION', label=np.nan,
                                                        attribute='TIME', relation='=', value=hsf_first_time)], 
                                actions_if=[Control._Action(kind='ORIFICE', label=rrr, action='SETTING', 
                                                    relation='=', value='0.0')], actions_else=[], priority=0)    

 
    def INT_setting(self, sim_type, raw_df):
        int_df = raw_df[self.INTERCEPTOR_ORIFICE_ch_list].ffill()
        for ccc, iii in zip(self.INTERCEPTOR_ORIFICE_ch_list, self.INTERCEPTOR_ORIFICE_list):
            c_open_list = []
            c_close_list = []
            for idx, val in zip(int_df.index,int_df[ccc].tolist()) :
                if val == 1:
                    c_open_list = c_open_list + [Control._Condition(logic='OR', object_kind='SIMULATION', label=np.nan, 
                                        attribute='TIME', relation='=', value= idx.strftime('%m/%d/%Y %H:%M:%S')) ]
                elif  val == 0:
                    c_close_list = c_close_list + [Control._Condition(logic='OR', object_kind='SIMULATION', label=np.nan, 
                                        attribute='TIME', relation='=', value= idx.strftime('%m/%d/%Y %H:%M:%S')) ]
        
            if len(c_open_list) >0 :
                c_open_list[0]['logic'] = 'IF'
                self.inp[sections.CONTROLS]['%s_%s_opn' %(iii,sim_type)] = Control(name='%s_%s_opn' %(iii,sim_type), 
                                conditions=c_open_list, 
                                actions_if=[Control._Action(kind='ORIFICE', label= iii, action='SETTING', 
                                                    relation='=', value='1.0')], actions_else=[], priority=0)
            if len(c_close_list) >0 :
                c_close_list[0]['logic'] = 'IF'
                self.inp[sections.CONTROLS]['%s_%s_clo' %(iii,sim_type)] = Control(name='%s_%s_clo' %(iii,sim_type), 
                            conditions=c_close_list, 
                            actions_if=[Control._Action(kind='ORIFICE', label= iii, action='SETTING', 
                                                relation='=', value='0.0')], actions_else=[], priority=0)
                    
        for in_name in self.INTERCEPTOR_ORIFICE_list:
            self.inp[sections.ORIFICES][in_name]['has_flap_gate'] = True

    def action_log_set(self): # target set log
        past_time_index = pd.date_range(start=self.time_dict['hsf_s_time'], end=self.time_dict['fct_s_time'] - pd.to_timedelta('00:10:00') , freq='10T')
        ch_name_1 = ['景美_P1抽水機運轉', '松山抽水機DP1', '昆陽抽水站 #1抽水泵 運轉/停止'] 
        ch_name_2 = ['士林紓流站_P1_抽水機運轉', '松信紓流站_P1_抽水機運轉', '新建紓流站_P1P2_抽水機運轉','忠孝紓流站_P1_抽水機運轉', 'B43抽水機1', 'B43抽水機2',\
                     'B43抽水機3', 'B43抽水機4', '六館電動閘門_全關']
        eng_name_1 = ['pump_GINMEI', 'pump_SONSHAN', 'pump_QUENYAN']
        eng_name_2 = ['pump_SULIN', 'pump_SONSHIN', 'pump_SINJAN', 'pump_ZUNSHAO', 'pump_B43_1', 'pump_B43_2',\
                      'pump_B43_3', 'pump_B43_4', 'ori_LOQUAN']
        # self=a 
        for time_idx in past_time_index:
            # open_num = (self.pump_df.loc[time_idx + pd.to_timedelta('00:10:00'), [f'迪化抽水機{idx+1}' for idx in np.arange(9)]] != 0).sum()
            open_num = (self.pump_df.loc[time_idx , [f'迪化抽水機{idx+1}' for idx in np.arange(9)]] != 0).sum()
            self.action_history_log.loc[time_idx, 'pump_open_num'] = open_num
            for eng, ch in zip(self.INTERCEPTOR_ORIFICE_list + eng_name_1 + eng_name_2, self.INTERCEPTOR_ORIFICE_ch_list + ch_name_1 + ch_name_2):
                # self.action_history_log.loc[time_idx, eng] = self.realtime_df.loc[time_idx + pd.to_timedelta('00:10:00'), ch]
                self.action_history_log.loc[time_idx, eng] = self.realtime_df.loc[time_idx , ch]
            self.action_history_log.loc[time_idx, ['ori_3031', 'ori_3041']] = 1.0
            self.action_history_log.loc[time_idx, 'ori_LOQUAN'] = 0.0
            self.action_history_log.loc[time_idx, 'ori_HURU'] = 0.0
        # abc = self.action_history_log['ori_LOQUAN']  ori_HURU
        
        

    def Diffuserup_setting(self, time_dict):
        self.inp[sections.CONTROLS]['Difup1'] = Control(name='Difup1', 
                            conditions=[Control._Condition(logic='IF', object_kind='SIMULATION', label=np.nan, attribute='TIME', 
                                                relation='>', value=time_dict['hsf_e_time'].strftime('%m/%d/%Y %H:%M:%S') )], 
                            actions_if=[
                                        # Control._Action(kind='PUMP', label='PUMP_DH1', action='STATUS', relation='=', value='ON'),
                                        # Control._Action(kind='PUMP', label='PUMP_DH2', action='STATUS', relation='=', value='ON'), 
                                        # Control._Action(kind='PUMP', label='PUMP_DH3', action='STATUS', relation='=', value='ON'), 
                                        # Control._Action(kind='PUMP', label='PUMP_DH4', action='STATUS', relation='=', value='ON'), 
                                        # Control._Action(kind='PUMP', label='PUMP_DH5', action='STATUS', relation='=', value='ON'), 
                                        # Control._Action(kind='PUMP', label='PUMP_DH6', action='STATUS', relation='=', value='ON'),
                                        Control._Action(kind='PUMP', label='PUMP_GINMEI', action='STATUS', relation='=', value='OFF'), 
                                        Control._Action(kind='PUMP', label='PUMP_SONSHAN', action='STATUS', relation='=', value='OFF'), 
                                        Control._Action(kind='PUMP', label='PUMP_QUENYAN', action='STATUS', relation='=', value='OFF'),
                                        Control._Action(kind='PUMP', label='PUMP_MUCHA', action='STATUS', relation='=', value='OFF')],
                            actions_else=[], priority=1)

    def PUMP_EMG_setting(self,time_dict):
        self.inp[sections.CONTROLS]['EMG1'] = Control(name='EMG1', 
                        conditions=[Control._Condition(logic='IF', object_kind='SIMULATION', label=np.nan, attribute='TIME', 
                                                        relation='>', value=time_dict['hsf_e_time'].strftime('%m/%d/%Y %H:%M:%S'))], 
                        actions_if=[Control._Action(kind='PUMP', label='PUMP_EMG_IN', action='STATUS', relation='=', value='ON'), 
                                    Control._Action(kind='PUMP', label='PUMP_EMG_OUT', action='STATUS', relation='=', value='ON'),
                                    Control._Action(kind='PUMP', label='PUMP-B43_OUT1', action='STATUS', relation='=', value='ON'),
                                    Control._Action(kind='PUMP', label='PUMP-B43_OUT2', action='STATUS', relation='=', value='ON'),
                                    Control._Action(kind='PUMP', label='PUMP-B43_OUT3', action='STATUS', relation='=', value='ON'),
                                    Control._Action(kind='PUMP', label='PUMP-B43_OUT4', action='STATUS', relation='=', value='ON')], 
                        actions_else=[], priority=0)
        

    def Difdown_setting(self,time_dict):
        self.inp[sections.CONTROLS][' Difdown'] = Control(name=' Difdown', 
                        conditions=[Control._Condition(logic='IF', object_kind='SIMULATION', label=np.nan, attribute='TIME', 
                                                        relation='>', value=time_dict['hsf_e_time'].strftime('%m/%d/%Y %H:%M:%S'))], 
                        actions_if=[Control._Action(kind='PUMP', label='PUMP_SULIN', action='STATUS', relation='=', value='ON'), 
                                    Control._Action(kind='PUMP', label='PUMP_SONSHIN', action='STATUS', relation='=', value='ON'), 
                                    Control._Action(kind='PUMP', label='PUMP_SINJAN', action='STATUS', relation='=', value='ON'), 
                                    Control._Action(kind='PUMP', label='PUMP_ZUNSHAO', action='STATUS', relation='=', value='ON'),
                                    Control._Action(kind='ORIFICE', label='HURU-Outlet', action='SETTING', relation='=', value='1.0'),
                                    Control._Action(kind='ORIFICE', label='LOQUAN-Outlet', action='SETTING', relation='=', value='1.0')],
                        actions_else=[], priority=0)
        
    def pump_0_setting(self):
        for pn in self.pump_name:
            if pn in ['PUMP-B43_OUT1', 'PUMP-B43_OUT2', 'PUMP-B43_OUT3', 'PUMP-B43_OUT4', 'PUMP_EMG_IN1', 'PUMP_EMG_IN2', 'PUMP_EMG_IN3', 'PUMP_EMG_OUT']:
                pass
            else:
                self.inp[sections.PUMPS][pn]['depth_on'] = 0
                self.inp[sections.PUMPS][pn]['depth_off'] = 0


    def pump_history_setting(self):
        out_df  = pd.read_excel(self.iotable_path,sheet_name='out')
        self.pump_name_df = out_df.loc[out_df.Type2=='pumps',['Object Name','Remark']]
        self.pump_df = self.realtime_raw_df.loc[self.time_dict['hsf_s_time']:self.time_dict['fct_e_time'],\
                                                self.pump_name_df[self.pump_name_df['Remark'].notna()]['Remark'].to_list()]
        # hs_pump = self.pump_name_df.loc[self.pump_name_df['Remark'].notna(),'Object Name'].to_list()
        # for i, name in enumerate(self.pump_df.columns):
        #     tmp_opn = []
        #     tmp_clo = []
        #     for time in self.pump_df.index:
        #         if self.pump_df.loc[time, name] > 0 and not pd.isnull(self.pump_df.loc[time, name]):
        #             tmp_opn.append(Control._Condition(logic='OR', object_kind='SIMULATION', label=np.nan, 
        #                                 attribute='TIME', relation='=', value= time.strftime('%m/%d/%Y %H:%M:%S')) )
        #         elif self.pump_df.loc[time, name] == 0 and not pd.isnull(self.pump_df.loc[time, name]):
        #             tmp_clo.append(Control._Condition(logic='OR', object_kind='SIMULATION', label=np.nan, 
        #                                 attribute='TIME', relation='=', value= time.strftime('%m/%d/%Y %H:%M:%S')) )
        #     if tmp_opn:
        #         tmp_opn[0]['logic'] = 'IF'
        #         self.inp[sections.CONTROLS][f'history_pump_opn_{hs_pump[i]}'] = Control(f'history_pump_opn_{hs_pump[i]}',
        #                 conditions=tmp_opn, 
        #                 actions_if=[Control._Action(kind='PUMP', label=hs_pump[i], action='STATUS', relation='=', value='ON'), ],
        #                 actions_else=[], priority=0)

        #     if tmp_clo:
        #         tmp_clo[0]['logic'] = 'IF'
        #         self.inp[sections.CONTROLS][f'history_pump_clo_{hs_pump[i]}'] = Control(f'history_pump_clo_{hs_pump[i]}',
        #                 conditions=tmp_clo, 
        #                 actions_if=[Control._Action(kind='PUMP', label=hs_pump[i], action='STATUS', relation='=', value='OFF'), ],
        #                 actions_else=[], priority=0)
                

    def pump_init_set(self, time_dict):
        self.inp[sections.PUMPS]['PUMP_GINMEI']['status'] = 'ON'
        self.inp[sections.PUMPS]['PUMP_SONSHAN']['status'] = 'ON'
        self.inp[sections.PUMPS]['PUMP_QUENYAN']['status'] = 'ON'
        # self.inp[sections.CONTROLS]['pump_init_set'] = Control(name='pump_init_set', 
        #                 conditions=[Control._Condition(logic='IF', object_kind='SIMULATION', label=np.nan, attribute='TIME', 
        #                                                 relation='>', value=time_dict['hsf_s_time'].strftime('%m/%d/%Y %H:%M:%S'))], 
        #                 actions_if=[Control._Action(kind='PUMP', label='PUMP_GINMEI', action='STATUS', relation='=', value='ON'), 
        #                             Control._Action(kind='PUMP', label='PUMP_SONSHAN', action='STATUS', relation='=', value='ON'), 
        #                             Control._Action(kind='PUMP', label='PUMP_QUENYAN', action='STATUS', relation='=', value='ON')],
        #                 actions_else=[], priority=0)




    def pump_power_set(self):
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA1']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA2']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA3']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA4']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA5']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA6']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA7']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA8']['kind'] = 'PUMP4'
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA9']['kind'] = 'PUMP4'

        # base_lv = 1.3
        # base_Mlv = 2.3
        # base_MHlv = 2.4  #2.9   
        # base_Hlv = 2.6 #2.5  dc 0.5
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA1']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [1.59, base_Mlv], [1.665	, base_Mlv], [1.765, base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA2']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [1.765, base_Mlv], [1.84	, base_Mlv], [1.94 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA3']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [1.915, base_Mlv], [1.99	, base_Mlv], [2.09 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA4']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.065, base_Mlv], [2.14	, base_Mlv], [2.24 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA5']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.215, base_Mlv], [2.29	, base_Mlv], [2.39 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA6']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.365, base_Mlv], [2.44	, base_Mlv], [2.54 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA7']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.515, base_Mlv], [2.59	, base_Mlv], [2.69 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA8']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.665, base_Mlv], [2.74	, base_Mlv], [2.84 , base_Hlv]]
        # self.inp[sections.CURVES]['PUMP_CURVE_DIHWA9']['points'] = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [2.815, base_Mlv], [2.89	, base_Mlv], [2.99 , base_Hlv]]
        # if self.DIHWA_tank.head <=-11:
        #     self.base_Qmax = 1.8
        #     self.base_Hlv = 1.8
        #     self.base_Mlv = 1.5
        #     self.base_Llv = 1.5
        # elif self.DIHWA_tank.head <=-10.5:
        #     self.base_Qmax = 2.2
        #     self.base_Hlv = 2.2
        #     self.base_Mlv = 1.9
        #     self.base_Llv = 1.9   
        # elif self.DIHWA_tank.head <=-10:
        #     self.base_Qmax = 2.5
        #     self.base_Hlv = 2.5
        #     self.base_Mlv = 2.2
        #     self.base_Llv = 2.2             
        # else:
        #     self.base_Qmax = 2.7
        #     self.base_Hlv = 2.7
        #     self.base_Mlv = 2.4
        #     self.base_Llv = 2.4
        # 水深 抽水量
        # tar_pump_power = [[0, 0], [1.0899, 0], [1.0899999999999999, base_lv], [1.59, base_Mlv], [1.665	, base_Mlv], [1.765, base_Hlv]]
        # tar_pump_power = [[0, 0], [3.01, 0], [3.010001, self.base_Qmax], [999,self.base_Qmax]]
        self.base_Hlv = 3
        self.base_Mlv = 2.7
        self.base_Llv = 2.5
        tar_pump_power = [[0, 0], [3.01, 0], [3.011, self.base_Hlv], [999, self.base_Hlv]]
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA1']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA2']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA3']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA4']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA5']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA6']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA7']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA8']['points'] = tar_pump_power
        self.inp[sections.CURVES]['PUMP_CURVE_DIHWA9']['points'] = tar_pump_power
        # 1.09   12.89 
        # -14.81  
        # -11.8 +14.81
    def time_index_set(self):
        self.hsf_fct_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['fct_e_time'],freq='10min')
        self.hsf_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['hsf_e_time'],freq='10min')
        self.fct_time_index = pd.date_range(self.time_dict['fct_s_time'],self.time_dict['fct_e_time'],freq='10min')
   
    def orifice_history_set(self):
        self.o_df = self.realtime_df.loc[:,['迪化SLG-3031主閘門全關', '迪化SLG-3041主閘門全關']]

    def storage_set(self):
        self.inp[sections.CURVES]['Storage_DIHWA']['points'] = [[0, 10], [0.01, 360.1], [8.81, 360.1], [8.82, 360.1]]
        # self.inp[sections.CURVES]['Storage_DIHWA']['points'] = [[0, 720], [0.01, 720], [8.81, 720], [8.82, 720]]
        
    def try_discharge_coef(self, dc):
        self.inp[sections.ORIFICES]['DIHWA_IN_3031']['discharge_coefficient'] = dc  ##123
        self.inp[sections.ORIFICES]['DIHWA_IN_3041']['discharge_coefficient'] = dc  ###123
        pass

    def try_infiltration(self, imper_ratio, CN):
        names = [
                    'AF_3944-0226', 'BJ_3947-0085', 'DA_4158-0019', 'AE_3741-0150', 'DD_3360-0083', 
                    'DD_3461-0005', 'DD_3559-0001', 'DE_3753-0004', 'BK_3642-0562', 'BK_3744-1148', 
                    'DC_3760-0003', 'DC_3761-0004', 'AD1_A04', 'AE_3842-0120', 'BK_3847-0002', 
                    'BK_3849-0065', 'DE1_3753-0004', 'D_3853-0055', 'DC_3857-0008', 'DC_3858-0004', 
                    'DC_3860-0059', 'DC_3862-0616', 'BJ1_3950-0164', 'BJ_3949-1077', 'CB_3953-0411', 
                    'DC_3955-0019', 'DC_3857-0044', 'DC_3956-0007', 'A_4041-0139', 'AD_A07', 
                    'AD_4044-0583', 'BI_4046-0853', 'BI_4049-0020', "CA'_4054-0013", 'CA_4055-0074', 
                    'BF_4141-0067', 'BH_4147-0007', 'BH_4148-0922', 'BG_4150-0007', 'DA_4156-0108', 
                    'DB_4157-0104', 'DB_4159-0021', 'DA1_4258-0404', 'DA2_4258-0404', 'DA3_4258-0404', 
                    'DA4_4258-0404', 'A_4236-0830', 'BF_4242-0003', 'BF_4244-0412', 'BF_4246-0105', 
                    'BF_4248-0830', 'BG_4250-0012', 'BE_4341-0047', 'BF_4343-0043', 'BF_4343-0776', 
                    'BE_4344-0579', 'BE_4346-0636', 'AC_4433-0059', 'A_4437-0082', 'BE_4443-0127', 
                    'BD_4446-0021', 'BD_4447-0064', 'AB_4534-0733', 'AC_4435-0023', 'A_4536-0073', 
                    'BE_4543-0104', 'BE_4543-0136', 'BD1_4549-0163', 'A_4635-0405', 'BE_4642-0057', 
                    'BC_4644-0024', 'B_4646-0983', 'AA_4734-0076', 'AA1_4734-0076', 'AA2_4734-0076', 
                    'A1_4737-0010', 'A_4737-0010', 'BC_4745-0106', 'BB_4845-0956', 'B_4847-0021', 
                    'BB_4945-0071', 'B_5047-0024', 'B_5147-0036', 'BA_5245-0015', 'BA_5246-0022', 
                    'sub5449-1', 'sub5348-3', 'subs5449-1', 'sub5550-2', 'sub5750-1', 'sub5851-1', 
                    'subs5850-1', 'sub5850-1', 'sub5650-3', 'sub5448-2', 'sub5650-2', 'sub5952-3', 
                    'sub5749-42', 'sub5549-3', 'sub5247-1', 'E_4251-0005'
                    ]
        for name in names:
            self.inp[sections.SUBCATCHMENTS][name]['imperviousness'] = imper_ratio[name[:2]]
            self.inp[sections.INFILTRATION][name]['curve_no'] = str(CN[name[:2]])

    def time_dict_correct(self):
        self.time_dict['fct_e_time'] = self.time_dict['fct_e_time'] - relativedelta(hours=1/6)

    # def out_inp(self, time_dict, strategy, save_name, save_path, dc, imper_ratio, CN):
                            
    def out_inp(self, time_dict, strategy, oripath, save_name, save_path, dc, Rfct_type,rain_bqpf_df):
        self.time_dict = time_dict
        self.oripath = oripath
        self.save_path = save_path
        self.strategy = strategy
        self.option_setting(self.time_dict)  #設定模式 絕對時間 和步長
        self.pump_init_set(self.time_dict)
        self.storage_set()
        self.time_dict_correct()  ###
        self.pump_power_set()
        self.try_discharge_coef(dc)
        # self.storage_init_set()
        self.dc = dc
        # self.try_infiltration(imper_ratio, CN)
        self.save_name = save_name
        self.time_index_set()
        # self.base_flow = self.basic_flow_cal()
        self.realtime_df = self.realtime_raw_df.loc[self.time_dict['hsf_s_time']:self.time_dict['fct_e_time'],:].bfill().ffill()
        self.realtime_hsf_df = self.realtime_raw_df.loc[self.time_dict['hsf_s_time']:self.time_dict['hsf_e_time'],:]
        
        # aaaa = self.realtime_hsf_df
        
        # self.realtime_raw_df['迪化SLG-3031主閘門全關']
        
        # 145/6
        # self.realtime_fct_df = self.realtime_raw_df.loc[self.time_dict['fct_s_time']:self.time_dict['fct_e_time'],:]
        self.realtime_fct_df = pd.DataFrame( columns = self.realtime_hsf_df.columns, index = self.fct_time_index)
        self.orifice_history_set()
        
        self.Rfct_type =  Rfct_type #未來雨量得選擇   0使用上帝視角雨量     1 使用NCDR雨量
        # hsf_e_time
        if self.Rfct_type == 1:
            rain_bqpf_df.index = pd.to_datetime(rain_bqpf_df.index)
            bqpf_col = rain_bqpf_df.columns
            temp_idx = list(set(list(rain_bqpf_df.index)).intersection(set(list(self.fct_time_index))))
            temp_idx.sort()
            self.realtime_fct_df.loc[temp_idx,:] = rain_bqpf_df.loc[temp_idx,:]
            if self.realtime_fct_df.index[-1].minute != 0:
                temp_time_add1h = self.realtime_fct_df.index[-1].replace(minute=0) + relativedelta(hours=1)  
                # time_ragne = pd.date_range(self.realtime_fct_df.index[-1], temp_time_add1h,freq='10min')[1:]
                # for rrr in time_ragne:
                    # self.realtime_fct_df.loc[rrr] = np.nan
              
                self.realtime_fct_df.loc[temp_time_add1h,bqpf_col] = rain_bqpf_df.loc[temp_time_add1h,bqpf_col]
                self.realtime_fct_df.loc[:,bqpf_col] = self.realtime_fct_df.loc[:,bqpf_col].bfill()
                self.realtime_fct_df.loc[:,bqpf_col] = self.realtime_fct_df.loc[:,bqpf_col] /6
                self.realtime_fct_df = self.realtime_fct_df.drop(pd.to_datetime(temp_time_add1h),axis=0)
            else:
                self.realtime_fct_df.loc[:,bqpf_col] = self.realtime_fct_df.loc[:,bqpf_col].bfill()
                self.realtime_fct_df.loc[:,bqpf_col] = self.realtime_fct_df.loc[:,bqpf_col] /6
        elif self.Rfct_type == 0: #使用觀測值  #要再修改
            self.realtime_fct_df = self.realtime_raw_df.loc[time_dict['fct_s_time']:time_dict['fct_e_time'],:]

        for iii in self.Sunday_inflow_list:
            tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
            self.a = tmp
            f_df = pd.DataFrame(index=self.hsf_fct_time_index,columns=['Time','Value','H'])
            f_df['Time'] = self.hsf_fct_time_index.strftime('%m/%d/%Y %H:%M:%S') 
            f_df['H'] = self.hsf_fct_time_index.hour
            f_df['Value'] = tmp.loc[f_df['H'],1].tolist()
            self.b = f_df
            self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)            
        print ('Strategy: %s' %(self.strategy) )
        print ('Load based file: %s' %(self.inp_path) )   
        inwt_set = self.realtime_fct_df.copy()

        if self.strategy == 99 : #歷史事件
            self.pump_history_setting()
            self.pump_0_setting()
            #3. inp設定雨量======================   
            type_sunwet = 1
            self.R_setting(type_sunwet)

            #4. inp設定紓流站 初始時間開關======================
            #關閉時間 '00:00:00'
            self.Diffuser_t0_setting()

            #5. inp設定 截流站開啟和關閉======================     
            # 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            
            # 預報 截流站設定
            #下面這行可能要ˇ改
            # self.realtime_fct_df = self.realtime_raw_df.loc[time_dict['fct_s_time']:time_dict['fct_e_time'],:]
            self.realtime_fct_df = self.realtime_raw_df.loc[self.fct_time_index,:]
            self.INT_setting('fct', self.realtime_fct_df)
            pass
        
        
            #4 紓流站停止抽水機  設定==================           
            self.Diffuserup_setting(self.time_dict)

            #5 啟動迪化緊急進流及紓流抽水站 設定==================           
            # self.PUMP_EMG_setting(self.time_dict)

            #6 啟動士林、松信、新建、忠孝、六館及葫蘆國小紓流站紓流 設定==================                 
            self.Difdown_setting(self.time_dict)



        elif self.strategy == 0 :  #無雨 無截流 R0_INT0_Difup0_EMG0_Difdown0 2024/2/20 21~
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 0
            self.R_setting(type_sunwet)

            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()

            #3. inp預報 截流站設定===0無截流==1截流全開===============     
            #3.1 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定===0無截流==1截流全開===============    
            inwt_set.loc[:,:] = 0
            self.INT_setting('fct', inwt_set)

        elif self.strategy == 1 :  #無雨 有截流 R0_INT1_Difup0_EMG0_Difdown0
            self.pump_history_setting()
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 1
            self.R_setting(type_sunwet)
            
            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()
            
            #3. inp預報 截流站設定 入流全開
            #3.1 熱啟動 截流站設定
            # self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定 ===0無截流==1截流全開===============   
            # inwt_set.loc[:,:] = 1
            # self.INT_setting('fct', inwt_set)

        elif  self.strategy == 2 :    #有雨 無截流  R1_INT0_Difup1_EMG0_Difdown0
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 1 
            self.R_setting(type_sunwet)

            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()

            #3. inp預報 截流站設定 入流全開
            #3.1 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定 ===0無截流==1截流全開===============   
            inwt_set.loc[:,:] = 0
            self.INT_setting('fct', inwt_set)


        elif  self.strategy == 3 :  # #R1_INT0_Difup1_EMG0_Difdown0   迪化抽水站啟動第6台，上游景美、松山、昆陽抽水站及內湖廠，啟動紓流模式
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 1 
            self.R_setting(type_sunwet)
            
            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()
            
            #3. inp預報 截流站設定 入流全開
            #3.1 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定 ===0無截流==1截流全開===============   
            inwt_set.loc[:,:] = 0
            self.INT_setting('fct', inwt_set)
            
            #4 紓流站停止抽水機  設定==================           
            self.Diffuserup_setting(time_dict)


        elif  self.strategy ==  4:  #R1INT0DIF1EMG1 #啟動迪化緊急進流及紓流抽水站
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 1 
            self.R_setting(type_sunwet)
            
            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()
            
            #3. inp預報 截流站設定 入流全開
            #3.1 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定 ===0無截流==1截流全開===============   
            inwt_set.loc[:,:] = 0
            self.INT_setting('fct', inwt_set)
            
            #4 紓流站停止抽水機  設定==================           
            self.Diffuserup_setting(time_dict)

            #5 啟動迪化緊急進流及紓流抽水站 設定==================           
            self.PUMP_EMG_setting(time_dict)

        elif  self.strategy == 5 :  #R1_INT0_Difup1_EMG1_Difdown1  #啟動士林、松信、新建、忠孝、六館及葫蘆國小紓流站紓流
            self.pump_0_setting()
            #1. inp設定雨量===0無雨==1有雨===============        
            type_sunwet = 1 
            self.R_setting(type_sunwet)
            
            #2. inp設定紓流站 初始時間開關====
            self.Diffuser_t0_setting()
            
            #3. inp預報 截流站設定 入流全開
            #3.1 熱啟動 截流站設定
            self.INT_setting('hsf', self.realtime_hsf_df)
            #3.2 預報 截流站設定 ===0無截流==1截流全開===============   
            inwt_set.loc[:,:] = 0
            self.INT_setting('fct', inwt_set)
            
            #4 紓流站停止抽水機  設定==================           
            self.Diffuserup_setting(time_dict)

            #5 啟動迪化緊急進流及紓流抽水站 設定==================           
            self.PUMP_EMG_setting(time_dict)

            #6 啟動士林、松信、新建、忠孝、六館及葫蘆國小紓流站紓流 設定==================                 
            self.Difdown_setting(time_dict)
            
        elif  self.strategy == 6 :  
            pass

        elif  self.strategy == 8 :  
            inptmp = SwmmInput.read_file("8_test.inp",encoding='big5')            
            self.inp[sections.CONTROLS] = inptmp[sections.CONTROLS]    
          
        # self.save_name = save_name
        # self.save_path = save_path
        
        # self=a
        if self.sys == 'windows':
            self.new_inp = "%s.inp" %(self.save_name)
        elif self.sys == 'linux':
            if __name__ == '__main__':
                self.new_inp = "test.inp"
            else:
                self.new_inp = "New.inp"  #%(self.save_name)
        self.inp.write_file(os.path.join(self.save_path,self.new_inp), encoding='big5')
        self.inp = self.original_inp.copy()

    
    def indicator_identify(self):
        # def cal_rain(rainfall_df):
        #     Rtmpall = rainfall_df.iloc[:,0:50].rolling(6).sum()
        #     Rtmp = Rtmpall.loc[self.hsf_fct_time_index] #.max(axis=1)
        #     Ridx = Rtmp.max(axis=1).idxmax()
        #     Rname = Rtmp.idxmax(axis=1)
        #     Rnamemax = Rname.loc[Ridx]
            
        #     # abbb = Rtmpall[Rnamemax].rolling(24*6).sum() /6
        #     tf_24h80 = (Rtmpall[Rnamemax].rolling(24*6).sum() /6 >=80).to_list()
        #     tf_1h40 = (Rtmpall[Rnamemax] >=40).tolist()
        #     tmp3h5 = Rtmpall[Rnamemax] >=5
        #     tf_3h5 = np.logical_and(tmp3h5,tmp3h5.shift(1),tmp3h5.shift(2)).tolist()
        #     tf_24h200 = (Rtmpall[Rnamemax].rolling(24*6).sum() /6 >=200).to_list()
        #     tf_3h100 = (Rtmpall[Rnamemax].rolling(3*6).sum() /6 >=100).to_list()
        #     indicator_df = pd.DataFrame(columns=['mode'], index=self.hsf_fct_time_index)
        #     indicator_df.loc[:,:] = '晴天模式'

        #     tmp_list2 = np.logical_or.reduce([tf_24h80,tf_1h40,tf_3h5],axis=0).tolist()
        #     if True in tmp_list2:
        #         True_first2 = tmp_list2.index(True)
        #         True_last2 = len( tmp_list2)-  tmp_list2[::-1].index(True)-1
        #         indicator_df.loc[self.hsf_fct_time_index[True_first2:True_last2],'mode'] = '大雨特報紓流模式'

        #         for i in range(True_first2, True_last2+1):
        #             if len(self.hsf_fct_time_index) > i >= 18:
        #                 two_hour_rain = Rtmpall[Rnamemax][i-18:i].sum()
        #                 if two_hour_rain < 1.5:
        #                     indicator_df.loc[self.hsf_fct_time_index[i], 'mode'] = '晴天模式'
            
        #     tmp_list = np.logical_or.reduce([tf_24h200,tf_3h100],axis=0).tolist()
        #     if True in tmp_list:
        #         True_first = tmp_list.index(True)
        #         True_last = len( tmp_list) - tmp_list[::-1].index(True)-1
        #         indicator_df.loc[self.hsf_fct_time_index[True_first:True_last],'mode'] = '豪雨特報紓流模式'
        #     return indicator_df
        
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
            indicator_df.loc[:, :] = '晴天模式'
            tmp_list2 = np.logical_or.reduce([tf_24h80, tf_1h40, tf_3h5], axis=0).tolist()

            if True in tmp_list2:
                True_first2 = tmp_list2.index(True)
                True_last2 = len(tmp_list2) - tmp_list2[::-1].index(True) - 1
                indicator_df.loc[self.hsf_fct_time_index[True_first2:True_last2], 'mode'] = '大雨特報紓流模式'

                for i in range(True_first2, True_last2+1):
                    if len(self.hsf_fct_time_index) > i >= 18:
                        two_hour_rain = Rtmpall[Rnamemax][i-18:i].sum()
                        if two_hour_rain < two_hour_threshold:
                            indicator_df.loc[self.hsf_fct_time_index[i], 'mode'] = '晴天模式'

            tmp_list = np.logical_or.reduce([tf_24h200, tf_3h100], axis=0).tolist()
            
            if True in tmp_list:
                True_first = tmp_list.index(True)
                True_last = len(tmp_list) - tmp_list[::-1].index(True) - 1
                indicator_df.loc[self.hsf_fct_time_index[True_first:True_last], 'mode'] = '豪雨特報紓流模式'
            print(indicator_df.loc[self.sim.current_time, 'mode'])
            return indicator_df

        def current_pump_open_num():
            return sum(pump.current_setting != 0 for pump in self.pumps)
        def current_pump_open_tarnum():
            return sum([ self.pump_setting[pump]  for pump in self.pumps])
        
        def indicator_transform(indicator):
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
        
        # rainfall_df = self.rainfall_df
        
        self.indicator_df = cal_rain(self.rainfall_df)
        self.indicator_df.loc[:,'indicator'] = np.nan
        current_open_num = current_pump_open_num()
        self.current_pump_open_tarnum = current_pump_open_tarnum()
        self.grad_wlv = self.DIHWA_tank.head - self.previous_wlv
        slope_wlv, _ = np.polyfit(np.arange(5), self.wlv_log[-5:], 1)
        print('slope_wlv',round(slope_wlv,2))
        self.grad_diffuser_wlv = self.node3850_0313S.head - self.previous_diffuser_wlv
        self.indicator = {indi:None for indi in ['0', '1', '2-1', '2-2', '3', '4', '5']}
        self.indicator['0'] = True
        print(current_open_num,round(self.current_pump_open_tarnum,2), round(self.DIHWA_tank.head,2), round(self.grad_wlv,2),self.base_Mlv)
        try:
            if (self.indicator_df.loc[self.sim.current_time:(self.sim.current_time + relativedelta(minutes=180)),'mode'] == '大雨特報紓流模式').any():
                self.indicator['1'] = True
        except KeyError:
            pass
        if current_open_num >= 5 and self.DIHWA_tank.head >= -10.1 and slope_wlv > -0.2 and self.indicator['1'] == True:
            self.indicator['2-1'] = True
        if self.indicator_df.loc[self.sim.current_time,'mode'] == '豪雨特報紓流模式':
            self.indicator['2-1'] = True
            self.indicator['2-2'] = True
        if current_open_num >= 6 and self.DIHWA_tank.head >= -10.1 and slope_wlv > -0.3:
            self.indicator['3'] = True
        if current_open_num >= 7 and self.DIHWA_tank.head >= -10.1 and slope_wlv > -0.3 and self.indicator['3'] == True:
            self.indicator['4'] = True
        if current_open_num >= 7 and self.DIHWA_tank.head >= -10.0 and slope_wlv > -0.3 and self.indicator['4'] == True:
            self.indicator['5'] = True
        # print(self.indicator)
        if (self.node3850_0313S.head > -8 and current_open_num >= 6) or(self.DIHWA_tank.head > -8.5 and current_open_num >= 4):
            self.indicator['3'] = True
        for indi, tf in reversed(self.indicator.items()):
            if tf == True:
                self.indicator_log_df.loc[self.sim.current_time,'indicator'] = indicator_transform(indi)
                return indi
    # aaaab=self.rainfall_df           #14:30 大雨
# aaaa=self.Result_df
    def indicator_algorithm(self, indicator, Kp, Ki, Kd):
        pumps_iter = iter(reversed(self.pumps))
        # cnt_p  = 1  
        # increase_ratio_pump = 0.3
        # decrease_ratio_orifice = 0.1
        def decrease_level_algo(increase_ratio_pump, decrease_ratio_orifice, pid, p_lowlim=1):
            current_number = sum(value != 0 for value in self.pump_setting.values())
            for pump in pumps_iter: 
                cnt_p = self.pumps.index(pump) + 1
                # if self.ori3031.current_setting + self.ori3041.current_setting < 2:
                #     self.pump_setting[pump] = 1
                #     if cnt_p==1: print('-1.特別限制 OP:閘門關閉，強制開啟9台')
                # if self.current_rain_intensity > 10 and self.pump_setting[pump] != 0 and current_number < 9:
                #     self.pump_setting[pump] = 1
                #     self.pump_setting[self.pumps[self.pumps.index(pump) + 1]] = 1
                #     break
                if self.DIHWA_tank_head_imperfect > -9.2 and self.grad_wlv > 0.5 and self.pump_setting[pump] != 0:
                    if current_number < 8:
                        rest_pump = 9 - current_number
                        self.pump_setting[pump] = 1
                        for kkk in range(1,rest_pump+1): #可能會六台 就關閘門
                            self.pump_setting[self.pumps[self.pumps.index(pump) + kkk]] = 1
                        # self.pump_setting[self.pumps[self.pumps.index(pump) + 2]] = 1

                        current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                        target_ori_total_open = current_ori_total_open - decrease_ratio_orifice if current_ori_total_open - decrease_ratio_orifice > 0.8 else 0.8
                        self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                        self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                        print('-1.1.COND:閘門增加%s為3031:%s, 3041:%s' %(round(decrease_ratio_orifice,2), round(self.ori3031.target_setting,2), round(self.ori3041.target_setting,2)))
                    
                    # if self.ori3031.current_setting != 0:
                    #     print('-1.1.COND:主閘門3031不等於0； OP:3031閘門減小%s為%s' %(decrease_ratio_orifice, self.ori3031.current_setting - decrease_ratio_orifice))
                    #     self.ori3031.target_setting = self.ori3031.current_setting - decrease_ratio_orifice if\
                    #         self.ori3031.current_setting - decrease_ratio_orifice >= 0 else 0
                    # elif self.ori3031.current_setting == 0 and self.ori3041.current_setting != 0:
                    #     print('-1.2.COND:主閘門3031等於0，3041大於0.3； OP:3041閘門減小%s為%s' %(decrease_ratio_orifice, (self.ori3041.current_setting - decrease_ratio_orifice)))
                    #     self.ori3041.target_setting = self.ori3041.current_setting - decrease_ratio_orifice if\
                    #         self.ori3041.current_setting - decrease_ratio_orifice >= 0.8 else 0.8
                    break
                elif self.pump_setting[pump] != 0 and self.pump_setting[self.pumps[8]] == 0:
                    if current_number <= 7 and self.pump_setting[pump] + increase_ratio_pump <= 1:
                        self.pump_setting[pump] = self.pump_setting[pump] + increase_ratio_pump # change flow rate gradually if pump 6 or pump 7 are not full load
                        print('-2.1. COND:第6-7台抽水機尚未滿載； OP::第%s台增加馬力%s' %(cnt_p,increase_ratio_pump))
                    elif current_number < 7 and self.pump_setting[pump] + increase_ratio_pump > 1:
                        print('-2.2.COND:總台數小於7，第%s台增加%s超過1' %(cnt_p,increase_ratio_pump))
                        if self.indicator in ('0', '1') and self.pump_cd == 0:
                            self.pump_setting[pump] = 1 # increase load to max
                            if current_number <= self.MaxPumpNum['1']:
                                # self.pump_setting[self.pumps[self.pumps.index(pump) + 1]] = self.base_Mlv / self.base_Hlv # open next pump
                                self.pump_setting[self.pumps[self.pumps.index(pump) + 1]] = 1 # open next pump
                                self.pump_cd = 6
                                # self.pump_cd = 0  if self.DIHWA_tank.head >-10 else 6
                                print('-2.2.1.1.COND:總台數小於7，指標1，第%s台增加%s超過1，cd=0； OP:拉滿%s台抽水機，並開啟第%s台抽水機base' %(cnt_p,increase_ratio_pump,cnt_p,cnt_p+1))
                            else:
                                print('-2.2.1.2.COND:指標%s，已達最大台數%s台； OP:不做動作' %( self.indicator,self.MaxPumpNum['1']))
                        elif self.indicator in ('0', '1') and self.pump_cd != 0:
                            self.pump_setting[pump] = 1 # increase load to max
                            print('-2.2.2.COND:總台數小於7，指標1，第%s台增加%s超過1，cd!=0； OP:拉滿%s台抽水機' %(cnt_p,increase_ratio_pump,cnt_p))                            
                        elif self.indicator not in ('0', '1'):
                            self.pump_setting[pump] = 1 # increase load to max
                            if current_number <= self.MaxPumpNum[self.indicator]:
                                # self.pump_setting[self.pumps[self.pumps.index(pump) + 1]] = self.base_Mlv / self.base_Hlv # open next pump
                                self.pump_setting[self.pumps[self.pumps.index(pump) + 1]] = 1 # open next pump
                                self.pump_cd = 6
                                # self.pump_cd = 0  if self.DIHWA_tank.head >-10 else 6
                                
                                print('-2.2.3.1.COND:總台數小於7，指標1，第%s台增加%s超過1，cd=0； OP:拉滿%s台抽水機，並開啟第%s台抽水機base' %(cnt_p,increase_ratio_pump,cnt_p,cnt_p+1))
                            else:
                                print('-2.2.3.2.COND:指標%s，已達最大台數%s台； OP:不做動作' %( self.indicator,self.MaxPumpNum['1']))
                        
                        if self.ori3031.current_setting + self.ori3041.current_setting < 2:
                            if self.pid_type == 0:
                                current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                                target_ori_total_open = current_ori_total_open - decrease_ratio_orifice if current_ori_total_open - decrease_ratio_orifice > 0.5 else 0.5
                                self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                                self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                                print('-2.2.35.COND:閘門增加%s為3031:%s, 3041:%s' %(decrease_ratio_orifice, self.ori3031.target_setting, self.ori3041.target_setting))
                            
                            elif self.pid_type in (1, 2):
                                current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                                target_ori_total_open = current_ori_total_open + pid if current_ori_total_open + pid > 0.5 else 0.5
                                self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                                self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                                print('-2.2.35.COND:閘門增加%s為3031:%s, 3041:%s' %(pid, self.ori3031.target_setting, self.ori3041.target_setting))
                        
                        try: #前兩小時 還是大雨的話  無CD時間
                            if (self.indicator_df.loc[self.sim.current_time:(self.sim.current_time + relativedelta(minutes=30)),'mode'] == '大雨特報紓流模式').any():
                                self.pump_cd = 0
                                print('-2.2.4.COND:大雨特報紓流模式的話CD為0')
                        except KeyError:
                            pass
                    elif current_number == 7 and self.pump_setting[pump] + increase_ratio_pump > 1:
                        print('-2.3.COND:已開啟7台； OP:拉滿第7台，並開啟第8台')
                        self.pump_setting[self.pumps[6]] = 1 # increase 7th load to max
                        self.pump_setting[self.pumps[7]] = 1 # open pump 8
                    elif self.grad_diffuser_wlv > 0.2:
                        print('-2.4.COND:Others； OP:開啟第8、第9台')
                        self.pump_setting[self.pumps[7]] = 1 # increase 8th load to max
                        self.pump_setting[self.pumps[8]] = 1 # open pump 9
                    break
                elif self.pump_setting[self.pumps[8]] == 1:
                    if self.pid_type == 0:
                        print('-3.COND:9台全開')
                        current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                        target_ori_total_open = current_ori_total_open - decrease_ratio_orifice if current_ori_total_open - decrease_ratio_orifice > 0.5 else 0.5
                        self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                        self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                        print('-3.1.COND:閘門增加%s為3031:%s, 3041:%s' %(decrease_ratio_orifice, self.ori3031.target_setting, self.ori3041.target_setting))
                    
                    elif self.pid_type in (1, 2):
                        print('-3.COND:9台全開')
                        current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                        target_ori_total_open = current_ori_total_open + pid if current_ori_total_open + pid > 0.5 else 0.5
                        self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                        self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                        print('-3.1.COND:閘門增加%s為3031:%s, 3041:%s' %(pid, self.ori3031.target_setting, self.ori3041.target_setting))

                    break

                else: # all pump close
                    self.pump_setting[self.pumps[0]] = 1
                    # self.pump_setting[self.pumps[1]] = 1
            
                

        def increase_level_algo(decrease_ratio_pump, increase_ratio_orifice, pid):
            for pump in pumps_iter:
                cnt_p = self.pumps.index(pump) + 1
                if (self.ori3041.current_setting + self.ori3031.current_setting != 2) and self.indicator != '0' and self.diffuser_head_imperfect > -9.5:
                    if self.pid_type == 0:
                        current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                        target_ori_total_open = current_ori_total_open + increase_ratio_orifice if current_ori_total_open + increase_ratio_orifice > 0.5 else 0.5
                        self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                        self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                        print('+1.COND:閘門增加%s為3031:%s, 3041:%s' %(round(increase_ratio_orifice,2), round(self.ori3031.target_setting,2), round(self.ori3041.target_setting,2)))
                    elif self.pid_type in (1, 2):
                        current_ori_total_open = self.ori3031.current_setting + self.ori3041.current_setting
                        target_ori_total_open = current_ori_total_open + pid if current_ori_total_open + pid > 0.5 else 0.5
                        self.ori3041.target_setting = 1 if target_ori_total_open > 1 else target_ori_total_open
                        self.ori3031.target_setting = target_ori_total_open - 1 if target_ori_total_open > 1 else 0
                        print('+1.COND:閘門增加%s為3031:%s, 3041:%s' %(round(pid,2), round(self.ori3031.target_setting,2), round(self.ori3041.target_setting,2)))
                    break
                elif pump in (self.PUMP_DH9, self.PUMP_DH8) and pump.current_setting != 0: # close 8, 9th pump
                    # if -9.5 - self.DIHWA_tank.head > 0.2:
                    if self.diffuser_head_imperfect < -9.5:
                        print(f'+3.1.COND: 緊繞水深{round(self.node3850_0313S.head, 2)}；第8第9台抽水機不等於0，3031全開； OP:關閉第8第9台抽水機' )
                        self.pump_setting[self.pumps[7]] = 0
                        self.pump_setting[self.pumps[8]] = 0
                        # if self.ori3031.current_setting != 1 and self.ori3041.current_setting != 1:
                        #     self.ori3041.target_setting = self.ori3041.current_setting + increase_ratio_orifice if\
                        #     self.ori3041.current_setting + increase_ratio_orifice <= 1 else 1
                        #     print('有1')
                    elif self.diffuser_head_imperfect < -9.2:
                        print('+3.2.COND:；第8第9台抽水機不等於0，3031全開； OP:關閉第8台抽水機' )
                        self.pump_setting[self.pumps[8]] = 0
                        # if self.ori3031.current_setting != 1 and self.ori3041.current_setting != 1:
                        #     self.ori3041.target_setting = self.ori3041.current_setting + increase_ratio_orifice if\
                        #     self.ori3041.current_setting + increase_ratio_orifice <= 1 else 1
                        #     print('有2')
                    break
                elif self.pump_setting[pump] != 0:
                    print('+4.COND:；第%s台抽水機不等於0' %(cnt_p))
                    if self.current_pump_open_tarnum - decrease_ratio_pump <= 1 + self.base_Llv / self.base_Hlv:
                    # if self.current_pump_open_tarnum - decrease_ratio_pump <= 1:
                        self.pump_setting[pump] = self.base_Llv / self.base_Hlv
                        # self.pump_setting[pump] = 1
                        print('+4.1.COND:指標%s，減小至最小台數%s台； OP:減小至最小台數or不能再減少' %( self.indicator,round(self.base_Llv / self.base_Hlv,2)))
                    elif self.pump_setting[pump] - decrease_ratio_pump >= self.base_Llv / self.base_Hlv: # decrease flow rate
                        print('+4.2.COND:第%s台抽水機減少%s抽水量 大於 開啟量(base)； OP:第%s台抽水機減少%s抽水量' %(cnt_p,decrease_ratio_pump,cnt_p,decrease_ratio_pump) )    
                        self.pump_setting[pump] -= decrease_ratio_pump
                        
                    elif self.indicator in ('0', '1', '2-1', '2-2') and self.pumps.index(pump) < 4 and \
                        (self.indicator_df.loc[(self.sim.current_time + relativedelta(minutes=60)):self.sim.current_time,'mode'].isin(['大雨特報紓流模式', '豪雨特報紓流模式'])).any() :
                            print('+4.2.1')
                            
                    else:# close pump
                        if self.pumps.index(pump) > 4 and self.DIHWA_tank_head_imperfect < -10:
                            if self.indicator in ('0', '1', '2-1', '2-2') and self.pump_cd == 0:
                                self.pump_setting[pump] = 0
                                self.pump_setting[self.pumps[self.pumps.index(pump) - 1]] = 0
                                self.pump_cd = 6
                                print('+4.3.1.1.COND:水位低開高，指標為1、2-1、2-2且CD為0； OP:關閉第%s, %s台抽水機' % (cnt_p, cnt_p-1))
                            elif self.indicator not in ('0', '1', '2-1', '2-2'):
                                self.pump_setting[pump] = 0
                                self.pump_setting[self.pumps[self.pumps.index(pump) - 1]] = 0
                                print('+4.3.1.2.COND:水位低開高，指標不是1、2-1、2-2； OP:關閉第%s, %s台抽水機' % (cnt_p, cnt_p-1))
                        else:
                            if self.indicator in ('0', '1', '2-1', '2-2') and self.pump_cd == 0:
                                self.pump_setting[pump] = 0
                                self.pump_cd = 6
                                print('+4.3.2.1.COND:指標為1、2-1、2-2且CD為0； OP:關閉第%s台抽水機' % (cnt_p))
                            elif self.indicator not in ('0', '1', '2-1', '2-2'):
                                self.pump_setting[pump] = 0
                                print('+4.3.2.2.COND:指標不是1、2-1、2-2； OP:關閉第%s台抽水機' % (cnt_p))

                        try: #前兩小時 還是大雨的話  無CD時間
                            # if (self.indicator_df.loc[(self.sim.current_time - relativedelta(minutes=30)):self.sim.current_time,'mode'] == '大雨特報紓流模式').any():
                            if self.indicator_df.loc[self.sim.current_time,'mode'] == '大雨特報紓流模式':
                                print('+4.3.3.COND:大雨特報紓流模式的話CD為0' )
                                self.pump_cd = 0
                        except KeyError:
                            pass

                        
                    if self.ori3031.current_setting != 1 and self.ori3041.current_setting == 1:
                        if self.pid_type == 0:
                            self.ori3031.target_setting = self.ori3031.current_setting + increase_ratio_orifice if\
                            self.ori3031.current_setting + increase_ratio_orifice <= 1 else 1
                        elif self.pid_type in (1, 2):
                            self.ori3031.target_setting = self.ori3031.current_setting + increase_ratio_orifice if\
                            self.ori3031.current_setting + increase_ratio_orifice <= 1 else 1
                    break
          

                
        def indi_1_measures(state):
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
# self= a
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
        
        def indi_0_control(current_pump_index, current_pump_ratio):
            self.pump_setting[self.pumps[current_pump_index-1]] = current_pump_ratio
            for i in range(0, current_pump_index-1): 
                self.pump_setting[self.pumps[i]] = 1                       
            for i in range(current_pump_index, 9):
                self.pump_setting[self.pumps[i]] = 0

        def pid_ratio_cal(total, kp, ki, kd):
            p = total * kp / (kp + ki + kd)
            i = total * ki / (kp + ki + kd)
            d = total * kd / (kp + ki + kd)
            return p, i, d
            
        def pid_auto_tune(target_level):
            if (np.abs(self.previous_grad_wlv) > 1.5 or np.abs(self.grad_wlv) > 1.5) and np.sign(self.previous_grad_wlv) != np.sign(self.grad_wlv):
                if all(lv > target_level for lv in [self.DIHWA_tank_head_imperfect, self.previous_wlv]) or\
                   all(lv < target_level for lv in [self.DIHWA_tank_head_imperfect, self.previous_wlv]):
                    self.p = self.p + 0.02 if self.p + 0.02 < 0.1 else 0.1
                    print('1.1')
                else:
                    self.p = self.p - 0.08 if self.p - 0.08 > 0.01 else 0.01
                    self.i = self.d - 0.04 if self.i - 0.04 > 0.01 else 0.01
                    self.d = self.d - 0.06 if self.d - 0.06 > 0.01 else 0.01
                    print('1.2')
            elif (np.abs(self.previous_grad_wlv) > 0.5 or np.abs(self.grad_wlv) > 0.5) and np.sign(self.previous_grad_wlv) != np.sign(self.grad_wlv):
                if all(lv > target_level for lv in [self.DIHWA_tank_head_imperfect, self.previous_wlv]) or\
                   all(lv < target_level for lv in [self.DIHWA_tank_head_imperfect, self.previous_wlv]):
                    self.p = self.p + 0.02 if self.p + 0.02 < 0.1 else 0.1
                    print('2.1')
                else:
                    self.p = self.p - 0.06 if self.p - 0.06 > 0.01 else 0.01
                    self.i = self.i - 0.03 if self.i - 0.03 > 0.01 else 0.01
                    self.d = self.d - 0.05 if self.d - 0.05 > 0.01 else 0.01
                    print('2.2')
            elif np.abs(self.DIHWA_tank_head_imperfect - target_level) > 0.2:
                if np.abs(self.grad_wlv) > 1.5:
                    self.p = self.p + 0.08 if self.p + 0.08 < 0.1 else 0.1
                    self.i = self.i - 0.04 if self.i - 0.04 > 0.01 else 0.01
                    self.d = self.d - 0.05 if self.d - 0.05 > 0.01 else 0.01
                    # self.d = self.d + 0.04 if self.d + 0.04 < 0.15 else 0.15
                    print('3.1')
                elif np.abs(self.grad_wlv) > 1:
                    self.p = self.p + 0.05 if self.p + 0.05 < 0.1 else 0.1
                    self.i = self.i - 0.03 if self.i - 0.03 > 0.01 else 0.01
                    self.d = self.d - 0.04 if self.d - 0.04 > 0.01 else 0.01
                    # self.d = self.d + 0.04 if self.d + 0.04 < 0.15 else 0.15
                    print('3.2')
                elif np.abs(self.grad_wlv) > 0.5:
                    self.p = self.p + 0.02 if self.p + 0.02 < 0.1 else 0.1
                    self.i = self.i - 0.02 if self.i - 0.02 > 0.01 else 0.01
                    self.d = self.d - 0.03 if self.d - 0.03 > 0.01 else 0.01
                    # self.d = self.d + 0.04 if self.d + 0.04 < 0.15 else 0.15
                    print('3.3')
                else:
                    self.p = self.p + 0.02 if self.p + 0.02 < 0.1 else 0.1
                    self.i = self.i - 0.02 if self.i - 0.02 > 0.01 else 0.01
                    self.d = self.d - 0.02 if self.d - 0.02 > 0.01 else 0.01
                    # self.d = self.d + 0.04 if self.d + 0.04 < 0.15 else 0.15
                    print('3.4')
            elif np.abs(self.DIHWA_tank_head_imperfect - target_level) < 0.2 and np.abs(self.previous_wlv - target_level) < 0.2:
                self.p = self.p - 0.01 if self.p - 0.01 > 0.01 else 0.01
                self.d = self.d + 0.02 if self.d + 0.02 < 0.1 else 0.1
                print('4.1')
            if np.abs(self.previous_grad_wlv) < 0.1 and np.abs(self.grad_wlv) < 0.1:
                self.i = self.i + 0.02 if self.i + 0.02 < 0.05 else 0.05
            print('p_grad, grad, p, i, d',round(self.previous_grad_wlv,2), round(self.grad_wlv,2), round(self.p,2), round(self.i,2), round(self.d,2))

            
           
        
        if indicator == '0':
            indi_1_measures('off')
            indi_3_measures('off')
            indi_4_measures('off')
            if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=30*1)):self.sim.current_time,'indicator'].isin( ['5','4','3'])).any():    
                target_level = -9
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1 if self.DIHWA_tank_head_imperfect > -11 else 0
                self.target_cd = 3
            elif self.previous_target == -9.5:
                target_level = -9
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1 if self.DIHWA_tank_head_imperfect> -11 else 0
            else:    
                target_level = -9
                buff_in = 0.1
                buff_de = 0.5
            # target_level = -9.5
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            print('0目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )

            if level_diff > 0.35 + buff_in:
                increase_level_algo(0.3, 0.3, 0)
            elif level_diff > 0.25 + buff_in and level_diff < 0.35 + buff_in:
                increase_level_algo(0.2, 0.2, 0)
            elif level_diff > 0.2 + buff_in :
                increase_level_algo(0.1, 0.1, 0)   
            elif level_diff < -0.3 - buff_de :
                decrease_level_algo(0.2, 0.2, 0,self.Rain_plowlim)
            elif level_diff > -0.3 - buff_de and level_diff < -0.25 - buff_de:
                decrease_level_algo(0.1, 0.1, 0,self.Rain_plowlim)
            else:
                pass

            if self.DIHWA_tank_head_imperfect < -11.8 and sum(value != 0 for value in self.pump_setting.values()) >= 5:
                pumps_iter = iter(reversed(self.pumps))
                for pump in pumps_iter:
                    if self.pump_setting[pump] != 0:
                        print('+4.3.4.COND:緊急關')
                        self.pump_setting[pump] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 1]] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 2]] = 0
                        break
            

            # self.previous_index = index
            # if TIME(16,0) <= self.sim.current_time.time() <= TIME(23,59,59):
            #     for inv_name in ['SUANYAN_IN', 'GINMEI_IN', 'SINJAN_IN']:
            #         self.link_object[inv_name].target_setting = 0
        
        elif indicator == '1' :
            indi_1_measures('on')
            indi_3_measures('off')
            indi_4_measures('off')
            # self.rainfall_df.iloc[:,0].isin['5','4','3','2-1','2-2'] .isin(('5','4','3','2-1','2-2'))
            # 前六小時 指標超過 2  後面要慢慢下降 不要馬上變回 從tar-9 拉回 -10.5  這樣會突然開很多台  先設定緩降成-10
            # if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*6)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2','1'])).any():
            # if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=30)):self.sim.current_time,'indicator'].isin( ['0'])).any():    
            #     target_level = -9.5
            #     buff_in = 0.5 if self.DIHWA_tank.head > -11 else 0
            #     buff_de = 0.5 if self.DIHWA_tank.head > -11 else 0
                
            if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*2)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1 if self.DIHWA_tank_head_imperfect > -11 else 0
            elif (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*3)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5
                buff_in = 0.4 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 0.9 if self.DIHWA_tank_head_imperfect > -11 else 0
            elif (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*4)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5
                buff_in = 0.3 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 0.8 if self.DIHWA_tank_head_imperfect > -11 else 0
            elif (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*5)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5
                buff_in = 0.2 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 0.7 if self.DIHWA_tank_head_imperfect > -11 else 0
            elif (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*6)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5 
                buff_in = 0.1 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 0.6 if self.DIHWA_tank_head_imperfect > -11 else 0
            elif (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=60*10)):self.sim.current_time,'indicator'].isin( ['5','4','3','2-1','2-2'])).any():    
                target_level = -10.5      
                buff_in = 0.1 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
            else: 
                target_level = -10.5
                buff_in = 0  #用來微調的
                buff_de = 0.5

            if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=30*1)):self.sim.current_time,'indicator'].isin( ['5','4','3'])).any():    
                target_level = -10
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1.3 if self.DIHWA_tank_head_imperfect > -11 else 0
                self.target_cd = 3
            elif self.previous_target in  (-9, -10) and self.target_cd != 0:
                target_level = -10
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                self.target_cd -= 1
            else:    
                pass
            
            # target_level = -9
            current_number = sum(value != 0 for value in self.pump_setting.values())
            # self.previous_grad_wlv
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            print('1目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )
            #目前抽水機數小於 雨量限制的最小抽水機數
            # if current_number < self.Rain_plowlim:
            #     decrease_level_algo(0.2, 0.2, 0, self.Rain_plowlim)
            #     print('try:1') level_diff = -1.43
            if level_diff > 0.35 + buff_in:
                if self.previous_grad_wlv < 0:
                    increase_level_algo(0.3, 0.3, 0)
            elif level_diff > 0.25 + buff_in and level_diff < 0.35 + buff_in:
                if self.previous_grad_wlv < 0:
                    increase_level_algo(0.2, 0.2, 0)
            elif level_diff > 0.2 + buff_in :
                if self.previous_grad_wlv < 0:
                    increase_level_algo(0.1, 0.1, 0)
            elif level_diff < -0.3 - buff_de :
                if self.previous_grad_wlv > 0.01 or self.DIHWA_tank_head_imperfect > -9:   
                    print('add-1',buff_de,buff_in)
                    decrease_level_algo(0.2, 0.2, 0, self.Rain_plowlim)
            elif level_diff > -0.3 - buff_de and level_diff < -0.25 - buff_de:
                if self.previous_grad_wlv > 0.01 or self.DIHWA_tank_head_imperfect > -9 :    
                    print('add-2',buff_de,buff_in)
                    decrease_level_algo(0.1, 0.1, 0, self.Rain_plowlim)
            else:
                pass

            if self.DIHWA_tank_head_imperfect < -11.8 and sum(value != 0 for value in self.pump_setting.values()) >= 5:
                pumps_iter = iter(reversed(self.pumps))
                for pump in pumps_iter:
                    if self.pump_setting[pump] != 0:
                        print('+4.3.4.COND:緊急關')
                        self.pump_setting[pump] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 1]] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 2]] = 0
                        break
            print('indicator: %s; TarWl_diff: %s' %(indicator, round(self.DIHWA_tank.head - target_level,2)))
            self.previous_target = target_level


        elif indicator in ('2-1', '2-2'):
            if (self.indicator_log_df.loc[(self.sim.current_time - relativedelta(minutes=30*1)):self.sim.current_time,'indicator'].isin( ['5','4','3'])).any():    
                target_level = -9.5
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1.3 if self.DIHWA_tank_head_imperfect > -11 else 0
                self.target_cd = 3
            elif self.previous_target in  (-9.5, -10) and self.target_cd != 0:
                target_level = -10
                buff_in = 0.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1.5 if self.DIHWA_tank_head_imperfect > -11 else 0
                self.target_cd -= 1
            else:    
                target_level = -10.5
                buff_in = 0 if self.DIHWA_tank_head_imperfect > -11 else 0
                buff_de = 1 if self.DIHWA_tank_head_imperfect > -11.3 else 0
                
            
            indi_1_measures('on'),
            indi_3_measures('off')
            indi_4_measures('off')
            # target_level = -9.5
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            print('2目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )

            if level_diff > 0.3 + buff_in:
                increase_level_algo(0.4, 0.3, 0)
            elif level_diff > 0.1 + buff_in and level_diff < 0.3 + buff_in:
                increase_level_algo(0.2, 0.2, 0)
            elif self.DIHWA_tank_head_imperfect < -11 and level_diff < 0.1 + buff_in:
                increase_level_algo(0.2, 0.2, 0)
            elif level_diff < -1.2: #-9.3
                decrease_level_algo(0.1, 0.8, 0, self.Rain_plowlim)
                print('迴圈3')
            elif level_diff < -0.1 - buff_de:
                decrease_level_algo(0.2, 0.1, 0, self.Rain_plowlim)
            elif level_diff > -0.1 - buff_de and level_diff < -0.05 - buff_de:
                decrease_level_algo(0.2, 0.1, 0, self.Rain_plowlim)
            else:
                pass

            if self.DIHWA_tank_head_imperfect < -11.8 and sum(value != 0 for value in self.pump_setting.values()) >= 5:
                pumps_iter = iter(reversed(self.pumps))
                for pump in pumps_iter:
                    if self.pump_setting[pump] != 0:
                        print('+4.3.4.COND:緊急關')
                        self.pump_setting[pump] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 1]] = 0
                        self.pump_setting[self.pumps[self.pumps.index(pump) - 2]] = 0
                        break

            self.previous_target = target_level
            # if self.DIHWA_tank.head < -11 and self.grad_wlv < -0.35:
            #     print('grad_wlv',self.grad_wlv)
            #     self.pump_cd -= 3 if self.pump_cd - 3 > 0 else 0


        elif indicator =='3':
            indi_1_measures('on')
            indi_3_measures('on')
            indi_4_measures('off')
            target_level = -9.5
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            total = 0.1
            if self.pid_type == 0:
                pid = 0
            elif self.pid_type == 1:
                kp, ki, kd = pid_ratio_cal(total, Kp, Ki, Kd)
                pid = self.pid_control(kp, ki, kd, target_level)
            elif self.pid_type == 2:
                pid_auto_tune(-9)
                pid = self.pid_control(self.p, self.i, self.d, -9)
            
            print('3目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )

            if level_diff > 0.5 :
                increase_level_algo(0.1, 0.4, pid)
                print('迴圈0')
            elif level_diff > 0.3:
                increase_level_algo(0.0, 0.2, pid)
                print('迴圈1')
            elif level_diff > 0.2 and level_diff < 0.3:
                increase_level_algo(0.00, 0.1, pid)
                print('迴圈2')
            elif level_diff < -1 or  self.previous_grad_wlv > 0.7:
                decrease_level_algo(0.5, 0.8, pid, self.Rain_plowlim)
                print('迴圈3')
            elif level_diff < -1 or  self.previous_grad_wlv > 0.5:
                decrease_level_algo(0.1, 0.8, pid, self.Rain_plowlim)
                print('迴圈3')
            elif level_diff > -1 and level_diff < -0.75:
                decrease_level_algo(0.05, 0.15, pid, self.Rain_plowlim)
                print('迴圈4')
            elif level_diff > -0.75 and level_diff < -0.5:
                decrease_level_algo(0.05, 0.1, pid, self.Rain_plowlim)
                print('迴圈4-1')
            elif level_diff > -0.5 and level_diff < -0.1:
                decrease_level_algo(0.05, 0.05, pid, self.Rain_plowlim)
                print('迴圈4-2')
            # elif level_diff > -0.1 and level_diff < -0.05:
            #     decrease_level_algo(0.05, 0.05, self.Rain_plowlim)
            #     print('迴圈5')
            else:
                print('迴圈7')
                pass
            self.previous_target = target_level
            self.target_cd = 3

        elif indicator == '4':
            indi_1_measures('on')
            indi_3_measures('on')
            indi_4_measures('on')
            target_level = -9.5
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            total = 0.1
            # if self.DIHWA_tank.head > -7:
            #     total += 0.08 if total + 0.08 < 0.2 else 0.25
            #     print('a1', total)
            # elif self.DIHWA_tank.head > -7.5:
            #     total += 0.05 if total + 0.05 < 0.2 else 0.25
            #     print('a2', total)
            # elif self.DIHWA_tank.head > -8:
            #     total += 0.02 if total + 0.02 < 0.2 else 0.25
            #     print('a3', total)

            # if np.abs(self.grad_wlv) > 3:
            #     total -= 0.05 if total - 0.05 > 0.05 else 0.05
            #     print('a4')
            # elif np.abs(self.grad_wlv) > 2:
            #     total -= 0.03 if total - 0.03> 0.05 else 0.05
            #     print('a5')
            # elif np.abs(self.grad_wlv) > 1:
            #     total -= 0.01 if total - 0.01 > 0.05 else 0.05
            #     print('a6')

            if self.pid_type == 0:
                pid = 0
            elif self.pid_type == 1:
                kp, ki, kd = pid_ratio_cal(total, Kp, Ki, Kd)
                pid = self.pid_control(kp, ki, kd, target_level)
            elif self.pid_type == 2:
                pid_auto_tune(-9)
                pid = self.pid_control(self.p, self.i, self.d, -9)
            # pid_auto_tune(-9)
            # pid = self.pid_control(self.p, self.i, self.d, -9)
            # kp, ki, kd = pid_ratio_cal(total, 8, 1, 3)
            # pid = self.pid_control(kp, ki, kd, -8.5)
            print('4目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )
            if level_diff > 0.4:
                increase_level_algo(0.1, 0.4, pid)
                print('迴圈0')
            elif 0.4 > level_diff > 0.3:
                increase_level_algo(0.1, 0.3, pid)
                print('迴圈1')
            elif level_diff > 0.1 and level_diff < 0.3:
                increase_level_algo(0.05, 0.1, pid)
                print('迴圈2')
            elif level_diff < -1: #-8.5
                decrease_level_algo(0.1, 0.8, pid, self.Rain_plowlim)
                print('迴圈3')
            elif level_diff > -1 and level_diff < -0.75:
                decrease_level_algo(0.05, 0.15, pid, self.Rain_plowlim)
                print('迴圈4')
            elif level_diff > -0.75 and level_diff < -0.5:
                decrease_level_algo(0.05, 0.05, pid, self.Rain_plowlim)
                print('迴圈4-1')
            elif level_diff > -0.5 and level_diff < -0.1:
                decrease_level_algo(0.05, 0.05, pid, self.Rain_plowlim)
                print('迴圈4-2')
            # elif level_diff > -0.1 and level_diff < -0.05:
            #     decrease_level_algo(0.05, 0.05, self.Rain_plowlim)
            #     print('迴圈5')
            else:
                print('迴圈7')
                pass
            self.previous_target = target_level
            self.target_cd = 3

        elif indicator == '5':
            indi_1_measures('on')
            indi_3_measures('on')
            indi_4_measures('on')
            target_level = -9.5
            level_diff = target_level - self.DIHWA_tank_head_imperfect
            total = 0.1
            # total = 0.1
            # if self.DIHWA_tank.head > -7:
            #     total += 0.08 if total + 0.08 < 0.2 else 0.25
            #     print('a1', total)
            # elif self.DIHWA_tank.head > -7.5:
            #     total += 0.05 if total + 0.05 < 0.2 else 0.25
            #     print('a2', total)
            # elif self.DIHWA_tank.head > -8:
            #     total += 0.02 if total + 0.02 < 0.2 else 0.25
            #     print('a3', total)

            # if np.abs(self.grad_wlv) > 3:
            #     total -= 0.05 if total - 0.05 > 0.05 else 0.05
            #     print('a4', total)
            # elif np.abs(self.grad_wlv) > 2:
            #     total -= 0.03 if total - 0.03> 0.05 else 0.05
            #     print('a5', total)
            # elif np.abs(self.grad_wlv) > 1:
            #     total -= 0.01 if total - 0.01 > 0.05 else 0.05
            #     print('a6', total)

            if self.pid_type == 0:
                pid = 0
            elif self.pid_type == 1:
                kp, ki, kd = pid_ratio_cal(total, Kp, Ki, Kd)
                pid = self.pid_control(kp, ki, kd, target_level)
            elif self.pid_type == 2:
                pid_auto_tune(-9)
                pid = self.pid_control(self.p, self.i, self.d, -9)
            # pid_auto_tune(-9)
            # pid = self.pid_control(self.p, self.i, self.d, -9)
            # kp, ki, kd = pid_ratio_cal(total, 8, 1, 3)
            # pid = self.pid_control(kp, ki, kd, -8.5)   
            buff_in = 0
            buff_de = 0.1 # 1
            print('5目標',round(target_level, 2),'差異',round(level_diff,2), '液位變化',round(self.previous_grad_wlv,2),self.base_Mlv )
            if level_diff > 0.4 + buff_in:
                increase_level_algo(0.1, 0.4, pid)
                print('迴圈0')
            elif level_diff > 0.3 + buff_in:
                increase_level_algo(0.1, 0.3, pid)
                print('迴圈1')
            elif level_diff > 0.2 + buff_in and level_diff < 0.3 + buff_in:
                increase_level_algo(0.05, 0.1, pid)
                print('迴圈2')
            elif level_diff < -1 - buff_de:
                decrease_level_algo(0.1, 0.8, pid, self.Rain_plowlim)
                print('迴圈3')
            elif level_diff > -1 - buff_de and level_diff < -0.75 - buff_de:
                decrease_level_algo(0.05, 0.8, pid, self.Rain_plowlim)
                print('迴圈4')
            elif level_diff > -0.75 - buff_de and level_diff < -0.5 - buff_de:
                decrease_level_algo(0.05, 0.05, pid, self.Rain_plowlim)
                print('迴圈4-1')
            elif level_diff > -0.5 - buff_de and level_diff < -0.1 - buff_de:
                decrease_level_algo(0.02, 0.05, pid, self.Rain_plowlim)
                print('迴圈4-2')
            # elif level_diff > -0.1 - buff_de and level_diff < -0.05 - buff_de:
            #     decrease_level_algo(0.05, 0.05, pid, self.Rain_plowlim)
            #     print('迴圈5')
            else:
                decrease_level_algo(0, 0, pid, self.Rain_plowlim)
                print('迴圈7')
                
            self.previous_target = target_level
            self.target_cd = 3

    def pid_control(self, kp, ki, kd, target):
        if self.sim.current_time == self.time_dict['fct_s_time']:
            self.integral = 0
            self.previous_error = 0
        dt = 1
        error = target - self.DIHWA_tank_head_imperfect
        self.integral += error * dt
        proportional = error
        derivative = (error - self.previous_error) / dt
        pid = kp * proportional + ki * self.integral + kd * derivative
        self.previous_error = error
        return pid

    def run(self, pid_type, Kp, Ki, Kd, imperfect_percent, random_seed):
        np.random.seed(random_seed)
        self.base_flow = self.basic_flow_cal(os.path.join(self.save_path,self.new_inp))
        self.sim = Simulation(os.path.join(self.save_path,self.new_inp))
        self.DIHWA_tank_levellog = []
        self.sim.step_advance(600)
        self.link_object = Links(self.sim)  # init link object
        # pump object
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
        self.PUMP_B43_1 = self.link_object['PUMP-B43_OUT1']
        self.PUMP_B43_2 = self.link_object['PUMP-B43_OUT2']
        self.PUMP_B43_3 = self.link_object['PUMP-B43_OUT3']
        self.PUMP_B43_4 = self.link_object['PUMP-B43_OUT4']

        # conduit object
        self.ori3031 = self.link_object['DIHWA_IN_3031']
        self.ori3041 = self.link_object['DIHWA_IN_3041']
        self.ori_LOQUAN = self.link_object['LOQUAN-Outlet']
        self.ori_HURU = self.link_object['HURU-Outlet']
        

        # node object
        node_object = Nodes(self.sim)
        self.DIHWA_tank = node_object['DIHWA']
        self.GINMEI_tank = node_object['GINMEI_TANK']
        self.ZUNZEN_tank = node_object['ZUNZEN_TANK']
        self.SONSHAN_tank = node_object['SONSHAN_TANK']
        self.QUENYAN_tank = node_object['QUENYAN_TANK']
        self.MUCHA_tank = node_object['MUCHA_TANK']
        self.TADURU_tank = node_object['TADURU_TANK']
        self.xizhi_tank = node_object['xizhi_TANK']
        self.node3850_0313S = node_object['3850-0313S'] #緊繞

        # raingage
        # Raingage_object = RainGages(self.sim)


        try:
            self.R17 = ['北投國小','陽明高中','太平國小','雙園','博嘉國小','中正國中','市政中心','留公國中',             
                   '桃源國中','奇岩','建國','民生國中','長安國小','台灣大學(新)','玉成','內湖','東湖國小']
            self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','A0A010','T15','C0A9F0','T014']
            self.R17_df = self.realtime_raw_df.loc[:, self.R17]
            # self.R17_df['mean'] = self.R17_df.mean(axis=1)
            # self.Result_df.loc[self.hsf_fct_time_index,nnn] =self.rainfall_df .loc[:,self.R17_id].rolling(6).sum().mean(axis=1)
        except:
            self.R17 = ['北投國小','陽明高中','太平國小','雙園','博嘉國小','中正國中','市政中心','瑠公國中',             
                   '桃源國中','奇岩','建國','民生國中','長安國小','台灣大學(新)','玉成','內湖','東湖國小']
         
            self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','T15','T014']
            self.R17_df = self.realtime_raw_df.loc[:, self.R17]
       # aaa = self.R17_df .mean(axis=1)

        cnt_sim=1
        # self.node3850_0313S.initial_depth = self.realtime_raw_df.loc[self.time_dict['hsf_s_time'], '緊急繞流井雷達波液位計'] - 12.39 + 12.39
        self.DIHWA_tank.initial_depth = 0
        self.integral = 0
        self.previous_error = 0
        self.sim.start()
        print(self.sim.current_time, self.DIHWA_tank.head)
        self.pumps = [self.PUMP_DH1, self.PUMP_DH2, self.PUMP_DH3, self.PUMP_DH4, self.PUMP_DH5,\
                                                self.PUMP_DH6, self.PUMP_DH7, self.PUMP_DH8, self.PUMP_DH9] 
        self.pump_setting = {pump:0 for pump in self.pumps}
        self.pump_cd = 0
        self.wlv_log = []
        self.indicator_log_df = pd.DataFrame(columns=['indicator'], index=self.hsf_fct_time_index)
        self.previous_target = -10
        self.target_cd = 0
        self.previous_ori_low = None
        self.p, self.i, self.d = 0.15, 0.01, 0.03
        self.previous_grad_wlv = 0

        self.MaxPumpNum = {'0':4,'1':5,'2-1':7,'2-2':7,'3':8,'4':9,'5':9}
        history_weight = 0
        self.base_pump= 1
        first_level = self.realtime_df.loc[self.time_dict['fct_s_time'], '迪化LT-1濕井液位高度'] - 12.89
        if first_level < -11.5:
            first_level = -11.5
        print('first_level', first_level) 
        start_modify  =False
        self.pid_type = pid_type # 0: none, 1: fixed, 2: adaptive

        integral = 0
        previous_error = 0     
        weight = 0.4
        while True:
            print('------------',self.sim.current_time,'--------------')
            if self.strategy != 99:
                self.R17sum = self.R17_df.mean(axis=1).loc[(self.sim.current_time - relativedelta(minutes=(60*1))):self.sim.current_time].sum()
                # print('R17:', self.R17sum)

                
                if self.pump_cd != 0:
                    self.pump_cd -= 1
                    
                
                if self.sim.current_time < self.time_dict['fct_s_time'] - relativedelta(minutes=10):
                    self.current_pump_open_tarnum = 4
                    if self.DIHWA_tank.head >= first_level and start_modify == False:
                        start_modify = True
                    if start_modify:
                        self.pump_setting[self.pumps[0]] = 1
                        self.pump_setting[self.pumps[1]] = 1
                        self.pump_setting[self.pumps[2]] = 1
                        self.pump_setting[self.pumps[3]] = 1
                        error = self.DIHWA_tank.head - first_level
                        integral += error * 1
                        proportional = error
                        derivative = (error - previous_error) / 1
                        pid = 0.25 * proportional + 0.0 * integral + 0.03 * derivative
                        weight += pid
                        # print(proportional, integral, derivative)
                        previous_error = error
                        self.current_pump_open_tarnum = 4
                           

                        if self.sim.current_time == self.time_dict['fct_s_time'] - relativedelta(minutes=20):
                            open_num = (self.realtime_df.loc[self.sim.current_time + pd.to_timedelta('00:10:00'), [f'迪化抽水機{idx+1}' \
                                                                                    for idx in np.arange(9)]] != 0).sum()
                            self.current_pump_open_tarnum = open_num
                            for idx in range(open_num):
                                self.pump_setting[self.pumps[idx]] = 1
                            for idx in range(open_num, 9):
                                self.pump_setting[self.pumps[idx]] = 0 
                            weight = 1
                        
                        for pump, ts in self.pump_setting.items():
                            pump.target_setting = ts * weight
                    
                    self.ori3031.target_setting = self.action_history_log.loc[self.sim.current_time, 'ori_3031']
                    self.ori3041.target_setting = self.action_history_log.loc[self.sim.current_time, 'ori_3041'] 
                    self.PUMP_GINMEI.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_GINMEI']
                    self.PUMP_SONSHAN.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_SONSHAN'] 
                    self.PUMP_QUENYAN.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_QUENYAN'] 
                    self.PUMP_SULIN.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_SULIN']
                    self.PUMP_SONSHIN.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_SONSHIN']
                    self.PUMP_SINJAN.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_SINJAN']
                    self.PUMP_ZUNSHAO.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_ZUNSHAO'] 
                    self.PUMP_B43_1.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_B43_1']
                    self.PUMP_B43_2.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_B43_2']
                    self.PUMP_B43_3.target_setting = self.action_history_log.loc[self.sim.current_time, 'pump_B43_3'] 
                    self.PUMP_B43_4.target_setting = self.action_history_log.loc[self.sim.current_time,'pump_B43_4'] 
                    
                    self.ori_LOQUAN.target_setting = self.action_history_log.loc[self.sim.current_time,'ori_LOQUAN']
                    
                    self.ori_HURU.target_setting = self.action_history_log.loc[self.sim.current_time,'ori_HURU'] 
                    for name in self.INTERCEPTOR_ORIFICE_list:
                        self.link_object[name].target_setting = self.action_history_log.loc[self.sim.current_time, name]

    


                    self.previous_wlv = self.DIHWA_tank.head
                    self.previous_diffuser_wlv = self.node3850_0313S.head
                    self.wlv_log.append(self.DIHWA_tank.head)
                    # self.previous_index = index
                    if self.sim.current_time == self.time_dict['fct_s_time'] - pd.to_timedelta('00:10:00') :
                        history_weight = 1
                    print('history', round(history_weight,2))
                    # for pump, ts in self.pump_setting.items():
                    #     pump.target_setting = ts * history_weight
                        # pump.target_setting = ts

                    if self.DIHWA_tank.head >= first_level and start_modify == False:
                        start_modify = True

                    if start_modify:
                        if self.DIHWA_tank.head >= first_level + 0.2:
                            history_weight = 1.5
                        elif self.DIHWA_tank.head >= first_level + 0.1:
                            history_weight = 1.2
                        elif self.DIHWA_tank.head >= first_level:
                            history_weight = 1
                        elif self.DIHWA_tank.head < first_level - 1:
                            history_weight = 0.5
                        elif self.DIHWA_tank.head < first_level:
                            history_weight = 0.6
                else:
                    self.indicator = self.indicator_identify()
                    print('indicator',self.indicator)
                    self.DIHWA_tank_head_imperfect = self.DIHWA_tank.head * np.random.uniform(1-imperfect_percent, 1+imperfect_percent)
                    # self.DIHWA_tank_head_imperfect = self.DIHWA_tank.head * np.random.uniform(1, 1)
                    # self.diffuser_head_imperfect = self.node3850_0313S.head * np.random.uniform(1-imperfect_percent, 1+imperfect_percent)
                    self.diffuser_head_imperfect = self.node3850_0313S.head * np.random.uniform(1, 1)
                    self.indicator_algorithm(self.indicator, Kp, Ki, Kd)


                    current_number = sum(value != 0 for value in self.pump_setting.values())
                    # R17sum = self.R17_df.mean(axis=1).loc[(self.sim.current_time - relativedelta(minutes=(60*1))):self.sim.current_time].sum()
                    data_in_range = self.R17_df.mean(axis=1).loc[(self.sim.current_time - relativedelta(minutes=(60*1))):self.sim.current_time]
                    random_factors = np.random.uniform(low=1-imperfect_percent, high=1+imperfect_percent, size=len(data_in_range))
                    # random_factors = np.random.uniform(low=1, high=1, size=len(data_in_range))
                    data_with_variation = data_in_range * random_factors
                    R17sum = data_with_variation.sum()
                    # R17sum = self.R17sum * np.random.uniform(0.95, 1.05)
                    if R17sum == 0: #沒下雨時持續記錄現在抽水機台數 紀錄起點
                        self.base_pump = current_number
                    elif open_num == 9:
                        self.base_pump = 1
                    self.Rain_plowlim = round(R17sum/10) -1  + self.base_pump 
                    print( 'R17sum',round(self.R17sum,1), 'plowlim',round(self.Rain_plowlim,1))
                    
                    if round(R17sum/10) -1  > 0:
                        # current_number_second = sum(value != 0 for value in self.pump_setting.values())
                        if current_number < self.Rain_plowlim:
                            for cnt in range(self.Rain_plowlim):
                                self.pump_setting[self.pumps[cnt]] = 1                    
                            print('-99.COND:未達到降雨(%s)時最低抽水機數，目前%s台，直接開滿%s台全速' %(round(self.R17sum,1),current_number,self.Rain_plowlim))

                    self.previous_wlv = self.DIHWA_tank.head
                    self.wlv_log.append(self.DIHWA_tank.head)
                    self.previous_diffuser_wlv = self.node3850_0313S.head
                    
                    pump_weight = 1




# y = -1630x2 - 27602x - 93039
# R² = 0.576
                    self.pump_type = 2
                    if self.pump_type == 1 :
                        if  current_number >= 4 and round(self.R17sum/10) -1  > 0:
                            if self.DIHWA_tank.head < -11.5:
                                pump_weight = 0.5
                            elif self.DIHWA_tank.head < -11:
                                pump_weight = 0.6
                            elif self.DIHWA_tank.head < -10.5:
                                pump_weight = 0.7
                            elif self.DIHWA_tank.head < -10.7:
                                pump_weight = 0.8   
                            elif self.DIHWA_tank.head < -10:
                                pump_weight = 0.95           
                    elif self.pump_type == 2 :
                        Hd = self.DIHWA_tank.head
                            # Hd = -9
                    # for current_number in range (2,10):
                        # Hd = -8
                            # current_number = 2
                            #3.2 2.8   0.4*3600  #1440    1440 / 30
                        if current_number == 2:
                            # y = -1630*Hd**2 - 27602*Hd - 93039  # R² = 0.576 #2
                            y = 651.07*Hd**3 + 16871*Hd**2 + 147166*Hd + 455791 #R² = 0.5783
                            ymax, ymin = 26728/2+200,  15272/2-200   
                        elif current_number==3:
                            y = 199.62*Hd**3 + 4995*Hd**2 + 45596*Hd + 185604   #R² = 0.5948  #3
                            ymax, ymin = 40140/3+200,  22727/3-500   
                        elif current_number == 4 :
                            y = -2.6306*Hd**4 + 452.61*Hd**3 + 12527*Hd**2 + 111122*Hd + 377371 #R² = 0.3714
                            ymax, ymin = 54923.5/4+200,  31000.9/4-200   
                            
                        elif current_number == 5 :
                            # y = -812.64*Hd**3 - 25017*Hd**2 - 245367*Hd - 716581 #R² = 0.9031 #5
                            y = -2387.8*Hd**2 - 36617*Hd - 78639  #R² = 0.8997 #5

                            ymax, ymin = 63869/5+300,  36749/5-300   
                        elif current_number == 6 :
                            y = 674.57*Hd**3 + 16595*Hd**2 + 138328*Hd + 454156 # R² = 0.6979 #6
                            ymax, ymin = 70920/6+400,  46442/6 -400                     
                        elif current_number == 7 :                            
                            y = -1920.4*Hd**2 + -28867*Hd + -32398  #R² = 0.6218  #7
                            ymax, ymin = 83700/7+500,  59274/7-500     
                            # 2317.6x4 + 86528x3 + 1E+06x2 + 7E+06x + 2E+07

                        elif current_number == 8 :
                            y = 5852.1*Hd + 132043 +800 #R² = 0.8154 #8
                            # y = 5320.1*Hd + 127801  #+ 500 
                            # y = 6712*Hd + 138712
                            ymax, ymin = 12000,  8500
                        elif current_number == 9:
                            y = 10426*Hd + 190377  #R² = 0.7931 #9   
                            ymax, ymin = 12000,  10141
                            
                        yone = y /current_number 
                        yone = min (ymax,yone)
                        yone = max (ymin,yone)
                        
                            # current_number = 9
                        # print(y/current_number)
                        pump_weight = yone/  (2.8* 3600)
                    # pump_weight=1
                    for pump, ts in self.pump_setting.items():
                        pump.target_setting = ts * pump_weight
                           # 71931/8     
# 9*7500
# 91271/9
# 105000/9
                    
            else: # strategy 99
                self.R17sum = self.R17_df.mean(axis=1).loc[(self.sim.current_time -
                                              relativedelta(minutes=60*10))].sum()
                print( 'R17sum',round(self.R17sum,2))
                if self.R17sum == 0: #沒下雨時持續記錄現在抽水機台數 紀錄起點
                    self.base_pump = open_num
                elif open_num == 9:
                    self.base_pump = 1
                self.Rain_plowlim = round(self.R17sum/10) -1  + self.base_pump 
                # self.Rain_plowlim = min(round(self.R17sum/10) + current_number, 9) #至少一台 雨量大於10多一台   2台
                print( 'R17sum',round(self.R17sum,1), 'plowlim',round(self.Rain_plowlim,1))
            
                # if self.sim.current_time < self.time_dict['fct_e_time']:
                #     for idx, pump in enumerate([self.PUMP_DH1, self.PUMP_DH2, self.PUMP_DH3, self.PUMP_DH4, self.PUMP_DH5,\
                #                                 self.PUMP_DH6, self.PUMP_DH7, self.PUMP_DH8, self.PUMP_DH9]):
                #         if self.pump_df.loc[self.sim.current_time + pd.to_timedelta('00:10:00'), f'迪化抽水機{idx+1}'] == 0:
                #             pump.target_setting = 0
                #         elif self.pump_df.loc[self.sim.current_time + pd.to_timedelta('00:10:00'), f'迪化抽水機{idx+1}'] != 0:
                #             pump.target_setting = 1
                if self.sim.current_time < self.time_dict['fct_e_time']:
                    open_num = (self.pump_df.loc[self.sim.current_time + pd.to_timedelta('00:10:00'), [f'迪化抽水機{idx+1}' \
                                                                                    for idx in np.arange(9)]] != 0).sum()
                    lv_interval = np.array([1.09, 1.4, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])
                    for idx in range(open_num):
                        self.pump_setting[self.pumps[idx]] = self.base_Llv / self.base_Hlv # not meant to open but for lv isn't consistent with history
                    
                    index = np.searchsorted(lv_interval, self.DIHWA_tank.depth)
                    
                    if index > 0 and index < 8:
                        self.pump_setting[self.pumps[index-1]] = 1 - ((lv_interval[index] - self.DIHWA_tank.depth)/(lv_interval[index] - lv_interval[index-1]) * (1 - self.base_Llv / self.base_Hlv))
                        for i in np.arange(0, index-1):
                            self.pump_setting[self.pumps[i]] = 1
                    elif index >= 8:
                        for i in np.arange(0, index):
                            self.pump_setting[self.pumps[i]] = 1

                    for idx in range(open_num, 9):
                        self.pump_setting[self.pumps[idx]] = 0 # history close
                    
                    previous_wlv = self.DIHWA_tank.depth
                    for pump, ts in self.pump_setting.items():
                        pump.target_setting = ts


            #if self.sim.current_time == self.time_dict['fct_s_time']:
            if True:
                # for name, pump in zip([f'pump{i+1}' for i in range(9)], self.pumps):
                #     self.action_history_log.loc[self.sim.current_time, name] = pump.current_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_open_num'] = sum(pump.target_setting != 0 for pump in self.pumps)
                self.action_history_log.loc[self.sim.current_time, 'ori_3031'] = self.ori3031.target_setting
                self.action_history_log.loc[self.sim.current_time, 'ori_3041'] = self.ori3041.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_GINMEI'] = self.PUMP_GINMEI.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_SONSHAN'] = self.PUMP_SONSHAN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_QUENYAN'] = self.PUMP_QUENYAN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_SULIN'] = self.PUMP_SULIN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_SONSHIN'] = self.PUMP_SONSHIN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_SINJAN'] = self.PUMP_SINJAN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_ZUNSHAO'] = self.PUMP_ZUNSHAO.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_B43_1'] = self.PUMP_B43_1.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_B43_2'] = self.PUMP_B43_2.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_B43_3'] = self.PUMP_B43_3.target_setting
                self.action_history_log.loc[self.sim.current_time, 'pump_B43_4'] = self.PUMP_B43_4.target_setting                
                self.action_history_log.loc[self.sim.current_time, 'ori_LOQUAN'] = self.ori_LOQUAN.target_setting
                self.action_history_log.loc[self.sim.current_time, 'ori_HURU'] = self.ori_HURU.target_setting
                for name in self.INTERCEPTOR_ORIFICE_list:
                    self.action_history_log.loc[self.sim.current_time, name] = self.link_object[name].target_setting






            print(cnt_sim ,len(self.hsf_fct_time_index),round(self.node3850_0313S.head,2),\
                  round(self.DIHWA_tank.head,2),'pump: ',round(self.current_pump_open_tarnum,2),\
                  round(self.ori3031.current_setting,2),\
                  round(self.ori3041.current_setting,2),self.dc, f'cd:{self.pump_cd}')
            cnt_sim= cnt_sim+1
            if self.sim.current_time > self.time_dict['fct_s_time']:
                self.previous_grad_wlv = self.grad_wlv
            if self.sim.current_time == (self.sim.end_time - pd.to_timedelta('00:10:00')):
                break
            else:
                self.sim.__next__()
# aaa = self.Result_df
      # aab = self.Data_com_df
        self.sim.close()
        # self=a
# a.load()



    def load(self):
    # def load(self, save_name, save_path, time_dict, strategy):
    #     self.save_name = save_name
    #     self.save_path = save_path
    #     self.time_dict = time_dict
    #     self.strategy = strategy
        self.time_index_set()
        # self.realtime_raw_df

        out_df  = pd.read_excel(self.iotable_path,sheet_name='out2')
        realtimedata_df  = pd.read_excel(self.iotable_path,sheet_name='realtimedata2')
        realtimedata_dict = { iii :realtimedata_df[iii].dropna().astype(str).tolist() for iii in realtimedata_df}
        if self.sys == 'windows':
            self.out_name = r'%s.out' %self.save_name
        elif self.sys == 'linux':
            if __name__ == '__main__':
                self.new_inp = "test.inp"
                self.out_name = r'test.out'
            else:
                self.new_inp = "New.inp"  #%(self.save_name)
                self.out_name = r'New.out'
        self.Out_file_df = swmm_api.out2frame(os.path.join(self.save_path,self.out_name))  
        self.Out_file_df.index = pd.to_datetime(self.Out_file_df.index)
        # Link = ['38500002-38500061', 'B42_3850-0001', '38510001-38500243', '38520010-38500247', '38480003-38500247']
        self.Result_df = pd.DataFrame(index = self.hsf_fct_time_index, 
                                columns = ['DateTime','Simulated time'] + out_df['Object Name (ch)'].tolist() 
                                +['Forecasted time'] + realtimedata_dict['realtimedata_ch'] )
        self.Result_df = self.Result_df.assign(DateTime=pd.to_datetime(self.hsf_fct_time_index))
        
        self.Result_df.loc[self.time_dict['hsf_s_time']:self.time_dict['hsf_e_time'],'Simulated time'] = 'hotstart'
        self.Result_df.loc[self.time_dict['fct_s_time']:self.time_dict['fct_e_time'],'Simulated time'] = 'forecast'
        self.Result_df.loc[self.time_dict['hsf_e_time'],'Simulated time'] = 'now'
# self.Result_df.columns
        self.Result_df.loc[self.time_dict['hsf_s_time']:self.time_dict['hsf_e_time'],'Forecasted time'] = 'observed data'
        self.Result_df.loc[self.time_dict['fct_s_time']:self.time_dict['fct_e_time'],'Forecasted time'] = 'forecast'
        self.Result_df.loc[self.time_dict['hsf_e_time'],'Forecasted time'] = 'now'
        # iii =1
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
        node_name = [('node', 'DIHWA', 'flooding')] + [('node', n, 'flooding') for n in nodes]
        sewage_node_name = [('link', n, 'flow') for n in ['38520010-38500247', '39510023-39510001', 'B42_3850-0001', '38500003-38500002', '38480003-38500247']]
        self.Result_df.loc[self.time_dict['fct_s_time'] - relativedelta(minutes=10):self.time_dict['fct_e_time'],'system_flooding'] = self.Out_file_df.loc[self.time_dict['fct_s_time'] - relativedelta(minutes=10):self.time_dict['fct_e_time'], node_name].sum(axis=1)
        self.Result_df.loc[self.time_dict['fct_s_time'] - relativedelta(minutes=10):self.time_dict['fct_e_time'],'sewage'] = self.base_flow[self.time_dict['fct_s_time'] - relativedelta(minutes=10):self.time_dict['fct_e_time']]
        
        for iii in range(len(out_df)): #OUT 頁面資料輸出
            nnn = out_df.iloc[iii,:].values
            if nnn[3] == 'DIHWA_inflow_acc': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn[5]] = self.Result_df.loc[:,out_df['Object Name (ch)'].iloc[0:5]].sum(axis=1)
            else:
                self.Result_df.loc[self.hsf_fct_time_index[1:],nnn[5]] = self.Out_file_df.loc[self.hsf_fct_time_index[1:], (nnn[1],nnn[3],nnn[4])].values
        self.Result_df.loc[:, '緊急繞流井雷達波液位(based -12.39)'] = self.Result_df.loc[:, '緊急繞流井液位'] - 12.39
        # self.Result_df.loc[self.hsf_fct_time_index[1:], '3850-0313S'] = self.Out_file_df.loc[self.hsf_fct_time_index[1:], ('node', '3850-0313S', 'head')]
        # self.Result_df.loc[:, '迪化抽水機總數'] = (self.realtime_df.loc[:, ['迪化抽水機1', '迪化抽水機2', '迪化抽水機3', '迪化抽水機4',\
        #                                                               '迪化抽水機5', '迪化抽水機6', '迪化抽水機7', '迪化抽水機8', '迪化抽水機9']]!=0).sum(axis=1)
# self=a
        for eee, nnn in zip(realtimedata_dict['realtimedata'],realtimedata_dict['realtimedata_ch']):
            if nnn == '平均時雨量': 
                Rsta2X = ["長安國小","民生國中",
                 "臺大","市政中心","留公國中","埤腹",
                 "奇岩","桃源國中","建國","碧湖國小",
                 "太平國小","雙園","北投國小","陽明高中",
                 "東湖國小","中正國中","博嘉國小","玉成",
                 "內湖","汐止","永和","新店","中和","板橋",
                 "三重","五堵","新莊","蘆洲"  ]
                
                self.R17 = ['北投國小','陽明高中','太平國小','雙園','博嘉國小','中正國中','市政中心','留公國中 ',             
                       '桃源國中','奇岩','建國','民生國中','長安國小','台灣大學(新)','玉成','內湖','東湖國小', '北政國中', '福德']
                try:
                    self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','A0A010','T15','C0A9F0','T014', 'T019', 'T31']
                    self.Result_df.loc[self.hsf_fct_time_index,nnn] =self.rainfall_df .loc[:,self.R17_id].rolling(6).sum().mean(axis=1)
                except:
                    self.R17_id = ['T004','T005','T006','T09','T018','T008','T017','T015','T003','T35','T22','T007','T020','T15','T014']
                    self.Result_df.loc[self.hsf_fct_time_index,nnn] =self.rainfall_df .loc[:,self.R17_id].rolling(6).sum().mean(axis=1)
            
            elif nnn == '平均10分鐘雨量': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = self.rainfall_df .loc[:,self.R17_id].mean(axis=1)
            elif nnn == '時雨量最大站':    
                Rtmp = self.rainfall_df.iloc[:,0:50].rolling(6).sum().loc[self.hsf_fct_time_index] #.max(axis=1)
                Rmax = Rtmp.max(axis=1).max()
                Ridx = Rtmp.max(axis=1).idxmax()
                Rname = Rtmp.idxmax(axis=1)
                Rnamemax = Rname.loc[Ridx]
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = Rnamemax
            elif nnn == '單站最大時雨量':  
                Rtmpall = self.rainfall_df .iloc[:,0:50].rolling(6).sum()
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = Rtmpall[Rnamemax]
            elif nnn == '豪大雨警報':  
                
                tmp24h80 = self.Result_df.loc[self.hsf_fct_time_index,'單站最大時雨量'].rolling(24*6).sum() /6 >=80
                tf_24h80 = (tmp24h80).tolist()
                tf_1h40 =  (self.Result_df.loc[self.hsf_fct_time_index,'單站最大時雨量'] >=40).tolist()
                tmp3h5 = self.Result_df.loc[self.hsf_fct_time_index,'單站最大時雨量'] >=5
                tf_3h5 = np.logical_and(tmp3h5,tmp3h5.shift(6),tmp3h5.shift(12)).tolist() 
                
                tmp24h200 = self.Result_df.loc[self.hsf_fct_time_index,'單站最大時雨量'].rolling(24*6).sum() /6 >=200
                tf_24h200 =  (tmp24h200).tolist()     
                tmp3h100 = self.Result_df.loc[self.hsf_fct_time_index,'單站最大時雨量'].rolling(3*6).sum() /6 >=100
                tf_3h100 =  (tmp3h100).tolist()
                
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = '晴天模式' 
                
                
                tmp_list2 = np.logical_or.reduce([tf_24h80,tf_1h40,tf_3h5],axis=0).tolist()
                if True in tmp_list2:
                    True_first2 = tmp_list2.index(True)
                    True_last2 = len( tmp_list2)-  tmp_list2[::-1].index(True)-1
                    self.Result_df.loc[self.hsf_fct_time_index[True_first2:True_last2],nnn] = '大雨特報紓流模式' 
                tmp_list = np.logical_or.reduce([tf_24h200,tf_3h100],axis=0).tolist()
                if True in tmp_list:
                    True_first = tmp_list.index(True)
                    True_last = len( tmp_list)-  tmp_list[::-1].index(True)-1
                    self.Result_df.loc[self.hsf_fct_time_index[True_first:True_last],nnn] = '豪雨特報紓流模式' 

            elif nnn == '迪化抽水機總數': 
                self.pump_list = ['迪化抽水機1', '迪化抽水機2', '迪化抽水機3', '迪化抽水機4',\
                                              '迪化抽水機5', '迪化抽水機6', '迪化抽水機7', '迪化抽水機8', '迪化抽水機9']
                self.Result_df.loc[:, nnn] = (self.Result_df.loc[:,self.pump_list]>0).sum(axis=1)

            elif nnn == '迪污進流量(cmh)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee].ffill().bfill()
            elif nnn == '超量污水送獅子頭': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = self.Result_df.loc[self.hsf_fct_time_index,'迪抽送水量(cmh)'] - self.Result_df.loc[self.hsf_fct_time_index,'迪污進流量(cmh)']
            elif nnn == 'B43人孔液位_升降': 
                ttmp = self.Result_df.loc[self.hsf_fct_time_index,'B43人孔液位'] 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = ttmp - ttmp.shift(1)
            elif nnn == '緊急繞流井液位_升降': 
                ttmp = self.Result_df.loc[self.hsf_fct_time_index,'緊急繞流井液位'] 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = ttmp - ttmp.shift(1)
            elif nnn == '迪化抽水站濕井液位_升降': 
                ttmp = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位'] 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = ttmp - ttmp.shift(1)
            elif nnn == '指標_截流站關閉': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn] = 0
                # tf_1 = self.Result_df.loc[self.hsf_fct_time_index, '豪大雨警報'] == '晴天模式'
                # if len(tf_1)>0:
                    # cnt = 170
                for cnt,ttt in enumerate(list(self.Result_df.index)):
                    # ttt= self.Result_df.index[cnt]
                    tf_len = len(self.Result_df.index>ttt)
                    bool_c = self.Result_df.loc[ttt,'迪化抽水機總數']>=5
                    bool_b = True in (self.Result_df['豪大雨警報'].iloc[cnt:min(cnt+6*12,tf_len)] != '晴天模式').tolist() #12小時內有 晴天模式以外的
                    if bool_b or bool_c:
                        self.Result_df.loc[ttt,nnn] = 1
                        # print(ttt,self.Result_df.loc[ttt,nnn] )
            elif nnn == '指標1': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                bool_a = self.Result_df.loc[self.hsf_fct_time_index,'豪大雨警報']!='晴天模式' 
                # bool_b = self.Result_df.loc[self.hsf_fct_time_index,'豪大雨警報']!='晴天模式'
                self.Result_df.loc[bool_a,nnn]=1
            elif nnn == '指標2-1': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                tf_1 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水機總數'] >=5
                tf_2 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位'] >=-10.1
                tf_3 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位_升降'] >=0 -0.3 #保守起見
                indi_1 = self.Result_df.loc[self.hsf_fct_time_index, '指標1'] == 1
                ccc_df = pd.concat([tf_1,tf_2,tf_3],axis=1)
                bool_a = np.logical_and.reduce([tf_1,tf_2,tf_3, indi_1],axis=0).tolist()
                self.Result_df.loc[bool_a,nnn]=1
            elif nnn == '指標2-2': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                bool_a = self.Result_df.loc[self.hsf_fct_time_index,'豪大雨警報']== '豪雨特報紓流模式' 
                self.Result_df.loc[bool_a,nnn]=1 
            elif nnn == '指標3': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                tf_1 = self.Result_df.loc[self.hsf_fct_time_index,'指標1'] == 1
                tf_2 = self.Result_df.loc[self.hsf_fct_time_index,'指標2-1'] == 1
                p6 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水機總數'] >= 6
                d8 = self.Result_df.loc[self.hsf_fct_time_index,'緊急繞流井液位'] >= -8
                w85 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位'] > -8.7
                bool_a = np.array(np.logical_and.reduce([tf_1, tf_2], axis=0))
                bool_b = np.array(np.logical_and.reduce([p6, d8], axis=0))
                bool_c = np.array(np.logical_and.reduce([p6, w85], axis=0))
                self.Result_df.loc[np.logical_or(bool_a, bool_b, bool_c),nnn]=1 
            elif nnn == '指標4': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                tf_1 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水機總數'] >=7
                tf_2 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位'] >=-10.1
                tf_3 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位_升降'] >=0 -0.3 #保守起見
                tf_4 = self.Result_df.loc[self.hsf_fct_time_index,'緊急繞流井液位_升降'] >=0 -0.3 #保守起見
                tf_5 = self.Result_df.loc[self.hsf_fct_time_index,'指標3'] == 1
                bool_a = np.logical_and.reduce([tf_1,tf_2,tf_3,tf_4,tf_5],axis=0).tolist()
                self.Result_df.loc[bool_a,nnn]=1
            elif nnn == '指標5': 
                self.Result_df.loc[self.hsf_fct_time_index,nnn]=0
                tf_1 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水機總數'] >=7
                tf_2 = self.Result_df.loc[self.hsf_fct_time_index,'緊急繞流井液位'] >=-10.0
                tf_3 = self.Result_df.loc[self.hsf_fct_time_index,'迪化抽水站濕井液位_升降'] >=0-0.3  #保守起見
                tf_4 = self.Result_df.loc[self.hsf_fct_time_index,'緊急繞流井液位_升降'] >=0-0.3 #保守起見
                tf_5 = self.Result_df.loc[self.hsf_fct_time_index,'指標4'] == 1
                bool_a = np.logical_and.reduce([tf_1,tf_2,tf_3,tf_4,tf_5],axis=0).tolist()
                self.Result_df.loc[bool_a,nnn]=1                
            elif nnn == '迪化抽水站總瞬間流量(cmh)': 
                self.Result_df.loc[:,nnn] = self.Result_df.loc[:,'迪化抽水站總瞬間流量(cms)']*3600
            elif nnn == '緊急繞流井液位(觀測)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]-12.39
            elif nnn == '迪化抽水站濕井液位(觀測)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]-11.8
            elif nnn == 'B43人孔液位(觀測)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]-11.2                      
            elif nnn == '迪化3031主閘門(觀測)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]
            elif nnn == '迪化3041主閘門(觀測)': 
                self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]                              
                
            else:
                try:
                    self.Result_df.loc[:,nnn] = self.realtime_df.loc[:,eee]  # self.hsf_fct_time_index
                except KeyError:
                    pass
                aaa = self.realtime_df

            self.Result_df['indicator'] = self.indicator_log_df['indicator']
            self.Result_df['rain_mode'] = self.indicator_df['mode']
        if self.sys == 'windows':
            self.csv_name = r'%s.csv' %self.save_name
        elif self.sys == 'linux':
            if __name__ == '__main__':
                self.csv_name = "test.csv"  #%(self.save_name)
                self.out_name = r'test.out'
            else:
                self.csv_name = "New.csv"  #%(self.save_name)
                self.out_name = r'New.out'
        self.Result_df.to_csv(os.path.join(self.save_path,self.csv_name),encoding='big5')          
        print('Output file: %s' %os.path.join(self.save_path,'%s.csv' %self.save_name))


    def load_only_fct(self):    
        out_name = r'%s.out' %self.save_name
        out_df  = pd.read_excel(self.iotable_path,sheet_name='out')
        Out_file_df = swmm_api.out2frame(os.path.join(self.save_path,out_name))  
        Out_fct_df = Out_file_df.loc[self.time_dict['fct_s_time']:self.time_dict['fct_e_time'],:]
        times = Out_fct_df.iloc[:,0].index
        Result_df = pd.DataFrame(index = times, columns = out_df['Object Name (ch)'])
        for nnn in out_df.values:
            if nnn[3] == 'DIHWA_inflow_acc': 
                Result_df[nnn[5]] = Result_df.iloc[:,0:5].sum(axis=1)
            else:
                Result_df[nnn[5]] = Out_fct_df[(nnn[1],nnn[3],nnn[4])]  
        Result_df.to_csv(os.path.join(self.save_path,'%s_fct.csv' %self.save_name),encoding='big5') 

# self = a


    def metric(self):
        from metric import metric_cal
        water_level = (self.Result_df.loc[self.fct_time_index, '迪化抽水站濕井液位'])
        pump = (self.Result_df.loc[:, ['迪化抽水機1', '迪化抽水機2', '迪化抽水機3', '迪化抽水機4', '迪化抽水機5', '迪化抽水機6', '迪化抽水機7', '迪化抽水機8', '迪化抽水機9']] != 0).sum(axis=1)
        column = ['指標1', '指標2-1', '指標2-2', '指標3', '指標4', '指標5']
        indicator = []
        for index, row in self.Result_df.loc[self.fct_time_index, column].iterrows():
            if row['指標5'] == 1:
                indicator.append(5)
            elif row['指標4'] == 1:
                indicator.append(4)
            elif row['指標3'] == 1:
                indicator.append(3)
            elif row['指標2-2'] == 1:
                indicator.append(2)
            elif row['指標2-1'] == 1:
                indicator.append(2)
            elif row['指標1'] == 1:
                indicator.append(1)
            else:
                indicator.append(0)
        orifice = self.Result_df.loc[self.fct_time_index, ['迪化3031主閘門', '迪化3041主閘門']].sum(axis=1)
        overflow = self.Result_df.loc[self.fct_time_index, 'system_flooding']
        run_log = {'water_level':water_level, 'pump':pump, 'indicator':indicator, 'orifice':orifice, 'overflow':overflow}
        level_metric, pump_metric, orifice_metric, overflow_metric, metric = metric_cal(run_log)
        return level_metric, pump_metric, orifice_metric, overflow_metric, metric   

    def basic_flow_cal(self, inp_path):
        inp = read_inp_file(inp_path)
        inflow_nodes = [name[0] for name in inp[sections.INFLOWS].keys() if name[0] not in 
                        ['INTWL_CHUNSAN', 'INTWL_DALON', 'INTWL_LOUQUAN', 'INTWL_SINJAN', 
                        'INTWL_ZUNSHAO', 'INTWL_HUANHE', 'INTWL_YANPING']]

        def DH_distance_cal(name): 
            DH_x, DH_y = inp[sections.COORDINATES]['DIHWA'].x, inp[sections.COORDINATES]['DIHWA'].y
            target_x, target_y = inp[sections.COORDINATES][name].x, inp[sections.COORDINATES][name].y
            return np.sqrt((DH_x - target_x)**2 + (DH_y - target_y)**2)

        distance_dict = {name: DH_distance_cal(name) for name in inflow_nodes}
        
        d1_name = [name for name, dist in distance_dict.items() if dist < 4000]
        d2_name = [name for name, dist in distance_dict.items() if 4000 <= dist < 9000]
        d3_name = [name for name, dist in distance_dict.items() if dist >= 9000]

        def get_time_series_sum(names):
            inflow_names = [inp[sections.INFLOWS][(name, 'FLOW')]['time_series'] for name in names]
            return sum(np.array(inp[sections.TIMESERIES][name]['data'])[:, 1] for name in inflow_names)

        d1_sum = get_time_series_sum(d1_name)
        d2_sum = get_time_series_sum(d2_name)
        d3_sum = get_time_series_sum(d3_name)
        d1 = pd.Series(d1_sum, index=self.hsf_fct_time_index)
        d2 = pd.Series(d2_sum, index=self.hsf_fct_time_index).shift(periods=30, freq='T')
        d3 = pd.Series(d3_sum, index=self.hsf_fct_time_index).shift(periods=60, freq='T')

        basic_flow = d1.add(d2, fill_value=0).add(d3, fill_value=0)
        return basic_flow            

    # self=a
    def plot(self):
        # self.save_path = save_path
        # self.save_name = save_name
        # self.strategy = strategy
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_list = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 99:'#e377c2'}
        obs_df = self.realtime_df
        obs_df.index = pd.to_datetime(obs_df.index)
        if self.sys == 'windows':
            self.csv_name = r'%s.csv' %self.save_name
        elif self.sys == 'linux':
            if __name__ == '__main__':
                self.csv_name = "test.csv"  #%(self.save_name)
                self.out_name = r'test.out'
            else:
                self.csv_name = "New.csv"  #%(self.save_name)
                self.out_name = r'New.out'
        
        file_path = os.path.join(self.save_path,self.csv_name )
        file = pd.read_csv(file_path,encoding='big5', index_col=0)
        cm = 1/2.54
        # 10/2.54
        fig, axes = plt.subplots(4, 1, figsize=(5*4*1.4*cm, 5*4*1.1*cm), sharex=False,gridspec_kw={'height_ratios': [1,1,1, 1]})
        file.index = pd.to_datetime(file.index)
        # x_range = file.index[file.index.get_loc(time_dict['fct_s_time']):]
        x_range = file.index[file.index>=self.time_dict['fct_s_time']]
        font_title = {'fontname': 'Times New Roman', 'fontsize': 15}
        font_label = {'fontname': 'Times New Roman', 'fontsize': 12}
        idxs = list(set(x_range).intersection(set(obs_df.index)))
        idxs.sort()
                    

        mpl.rc('axes', labelsize=14, titlesize=16)
        fontsize =  10
        # font_prop_lg = FontProperties(size=10)     
        font_prop_lg = FontProperties(family='Times New Roman', size=11)
        font_prop_title = FontProperties( size=16) 
        # axes.xlim

        axes_0 = axes[0].twinx()
        # sewer = axes_0.plot(idxs,self.base_flow[self.fct_time_index], color='blue', linewidth=2)
        sewer = axes_0.plot(idxs, [10 for _ in range(len(idxs))], color='blue', linewidth=2)
        axes_0.set_ylabel('inflow rate (CMS)', fontdict=font_label)
        axes_0.set_ylim(0, 21)
        axes_0.set_yticks([0, 5, 10, 15, 20])
        axes_0.set_yticklabels([0, 5, 10, 15, 20], fontproperties=FontProperties(family='Times New Roman'))
        locations = ['桃源國中', '北投國小', '陽明高中', '太平國小', '民生國中', '中正國中', 
             '三興國小', '格致國中', '東湖國小', '留公國中', '舊莊國小', '市政中心', 
             '博嘉國小', '北政國中', '雙園', '玉成', '建國', '福德', '奇岩', '中洲', 
             '磺溪橋', '中和橋', '白馬山莊', '望星橋', '宜興橋', '長安國小', '萬華國中', 
             '永建國小', '五常國小', '仁愛國小', '興華國小', '南港高工', '台灣大學(新)', 
             '臺北', '科教館', '天母', '汐止', '松山', '石牌', '關渡']
        max_rain_st = obs_df.loc[self.fct_time_index, locations].sum(axis=0).idxmax()
        rain = axes[0].bar(self.fct_time_index, obs_df.loc[self.fct_time_index, max_rain_st], width=0.005, color='skyblue')
        axes[0].set_title('Rainfall and Sewer water inflow rate',pad=23,loc='center', fontdict=font_title)
        axes[0].set_ylabel('rainfall (mm)', fontdict=font_label)
        axes[0].set_ylim(0, 41)
        axes[0].set_yticks([0, 10, 20, 30, 40])
        axes[0].set_yticklabels([0, 10, 20, 30, 40], fontproperties=FontProperties(family='Times New Roman'))

        
        axes[1].axhspan(ymax = -8, ymin=-9,xmin=0 , xmax=1, color='blue', alpha=0.3, linewidth=0)
        axes[1].axhspan(-10.1, -9, color='green', alpha=0.3, linewidth=0)
        axes[1].axhspan(-11.8, -10.1, color='orange', alpha=0.3, linewidth=0)
        # axes[0]
        if len(idxs) >0:
            Lobs_diff = axes[1].plot(idxs, obs_df.loc[idxs, '緊急繞流井雷達波液位計'] - 12.39, linewidth=2, label='緊繞液位觀測',color='#0000FF')   #觀測
            Lobs_tkwl = axes[1].plot(idxs, obs_df.loc[idxs,'迪化LT-1濕井液位高度'] - 11.8, linewidth=2, label='濕井液位觀測',color='#444444') #####000000   #觀測
            
        Lfct_tkwl = axes[1].plot(self.fct_time_index, file.loc[self.fct_time_index,'迪化抽水站濕井液位'], linewidth=2, label='濕井液位預報',color='#FF3333') #color_list[self.strategy]
        Lfct_diff =  axes[1].plot(self.fct_time_index, file.loc[self.fct_time_index, '緊急繞流井液位'], linewidth=2, label='緊繞液位預報',color='#FF8800')

        axes[1].set_title('Water level of wet well and diffuser',pad=23,loc='center', fontdict=font_title)

        # axes[0].set_ylabel('液位 (m)')
        axes[1].set_ylabel('water level (m)', fontdict=font_label)

        axes[1].set_ylim(-12.5,0)
        # axes[0].set_yticks([-13,-11, -9, -7, -5,-3, -1])
        axes[1].set_yticks([-12, -10, -8, -6, -4, -2, 0])
        axes[1].set_yticklabels([-12, -10, -8, -6, -4, -2, 0], fontproperties=FontProperties(family='Times New Roman'))    
        if len(idxs) >0:
            Lobs_q = axes[2].plot(idxs, obs_df.loc[idxs, '迪化抽水站總瞬間流量(cmh)'].to_list(), color='#444444', label='observation', linewidth=2) #觀測
        # axes[1].plot(x_range, file.loc[x_range, '迪化抽水站濕井入流量(累加)'], color='#78C0E0', label='DIHWA accumlative', linewidth=2)
        # axes[1].plot(x_range,file.loc[x_range,'迪化抽水站濕井入流量(濕井)'], color='#78C0E0', label='DIHWA simulation', linewidth=2)
        Lfct_q = axes[2].plot(self.fct_time_index, file.loc[self.fct_time_index, '迪化抽水站總瞬間流量(cmh)'], label='simulation',color='#FF3333', linewidth=2)

        # axes[1].set_title('迪抽瞬間抽水量',pad=23,loc='center')
        # axes[1].set_ylabel('瞬間抽水量(CMH)')
        axes[2].set_title('Pumping rate of Dihwa pumping station', pad=23, loc='center', fontdict=font_title)
        axes[2].set_ylabel('Pumping rate (CMH)', fontdict=font_label)
        # axes[1].set_title('Pumping rate of Dihwa pumping station',pad=23,loc='center')
        # axes[1].set_ylabel('Pumping rate (CMH)')
        
        axes[2].set_yticks([0,20000, 40000, 60000,80000,100000,120000])
        axes[2].set_ylim(0,120000)
        axes[2].set_yticklabels([0, 20000, 40000, 60000, 80000, 100000, 120000], fontproperties=FontProperties(family='Times New Roman'))
        # axes[1].legend(loc='best', bbox_to_anchor=(0.728, 1.45), fontsize=fontsize, frameon=False)
        
        
        if len(idxs) >0:
            Lobs_pump =  axes[3].step(idxs, obs_df.loc[idxs, self.pump_list].sum(axis=1), where='mid', color='#444444', label='歷史運作台數', linewidth=2)   #觀測
        Lfct_pump = axes[3].step(self.fct_time_index, (file.loc[self.fct_time_index, '迪化抽水機總數']), where='mid', color='#FF3333', label='預報運作台數', linewidth=2)
        
        # aaa = file.loc[idxs, '迪化抽水機總數']
        # axes[2].set_title('迪抽抽水機運轉台數',pad=23,loc='center')
        # axes[2].set_ylabel('抽水機運轉台數')

        axes[3].set_title('Number of working pumps',pad=23,loc='center', fontdict=font_title)
        axes[3].set_ylabel('Number', fontdict=font_label)
        axes[3].set_ylim(0,10)
        axes[3].set_yticks([0,3,6,9])
        axes[3].set_yticklabels([0,3,6,9], fontproperties=FontProperties(family='Times New Roman'))
        # axes[2].legend(loc='best', bbox_to_anchor=(0.73, 1.45), fontsize=fontsize, frameon=False)
        date_labels = [date.strftime('%Y-%m-%d') for date in x_range]
        time_labels = [date.strftime('%H:%M') for date in x_range]
        axes[3].set_xticks(x_range)

        hoursLoc = mpl.dates.HourLocator   (byhour=[0,12], interval=1, tz=None ) #为6小时为1副刻度  byminute= range(60*10*6*100), 
        # monsLoc = mdates.MinuteLocator( interval=24                                         
        #                                , tz=None)
        axes[3].xaxis.set_major_locator(hoursLoc)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H\n%Y/%m/%d'))
        hoursLoc2 = mpl.dates.HourLocator(byhour=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], interval=1, tz=None ) #为6小时为1副刻度
        # hoursLoc = mpl.dates.HourLocator   (byhour=[0,12], interval=1, tz=None ) #为6小时为1副刻度
        font_prop = FontProperties(family='Times New Roman')
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
        for label in axes[1].get_xticklabels(which='both'):
            label.set_fontproperties(font_prop)
 
        
        axes[3].xaxis.set_minor_locator(hoursLoc2)
        axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        axes[3].tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 12, length=10,width=.4)  
        axes[3].tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 12, length=4,width=.4)   
        for label in axes[2].get_xticklabels(which='both'):
            label.set_fontproperties(font_prop)

        import matplotlib.ticker as ticker
        axes[2].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))  # FormatStrFormatter('%.f') 
        

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
        L9 = plt.legend(sewer, [ 'sewer' ],loc='upper center',  bbox_to_anchor=(0.7, 6.43-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
        plt.gca().add_artist(L9); 
        L10 = plt.legend(rain, [ 'rainfall' ],loc='upper center',  bbox_to_anchor=(0.3, 6.43-5.12),frameon=False,fontsize=15,prop = font_prop_lg, handlelength=1.5 ,handleheight=0.8 , handletextpad=0.3)   #'upper center
        plt.gca().add_artist(L10);
        
        # self=a
        # fig.suptitle(f'Strategy {self.strategy} Simulation', fontsize=16)
        # plt.subplots_adjust(hspace=0.6)
        plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=.8)
        # fig.savefig(os.path.join(self.save_path,f'test.png'),dpi=300)
        fig.savefig(os.path.join(self.save_path,f'{self.save_name}_strategy{self.strategy}.png'),dpi=300)
        plt.close()
        
    def plot_all(self, strategy_iput: list ,save_path, time_dict):
        self.save_path = save_path
        self.time_dict = time_dict
        fct_length = int((self.time_dict['fct_e_time'] - self.time_dict['fct_s_time']).total_seconds()/3600)
        file_names = {
            0:f'0.Dihwa_R0_INT0_Difup0_EMG0_Difdn0_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv',
            1:f'1.Dihwa_R0_INT1_Difup0_EMG0_Difdn0_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv',
            2:f'2.Dihwa_R1_INT0_Difup0_EMG0_Difdn0_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv',
            3:f'3.Dihwa_R1_INT0_Difup1_EMG0_Difdn0_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv',
            4:f'4.Dihwa_R1_INT0_Difup1_EMG1_Difdn0_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv',
            5:f'5.Dihwa_R1_INT0_Difup1_EMG1_Difdn1_h6f18_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.csv'
        }
        color_list = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 99:'#e377c2'}
        all_exist = True
        # for file_name in file_names:
        #     file_path = os.path.join(self.save_path, file_name)
        #     if not os.path.isfile(file_path):
        #         all_exist = False
        #         break
        if all_exist:
            file0 = pd.read_csv(os.path.join(self.save_path, file_names[0]), encoding='big5', index_col=0)
            file1 = pd.read_csv(os.path.join(self.save_path, file_names[1]), encoding='big5', index_col=0)
            file2 = pd.read_csv(os.path.join(self.save_path, file_names[2]), encoding='big5', index_col=0)
            file3 = pd.read_csv(os.path.join(self.save_path, file_names[3]), encoding='big5', index_col=0)
            file4 = pd.read_csv(os.path.join(self.save_path, file_names[4]), encoding='big5', index_col=0)
            file5 = pd.read_csv(os.path.join(self.save_path, file_names[5]), encoding='big5', index_col=0)
            file0.index = pd.to_datetime(file0.index)
            file1.index = pd.to_datetime(file1.index)
            file2.index = pd.to_datetime(file2.index)
            file3.index = pd.to_datetime(file3.index)
            file4.index = pd.to_datetime(file4.index)
            file5.index = pd.to_datetime(file5.index)
            x_range = file1.index[file1.index.get_loc(time_dict['fct_s_time']):]
            x_range=x_range
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            # axes[1].plot(x_range, file0.loc[x_range,'PUMP_accum''], linewidth=1.5,color=color_list[0], label='strategy 0')
            axes[1].plot(x_range, file1.loc[x_range,'PUMP_accum'], linewidth=1.5,color=color_list[1], label='strategy 1')
            axes[1].plot(x_range, file2.loc[x_range,'PUMP_accum'], linewidth=1.5,color=color_list[2], label='strategy 2')
            axes[1].plot(x_range, file3.loc[x_range,'PUMP_accum'], linewidth=1.5,color=color_list[3], label='strategy 3',linestyle='--')
            axes[1].plot(x_range, file4.loc[x_range,'PUMP_accum'], linewidth=1.5,color=color_list[4], label='strategy 4',linestyle='-.')
            axes[1].plot(x_range, file5.loc[x_range,'PUMP_accum'], linewidth=1.5,color=color_list[5], label='strategy 5',linestyle=':')
            axes[1].plot(x_range, file0.loc[x_range,'迪化抽水站總瞬間流量(cms)'], linewidth=1.5, label='observation',color='gray')
            axes[1].set_title('DIHWA Pumping Flow Rate')
            axes[1].set_ylabel('Flow Rate (CMS)')
            axes[1].set_yticks([0, 10, 20])
            axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), fontsize=9)
            axes[0].axhspan(-8, -7, color='blue', alpha=0.3, linewidth=0)
            axes[0].axhspan(-9.1, -8, color='green', alpha=0.3, linewidth=0)
            axes[0].axhspan(-10.8, -9.1, color='orange', alpha=0.3, linewidth=0)
            # axes[0].plot(x_range, file0.loc[x_range,'DIHWA_head'],color=color_list[0], linewidth=1.5, label='strategy 0')
            axes[0].plot(x_range, file1.loc[x_range,'DIHWA_head'], linewidth=1.5,color=color_list[1], label='strategy 1',linestyle='-.')
            axes[0].plot(x_range, file2.loc[x_range,'DIHWA_head'], linewidth=1.5,color=color_list[2], label='strategy 2',linestyle='--')
            axes[0].plot(x_range, file3.loc[x_range,'DIHWA_head'], linewidth=1.5,color=color_list[3], label='strategy 3')
            axes[0].plot(x_range, file4.loc[x_range,'DIHWA_head'], linewidth=1.5,color=color_list[4], label='strategy 4',linestyle='-.')
            axes[0].plot(x_range, file5.loc[x_range,'DIHWA_head'], linewidth=1.5,color=color_list[5], label='strategy 5',linestyle=':')
            axes[0].plot(x_range, file0.loc[x_range,'濕井液位'] - 10.8, linewidth=1.5, label='observation',color='gray')
            axes[0].set_title('DIHWA Tank Head')
            axes[1].set_xlabel('Time')
            axes[0].set_ylabel('Head (m)')
            axes[0].set_ylim(-12,-3)
            axes[0].set_yticks([-12, -10, -8, -6, -4])
            axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), fontsize=9)
            date_labels = [date.strftime('%Y-%m-%d') for date in x_range]
            time_labels = [date.strftime('%H:%M') for date in x_range]
            axes[1].set_xticks(x_range)
            axes[1].set_xticklabels([f'{date}\n{time}' for date, time in zip(date_labels, time_labels)], fontsize=8)
            axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
            # axes[1].xaxis.set_major_locator(plt.MaxNLocator(9))
            fig.suptitle(f'All Strategy Simulations', fontsize=16)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_path,f'All_strategy_{self.time_dict["fct_s_time"].strftime("%Y%m%d_%H%M")}.png'),dpi=300)
            plt.close()

        else:
            print('do simulation 0~5!')
# self=a
    def plot_b43_emg(self):
        b43_columns = [('link', 'PUMP-B43_OUT1', 'flow'),
                       ('link', 'PUMP-B43_OUT2', 'flow'),
                       ('link', 'PUMP-B43_OUT3', 'flow'),
                       ('link', 'PUMP-B43_OUT4', 'flow')]
        B43 = self.Out_file_df.loc[self.hsf_fct_time_index[1:], b43_columns]
        b43_columns_obs = ['B43抽水機1', 'B43抽水機2', 'B43抽水機3', 'B43抽水機4']
        B43_obs = self.realtime_df.loc[:, b43_columns_obs].ffill()
        # dddd = self.realtime_df
# B43_obs.sum()
        emg_columns = [('link', 'PUMP_EMG_IN1', 'flow'),
                       ('link', 'PUMP_EMG_IN1', 'flow'),
                       ('link', 'PUMP_EMG_IN1', 'flow')]
        emg = self.Out_file_df.loc[self.hsf_fct_time_index[1:], emg_columns]
        emg_columns_obs = ['迪化緊急進流抽水站_PUMP1_抽水機運轉', '迪化緊急進流抽水站_PUMP2_抽水機運轉', '迪化緊急進流抽水站_PUMP3_抽水機運轉']
        emg_obs = self.realtime_df.loc[:, emg_columns_obs].ffill()
        x_range = B43.index[B43.index.get_loc(self.time_dict['fct_s_time']):]
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        axes[0].step(self.hsf_time_index, (B43_obs.loc[self.hsf_time_index,:]!=0).sum(axis=1), linewidth=2, where='mid', label='B43 pump observation')
        axes[0].step(self.hsf_fct_time_index[1:], (B43.loc[self.hsf_fct_time_index[1:],:]!=0).sum(axis=1), linewidth=2, where='mid', label='B43 pump simulation')
       
        
        axes[0].set_title('B43 simultaion and observation')
        axes[0].set_ylabel('number of working')
        axes[0].set_ylim(-0.1, 4.1)
        axes[0].set_yticks([0, 1, 2, 3, 4])
        axes[0].legend(loc='upper left', bbox_to_anchor=(0.73, 1.5), fontsize=7, frameon=False)
        
        
        
        axes[1].step(self.hsf_time_index, (emg_obs.loc[self.hsf_time_index,:]!=0).sum(axis=1), linewidth=2, where='mid', label='EMG pump observation')
        axes[1].step(self.hsf_fct_time_index[1:], (emg.loc[self.hsf_fct_time_index[1:],:]!=0).sum(axis=1), linewidth=2, where='mid', label='EMG pump simulation')
        axes[1].set_title('EMG simultaion and observation')
        axes[1].set_ylabel('number of working')
        axes[1].set_ylim(-0.1,3.1)
        axes[1].set_yticks([0,1,2,3])
        axes[1].legend(loc='upper left', bbox_to_anchor=(0.73, 1.45), fontsize=7, frameon=False)
        date_labels = [date.strftime('%Y-%m-%d') for date in x_range]
        time_labels = [date.strftime('%H:%M') for date in x_range]
        axes[1].set_xticks(x_range)
        axes[1].set_xticklabels([f'{date}\n{time}' for date, time in zip(date_labels, time_labels)], fontsize=6)
        axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axes[1].set_xlabel('Time')
        fig.suptitle(f'B43 and Emergency {self.strategy} Simulation', fontsize=16)
        plt.subplots_adjust(hspace=0.6)
        # fig.savefig(os.path.join(self.save_path,f'{self.save_name}_strategy{self.strategy}.png'),dpi=300)
        plt.close()
        plt.show()

    def plot_wl(self, level_path):

        
        if self.time_dict['t_time']<  datetime(2024, 5, 31, 0, 0):
            self.level_df = pd.read_excel(level_path,sheet_name = '202405')
        elif self.time_dict['t_time']>=  datetime(2024, 6, 1, 0, 0):
            self.level_df = pd.read_excel(level_path,sheet_name = '202406')
            
            
        self.level_index1 = self.level_df.loc[:,'編號'].tolist()
        self.level_index2 = self.level_df.loc[:,'人孔編號'].tolist()
        
        
        # len(self.level_index1)
        # len(self.level_index2)
        # len(level_dict[1])
        # ccc = self.realtime_df
        # self.realtime_df.loc[:,'CB10']
        level_dict = {1:[], 2:[]}
        # #內湖 E 和 大直F兩邊都沒建立
        # self.level_index1.index('CB10')
        # li1 = 'CB10'
        # self.level_index2[27]
        # li2 = '3954-0233'
        for li1, li2 in zip(self.level_index1, self.level_index2):
            if f'液位_迪化_{li1}' in self.realtime_df.columns and ('node', f'{li2}','depth') in self.Out_file_df.columns:
                level_dict[1].append(f'液位_迪化_{li1}')
                level_dict[2].append(('node', f'{li2}','depth'))
            # else:
                # print(li1, li2)
        self.wl_obs_df = self.realtime_df.loc[:,level_dict[1]]
        self.wl_simu_df = self.Out_file_df.loc[self.hsf_fct_time_index[1:], level_dict[2]]
        x_range = self.wl_simu_df.index[self.wl_simu_df.index.get_loc(self.time_dict['fct_s_time']):]
        # x_range = B43.index[B43.index.get_loc(self.time_dict['fct_s_time']):]
        num_plots = 9 #len(self.wl_obs_df.columns) 
        num_rows = 3  #3*3  
        num_cols = 3
        
        kkk = 0
        hoursLoc = mpl.dates.HourLocator   (byhour=[0,12], interval=1, tz=None ) #为6小时为1副刻度  byminute= range(60*10*6*100), 
        hoursLoc2 = mpl.dates.HourLocator   (byhour=[iii for iii in range (0,23,2)], interval=1, tz=None ) #为6小时为1副刻度
        
        # hoursLoc2 = mpl.dates.HourLocator   (byhour=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], interval=1, tz=None ) #为6小时为1副刻度
        for kkk in range(0,4):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*3), constrained_layout=True)
            axes = axes.reshape(-1)
            
            for idx, (li1, li2) in enumerate(zip(self.wl_obs_df.columns[kkk*9:(kkk+1)*9], self.wl_simu_df.columns[kkk*9:(kkk+1)*9])):
                # print(idx, (li1, li2))
                ax = axes[idx]
                if li1.split('_')[-1] == 'B43':
                    ax.plot(self.hsf_fct_time_index[1:], self.wl_obs_df.loc[self.hsf_fct_time_index[1:], li1], label='觀測')
                else:
                    ax.plot(self.hsf_fct_time_index[1:], self.wl_obs_df.loc[self.hsf_fct_time_index[1:], li1] / 1000, label='觀測')
                ax.plot(self.hsf_fct_time_index[1:], self.wl_simu_df.loc[self.hsf_fct_time_index[1:], li2], label='預報')
                if idx in [6,7,8]:
                    ax.xaxis.set_major_locator(hoursLoc)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H\n%Y/%m/%d'))
                    ax.xaxis.set_minor_locator(hoursLoc2)
                    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
                    ax.tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 10, length=8,width=.4)  
                    ax.tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 8, length=4,width=.4)   
                else:
                    ax.xaxis.set_major_locator(hoursLoc)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                    ax.xaxis.set_minor_locator(hoursLoc2)
                    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
                    ax.tick_params(axis='x',which= 'major', direction= 'out',pad=1, labelsize= 10, length=8,width=.4)  
                    ax.tick_params(axis='x',which= 'minor', direction= 'out', pad=1, labelsize= 8, length=4,width=.4)   
      
                ax.legend()
                ax.set_title(f"{li1.split('_')[-1]}")
                
                if idx % 3 ==0:
                    ax.set_ylabel('液位(m)')
                ax.set_ylim(0,11)
                
                
                
                
            for idx in range(num_plots, num_rows * num_cols):
                fig.delaxes(axes[idx])
    
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"{self.save_name}_waterlevel_%s.png" %kkk) , dpi=300)
            plt.close()
            plt.show()
        
    def df2sql(self,name, db_params, df,exist):
        engine = create_engine(
           f"{db_params['dialect_driver']}://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
        )
        with engine.connect() as conn:
            df.to_sql(name, conn, index=False,if_exists=exist)         
    # def df2sql(self,name, db_params, df):
    #     engine = create_engine(
    #        f"{db_params['dialect_driver']}://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    #     )
    #     with engine.connect() as conn:
    #         df.to_sql(name, conn, index=False,if_exists='replace')   
    
    def summarySuggest2(self):
        
        fct_time = self.fct_time_index
        if '海上陸上颱風紓流模式'in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '海上陸上颱風紓流模式' 
            pass
        elif '豪雨特報紓流模式' in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '豪雨特報紓流模式'
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest豪雨')
        elif '大雨特報紓流模式' in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '大雨特報紓流模式'
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest大雨')
        else:
            self.mode_type = '晴天模式' 
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest晴天')
            # self.Result_df['豪大雨警報']

        
        strategy_name_list = ['晴天模式', '截流關閉','景美,松山,昆陽,內湖紓流','B43,緊急進/紓流','士林,松信,新建,忠孝,六館,葫蘆紓流']
        idx_list = ['DateTime',"模式","模擬時間","最大平均時雨量(mm)","最大時雨量時間",
                    "尖峰進流量(CMH)","尖峰時間","尖峰抽水機啟動數", '現況指標','現況指標說明',
                    '指標_截流站關閉','指標1','指標2-1','指標2-2','指標3','指標4','指標5',
                    "水情分析"]
        self.weather_df = pd.DataFrame( index =idx_list, columns = ['項次','說明'])
        
        self.weather_df['項次'] = idx_list
        
        self.weather_df.loc ['DateTime','說明'] = self.time_dict['t_time']
        self.weather_df.loc ['模式','說明'] = self.mode_type
        self.weather_df.loc['模擬時間','說明'] = self.time_dict['t_time'].strftime("%Y/%m/%d %H:%M")
# self.Result_df['平均時雨量']
        self.weather_df.loc["最大平均時雨量(mm)",'說明'] =self.Result_df.loc[fct_time,'平均時雨量'].astype(float).max().round(2)
        self.weather_df.loc["最大時雨量時間",'說明'] =self.Result_df.loc[fct_time,'平均時雨量'].astype(float).idxmax().strftime("%Y/%m/%d %H:%M")
        self.weather_df.loc["尖峰進流量(CMH)",'說明'] =self.Result_df.loc[fct_time,'迪化抽水站總瞬間流量(cmh)'].astype(float).max().round(0)
        self.weather_df.loc["尖峰時間",'說明'] =self.Result_df.loc[fct_time,'迪化抽水站總瞬間流量(cmh)'].astype(float).idxmax() .strftime("%Y/%m/%d %H:%M")       
        self.weather_df.loc["尖峰抽水機啟動數",'說明'] = round(self.Result_df.loc[self.weather_df.loc["尖峰時間",'說明'],'迪化抽水機總數'],0)
        tsim = self.weather_df.loc['模擬時間','說明']; tRmax= self.weather_df.loc["最大時雨量時間",'說明'];tQmax=self.weather_df.loc["尖峰時間",'說明']
        Rmax= self.weather_df.loc["最大平均時雨量(mm)",'說明'];Qmax=int(self.weather_df.loc["尖峰進流量(CMH)",'說明']);Nmax= int(self.weather_df.loc["尖峰抽水機啟動數",'說明'])
                
        self.weather_df.loc["水情分析",'說明'] = f"""{tsim} 預報未來24小時，
最大雨量{tRmax} {Rmax} mm，
最大總瞬間流量：{tQmax} {Qmax} CMH(啟動{Nmax}台)"""
        

        ind_list = ['指標_截流站關閉','指標1','指標2-1','指標2-2','指標3','指標4','指標5']
        t_time_ind =  self.Result_df.loc[ self.time_dict['t_time'],ind_list].sum()
        if t_time_ind>0:
            self.weather_df.loc['現況指標','說明'] = ind_list[t_time_ind-1]
            self.weather_df.loc['現況指標說明','說明'] = sug_raw_df.loc[ind_list[t_time_ind-1],'指標說明']


        for ind in ind_list:
            if self.Result_df[ind].sum() > 0:
                abb = self.Result_df.loc[:,ind] == 1
                self.weather_df.loc[ind,'說明'] = self.Result_df.loc[:,ind] .loc[abb.tolist()] .index[0]

        strategy_name_list = ['DateTime','OpTime','操作建議','指標']
        self.suggest_df = pd.DataFrame( columns =strategy_name_list)
        cnt=0
        # self=a
        if self.mode_type == '晴天模式':
            ttmp = self.time_dict['t_time'].strftime("%m/%d %H:%M")
            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
            self.suggest_df.loc[cnt,'OpTime'] = ttmp
            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc['指標_截流站關閉','操作建議']
            
        
        else:      
            ttmp = self.time_dict['t_time'].strftime("%m/%d %H:%M")
            self.suggest_df.loc[cnt,'OpTime'] = ttmp
            self.suggest_df.loc[cnt,'操作建議'] =  self.weather_df.loc['現況指標說明','說明'] 
            self.suggest_df.loc[cnt,'指標'] =     self.weather_df.loc['現況指標','說明'] 
            cnt=cnt+1
            for ind in ind_list:
                pass
                if self.mode_type != '豪雨特報紓流模式' and ind == '指標2-2': 
                    continue
                else:
                    if not pd.isnull(self.weather_df.loc[ind,'說明']):
                        if self.weather_df.loc[ind,'說明'] > self.time_dict['t_time']:
                            ttmp =  self.weather_df.loc[ind,'說明'].strftime("%m/%d %H:%M")
                            # self.suggest_df.loc[cnt,'DateTime'] = ttmp
                            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
                            self.suggest_df.loc[cnt,'OpTime'] = ttmp
                            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc[ind,'指標說明']
                            self.suggest_df.loc[cnt,'指標'] = ind
                            cnt=cnt+1
                            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
                            self.suggest_df.loc[cnt,'OpTime'] = ttmp
                            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc[ind,'操作建議']
                            self.suggest_df.loc[cnt,'指標'] = ind
                            cnt=cnt+1

        file_names = [os.path.join(self.save_path,'%s.csv' %self.save_name)]
        all_exist = True
        for file_name in file_names:
            file_path = os.path.join(self.save_path, file_name)
            if not os.path.isfile(file_path):
                all_exist = False
                break
        fct_len = 24
        self.sel_hsf_fct_time_index = self.hsf_fct_time_index[:len(self.hsf_time_index)+6*fct_len] #181
        self.sel_fct_time_index = self.fct_time_index[:6*fct_len] #181
        strategy_name_list = ['DateTime','RecordTime','數值說明','歷史降雨','預報雨量','歷史迪抽總瞬間流量','預報迪抽總瞬間流量','歷史台數','預報台數',
                              '歷史濕井液位','預報濕井液位','歷史緊繞液位','預報緊繞液位',  
                              '歷史B43液位','預報B43液位',  
                              '歷史閘門3031','預報閘門3031','歷史閘門3041','預報閘門3041']
        # file_dict[iii] = pd.read_csv(os.path.join(self.save_path, file_names[iii]), encoding='big5', index_col=0)
        # file_dict[iii].index = pd.to_datetime(file_dict[iii].index)
        self.Data_com_df = pd.DataFrame(index = self.sel_hsf_fct_time_index, columns=strategy_name_list)
        self.Data_com_df = self.Data_com_df.assign(DateTime=pd.to_datetime(self.time_dict['t_time']))
        self.Data_com_df = self.Data_com_df.assign(RecordTime=pd.to_datetime(self.sel_hsf_fct_time_index))
        self.Data_com_df['數值說明'] = self.Result_df['Simulated time']
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史降雨'] = self.Result_df.loc[self.hsf_time_index,'單站最大時雨量'].fillna(0).round(2)        
        self.Data_com_df.loc[self.sel_fct_time_index,'預報雨量'] = self.Result_df.loc[self.sel_fct_time_index,'單站最大時雨量'].astype(float).round(2)        
        self.Data_com_df.loc[self.hsf_time_index,'歷史迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)(觀測)'].fillna(0).round(2)               
        self.Data_com_df.loc[self.sel_fct_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float).round(2)         
        self.Data_com_df.loc[self.hsf_time_index,'歷史迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)(觀測)'].fillna(0).round(2)   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float)      .round(2)
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史台數'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水機總數(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.sel_fct_time_index,'預報台數'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水機總數']      

        self.Data_com_df.loc[self.hsf_time_index,'歷史濕井液位'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站濕井液位(觀測)'].fillna(0).round(2)               
        self.Data_com_df.loc[self.sel_fct_time_index,'預報濕井液位'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站濕井液位'].astype(float).round(2)         
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史緊繞液位'] = self.Result_df.loc[self.hsf_time_index,'緊急繞流井液位(觀測)'].fillna(0).round(2)   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報緊繞液位'] = self.Result_df.loc[self.sel_fct_time_index,'緊急繞流井液位'].astype(float).round(2)   
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史B43液位'] = self.Result_df.loc[self.hsf_time_index,'B43人孔液位(觀測)'].fillna(0).round(2)   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報B43液位'] = self.Result_df.loc[self.sel_fct_time_index,'B43人孔液位'].astype(float).round(2)   
        
              
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史閘門3031'] = self.Result_df.loc[self.hsf_time_index,'迪化3031主閘門(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.sel_fct_time_index,'預報閘門3031'] = self.Result_df.loc[self.sel_fct_time_index,'迪化3041主閘門']   *100   
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史閘門3041'] = self.Result_df.loc[self.hsf_time_index,'迪化3041主閘門(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.sel_fct_time_index,'預報閘門3041'] = self.Result_df.loc[self.sel_fct_time_index,'迪化3041主閘門'] *100

        if self.sys == 'windows':
            self.xlsxname = f'Suggest.%s.xlsx' %self.save_name
            # filename_Exl_com=os.path.join('.xlsx' %(sewer_thold,R_thold,WL_ver,discharge_thold)  )
            # filename_Exl_com=os.path.join('1.Output_2011_AAC.xlsx'  )
            writer_combination = pd.ExcelWriter(os.path.join(self.save_path,self.xlsxname ) )
            self.weather_df.to_excel(writer_combination, '分析') 
            self.suggest_df.to_excel(writer_combination, '操作建議')  
            self.Data_com_df.to_excel(writer_combination, '資料')  
            writer_combination.close()
        elif self.sys == 'linux':
            
            # self.Result_df
                
            DB_PARAMS_analysis = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_analysis',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            
            DB_PARAMS_operation= {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_operation',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            DB_PARAMS_data = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_data',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            DB_PARAMS_Result = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'swmm_result',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            
            
            self.df2sql(self.save_name, DB_PARAMS_analysis, self.weather_df.astype(str),'replace')
            self.df2sql(self.save_name, DB_PARAMS_operation, self.suggest_df,'replace')
            self.df2sql(self.save_name, DB_PARAMS_data, self.Data_com_df,'replace')
            self.df2sql(self.save_name, DB_PARAMS_Result, self.Result_df,'replace')
            
            
            self.weather_df_T = self.weather_df['說明'].T
            # self.suggest_df = self.suggest_df['說明'].T
            # self.Data_com_df = self.Data_com_df['說明'].T
            DB_PARAMS_suggests =  {
                    "dialect_driver": "mssql+pymssql",
                    "host": "192.168.32.106",
                    "database": 'suggests' ,
                    "user": 'sa',
                    "password": 'ssofms%40TP', #@
                    "port": '1433',
                    "show_info": False
                    }
            DB_tbls = {}
            DB_tbls_list = ['sug_analysis_new','sug_analysis_his',
                            'sug_operation_new','sug_operation_his',
                            'sug_data_new','sug_data_his', 
                            'swmm_result_new','swmm_result_his',
                            ]
            for tbl in DB_tbls_list:
                tbl_ = tbl.split('_')
                if tbl_[2] == 'his':
                    ifexist = 'append'
                elif tbl_[2] == 'new':
                    ifexist = 'replace'
                if tbl.split('_')[1] == 'analysis':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.weather_df_T.astype(str),ifexist)
                elif tbl.split('_')[1] == 'operation':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.suggest_df.astype(str),ifexist)                   
                elif tbl.split('_')[1] == 'data':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.Data_com_df.astype(str),ifexist)   
                elif tbl.split('_')[1] == 'result':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.Result_df.astype(str),ifexist) 
    
    def summarySuggest(self):
        fct_time = self.fct_time_index
        if '海上陸上颱風紓流模式'in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '海上陸上颱風紓流模式' 
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest颱風')
            pass
        elif '豪雨特報紓流模式' in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '豪雨特報紓流模式'
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest豪雨')
        elif '大雨特報紓流模式' in self.Result_df.loc[fct_time,'豪大雨警報'].tolist():
            self.mode_type = '大雨特報紓流模式'
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest大雨')
        else:
            self.mode_type = '晴天模式' 
            sug_raw_df = pd.read_excel(self.iotable_path,index_col=0,sheet_name='Suggest晴天')
            # self.Result_df['豪大雨警報']
            
        
        strategy_name_list = ['晴天模式', '截流關閉','景美,松山,昆陽,內湖紓流','B43,緊急進/紓流','士林,松信,新建,忠孝,六館,葫蘆紓流']
        idx_list = ['DateTime',"模式","模擬時間","最大平均時雨量(mm)","最大時雨量時間",
                    "尖峰進流量(CMH)","尖峰時間","尖峰抽水機啟動數", '現況指標','現況指標說明',
                    '指標_截流站關閉','指標1','指標2-1','指標2-2','指標3','指標4','指標5',
                    "水情分析"]
        self.weather_df = pd.DataFrame( index =idx_list, columns = ['項次','說明'])
        
        self.weather_df['項次'] = idx_list
        self.weather_df.loc ['DateTime','說明'] = self.time_dict['t_time']
        self.weather_df.loc ['模式','說明'] = self.mode_type
        self.weather_df.loc['模擬時間','說明'] = self.time_dict['t_time'].strftime("%Y/%m/%d %H:%M")
    
        self.weather_df.loc["最大平均時雨量(mm)",'說明'] =self.Result_df.loc[fct_time,'平均時雨量'].astype(float).max().round(2)
        self.weather_df.loc["最大時雨量時間",'說明'] =self.Result_df.loc[fct_time,'平均時雨量'].astype(float).idxmax().strftime("%Y/%m/%d %H:%M")
        self.weather_df.loc["尖峰進流量(CMH)",'說明'] =self.Result_df.loc[fct_time,'迪化抽水站總瞬間流量(cmh)'].astype(float).max().round(0)
        self.weather_df.loc["尖峰時間",'說明'] =self.Result_df.loc[fct_time,'迪化抽水站總瞬間流量(cmh)'].astype(float).idxmax() .strftime("%Y/%m/%d %H:%M")       
        self.weather_df.loc["尖峰抽水機啟動數",'說明'] = round(self.Result_df.loc[self.weather_df.loc["尖峰時間",'說明'],'迪化抽水機總數'],0)
        tsim = self.weather_df.loc['模擬時間','說明']; tRmax= self.weather_df.loc["最大時雨量時間",'說明'];tQmax=self.weather_df.loc["尖峰時間",'說明']
        Rmax= self.weather_df.loc["最大平均時雨量(mm)",'說明'];Qmax=int(self.weather_df.loc["尖峰進流量(CMH)",'說明']);Nmax= int(self.weather_df.loc["尖峰抽水機啟動數",'說明'])
                
        self.weather_df.loc["水情分析",'說明'] = f"""{self.mode_type}：{tsim} 預報未來24小時
    最大10分鐘平均雨量：{tRmax} {Rmax} mm
    最大總瞬間流量：{tQmax} {Qmax} CMH(啟動{Nmax}台)"""
        
    
        ind_list = ['指標_截流站關閉','指標1','指標2-1','指標2-2','指標3','指標4','指標5']
        t_time_ind =  self.Result_df.loc[ self.time_dict['t_time'],ind_list].sum()
        if t_time_ind>0:
            self.weather_df.loc['現況指標','說明'] = ind_list[t_time_ind-1]
            self.weather_df.loc['現況指標說明','說明'] = sug_raw_df.loc[ind_list[t_time_ind-1],'指標說明']

        for ind in ind_list:
            if self.Result_df[ind].sum() > 0:
                abb = self.Result_df.loc[:,ind] == 1
                self.weather_df.loc[ind,'說明'] = self.Result_df.loc[:,ind] .loc[abb.tolist()] .index[0]
        file_names = [os.path.join(self.save_path,'%s.csv' %self.save_name)]
        all_exist = True
        for file_name in file_names:
            file_path = os.path.join(self.save_path, file_name)
            if not os.path.isfile(file_path):
                all_exist = False
                break
        # file_dict = {}
        # len(self.fct_time_index[:6*24])
        # 24*6
        # self.time_dict
        fct_len = 24
        # self.Result_df.index
        # 180/6
        self.sel_hsf_fct_time_index = self.hsf_fct_time_index[:len(self.hsf_time_index)+6*fct_len] #181
        self.sel_fct_time_index = self.fct_time_index[:6*fct_len] #181
        
        
        
        strategy_name_list = ['DateTime','RecordTime','數值說明','歷史降雨','預報雨量','歷史迪抽總瞬間流量','預報迪抽總瞬間流量','歷史台數','預報台數',
                              '歷史濕井液位','預報濕井液位','歷史緊繞液位','預報緊繞液位',  
                              '歷史B43液位','預報B43液位',  
                              '歷史閘門3031','預報閘門3031','歷史閘門3041','預報閘門3041']
        # file_dict[iii] = pd.read_csv(os.path.join(self.save_path, file_names[iii]), encoding='big5', index_col=0)
        # file_dict[iii].index = pd.to_datetime(file_dict[iii].index)
        self.Data_com_df = pd.DataFrame(index = self.sel_hsf_fct_time_index, columns=strategy_name_list)
        self.Data_com_df = self.Data_com_df.assign(DateTime=pd.to_datetime(self.time_dict['t_time']))
        self.Data_com_df = self.Data_com_df.assign(RecordTime=pd.to_datetime(self.sel_hsf_fct_time_index))
        self.Data_com_df['數值說明'] = self.Result_df['Simulated time']
        # self=a 
        # aaaaaa  =self.Result_df
        
        # self.realtime_raw_df ['']
        # self.rainfall_df
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史降雨'] = self.Result_df.loc[self.hsf_time_index,'平均10分鐘雨量'].fillna(0).round(2)        
        self.Data_com_df.loc[self.sel_fct_time_index,'預報雨量'] = self.Result_df.loc[self.sel_fct_time_index,'平均10分鐘雨量'].astype(float).round(2)        
        self.Data_com_df.loc[self.sel_fct_time_index,'預報雨量'] = self.Result_df.loc[self.sel_fct_time_index,'平均10分鐘雨量'].astype(float).round(2)         
        # 平均10分鐘雨量
        self.Data_com_df.loc[self.hsf_time_index,'歷史迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)(觀測)'].fillna(0).round(2)               
        self.Data_com_df.loc[self.hsf_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float).round(2)         
        self.Data_com_df.loc[self.sel_fct_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float).round(2)         
        #繪圖平緩 取歷史最後一點 跟預報第二點平均 取代第一點
        self.Data_com_df.loc[self.sel_fct_time_index[0],'預報迪抽總瞬間流量'] = (self.Data_com_df.loc[self.hsf_time_index[-1],'歷史迪抽總瞬間流量']+self.Data_com_df.loc[self.sel_fct_time_index[1],'預報迪抽總瞬間流量'])/2
        # self.Data_com_df.loc[self.hsf_time_index,'歷史迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)(觀測)'].fillna(0).round(2)   
        # self.Data_com_df.loc[self.hsf_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float)      .round(2)
        # self.Data_com_df.loc[self.sel_fct_time_index,'預報迪抽總瞬間流量'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站總瞬間流量(cmh)'].astype(float)      .round(2)
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史台數'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水機總數(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.hsf_time_index,'預報台數'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水機總數']      
        self.Data_com_df.loc[self.sel_fct_time_index,'預報台數'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水機總數']      
    
        self.Data_com_df.loc[self.hsf_time_index,'歷史濕井液位'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站濕井液位(觀測)'].fillna(0).round(2)               
        self.Data_com_df.loc[self.hsf_time_index,'預報濕井液位'] = self.Result_df.loc[self.hsf_time_index,'迪化抽水站濕井液位'].astype(float).round(2)         
        self.Data_com_df.loc[self.sel_fct_time_index,'預報濕井液位'] = self.Result_df.loc[self.sel_fct_time_index,'迪化抽水站濕井液位'].astype(float).round(2)         
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史緊繞液位'] = self.Result_df.loc[self.hsf_time_index,'緊急繞流井液位(觀測)'].fillna(0).round(2)   
        self.Data_com_df.loc[self.hsf_time_index,'預報緊繞液位'] = self.Result_df.loc[self.hsf_time_index,'緊急繞流井液位'].astype(float).round(2)   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報緊繞液位'] = self.Result_df.loc[self.sel_fct_time_index,'緊急繞流井液位'].astype(float).round(2)   
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史B43液位'] = self.Result_df.loc[self.hsf_time_index,'B43人孔液位(觀測)'].fillna(0).round(2)   
        self.Data_com_df.loc[self.hsf_time_index,'預報B43液位'] = self.Result_df.loc[self.hsf_time_index,'B43人孔液位'].astype(float).round(2)   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報B43液位'] = self.Result_df.loc[self.sel_fct_time_index,'B43人孔液位'].astype(float).round(2)   
        
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史閘門3031'] = self.Result_df.loc[self.hsf_time_index,'迪化3031主閘門(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.hsf_time_index,'預報閘門3031'] = self.Result_df.loc[self.hsf_time_index,'迪化3041主閘門']   *100   
        self.Data_com_df.loc[self.sel_fct_time_index,'預報閘門3031'] = self.Result_df.loc[self.sel_fct_time_index,'迪化3041主閘門']   *100   
        
        self.Data_com_df.loc[self.hsf_time_index,'歷史閘門3041'] = self.Result_df.loc[self.hsf_time_index,'迪化3041主閘門(觀測)'].fillna(0)            
        self.Data_com_df.loc[self.hsf_time_index,'預報閘門3041'] = self.Result_df.loc[self.hsf_time_index,'迪化3041主閘門'] *100
        self.Data_com_df.loc[self.sel_fct_time_index,'預報閘門3041'] = self.Result_df.loc[self.sel_fct_time_index,'迪化3041主閘門'] *100
        
        Q_df = self.Data_com_df.loc[:,'預報迪抽總瞬間流量'] 
        p_df = self.Data_com_df.loc[:,'預報台數'] 
        w_df = self.Result_df.loc[:,'豪大雨警報'] 
        strategy_name_list = ['DateTime','OpTime','操作建議','指標','Order']
        self.suggest_df = pd.DataFrame( columns =strategy_name_list)
        cnt=0
        # self=a
        if self.mode_type == '晴天模式':                
            ttmp = self.time_dict['t_time'].strftime("%m/%d %H:%M")
            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
            self.suggest_df.loc[cnt,'OpTime'] = ttmp
            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc['指標_截流站關閉','操作建議']
            self.suggest_df.loc[cnt,'Order'] = 1
            
        else:      
            for ind in ind_list:
                if self.mode_type != '豪雨特報紓流模式' and ind == '指標2-2': 
                    continue
                else:
                    # # self.Result_df[ind].tolist().index(0)
                    # if int(self.Result_df[ind].astype(float).idxmax().strftime("%m"))<10:
                    #     ttmp = self.Result_df[ind].astype(float).idxmax().strftime("%m/%d %H:%M")[1:]
                    # else:
                    if not pd.isnull(self.weather_df.loc[ind,'說明']):
                        if self.weather_df.loc[ind,'說明'] > self.time_dict['t_time']:
                            ttmp =  self.weather_df.loc[ind,'說明'].strftime("%m/%d %H:%M")
                            # self.suggest_df.loc[cnt,'DateTime'] = ttmp
                            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
                            self.suggest_df.loc[cnt,'OpTime'] = ttmp
                            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc[ind,'指標說明']
                            self.suggest_df.loc[cnt,'指標'] = ind
                            self.suggest_df.loc[cnt,'Order'] = 1
                            cnt=cnt+1
                            
                            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
                            self.suggest_df.loc[cnt,'OpTime'] = ttmp
                            self.suggest_df.loc[cnt,'操作建議'] = sug_raw_df.loc[ind,'操作建議']
                            self.suggest_df.loc[cnt,'指標'] = ind
                            self.suggest_df.loc[cnt,'Order'] = 3
                            cnt=cnt+1    
            
        sug_fct_time = 12                            
                        
        mode_type_list = ['晴天模式','大雨特報紓流模式','豪雨特報紓流模式','海上陸上颱風紓流模式']
        delw = w_df == w_df.shift(1) # .iloc[1:]
        delw_v = delw[delw==False]
        t12= self.time_dict['t_time'] + pd.Timedelta(hours=sug_fct_time)
        bool_d = np.logical_and(delw_v.index>=self.time_dict['t_time'], delw_v.index<=t12)
        delw_vt = delw_v.loc[bool_d]
            # ddd = 0
            
            
        for ddd in range(len(delw_vt)):
            total_w = w_df.loc[delw_vt.index[ddd]]
            total_w_1 = w_df.loc[delw_vt.index[ddd]-pd.Timedelta(minutes=10)]
            
            # total_Q = int(Q_df.loc[delp_vt.index[ddd]])
            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
            self.suggest_df.loc[cnt,'OpTime'] = delw_vt.index[ddd].strftime("%m/%d %H:%M")
            # abs_v = abs(delp_vt.iloc[ddd])
            
            if mode_type_list.index(total_w) >mode_type_list.index(total_w_1):
                self.suggest_df.loc[cnt,'操作建議'] = f"""由 {total_w_1} 升級為 {total_w} """
            elif mode_type_list.index(total_w) < mode_type_list.index(total_w_1):
                self.suggest_df.loc[cnt,'操作建議'] = f"""由 {total_w_1} 降級為 {total_w} """
            self.suggest_df.loc[cnt,'Order'] = 2
                
            # self.suggest_df.loc[cnt,'指標'] = ind
            cnt=cnt+1                 
            
            
           
          
        delp = (p_df - p_df.shift(1)).dropna()
        delp_v = delp[delp!=0]
        t12= self.time_dict['t_time'] + pd.Timedelta(hours=sug_fct_time)
        bool_d = np.logical_and(delp_v.index>=self.time_dict['t_time'], delp_v.index<=t12)
        delp_vt = delp_v.loc[bool_d]
        
        for ddd in range(len(delp_vt)):
            total_p = p_df.loc[delp_vt.index[ddd]]
            total_Q = int(Q_df.loc[delp_vt.index[ddd]])
            self.suggest_df.loc[cnt,'DateTime']  = self.time_dict['t_time']
            self.suggest_df.loc[cnt,'OpTime'] = delp_vt.index[ddd].strftime("%m/%d %H:%M")
            abs_v = abs(delp_vt.iloc[ddd])
            if delp_vt.iloc[ddd] >0:
                self.suggest_df.loc[cnt,'操作建議'] = f"""加開{abs_v}台抽水機(共啟動{total_p}台)，抽水量達{total_Q} CMH"""
            elif delp_vt.iloc[ddd] <0:
                self.suggest_df.loc[cnt,'操作建議'] = f"""關閉{abs_v}台抽水機(共啟動{total_p}台)，抽水量達{total_Q} CMH"""
            self.suggest_df.loc[cnt,'Order'] = 4
            cnt=cnt+1  
    
    
     
            
        self.suggest_df = self.suggest_df.sort_values(by=['OpTime','Order'])     
        
    
        if self.sys == 'windows':
            self.xlsxname = f'Suggest.%s.xlsx' %self.save_name
            # filename_Exl_com=os.path.join('.xlsx' %(sewer_thold,R_thold,WL_ver,discharge_thold)  )
            # filename_Exl_com=os.path.join('1.Output_2011_AAC.xlsx'  )
            writer_combination = pd.ExcelWriter(os.path.join(self.save_path,self.xlsxname ) )
            self.weather_df.to_excel(writer_combination, '分析') 
            self.suggest_df.to_excel(writer_combination, '操作建議')  
            self.Data_com_df.to_excel(writer_combination, '資料')  
            writer_combination.close()
        elif self.sys == 'linux':
            
            # self.Result_df
                
            DB_PARAMS_analysis = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_analysis',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            
            DB_PARAMS_operation= {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_operation',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            DB_PARAMS_data = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'sug_data',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            DB_PARAMS_Result = {
                "dialect_driver": "mssql+pymssql",
                "host": "192.168.32.106",
                "database": 'swmm_result',
                "user": 'sa',
                "password": 'ssofms%40TP', #@
                "port": '1433',
                "show_info": False
            }
            
            
            self.df2sql(self.save_name, DB_PARAMS_analysis, self.weather_df.astype(str),'replace')
            self.df2sql(self.save_name, DB_PARAMS_operation, self.suggest_df,'replace')
            self.df2sql(self.save_name, DB_PARAMS_data, self.Data_com_df,'replace')
            self.df2sql(self.save_name, DB_PARAMS_Result, self.Result_df,'replace')
            
            # self=a
            self.weather_df_T = self.weather_df[['說明']].T
            # self.suggest_df = self.suggest_df['說明'].T
            # self.Data_com_df = self.Data_com_df['說明'].T
            DB_PARAMS_suggests =  {
                    "dialect_driver": "mssql+pymssql",
                    "host": "192.168.32.106",
                    "database": 'sug_opr' ,
                    "user": 'sa',
                    "password": 'ssofms%40TP', #@
                    "port": '1433',
                    "show_info": False
                    }
            DB_tbls = {}
            
            DB_tbls_list = [
                            'sug_analysis_new','sug_analysis_his',
                            'sug_operation_new','sug_operation_his',
                            'sug_data_new','sug_data_his', 
                            'swmm_result_new','swmm_result_his',
                            ]
            for tbl in DB_tbls_list:
                print(tbl)
                tbl_ = tbl.split('_')
                if tbl_[2] == 'his':
                    ifexist = 'append'
                elif tbl_[2] == 'new':
                    ifexist = 'replace'
                if tbl.split('_')[1] == 'analysis':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.weather_df_T.astype(str),ifexist)
                elif tbl.split('_')[1] == 'operation':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.suggest_df.astype(str),ifexist)                   
                elif tbl.split('_')[1] == 'data':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.Data_com_df.astype(str),ifexist)  
                elif tbl.split('_')[1] == 'result':
                    self.df2sql(tbl, DB_PARAMS_suggests, self.Result_df.astype(str),ifexist)  
                    
def setting(): 
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
        save_path = os.path.join(oripath,'00Result')
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
        fct_s_time17 = datetime(2024, 7, 24, 0, 0)        
        fct_s_time18 = datetime(2024, 7, 25, 0, 0)               
    
        fct_s_time81 = datetime(2024, 8, 15, 0, 0)     
        fct_s_time82 = datetime(2024, 8, 19, 0, 0)    #9        
        fct_s_time83 = datetime(2024, 8, 20, 0, 0)    #9        
        fct_s_time84 = datetime(2024, 8, 23, 0, 0)    #9           
        fct_s_time85 = datetime(2024, 8, 29, 0, 0)    #9
        
        # 8月15、19、20、23、29日
        
        # runcell(0, 'E:/E_Program/SSO_swmm/00Code/Sewer_simulation3/run_sewer_pyswmm0906.py')
        # runcell(0, 'E:/E_Program/SSO_swmm/00Code/Sewer_simulation3/run_sewer_pyswmm0815.py')
        # fct_s_time = fct_s_time83
        fct_s_time = fct_s_time16
        
        t_time =  fct_s_time - relativedelta(minutes=10)  
        
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
                        'load_data_e_time': t_time+pd.Timedelta(hours=fct_lead_time_len)
                        }
    # t_time =  datetime(2024, 7, 5, 3, 10)

        Rfct_type = 0
        
        list_fct_s_time = [fct_s_time]
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
        Realtime_data_df = pd.read_csv('Realtime_data_201901_202408.csv', encoding = 'big5', index_col=0)
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
        
        # dc = 0.45  # 閘門入流細數
        dc = 0.5
        self=a
        for strategy in strategy_list:
            # for r in range(0.02, 0.1, 0.01):
                # for dc in np.arange(0.2, 0.9, 0.05):
                save_name =  '%s_h%01df%02d_%s' %(savename_dict[strategy],
                                                  hsf_time_len,fct_lead_time_len,\
                                                        t_time.strftime("%Y%m%d_%H%M"))
                # savename =  '%s_h%01df%02d_%s_%s' %(savename_dict[strategy],hsf_time_len,fct_lead_time_len,\
                #                                     fct_s_time.strftime("%Y%m%d_%H%M"),dc)
                a.out_inp(time_dict, strategy, oripath, save_name,save_path, dc,Rfct_type,rain_bqpf_df)
                log_write_not_done = True
                if log_write_not_done:
                    a.action_log_set()
                    log_write_not_done = False
                # T1 = time.time()
                a.run()
                a.action_history_log.to_csv('00Result\log_%s.csv' %save_name)
                a.load()
                # a.load(save_name, save_path, time_dict, strategy)
                # if platform.system().lower() == 'windows':
                a.summarySuggest()
                if __name__ == '__main__':
                    pass

                    a.plot_b43_emg()
                    a.plot_wl(level_path)
                    a.plot()

                # a.summarySTData()
        # a.plot_all(save_path, time_dict)
        # self.sim.close()
        
        T2 = time.time()
        self=a
        print('Done! Run time: %.3f min' %( (T2-T1)/60))
            
        
        #%%
if __name__ == '__main__':
    a= setting()
    # import platform
    self = a
    # self.pump_type
