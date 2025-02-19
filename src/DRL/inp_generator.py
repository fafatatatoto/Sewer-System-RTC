from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import swmm,os
from swmm_api import SwmmInput
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
from swmm_api.input_file.sections import TimeseriesData, Timeseries
from dateutil.relativedelta import relativedelta 
import time

class inp_generate():
    def __init__(self,path_inp, path_realdata, path_iotable):
        # read files needed
        self.inp_path = path_inp
        self.original_inp = SwmmInput.read_file(path_inp,encoding='big5') # inp file
        self.inp = self.original_inp.copy()
        self.iotable_path = path_iotable
        self.realdata_path = path_realdata

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

        self.realtime_raw_df = pd.read_csv(self.realdata_path, encoding = 'big5', index_col=0)
        # self.realtime_raw_df = pd.read_csv(self.realdata_path, index_col=0)
        self.realtime_raw_df.index = pd.to_datetime(self.realtime_raw_df.index)

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
 
    def option_setting(self, time_dict, sum_type='hsf_fct'):
        if sum_type == 'hsf_fct':
            time_dict['fct_e_time'] = time_dict['fct_e_time'] + relativedelta(hours=1/6)
            self.inp[sections.OPTIONS]['START_DATE'] = datetime.date(time_dict['hsf_s_time'])
            self.inp[sections.OPTIONS]['START_TIME'] = datetime.time(time_dict['hsf_s_time'])
            self.inp[sections.OPTIONS]['END_DATE'] = datetime.date(time_dict['fct_e_time'])
            self.inp[sections.OPTIONS]['END_TIME'] = datetime.time(time_dict['fct_e_time'])
            self.inp[sections.OPTIONS]['REPORT_START_DATE'] = datetime.date(time_dict['hsf_s_time'])
            self.inp[sections.OPTIONS]['REPORT_START_TIME'] = datetime.time(time_dict['hsf_s_time'])
            # datetime.time 會出錯 所以虛擬一個時間
            self.inp[sections.OPTIONS]['WET_STEP'] = datetime.time(datetime(2020,10,10,0,10))
            self.inp[sections.OPTIONS]['DRY_STEP'] = datetime.time(datetime(2020,10,10,0,10))
        for kkk in self.inp[sections.RAINGAGES].keys():
            self.inp[sections.RAINGAGES][kkk]['interval'] = self.inp[sections.OPTIONS]['WET_STEP'] #虛擬時間 取10分鐘


    # def R_setting(self, type_sunwet):
    #     set_rain_index = pd.date_range(self.hsf_fct_time_index[0], periods=len(self.hsf_fct_time_index)+3, freq='10T') 
    #     self.rainfall_df = pd.DataFrame(index=set_rain_index)
    #     for iii in self.R_timeseries_list:
    #         self.inp[sections.RAINGAGES][iii]['form'] = 'VOLUME'
    #         tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
    #         f_df = pd.DataFrame(index=set_rain_index, columns=['Time','Value'])
    #         col_ch = self.col_df.loc[iii,'cols']
    #         f_df['Time'] = set_rain_index.strftime('%m/%d/%Y %H:%M:%S') 
    #         if type_sunwet == 0:
    #             f_df.loc[set_rain_index,'Value'] = 0
    #         elif type_sunwet == 1:
    #             f_df.loc[set_rain_index,'Value'] = self.realtime_raw_df.loc[set_rain_index, col_ch]
    #             # f_df.loc[self.hsf_fct_time_index,'Value'] = self.realtime_df.loc[self.hsf_fct_time_index, col_ch]*0.5
    #         self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
    #         self.rainfall_df[iii] = f_df.Value
    #     self.rainfall_df.replace(np.nan, 0, inplace=True)


    def R_setting(self, type_sunwet, Rfct_type=0):
        self.Rfct_type = Rfct_type
        self.R_list =  pd.read_excel(r'C:\Users\xul51\cleanrl\ID_tables.xlsx',sheet_name='Rainfall')['編號'].dropna().to_list()#.toslit()
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

                    self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)
                self.rainfall_df.loc[self.hsf_time_index,iii] = self.realtime_raw_df.loc[self.hsf_time_index, col_ch]
                self.rainfall_df.loc[self.fct_time_index,iii] = self.realtime_raw_df.loc[self.fct_time_index, col_ch]                    
                     
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

    
    def time_index_set(self):
        self.hsf_fct_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['fct_e_time'],freq='10min')
        self.hsf_time_index = pd.date_range(self.time_dict['hsf_s_time'],self.time_dict['hsf_e_time'],freq='10min')
        self.fct_time_index = pd.date_range(self.time_dict['fct_s_time'],self.time_dict['fct_e_time'],freq='10min')
    
    def pump_power_set(self, base_Hlv):
        pump_names = ['PUMP_CURVE_DIHWA1', 'PUMP_CURVE_DIHWA2', 'PUMP_CURVE_DIHWA3', 'PUMP_CURVE_DIHWA4', 'PUMP_CURVE_DIHWA5', 'PUMP_CURVE_DIHWA6',\
                      'PUMP_CURVE_DIHWA7', 'PUMP_CURVE_DIHWA7', 'PUMP_CURVE_DIHWA8', 'PUMP_CURVE_DIHWA9']
        pump_names2 = ['PUMP_DH1', 'PUMP_DH2', 'PUMP_DH3', 'PUMP_DH4', 'PUMP_DH5', 'PUMP_DH6', 'PUMP_DH7', 'PUMP_DH8', 'PUMP_DH9'] 
        self.base_Hlv = base_Hlv
        for p in pump_names:
            self.inp[sections.CURVES][p]['kind'] = 'PUMP4'
            # self.inp[sections.CURVES][p]['points'] = [[0, 0], [0.0899, 0], [0.089999999, self.base_Hlv], [3, self.base_Hlv]]
            self.inp[sections.CURVES][p]['points'] =  [[0, 0], [3.01, 0], [3.011, self.base_Hlv], [999, self.base_Hlv]]
        for p in pump_names2: 
            self.inp[sections.PUMPS][p]['depth_on'] = 0
            self.inp[sections.PUMPS][p]['depth_off'] = 0

    def basic_flow(self):
        for iii in self.Sunday_inflow_list:
            tmp = pd.DataFrame(self.inp[sections.TIMESERIES][iii]['data'])
            f_df = pd.DataFrame(index=self.hsf_fct_time_index, columns=['Time','Value','H'])
            f_df['Time'] = self.hsf_fct_time_index.strftime('%m/%d/%Y %H:%M:%S') 
            f_df['H'] = self.hsf_fct_time_index.hour
            f_df['Value'] = tmp.loc[f_df['H'],1].tolist()
            self.inp[sections.TIMESERIES][iii] = TimeseriesData(name=iii, data=f_df[['Time','Value']].values)

    def ini_level_0(self):
        self.inp[sections.STORAGE]['DIHWA']['depth_init'] = 0
    

    def out_inp(self, fct_s_time, save_path, base_Hlv):
        self.save_path = save_path
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
        self.inp[sections.REPORT]['CONTROLS'] = 'NO'
        self.option_setting(self.time_dict)
        self.time_index_set()
        self.pump_power_set(base_Hlv)
        self.R_setting(type_sunwet=1)
        self.basic_flow()
        self.ini_level_0()
        savename = 'rain_h%01df%02d_%s' %(hsf_time_len,fct_lead_time_len,fct_s_time.strftime("%Y%m%d_%H%M"))
        inp = "%s.inp" %(savename)
        self.inp.write_file(os.path.join(self.save_path, inp), encoding='big5')



if __name__=='__main__':
    path_inp = r'C:\Users\309\DLPA\history_h6f18_20230630_1200_0.55_modA_D_loquan_20240820.inp'
    path_realdata = r'C:\Users\309\DLPA\Realtime_data_202301_202408.csv'
    path_iotable = r'C:\Users\309\DLPA\IOtable.xlsx'
    a = inp_generate(path_inp, path_realdata, path_iotable)
    # os.makedirs(r'C:\Users\309\DLPA\envs', exist_ok=True)
    save_path = r'C:\Users\309\cleanrl\envs'
    fct_s_time = datetime(2024, 4, 13, 0, 0)
    a.out_inp(fct_s_time, save_path, 3)