import numpy as np
import pandas as pd
import glob

from soil_moisture.utils import preprocess, data_dir
fldr = data_dir + '/COSMOS/'


class CosmosData():
  
    def __init__(self, SID):
        self.SID = SID
        self.daily = None
        self.hourly = None
        self.subhourly = None

    def read_all_data(self):
        self.read_daily()
        self.read_hourly()
        self.read_subhourly()
    
    def read_QC_flags(self):
        self.qc_flags = pd.read_csv(glob.glob(fldr + '/QC_flags/COSMOS-UK_' + self.SID + '_HydroSoil_SH_*_QC_Flags.csv')[0])
    
    def read_daily(self):
        self.daily = pd.read_csv(glob.glob(fldr + '/daily/COSMOS-UK_' + self.SID + '_HydroSoil_Daily_*.csv')[0])
  
    def read_hourly(self):
        self.hourly = pd.read_csv(glob.glob(fldr + '/hourly/COSMOS-UK_' + self.SID + '_HydroSoil_Hourly_*.csv')[0])
  
    def read_subhourly(self):
        self.subhourly = pd.read_csv(glob.glob(fldr + '/subhourly/COSMOS-UK_' + self.SID + '_HydroSoil_SH_*.csv')[0])
      
    def preprocess_all(self, missing_val=np.nan, dt_name='DATE_TIME', use_qc_flags=False):
        try:
            self.daily = preprocess(self.daily, missing_val, dt_name, add_stamp='T00:00:00Z')
        except:
            print(f'Failed to preprocess daily for {self.SID}')
        try:
            self.hourly = preprocess(self.hourly, missing_val, dt_name)
        except:
            print(f'Failed to preprocess hourly for {self.SID}')
        try:
            if use_qc_flags is True:
                self.read_QC_flags()
                cl = self.qc_flags.columns[2:]
                clv = self.subhourly.columns[2:]
                clmap = {oldname:oldname[:-7] for oldname in cl}
                mask = self.qc_flags.loc[:, cl] != 0.
                self.subhourly.loc[:, clv][mask.rename(columns = clmap)] = missing_val                
            self.subhourly = preprocess(self.subhourly, missing_val, dt_name)
        except:
            print(f'Failed to preprocess subhourly for {self.SID}')
        try:
            self.subhourly.loc[self.subhourly['SWIN']<0, 'SWIN'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['SWOUT']<0, 'SWOUT'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['LWIN']<0, 'LWIN'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['LWOUT']<0, 'LWOUT'] = 0 # cannot have negative
            self.subhourly['TA'] = self.subhourly['TA'] + 273.15 # convert to Kelvin to remove -ve
        except:
            pass
    
    
class CosmosMetaData():
  
    def __init__(self):
        self.daily = pd.read_csv(fldr + '/daily/COSMOS-UK_HydroSoil_Daily_2013-2019_Metadata.csv')
        self.hourly = pd.read_csv(fldr + '/hourly/COSMOS-UK_HydroSoil_Hourly_2013-2019_Metadata.csv')
        self.subhourly = pd.read_csv(fldr + '/subhourly/COSMOS-UK_HydroSoil_SH_2013-2019_Metadata.csv')
        self.site = pd.read_csv(fldr + '/COSMOS-UK_SiteMetadata_2013-2019.csv')
        self.tdt_config = pd.read_csv(fldr + '/TDT_CONFIG.csv')
    
    
