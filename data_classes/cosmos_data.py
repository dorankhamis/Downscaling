import numpy as np
import pandas as pd
import glob

from soil_moisture.utils import replace_missing, indexify_datetime, data_dir
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
        self.daily = pd.read_csv(glob.glob(fldr + '/daily/COSMOS-UK_' + self.SID + '_HydroSoil_Daily*.csv')[0])
  
    def read_hourly(self):
        self.hourly = pd.read_csv(glob.glob(fldr + '/hourly/COSMOS-UK_' + self.SID + '_HydroSoil_Hourly*.csv')[0])
  
    def read_subhourly(self):
        self.subhourly = pd.read_csv(glob.glob(fldr + '/subhourly/COSMOS-UK_' + self.SID + '_HydroSoil_SH*.csv')[0])
      
    def preprocess_all(self, missing_val= -9999.0, dt_name='DATE_TIME', use_qc_flags=False):
        if not self.daily is None:
            self.daily = replace_missing(self.daily, missing_val)
            self.daily = indexify_datetime(self.daily, dt_name, drop=True, utc=True)

        if not self.hourly is None:
            self.hourly = replace_missing(self.hourly, missing_val)
            self.hourly = indexify_datetime(self.hourly, dt_name, drop=True, utc=True)
        
        if not self.subhourly is None:
            self.subhourly = replace_missing(self.subhourly, missing_val)
            self.subhourly = indexify_datetime(self.subhourly, dt_name, drop=True, utc=True)

            self.subhourly.loc[self.subhourly['SWIN']<0, 'SWIN'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['SWOUT']<0, 'SWOUT'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['LWIN']<0, 'LWIN'] = 0 # cannot have negative
            self.subhourly.loc[self.subhourly['LWOUT']<0, 'LWOUT'] = 0 # cannot have negative
            self.subhourly['TA'] = self.subhourly['TA'] + 273.15 # convert to Kelvin to remove -ve

    
    
class CosmosMetaData():
  
    def __init__(self):
        self.daily = pd.read_csv(fldr + '/daily/COSMOS-UK_HydroSoil_Daily_2013-2019_Metadata.csv')
        self.hourly = pd.read_csv(fldr + '/hourly/COSMOS-UK_HydroSoil_Hourly_2013-2019_Metadata.csv')
        self.subhourly = pd.read_csv(fldr + '/subhourly/COSMOS-UK_HydroSoil_SH_2013-2019_Metadata.csv')
        self.site = pd.read_csv(fldr + '/COSMOS-UK_SiteMetadata_2013-2019.csv')
        self.tdt_config = pd.read_csv(fldr + '/TDT_CONFIG.csv')
    
    
