import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
import os
import glob
from shapely.geometry import Point
import functions as jj

# setting maximum display rows and columns
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
# ignore warnings
warnings.filterwarnings('ignore')

# setting working directory
os.chdir('/Users/jeremy/Downloads/working/GlobalPorts')

if __name__ == '__main__':

    file_global_ports = 'output_pt2/metadata/GlobalPorts_v20240819.xlsx'
    df_ports = pd.read_excel(file_global_ports)

    # Asia
    # Skip, Asia has been processed in the previous script.

    # North America
    file_NA = './output_pt2/processing_list/North America/North America_address_adjusted.xlsx'
    df_NA = pd.read_excel(file_NA)
    df_NA.loc[~df_NA['address'].str.contains('\+'), 'address'] = 'HKG+HKG ' + df_NA.loc[~df_NA['address'].str.contains('\+'), 'address'] 
    
    df_NA = jj.split_address(df_NA)   

    df_NA = jj.P4_main_NA(df_NA)
    jj.get_info(df_NA, './output_pt2/processing_list/North America/NA_address_split_info.txt')

    # df_NA = jj.city_to_en(df_NA)
    df_NA['city_en'] = df_NA['city']        # all are in English or Spanish, no need to convert     
    jj.check_UNLOCODE_complete(df_NA, df_ports)

    df_NA['continent'] = 'North America'
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'Argentina', 'South America')
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'India', 'Asia')
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'Russia', 'Europe')
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'Spain', 'Europe')
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'Uruguay', 'South America')
    df_NA['continent'] = df_NA['continent'].mask(df_NA['part_1'] == 'Venezuela', 'South America')
    df_NA['country'] = df_NA['part_1']                                            
    df_NA['subdivision'] = df_NA['part_2']

    df_NA = df_NA.merge(df_ports[['UNLOCODE', 'lat', 'lon']], on=['lat', 'lon'], how='left')
    df_NA.to_excel('./output_pt2/processing_list/North America/North America_address_adjusted2.xlsx',index=False)    