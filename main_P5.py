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

    # Installation on water
    df_ports['continent'] = df_ports['continent'].mask(df_ports['continent_old'] == 'Others', 'Others')
    df_ports['country'] = df_ports['country'].mask(df_ports['continent_old'] == 'Others', 'Installations in International Waters')
    df_ports['subdivision'] = df_ports['subdivision'].mask(df_ports['continent_old'] == 'Others', 'Installations in International Waters')
    df_ports['city'] = df_ports['city'].mask(df_ports['continent_old'] == 'Others', df_ports['PortName'])
    df_ports['city_en'] = df_ports['city_en'].mask(df_ports['continent_old'] == 'Others', df_ports['PortName'])
    df_ports_OTHERS = df_ports[df_ports['continent'] == 'Others']
    df_ports_OTHERS = df_ports_OTHERS.sort_values(by=['country', 'subdivision', 'city'])
    df_ports_OTHERS.to_excel('./output_pt2/metadata/GlobalPorts_v20240819_Others.xlsx', index=False)

    # Asia
    df_ports['country'] = df_ports['country'].mask(df_ports['country'] == 'United Arab Emirates', 'United Arab Emirates (the)')
    df_ports['subdivision'] = df_ports['subdivision'].mask(df_ports['subdivision'] == '- Zayed City - Abu Dhabi', 'Abu Dhabi')

    df_ports_Asia = df_ports[df_ports['continent'] == 'Asia']
    df_ports_Asia = df_ports_Asia.sort_values(by=['country', 'subdivision', 'city'])
    df_ports_Asia.to_excel('./output_pt2/metadata/GlobalPorts_v20240819_Asia.xlsx', index=False)

    # North America
    usecols = ['UNLOCODE', 'continent', 'country', 'subdivision', 'city', 'city_en']
    df_NA = pd.read_excel('./output_pt2/processing_list/North America/North America_address_adjusted2.xlsx')
    df_NA = df_NA[usecols]
    df_ports = pd.merge(df_ports, df_NA, on='UNLOCODE', how='left', suffixes=('', '_NA'))
    for col in ['continent', 'country', 'subdivision', 'city', 'city_en']:

        col1 = col
        col2 = col + '_NA'
        df_ports.loc[df_ports[col2].notnull(), col1] = df_ports[col2]
    df_ports = df_ports.drop(columns=[col + '_NA' for col in ['continent', 'country', 'subdivision', 'city', 'city_en']])
    df_ports_NA = df_ports[df_ports['continent'] == 'North America']
    df_ports_NA = df_ports_NA.sort_values(by=['country', 'subdivision', 'city'])
    df_ports_NA.to_excel('./output_pt2/metadata/GlobalPorts_v20240819_NA.xlsx', index=False)

    # Code reference table
    df_code = df_ports[['UNLOCODE', 'continent', 'country', 'subdivision', 'city', 'city_en']]
    df_code['FullPortStyle'] = df_code['UNLOCODE']
    df_code = df_code.drop_duplicates(subset=['UNLOCODE'])
    df_code = df_code.sort_values(by=['UNLOCODE'])
    df_code = df_code.reset_index(drop=True)
    df_code = df_code.fillna('TBF')
    df_code.to_excel('./output_pt2/metadata/UNLOCODE_reference.xlsx', index=False)

    # Save the final file
    ### Some continents have not been processed yet ###
    df_ports['FullPortStyle_old'] = df_ports['FullPortStyle']
    df_ports['FullPortStyle'] = df_ports['UNLOCODE']
    df_ports = df_ports[['UNLOCODE', 'PortName', 'FullPortStyle', 'lat', 'lon']]
    df_ports = df_ports.sort_values(by=['UNLOCODE','lat', 'lon'])
    df_ports.to_excel('./output_pt2/metadata/GlobalPorts_FullPortStyle_v20240819.xlsx', index=False)