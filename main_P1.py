# Based on the old script, this part continues the work.
import pandas as pd
import numpy as np
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

# In
file_global_ports = 'output/P6_cleaned/All/GlobalPorts_v20240814.xlsx'
df_ports = pd.read_excel(file_global_ports)

# Out
dir_output_all = './output_pt2/metadata'
jj.io_create_new_folder(dir_output_all)

print('Data without continents_old', df_ports['continent_old'].isnull().sum())
continents = list(df_ports['continent_old'].unique())
continents = sorted(continents) 
print('Continents_old:', continents)

for continent in continents:
    print('Processing:', continent)

    dir_output = f'./output_pt2/processing_list/{continent}'
    jj.io_create_new_folder(dir_output)

    df_continent = df_ports[df_ports['continent_old'] == continent]

    df_with_code = df_continent[df_continent['UNLOCODE'].notnull()]
    df_with_code = df_with_code.drop_duplicates(subset=['UNLOCODE'])
    df_without_code = df_continent[df_continent['UNLOCODE'].isnull()]

    df_combined = pd.concat([df_with_code, df_without_code], axis=0)
    df_combined = df_combined.sort_values(by=['UNLOCODE', 'PortName'])
    df_combined = df_combined.reset_index(drop=True)

    # Fill missing UNLOCODE
    df_combined['index'] = df_combined.index.astype(str)
    df_combined['UNLOCODE_fillna'] = df_combined['continent_old'].str[:2].str.upper() + df_combined['index'].str.zfill(3)
    df_combined['UNLOCODE'] = df_combined['UNLOCODE'].fillna(df_combined['UNLOCODE_fillna'])
    df_combined = df_combined.drop(columns=['index', 'UNLOCODE_fillna'])
    print('UNLOCODE null:', df_combined['UNLOCODE'].isnull().sum())

    df_code_lat_lon = df_combined[['UNLOCODE', 'lat', 'lon']]
    df_code_lat_lon = df_code_lat_lon.rename(columns={'UNLOCODE': 'UNLOCODE_new'})
    df_ports = df_ports.merge(df_code_lat_lon, on=['lat', 'lon'], how='left')
    df_ports.loc[df_ports['UNLOCODE'].isnull(), 'UNLOCODE'] = df_ports.loc[df_ports['UNLOCODE'].isnull(), 'UNLOCODE_new']
    df_ports = df_ports.drop(columns=['UNLOCODE_new'])

    df_lat_lon = df_combined[['lat', 'lon']]

    if continent == 'Asia':     # Asia has been processed in the previous script.

        file_combined = f'{dir_output}/{continent}_combined.xlsx'
        df_combined.to_excel(file_combined, index=False)

    file_combined = f'{dir_output}/{continent}_check.csv'
    file_lat_lon = f'{dir_output}/{continent}_lat_lon.csv'

    df_combined.to_csv(file_combined, index=False)
    df_lat_lon.to_csv(file_lat_lon, index=False)

    print(df_ports['UNLOCODE'].isnull().sum())
    
df_ports.to_excel(f'{dir_output_all}/GlobalPorts_v20240819.xlsx', index=False)