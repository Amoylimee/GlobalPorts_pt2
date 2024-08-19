# Based on the old script, this part continues the work.
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

    # Asia
    # Skip, Asia has been processed in the previous script.

    # North America
    file_NA = '/Users/jeremy/Downloads/working/GlobalPorts/output_pt2/processing_list/North America/North America_address.xlsx'
    df_NA = pd.read_excel(file_NA)
    df_NA = df_NA.rename(columns={'参数2_文本': 'coordinates', '参数3_文本': 'address'})

    file_redirect_processed = './output_pt2/processing_list/North America/North America_redirected_processed.xlsx'
    df_redirected_processed_NA = pd.read_excel(file_redirect_processed)
    df_redirected_processed_NA = df_redirected_processed_NA.rename(columns={'参数2_文本': 'coordinates_new', '参数3_文本': 'address_new'})


    if len(df_redirected_processed_NA[~df_redirected_processed_NA['address_new'].str.contains(',')]) > 0:
        raise ValueError('There are incomplete addresses in the redirected file.')

    df_NA = df_NA.merge(df_redirected_processed_NA, on=['lat', 'lon'], how='left')

    df_NA.loc[df_NA['address_new'].notnull(), 'address'] = df_NA.loc[df_NA['address_new'].notnull(), 'address_new']

    df_NA = df_NA[['lat', 'lon', 'coordinates', 'address']]

    df_NA.to_excel('./output_pt2/processing_list/North America/North America_address_adjusted.xlsx',index=False)