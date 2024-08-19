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
    jj.redirect_incomplete_address(file_NA)