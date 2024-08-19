import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
import geopandas as gpd
from openai import AzureOpenAI

# set working directory
os.chdir('/Users/jeremy/Downloads/working/GlobalPorts')

def find_ports_points(df, 
                      by_col: str, 
                      input: str):

    df1 = df.copy()
    df1 = df1[df1[by_col] == input]

    return df1

def io_create_new_folder(u_path: str) -> None:
    """ Create a new folder/directory if the folder/directory does not exist.
    
    io_create_new_folder(/disk/r046/jchenhl/Project_RBF/)
    """
    if os.path.isdir(u_path):
        pass
    else:
        os.makedirs(u_path)
    return

def adjust_address(df: pd.DataFrame, indices: list, new_value: str) -> pd.DataFrame:
 
    df1 = df.copy()
    
    for i in indices:
        df1.loc[df1['index'] == i, 'address'] = new_value
        print(f'Index {i} has been adjusted to {new_value}')
    
    return df1

def find_nearest(src_points, candidates):
    """在candidates中找到与src_points最近的点并返回索引"""
    nearest_points = []
    for src_point in src_points:
        distances = candidates.geometry.distance(src_point)
        nearest_point_idx = distances.idxmin()
        nearest_points.append(nearest_point_idx)
    return nearest_points

def merge_nearest_points(gdf1, gdf2):
    """对gdf1中的每个点找到gdf2中最近的点并合并其信息

    Args:
        gdf1 (GeoDataFrame): 包含选择点的GeoDataFrame
        gdf2 (GeoDataFrame): 包含候选最近点的GeoDataFrame

    Returns:
        GeoDataFrame: 合并了最近点信息的gdf1
    """
    # 确保两个GeoDataFrame使用相同的CRS
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)

    # 找到gdf2中与gdf1最近的点的索引
    nearest_indices = find_nearest(gdf1.geometry, gdf2)

    # 通过索引获取最近点的信息
    nearest_points = gdf2.iloc[nearest_indices].copy()
    nearest_points = nearest_points.reset_index(drop=True)
    
    # 重命名近点的geometry列为recent_geometry，避免与初始化列错合并问题
    nearest_points = nearest_points.rename(columns={'geometry': 'nearest_geometry'})

    # 合并信息
    combined = pd.concat([gdf1.reset_index(drop=True), nearest_points], axis=1)
    combined['distance_to_nearest'] = gdf1.geometry.distance(nearest_points['nearest_geometry'])

    return gpd.GeoDataFrame(combined, crs=gdf1.crs)

def redirect_incomplete_address(path_to_file_address:str):

    continent = path_to_file_address.split('/')[-2]

    file_worldcities = './data/worldcities.csv'
    df_worldcities = pd.read_csv(file_worldcities, usecols = ['city_ascii', 'lat', 'lng'])
    df_worldcities = df_worldcities.rename(columns={'lat': 'lat2', 'lng': 'lon2'})

    df_address = pd.read_excel(path_to_file_address)
    df_address = df_address.rename(columns={'参数2_文本': 'coordinates', '参数3_文本': 'address'})
    df_address = df_address.dropna(subset=['coordinates'])
    df_address_incomplete = df_address[~df_address['address'].str.contains(',')]
    df_address_incomplete = df_address_incomplete.reset_index(drop=True)

    print('Continent:', continent)
    print('Total address:', len(df_address))
    print('Incomplete address:', len(df_address_incomplete))

    gdf_address_incomplete = gpd.GeoDataFrame(df_address_incomplete, geometry=gpd.points_from_xy(df_address_incomplete['lon'], df_address_incomplete['lat']))
    gdf_worldcities = gpd.GeoDataFrame(df_worldcities, geometry=gpd.points_from_xy(df_worldcities['lon2'], df_worldcities['lat2']))

    df_find_nearest = merge_nearest_points(gdf_address_incomplete, gdf_worldcities)
    df_find_nearest = df_find_nearest[['lat', 'lon', 'lat2', 'lon2']]
    df_find_nearest.to_excel(f'./output_pt2/processing_list/{continent}/{continent}_redirected.xlsx', index=False)

    df_google = df_find_nearest[['lat2', 'lon2']]
    df_google.to_csv(f'./output_pt2/processing_list/{continent}/{continent}_to_google.csv', index=False)

    return 

def get_info(df: pd.DataFrame, output_file: str, col_name: str = 'part_1', col_city: str = 'city') -> None:
    with open(output_file, 'w', encoding='utf-16') as f:
        f.write(f'part_1 null: {df[col_name].isnull().sum()}\n')
        f.write(' \n')

        df1 = df[df[col_city].isnull()]     # filter out countries that still have null city values

        countries = list(df1[col_name].unique())
        countries = sorted(countries)

        for country in countries:
            f.write(f'{country}\n')
            df_country = df1[df1[col_name] == country]
            parts_col = [col for col in df_country.columns if 'part' in col]
            for part in parts_col:
                df_country[part] = df_country[part].fillna('none')
                part_unique = list(df_country[part].unique())
                part_unique = sorted(part_unique)
                f.write(f'{part} {part_unique}\n')
            f.write(' \n')
    print(f"output save to: {output_file}")

def split_address(df: pd.DataFrame) -> pd.DataFrame:

    # # 使用正则表达式去除前缀 code
    df['address'] = df['address'].str.replace(r'^[A-Z0-9\+\+]{4,10}\s+', '', regex=True)
    # 使用逗号拆分address列，倒序分割，并将结果转换为一个新的DataFrame
    split_df = df['address'].apply(lambda x: x.split(',')[::-1]).apply(pd.Series)
    # 生成新的DataFrame列名，按照part_n命名方便理解
    split_df.columns = [f'part_{i+1}' for i in range(split_df.shape[1])]
    # 将新的DataFrame与原始DataFrame连接起来
    df = pd.concat([df, split_df], axis=1)
    # strip
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.sort_values(by=['part_1', 'part_2', 'part_3'])
    return df


def city_to_en(df_city: pd.DataFrame) -> pd.DataFrame:

    def translator(df):

        prompt_template = """
        你是个翻译专家，请帮我把这些城市名翻译成英文,回答的时候请写成这样的形式：
        city_name,city_name_in_english
        例如：
        北京,Beijing
        """

        client = AzureOpenAI(
            api_key = '68dc0f438fbd464bb8068040c93d659e',
            api_version='2023-05-15',
            azure_endpoint='https://myexperts.openai.azure.com/',
        )

        response = client.chat.completions.create(
            model = 'gpt-4o-mini-0718',
            messages = [
                {
                    'role': 'system',
                    'content': prompt_template
                },
                {
                    'role': 'user',
                    'content': df
                }
            ],
            max_tokens = 16384,
            temperature = 0.3
        )

        output = response.choices[0].message.content  # 使用点语法

        if not output:  # 添加错误处理
            print("API 响应为空，请检查请求参数或API服务状态。")
            return None

        return output

    all_cities = list(df_city['city'].unique())
    all_cities = sorted(all_cities)
    print(len(all_cities))

    rows_num = len(all_cities)
    batch_size = 30

    df_translated = pd.DataFrame()

    for i in range(0, rows_num, batch_size):

        print(f'Processing rows from {i} to {i + batch_size}...')

        batch = all_cities[i:i + batch_size]
        batch = pd.DataFrame(batch)

        output = translator(batch.to_string())

        # 将列表中的字符串拆分成行
        rows = output.split('\n')
        # 初始化列表存储处理后的数据
        processed_data = []
        # 处理每一行
        for row in rows:
            if row:  # 检查行是否为空
                # 以逗号为分隔符，将数据拆分成两部分
                parts = row.split(',')
                # 移除多余的空格和引号
                parts = [part.strip().strip("'") for part in parts]
                processed_data.append(parts)
        # 将处理好的数据转换成DataFrame
        df_trans_processed = pd.DataFrame(processed_data, columns=['city', 'city_en'])
        print(df_trans_processed)
        # 将处理好的数据添加到df_translated
        df_translated = pd.concat([df_translated, df_trans_processed], ignore_index=True)

        del batch, output, rows, processed_data

    df_city = df_city.merge(df_translated, on='city', how='left')

    return df_city

def check_UNLOCODE_complete(df_continent, df_ports):

    df1 = df_continent.copy()

    df1 = df1.merge(df_ports[['UNLOCODE', 'lat', 'lon']], on=['lat', 'lon'], how='left')
    print('UNLOCODE null:', df1['UNLOCODE'].isnull().sum())

    pass

def P4_main_NA(df:pd.DataFrame) -> pd.DataFrame:
    
    # Get city name
    df['city'] = np.nan

    df.loc[
    (df['part_1'].notnull()) &
    (df['part_2'].isnull()) &
    (df['part_3'].isnull()) &
    (df['part_3'].isnull()) &
    (df['part_4'].isnull()), 
    'city'] = df['part_1']      # small island special administration etc. where the whole island is identified as city

    df.loc[
        (df['part_2'].notnull()) &
        (df['part_3'].isnull()) &
        (df['part_3'].isnull()) &
        (df['part_4'].isnull()),
        'city'] = df['part_2']      # city name is in part_2 if there is no part_3, part_4, part_5, part_6

    # Argentina
    df.loc[
        (df['part_1'] == 'Argentina'), 'city'] = df['part_3'] 
    
    # Canada
    if not os.path.exists('./output_pt2/processing_list/North America/canada_shp/canada.json'):

        print('Cutting Canada shapefile...')

        df_CA = df[df['part_1'] == 'Canada']
        df_CA = df_CA.reset_index(drop=True)
        gdf_ca = gpd.GeoDataFrame(df_CA, geometry=gpd.points_from_xy(df_CA.lon, df_CA.lat), crs='EPSG:4236')
        gdf_ca = gdf_ca.to_crs('EPSG:3347')
        gdf_ca.to_file('./output_pt2/processing_list/North America/canada_shp/canada.json', driver='GeoJSON')

        raise ValueError('Please check the Canada shp.')
    else:
        df_CA = gpd.read_file('./output_pt2/processing_list/North America/canada_shp/canada_shortest_distance.geojson')
        df_CA['part_3'] = df_CA['CDNAME']
        df_CA = df_CA[['lat', 'lon', 'part_3']]
        df_CA = df_CA.rename(columns={'part_3': 'PART_3_ca'})
        df = df.merge(df_CA, on=['lat', 'lon'], how='left')
        df.loc[df['PART_3_ca'].notnull(), 'part_3'] = df.loc[df['PART_3_ca'].notnull(), 'PART_3_ca']
        df = df.drop(columns=['PART_3_ca'])
    df.loc[
        (df['part_1'] == 'Canada'), 'city'] = df['part_3']

    # Costa Rica
    df.loc[
        (df['part_1'] == 'Costa Rica'), 'city'] = df['part_3']
    
    # India
    df.loc[
        (df['part_1'] == 'India'), 'city'] = df['part_3']

    # Mexico
    df.loc[
        (df['part_1'] == 'Mexico'), 'city'] = df['part_3']
    df.loc[
        (df['part_1'] == 'Mexico') &
        (df['part_2'] == 'Campeche'), 
        'city'] = 'Campeche'
    
    # Puerto Rico
    df.loc[
        (df['part_1'] == 'Puerto Rico'), 'city'] = df['part_2']
    
    # RMI
    df.loc[
        (df['part_1'] == 'RMI'), 'city'] = df['part_2']     # Marshall Islands
    
    # Russia
    df.loc[
        (df['part_1'] == 'Russia'), 'city'] = df['part_3'] 

    # USA
    if not os.path.exists('./output_pt2/processing_list/North America/usa_shp/usa.json'):  

        print('Cutting USA shapefile...')  
        df_USA = df[df['part_1'] == 'USA']
        gdf_USA = gpd.GeoDataFrame(df_USA, geometry=gpd.points_from_xy(df_USA['lon'], df_USA['lat']))
        gdf_USA.crs = 'EPSG:4326'
        gdf_USA.to_file('./output_pt2/processing_list/North America/usa_shp/usa.geojson', driver='GeoJSON')    
    else:
        df_USA = gpd.read_file('./output_pt2/processing_list/North America/usa_shp/usa_shortest_city.geojson')
        df_USA['part_3'] = df_USA['HubName']
        df_USA = df_USA[['lat', 'lon', 'part_3']]
        df_USA = df_USA.rename(columns={'part_3': 'PART_3_usa'})
        df = df.merge(df_USA, on=['lat', 'lon'], how='left')
        df.loc[df['PART_3_usa'].notnull(), 'part_3'] = df.loc[df['PART_3_usa'].notnull(), 'PART_3_usa']
        df = df.drop(columns=['PART_3_usa'])
    df.loc[
        (df['part_1'] == 'USA'), 'city'] = df['part_3']

    # USVI
    df.loc[
        (df['part_1'] == 'USVI'), 'city'] = df['part_2']        # US Virgin Islands
    
    # Uruguay
    df.loc[
        (df['part_1'] == 'Uruguay'), 'city'] = df['part_3']
    
    # Venezuela
    df.loc[
        (df['part_1'] == 'Venezuela'), 'city'] = df['part_3']
    df.loc[
        (df['part_1'] == 'Venezuela') &
        (df['part_2'] == 'Nueva Esparta'),
        'city'] = 'Punta de Mangle'

    return df


if __name__ == '__main__':
    
    # adjust_address
    gdf = gpd.read_file('/Users/jeremy/Downloads/working/GlobalPorts/output/P4_spider/cleaned/Asia/UNLOCODE_Address_Asia_incomplete.json')
    gdf = gdf[(gdf['index'] == 30) | (gdf['index'] == 16)]
    gdf = adjust_address(gdf, [30], 'testing address')
    print(gdf['address'].iloc[0])
