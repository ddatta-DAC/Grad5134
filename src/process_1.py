import pandas as pd
import numpy as np
import config
from scipy.sparse import csr_matrix

def get_user_ids():
    user_df =  pd.read_csv(config.data_dir_data + 'user.csv')
    uid_list = list(set(list(user_df['FakeID'])))
    return uid_list

def get_loc_ids():
    loc_df =  pd.read_csv(config.data_dir_data + 'venue.csv')
    uid_list = list(set(list(loc_df['FakeID'])))
    return uid_list

def create_user_item_matrix():
    df_ratings = pd.read_csv(config.data_dir_processed + 'ratings.csv')
    print df_ratings.columns

    users = get_user_ids()
    items = get_loc_ids()

    # create matrix
    num_users = len(users)
    num_items = len(items)
    matrix = np.zeros(shape=[num_users,num_items], dtype = np.float)
    print matrix.shape

    for i,row in df_ratings.iterrows():
        uid = row['uid']
        u_idx = uid-1
        lid = row['lid']
        i_idx = lid -1
        matrix[u_idx][i_idx] += 1

    print 'Compressed Represeentation of User-Item Matrix ', csr_matrix(matrix)
    return matrix


create_user_item_matrix()
