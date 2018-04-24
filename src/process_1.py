import pandas as pd
import numpy as np
import config
from scipy.sparse import csr_matrix
import cPickle
import os
ui_matrix_file = 'ui_matrix.dat'

TEST_FLAG = False


#TEST
def get_fake_matrix():

    z = np.random.multinomial(5, [1 / 20.0] * 20, size=25)
    for i in range(25):
        for j in range(20):
            if z[i][j] > 0 :
                z[i][j] = 1
    return z



def get_user_ids():
    user_df = pd.read_csv(config.data_dir_data + 'user.csv')
    uid_list = list(set(list(user_df['FakeID'])))
    return uid_list


def get_loc_ids():
    loc_df = pd.read_csv(config.data_dir_data + 'venue.csv')
    uid_list = list(set(list(loc_df['FakeID'])))
    return uid_list


def create_user_item_matrix():
    global ui_matrix_file
    global TEST_FLAG

    if TEST_FLAG :
        return get_fake_matrix()

    if os.path.exists(ui_matrix_file):
        file = open(ui_matrix_file, 'r')
        matrix = cPickle.load(file)
        file.close()
        return matrix

    df_ratings = pd.read_csv(config.data_dir_processed + 'ratings.csv')
    users = get_user_ids()
    items = get_loc_ids()

    # create matrix
    num_users = len(users)
    num_items = len(items)
    matrix = np.zeros(shape=[num_users, num_items], dtype=np.float)
    for i, row in df_ratings.iterrows():

        uid = row['uid']
        u_idx = uid - 1
        lid = row['lid']
        i_idx = lid - 1
        try:
            matrix[u_idx][i_idx] += 1
        except:
            pass

    file = open(ui_matrix_file, 'w')
    cPickle.dump(matrix, file)
    file.close()
    print 'Compressed Represeentation of User-Item Matrix ', csr_matrix(matrix)

    return matrix


create_user_item_matrix()
