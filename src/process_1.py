import pandas as pd
import numpy as np
import config
from scipy.sparse import csr_matrix
import cPickle
import os
import pprint
ui_matrix_file = 'ui_matrix.dat'

<<<<<<< HEAD
TEST_FLAG = False
item_comm_area_map = {}
comm_area_item_map = {}
=======
TEST_FLAG = True
>>>>>>> 8b20c579c88406e48392ef30ce8933516e0fcce6


# TEST
def get_fake_matrix():
    z = np.random.multinomial(5, [1 / 20.0] * 20, size=25)
    for i in range(25):
        for j in range(20):
            if z[i][j] > 0:
                z[i][j] = 1
    return z


def get_user_ids():
    user_df = pd.read_csv(config.data_dir_data + 'user.csv')
    uid_list = list(set(list(user_df['FakeID'])))
<<<<<<< HEAD

    return uid_list
=======
    return uid_list[0:25]
	return uid_list
>>>>>>> 8b20c579c88406e48392ef30ce8933516e0fcce6


def get_loc_ids():
    loc_df = pd.read_csv(config.data_dir_data + 'venue.csv')
    uid_list = list(set(list(loc_df['FakeID'])))
<<<<<<< HEAD

    return uid_list
=======
    return uid_list[0:20]
	return uid_list
>>>>>>> 8b20c579c88406e48392ef30ce8933516e0fcce6


def create_user_item_matrix():
    global ui_matrix_file
    global TEST_FLAG

    if TEST_FLAG:
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


def setup_loc_id_commu_file():
    global items

    # File with venue and community area :
    # venue_comm.csv
    target_file_name = config.data_dir_data + 'venue_comm.csv'
    if os.path.isfile(target_file_name):
        return

    df_venue = pd.read_csv(config.data_dir_data + 'venue.csv')
    df_checkins = pd.read_csv(config.data_dir_data + 'chicago_checkins_data.csv')
    df_checkins = df_checkins[['lid', 'commu_area']]

    # lid :: community area
    lid_commu = {}
    for i, row in df_checkins.iterrows():
        lid_commu[row['lid']] = row['commu_area']

    def get_comm_area(row):
        return lid_commu[row['lid']]

    df_venue['commu_area'] = df_venue.apply(get_comm_area, axis=1)
    # Write to file
    df_venue.to_csv(target_file_name)
    return


def setup_commu_area_lid():
    global item_comm_area_map
    global comm_area_item_map
    item_comm_area_map_file = 'item_comm_area_map.dat'
    comm_area_item_map_file = 'comm_area_item_map.dat'

    count = 0
    if os.path.isfile(item_comm_area_map_file):
        file = open(item_comm_area_map_file, 'r')
        item_comm_area_map = cPickle.load(file)
        file.close()
        count += 1

    if os.path.isfile(comm_area_item_map_file):
        file = open(comm_area_item_map_file, 'r')
        comm_area_item_map = cPickle.load(file)
        file.close()
        count += 1

    if count == 2:
        return


    f_name = config.data_dir_data + 'venue_comm.csv'
    df_venue = pd.read_csv(f_name)

    for i, row in df_venue.iterrows():
        item = row['FakeID']
        ca = row['commu_area']

        item_comm_area_map[item] = ca

        if ca in comm_area_item_map.keys():
            comm_area_item_map[ca].append(item)
        else:
            comm_area_item_map[ca] = [item]

    file = open(comm_area_item_map_file,'w')
    cPickle.dump(comm_area_item_map,file)
    file.close()

    file = open(item_comm_area_map_file, 'w')
    cPickle.dump(item_comm_area_map, file)
    file.close()

    return


create_user_item_matrix()
setup_loc_id_commu_file()
setup_commu_area_lid()
