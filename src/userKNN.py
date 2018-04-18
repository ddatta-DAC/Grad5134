import process_1
import numpy as np
import math
import operator
import itertools
from collections import OrderedDict
import cPickle
import os
import multiprocessing as mp


# userKNN #
# User based Neighborhood model #
# Refer : Charu Aggarwal's Recommender system book #


def pearson_sim(vec_u, vec_v):
    mean_u = np.mean(vec_u)
    mean_v = np.mean(vec_v)
    num = 0.0
    den_1 = 0.0
    den_2 = 0.0
    _lambda = 10

    for r_uk, r_vk in zip(vec_u, vec_v):
        if r_uk == 0.0 or r_vk == 0.0:
            continue
        num += (r_uk - mean_u) * (r_vk - mean_v)
        den_1 = math.pow((r_uk - mean_u), 2)
        den_2 = math.pow((r_vk - mean_v), 2)
    den = math.sqrt(den_1) * math.sqrt(den_2)
    if den == 0.0:
        den += _lambda

    res = num / den
    return res


# Input 2 vectors
# Return : similarity score
def similarity(vec_a, vec_b):
    res = pearson_sim(vec_a, vec_b)
    return res


def aux_get_sim(ui_matrix, similarity_scores, idx1, idx2, lock):
    score = similarity(ui_matrix[idx1, :], ui_matrix[idx2, :])
    lock.acquire()
    print ' In aux_get_sim ', idx1, idx2, score
    similarity_scores[idx1][idx2] = score
    lock.release()
    return


class top_k:

    def __init__(self, closest_user_k, closest_items_k):

        self.sim_matrix_file = 'user_sim_matrix.dat'
        self.rating_matrix_file = 'rating_1_matrix.dat'

        self.closest_user_k = closest_user_k
        self.closest_items_k = closest_items_k

        self.ui_matrix = process_1.create_user_item_matrix()
        self.users = process_1.get_user_ids()
        self.items = process_1.get_loc_ids()
        self.num_users = len(self.users)
        self.num_items = len(self.items)

        self.sim_init_val = -1000.00
        self.similarity_scores = np.full([self.num_users, self.num_users], self.sim_init_val, np.float)
        self.rating_matrix = None

        self.setup_similarity_matrix()
        self.setup_rating_matrix()

        return

    # set up the similarity score matrix
    def setup_similarity_matrix(self):

        if os.path.exists(self.sim_matrix_file):
            file = open(self.sim_matrix_file, 'r')
            self.similarity_scores = cPickle.load(file)
            file.close()
            return
        cur_len = 0
        max_len = self.num_users / 2 + 2
        for user in self.users:
            # Use the fact its a symmetric matrix
            # Truncate if cur_len > max_len
            if cur_len > max_len:
                break
            cur_len += 1

            user_idx = user - 1
            other_users = list(self.users)
            other_users.remove(user)
            ui_matrix = np.array(self.ui_matrix)
            lock = mp.Lock()

            processes = [
                mp.Process(target=aux_get_sim, args=(ui_matrix, self.similarity_scores, user_idx, other_user - 1, lock))
                for other_user in other_users]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        # Write the similarity matrix to file
        file = open(self.sim_matrix_file, 'w')
        cPickle.dump(self.similarity_scores, file)
        file.close()
        return

    def setup_rating_matrix(self):

        if os.path.exists(self.rating_matrix_file):
            file = open(self.rating_matrix_file, 'r')
            self.rating_matrix = cPickle.load(file)
            file.close()

        ui_matrix = self.ui_matrix
        rating_matrix = np.zeros([self.num_users, self.num_items])
        # set up a dictionary for each item:users who have rating for it
        # This stores item_id : user_id
        item_user_dict = {}
        for item in self.items:
            item_idx = item - 1
            # set of users where j_idx is non zero
            j_users_idx = list(np.nonzero(ui_matrix[:, item_idx]))
            item_user_dict[item] = [y + 1 for y in j_users_idx]

        # set up the top k neighbors for each user
        for user in self.users:
            user_idx = user - 1
            row_vec = self.ui_matrix[user_idx]
            mean_u = np.mean(row_vec)
            sim_vec = self.similarity_scores[user_idx]

            # Find the z closest users to user u
            # where item j has rating
            # for each item
            for j in self.items:

                j_idx = j - 1
                # set of users where j_idx is non zero
                user_id_list = item_user_dict[j]
                j_users_idx = [y - 1 for y in user_id_list]

                #  This stores user_index : score
                k_closest_dict = OrderedDict()

                for j_user_idx in j_users_idx:
                    # do not include self - similarity score
                    if j_user_idx == user_idx:
                        continue
                    k_closest_dict[j_user_idx] = sim_vec[j_user_idx]

                k_closest_dict = sorted(
                    k_closest_dict.items(),
                    key=operator.itemgetter(1)
                )

                k_closest_dict = itertools.islice(k_closest_dict.items(), 0, self.closest_user_k)
                k_closest_user_idx_j = k_closest_dict.keys()

                # Calculate the rating of item j for user u
                num = 0.0
                den = 0.0
                for user_v_idx, sim_score in k_closest_dict.iteritems():
                    z = (self.ui_matrix[user_v_idx][j_idx] - np.mean(self.ui_matrix[user_v_idx]))
                    num += sim_score * z
                    den += abs(z)
                r_uj = mean_u + (num / den)

                rating_matrix[user_idx][j_idx] = r_uj

        self.rating_matrix = rating_matrix
        # Save the rating matrix
        file = open(self.rating_matrix_file, 'w')
        cPickle.dump(self.rating_matrix, file)
        file.close()
        return


# def get_top_k_obj(closest_user_k , closest_items_k) :
#     obj_file = 'userKNN_obj_'+str(closest_user_k) + '_' + str(closest_items_k) + '.dat'
#
#     if os.path.exists(obj_file):
#         file = open(obj_file, 'r')
#         obj = cPickle.load(file)
#         file.close()
#     else:
#         file = open(obj_file , 'w')
#         obj = top_k(closest_user_k, closest_items_k)
#         cPickle.dump(obj,file)
#         file.close()
#     return obj


top_k(10, 10)
