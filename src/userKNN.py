# coding=utf-8
import process_1
import numpy as np
import math
import operator
import itertools
from collections import OrderedDict
import cPickle
import os
import multiprocessing as mp
import pprint
from multiprocessing import Queue
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# userKNN #
# User based Neighborhood model #
# Refer : Charu Aggarwal's Recommender system book #

'''
I_u ∩ I_v represents the set of item indices for which 
both user u and user v have specified ratings, 
the coefficient is computed only on this set of items

First step is to compute the mean rating μ_u for each user u
using his specified ratings

μ_u = k∈I_u Sum(r_uk) /|I_u|

'''
def pearson_sim(vec_u, vec_v):

    # mean_u = float(np.sum(vec_u)) / len(np.nonzero(vec_u)[0])
    # mean_v = float(np.sum(vec_v)) / len(np.nonzero(vec_v)[0])
    num = 0.0
    den_1 = 0.0
    den_2 = 0.0
    _lambda = 1

    for r_uk, r_vk in zip(vec_u, vec_v):
        # No rating over item k for user u or v, skip it!
        if r_uk == 0.0 or r_vk == 0.0:
            continue
        k1 = r_uk
        k2 = r_vk
        num += k1 * k2
        den_1 += math.pow(k1, 2)
        den_2 += math.pow(k2, 2)
    den = math.sqrt(den_1) * math.sqrt(den_2)

    if den == 0.0:
        den += _lambda

    res = float(num) / den
    return res

def cosine_sim(vec_u, vec_v):
    res = cosine_similarity([vec_u],[vec_v])[0][0]
    return res

# Input 2 vectors
# Return : similarity score
def similarity(vec_a, vec_b):
    res = cosine_sim(vec_a, vec_b)
    return res


def aux_get_sim(q, lock, ui_matrix, idx1, idx2):
    score = similarity(ui_matrix[idx1, :], ui_matrix[idx2, :])
    lock.acquire()
    q.put([idx1, idx2, score])
    lock.release()
    return


class userKnn:

    def __init__(self, closest_user_k):

        print ' Constructing userKNN class with Number of neighborhood users considered ',closest_user_k
        self.sim_matrix_file = 'user_sim_matrix.dat'
        self.rating_matrix_file = 'rating_userKNN_'+ str(closest_user_k) + '.dat'

        self.closest_user_k = closest_user_k

        self.ui_matrix = process_1.create_user_item_matrix()
        self.users = process_1.get_user_ids()
        self.items = process_1.get_loc_ids()
        self.num_users = len(self.users)
        self.num_items = len(self.items)

        self.sim_init_val = 0.00
        self.similarity_scores = np.full(
            [self.num_users, self.num_users],
            self.sim_init_val,
            np.float
        )

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
            print csr_matrix(self.similarity_scores)
            return

        cur_len = 0
        max_len = (self.num_users / 2)+ 2

        for user in self.users:
            # Use the fact its a symmetric matrix
            # Truncate if cur_len > max_len
            if cur_len > max_len:
                break
            cur_len += 1

            lock = mp.Lock()
            q = Queue()
            ui_matrix = np.asarray(self.ui_matrix)
            user_idx = user - 1
            other_users = list(self.users)
            other_users.remove(user)

            processes = [
                mp.Process(
                    target=self.aux_get_sim,
                    args=(
                        q,
                        lock,
                        ui_matrix,
                        user_idx,
                        other_user - 1
                    )
                )
                for other_user in other_users
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            while q.empty() == False:
                res = q.get()
                idx_1 = res[0]
                idx_2 = res[1]
                score = res[2]
                self.similarity_scores[idx_1][idx_2] = score
                self.similarity_scores[idx_2][idx_1] = score

        # Write the similarity matrix to file
        file = open(self.sim_matrix_file, 'w')
        cPickle.dump(self.similarity_scores, file)
        file.close()

        return


    def setup_rating_matrix(self):

        print ' In setup_rating_matrix . . . '

        if os.path.exists(self.rating_matrix_file):
            file = open(self.rating_matrix_file, 'r')
            self.rating_matrix = cPickle.load(file)
            file.close()
            return

        rating_matrix = np.zeros([self.num_users, self.num_items])

        # Set up a dictionary for each item:users who have rating for it
        # This stores item_id : user_id
        item_user_dict = {}

        for item in self.items:
            item_idx = item - 1
            # set of users where j_idx is non zero
            j_users_idx = list(np.nonzero(self.ui_matrix[:, item_idx]))[0]
            item_user_dict[item] = [y + 1 for y in j_users_idx]

        print '---'
        print item_user_dict
        print '---'

        # set up the top k neighbors for each user
        for user in self.users:
            print 'User ', user
            user_idx = user - 1
            row_vec = self.ui_matrix[user_idx]

            if len(np.nonzero(row_vec)[0]) > 0 :
                mean_u = float(np.sum(row_vec)) / len(np.nonzero(row_vec)[0])
            else :
                mean_u = 0

            sim_vec = self.similarity_scores[user_idx]

            # Find the z closest users to user u
            # where item j has rating
            # for each item

            for j in self.items:

                j_idx = j - 1
                # set of users where j_idx is non zero
                user_id_list = list(item_user_dict[j])
                j_users_idx = [y-1 for y in user_id_list]


                #  This stores user_index : score
                k_closest_dict = OrderedDict()

                for j_user_idx in j_users_idx:
                    # do not include self - similarity score
                    if j_user_idx == user_idx:
                        continue
                    k_closest_dict[j_user_idx] = sim_vec[j_user_idx]

                sorted_k_closest_dict_key_val = sorted(
                    k_closest_dict.items(),
                    key=operator.itemgetter(1),
                    reverse=True
                )

                k_closest_dict = OrderedDict()

                for k1_v1 in sorted_k_closest_dict_key_val:
                    k_closest_dict[k1_v1[0]] = k1_v1[1]

                k_closest_dict = itertools.islice(k_closest_dict.items(), 0, self.closest_user_k)
                k_closest_dict = OrderedDict(k_closest_dict)

                # Calculate the rating of item j for user u
                num = 0.0
                den = 0.0

                for user_v_idx, sim_score in k_closest_dict.iteritems():
                    k1 = self.ui_matrix[user_v_idx][j_idx]
                    k2 = np.mean(self.ui_matrix[user_v_idx])
                    z = (k1) # tried k1-k2
                    num += sim_score * z
                    den += abs(z)
                if den == 0 :
                    den += 1

                r_uj =  (num / den)
                rating_matrix[user_idx][j_idx] = r_uj


        self.rating_matrix = rating_matrix
        # Save the rating matrix
        file = open(self.rating_matrix_file, 'w')
        cPickle.dump(self.rating_matrix, file)
        print 'Rating matrix', csr_matrix(rating_matrix)
        file.close()
        return

    def recommend_items(self, user_id, num_items):
        rec_vec = self.rating_matrix[user_id - 1]
        # sort the items by recommendation
        # this dictionary will have  item : score
        rec_dict = {}
        for item_idx in range(len(rec_vec)):
            if rec_vec[item_idx] > 0.0:
                rec_dict[item_idx + 1] = rec_vec[item_idx]

        sorted_item_score = sorted(
            rec_dict.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
        # Return the top k
        # k = num_items
        ordered_items = [x[0] for x in sorted_item_score]
        return ordered_items[0:num_items]


userKnn_obj = userKnn(25)
print userKnn_obj.recommend_items(user_id=75,num_items=2)
