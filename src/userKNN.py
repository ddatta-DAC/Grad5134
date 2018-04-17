import process_1
import numpy as np
import pandas as pd
import math
import operator
import itertools
from collections import OrderedDict
import cPickle
import os
from joblib import Parallel, delayed

# userKNN #
# User based Neighborhood model #
# Refer : Charu Aggarwal's Recommender system book #


def pearson_sim(vec_u,vec_v):
    mean_u = np.mean(vec_u)
    mean_v = np.mean(vec_v)
    num = 0.0
    den_1 = 0.0
    den_2 = 0.0
    _lambda = 10

    for r_uk,r_vk in zip (vec_u,vec_v):
        if r_uk == 0.0 or r_vk == 0.0 :
            continue
        num +=  (r_uk - mean_u) * (r_vk - mean_v)
        den_1 = math.pow((r_uk - mean_u),2)
        den_2 = math.pow((r_vk - mean_v),2)
    den = math.sqrt(den_1) * math.sqrt(den_2)
    if den == 0.0 :
        den += _lambda

    res = num/den
    return res

# Input 2 vectors
# Return : similarity score
def similarity(vec_a,vec_b) :
    res = pearson_sim(vec_a,vec_b)
    return res

class top_k:

    def __init__(self,closest_user_k, closest_items_k):
        self.sim_matrix_file = 'user_sim_matrix.dat'
        self.closest_user_k = closest_user_k
        self.ui_matrix = process_1.create_user_item_matrix()
        self.users = process_1.get_user_ids()
        self.items = process_1.get_loc_ids()
        self.num_users = len(self.users)
        self.sim_init_val = -1000.00
        self.similarity_scores = np.full([self.num_users,self.num_users],self.sim_init_val , np.float )
        self.setup_similarity_matrix()
        self.setup_top_k()

    # set up the similarity score matrix
    def setup_similarity_matrix(self):

        if os.path.exists(self.sim_matrix_file):
            file = open(self.sim_matrix_file, 'r')
            self.similarity_scores = cPickle.load(file)
            file.close()
            return

        for user in self.users:
            user_idx = user-1
            other_users = list(self.users)
            other_users.remove(user)

            for other_user in other_users :
                other_user_idx = other_user - 1
                if self.similarity_scores[user_idx, other_user_idx] != self.sim_init_val :
                    continue
                score = similarity(self.ui_matrix[user_idx], self.ui_matrix[other_user_idx])
                self.similarity_scores[user_idx, other_user_idx] = score
                self.similarity_scores[other_user_idx, user_idx ] = score
                # print 'user_idx, other_user_idx  = score' , user_idx +1 ,' : ',  other_user_idx +1 , score

        # Write the similarity matrix to file
        file = open(self.sim_matrix_file,'w')
        cPickle.dump(self.similarity_scores)
        file.close()
        return


    def setup_top_k( self ):
        ui_matrix = self.ui_matrix

        # set up the top k neighbors for each user
        for user in self.users:
            user_idx = user - 1
            row_vec = self.ui_matrix[user_idx]
            mean_u = np.mean(row_vec)
            sim_vec = self.similarity_scores[user_idx]

            # Find the k closest
            k_closest_dict = OrderedDict()
            for _ui, val in zip(range(0,self.num_users), sim_vec):
                # do not include self - similarity score
                if _ui == user_idx :
                    continue
                k_closest_dict [_ui+1] = val

            sorted_k_closest_dict = sorted(k_closest_dict.items(),
                                               key= operator.itemgetter(1))

            k_closest_dict = itertools.islice(sorted_k_closest_dict.items(), 0, self.closest_user_k)
            print k_closest_dict

        return



def get_top_k_obj(closest_user_k , closest_items_k) :
    obj_file = 'userKNN_obj_'+str(closest_user_k) + '_' + str(closest_items_k) + '.dat'

    if os.path.exists(obj_file):
        file = open(obj_file, 'r')
        obj = cPickle.load(file)
        file.close()
    else:
        file = open(obj_file , 'w')
        obj = top_k(closest_user_k, closest_items_k)
        cPickle.dump(obj,file)
        file.close()
    return obj


get_top_k_obj(10 , 10)