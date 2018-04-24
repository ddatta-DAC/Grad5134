import process_1
import numpy as np
from collections import OrderedDict
import cPickle
import os
import multiprocessing as mp
import math
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Queue
import operator
import itertools
import pprint


# ItemKNN #

# def adj_cosine_sim(vec_i, vec_j):
#
#     num = 0.0
#     den_1 = 0.0
#     den_2 = 0.0
#     _lambda = 1
#
#     for r_ui, r_uj in zip(vec_i, vec_j):
#         if r_ui == 0.0 or r_uj == 0.0:
#             continue
#         num += (r_ui - mean_i) * (r_uj - mean_j)
#         den_1 = math.pow((r_ui - mean_i), 2)
#         den_2 = math.pow((r_uj - mean_j), 2)
#
#     den = math.sqrt(den_1) * math.sqrt(den_2)
#     if den == 0.0:
#         den += _lambda
#
#     res = num / den
#     return res

def cosine_sim(vec_u, vec_v):
    res = cosine_similarity([vec_u], [vec_v])[0][0]
    return res


# Input 2 vectors
# Return : similarity score
def similarity(vec_a, vec_b):
    res = cosine_sim(vec_a, vec_b)
    return res


def aux_get_sim(ui_matrix, similarity_scores, idx1, idx2, lock):
    score = similarity(ui_matrix[idx1, :], ui_matrix[idx2, :])
    lock.acquire()
    print ' In aux_get_sim ', idx1, idx2, score
    similarity_scores[idx1][idx2] = score
    lock.release()
    return


class itemKnn:

    def __init__(self, closest_items_k):
        print ' Constructing userKNN class with Number of neighborhood users considered ', closest_items_k

        self.sim_matrix_file = 'item_sim_matrix.dat'
        self.rating_matrix_file = 'rating_itemKNN_' + str(closest_items_k) + '.dat'
        self.closest_items_k = closest_items_k

        self.ui_matrix = process_1.create_user_item_matrix()
        self.users = process_1.get_user_ids()
        self.items = process_1.get_loc_ids()
        self.num_users = len(self.users)
        self.num_items = len(self.items)

        self.sim_init_val = 0.0
        self.similarity_scores = np.full([self.num_items, self.num_items], self.sim_init_val, np.float)
        self.rating_matrix = None
        self.setup_similarity_matrix()
        self.setup_rating_matrix()
        return

    def aux_get_sim(self, q, idx1, idx2):
        score = similarity(self.ui_matrix[:, idx1], self.ui_matrix[:, idx2])
        print 'Score ', idx1, idx2, score
        q.put([idx1, idx2, score])
        return

    # set up the similarity score matrix
    def setup_similarity_matrix(self):

        if os.path.exists(self.sim_matrix_file):
            file = open(self.sim_matrix_file, 'r')
            self.similarity_scores = cPickle.load(file)
            file.close()
            return

        cur_len = 0
        max_len = (self.num_items / 2) + 2

        for item in self.items:
            # Use the fact its a symmetric matrix
            # Truncate if cur_len > max_len
            if cur_len > max_len:
                break
            cur_len += 1

            item_idx = item - 1
            other_items = list(self.items)
            other_items.remove(item)

            q = Queue()
            processes = [
                mp.Process(
                    target=self.aux_get_sim,
                    args=(
                        q,
                        item_idx,
                        other_item - 1
                    )
                )
                for other_item in other_items
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

    def sort_by_value_k(self, ordDict):

        # sort
        sorted_k_closest_dict_key_val = sorted(
            ordDict.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        k_closest_dict = OrderedDict()

        for k1_v1 in sorted_k_closest_dict_key_val:
            k_closest_dict[k1_v1[0]] = k1_v1[1]

        # pick top k
        k_closest_dict = itertools.islice(k_closest_dict.items(), 0, self.closest_items_k)
        k_closest_dict = OrderedDict(k_closest_dict)

        return k_closest_dict

    def setup_rating_matrix(self):

        print ' In setup_rating_matrix . . . '

        if os.path.exists(self.rating_matrix_file):
            file = open(self.rating_matrix_file, 'r')
            self.rating_matrix = cPickle.load(file)
            file.close()
            return

        rating_matrix = np.zeros([self.num_users, self.num_items])

        for item in self.items:
            item_idx = item - 1
            col_vec = self.ui_matrix[item_idx]

            if len(np.nonzero(col_vec)[0]) > 0:
                mean_i = float(np.sum(col_vec)) / len(np.nonzero(col_vec)[0])
            else:
                mean_i = 0

            other_items = list(self.items)
            other_items.remove(item)

            vec_item = list(self.similarity_scores[item_idx])

            for user in self.users:
                # if rating is already present skip!
                if self.ui_matrix[user-1][item_idx] != 0.0 :
                    continue

                # find the items which are rated by user u,
                # From them select the top k matching ones
                user_idx = user - 1
                row_vec = self.ui_matrix[user_idx]
                user_rated_items = np.nonzero(row_vec)[0]
                user_rated_items_idx = [w - 1 for w in user_rated_items]

                # get the similarity scores of these items wrt 'item'
                sim_items = {}
                for j in user_rated_items_idx:
                    key = j + 1
                    sim_items[key] = vec_item[j]

                # sort them and get top k!
                k_closest_dict = self.sort_by_value_k(sim_items)

                # Do a weighted avg over these scores
                num = 0.0
                den = 0.0
                score = 0.0
                for item_j, item_j_sim_score in k_closest_dict.items():
                    num += item_j_sim_score * self.ui_matrix[user_idx][item_j-1]
                    den += item_j_sim_score
                if den == 0 :
                    score = 0
                else :
                    score = num/den

                rating_matrix[user_idx][item_idx] = score

        self.rating_matrix = rating_matrix
        # Save the rating matrix
        file = open(self.rating_matrix_file, 'w')
        cPickle.dump(self.rating_matrix, file)
        file.close()
        pprint.pprint(rating_matrix)
        return

    def recommend_items(self, user_id, num_items):
        rec_vec = self.rating_matrix[user_id - 1]
        print rec_vec
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


itemKnn_obj = itemKnn(25)


