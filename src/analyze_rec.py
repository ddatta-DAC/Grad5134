import pandas as pd
import json
import pprint
import process_1
import config
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn

# -------------------- #

items = process_1.get_loc_ids()
users = process_1.get_user_ids()

# get a dictionary of
# user_id :: locations recommended
def get_rec_dict():
    user_rec_file = 'userKNN_100rec.json'
    f = open(user_rec_file,'r')
    s = f.read()
    f.close()
    _dict = json.loads(s)
    return _dict


def get_commu_area_lid(item_list):
    _dict = process_1.item_comm_area_map
    res = []
    for item in item_list:
        res.append(_dict[item])
    return res


# See the distribution of recommendations by community area.
def analyze_rec_comm():
    # Top 10
    top_list = [5,10,15,25]

    for top in top_list:
        user_rec_dict = get_rec_dict()
        all_results = []
        for user_id, item_list in user_rec_dict.iteritems():
            r = get_commu_area_lid(item_list)
            r = r[0:top]
            all_results.extend(r)

        rec_comm_count_dict = dict(Counter(all_results))
        df = pd.DataFrame(rec_comm_count_dict.items(),
                          columns=[
                                'Community Area',
                                'Recommendation count'
                          ]
                          )
        df = df.sort_values(by='Recommendation count')
        df = df.head(25)

        plt.title('Top 25 community areas by recommendation count, with top '+ str(top) + 'recommendations across all users' , fontsize=24)
        plt.xlabel('Community Area')
        plt.ylabel('Count', )
        seaborn.barplot(
                x="Community Area",
                y="Recommendation count",
                data=df,
                palette="pastel"
            )
        # plt.show()

        df = pd.DataFrame(rec_comm_count_dict.items(),
                          columns=[
                              'commu_area',
                              'count'
                          ]
                          )
        file_name = 'pred_checkins_userKNN'+str(top)+'.csv'
        df.to_csv(file_name)

analyze_rec_comm()
