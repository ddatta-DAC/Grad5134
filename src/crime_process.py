import pandas as pd
import config
import pprint
import seaborn
import matplotlib.pyplot as plt


def get_user_ids():
    user_df = pd.read_csv(config.data_dir_data + 'user.csv')
    uid_list = list(set(list(user_df['FakeID'])))
    return uid_list


def get_loc_ids():
    loc_df = pd.read_csv(config.data_dir_data + 'venue.csv')
    uid_list = list(set(list(loc_df['FakeID'])))
    return uid_list


def process_crime_data():
    df = pd.read_csv(config.data_dir_processed + 'crime_commu_sta.csv')
    cols = list(df.columns)

    for x in ['commu_area', 'user_cnt', 'venue_cnt', 'check-ins']:
        cols.remove(x)

    pprint.pprint(cols)
    violent_crimes = ['PUBLIC PEACE VIOLATION', 'KIDNAPPING', 'BURGLARY', 'CRIMINAL DAMAGE', 'WEAPONS VIOLATION',
                      'OBSCENITY', 'ARSON', 'NARCOTICS', 'SEX OFFENSE', 'INTIMIDATION', 'BATTERY',
                      'NON - CRIMINAL', 'STALKING', 'ASSAULT', 'CRIM SEXUAL ASSAULT']

    cols_1 = list(violent_crimes)
    cols_1.append('commu_area')

    # Violent crimes by community areas
    violent_crime_dict = {}
    vc_df = df[cols_1]
    print vc_df.columns

    def total_violent(row):
        s = 0
        for x in violent_crimes:
            s += row[x]
        return s

    vc_df['violent_crime'] = df.apply(total_violent, axis=1)
    print vc_df

    file_name = 'violent_crimes.csv'
    vc_df.to_csv(file_name, index=True)

    plt.figure()
    plt.tight_layout()
    plt.title(' Community Area vs Violent Crimes ', fontsize=28)
    seaborn.barplot(x="commu_area", y="violent_crime", data=vc_df, palette="Greens_d");
    plt.xlabel('Community Area', fontsize=24)
    plt.ylabel('Count of Violent Crimes', fontsize=24)
    plt.xticks(fontsize=12, rotation=90)
    # plt.show()
    plt.close()

    # Top 15 most crime prone areas heatmap
    plt.figure(figsize=[16,7])
    plt.tight_layout()
    top15_vc = vc_df.sort_values(by='violent_crime')
    top15_vc = top15_vc.rename({'violent_crime': 'Total Violent Crimes'},axis=1)
    top15_vc = top15_vc.head(15)
    top15_vc = top15_vc.sort_values(by='commu_area')
    top15_vc = top15_vc.set_index('commu_area')
    print top15_vc
    del top15_vc['Total Violent Crimes']
    seaborn.heatmap(
        top15_vc,
        linewidths=.5,
        linecolor="blue",
        cmap='YlGnBu'
    )
    plt.ylabel('Community Area', fontsize=24)
    plt.xlabel('Crime Type', fontsize=24)
    plt.xticks(fontsize=10, rotation=40)
    plt.yticks(fontsize=14, rotation=45)
    # plt.show()


process_crime_data()
