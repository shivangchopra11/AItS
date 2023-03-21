import pandas as pd
import numpy as np

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

if __name__ == "__main__":
    df = pd.read_csv('data/video_items.csv')
    print(df.values)

    lists = df.values[:,1]

    fin_list = []
    for pick in lists:
        # print(pick)
        pick_len = len(pick)
        # print(pick_len)
        cur_order = []
        for i in range(0,len(pick),2):
            # print(pick[i])
            # try:
            cur_order.append(pick[i])
            # except:
            #     cur_order.append('')
        # print(cur_order)
        fin_list.append(cur_order)

    ids = df.values[:,0]
    fin_list = np.array(fin_list)
    all_negatives = []
    for pid, pick in zip(ids, fin_list):
        cur_negatives = []
        for idx in range(len(ids)):
            inter = intersection(pick, fin_list[idx])
            if len(inter) == 0:
                cur_negatives.append(ids[idx])
        all_negatives.append(cur_negatives)
        # print(pid, pick)
    # print(all_negatives)

    df['colors'] = fin_list
    df['negatives'] = all_negatives

    df.to_csv('data/picklist_negatives.csv', index=False)

    print(df.head())

    # print(fin_list.shape, ids.shape)
