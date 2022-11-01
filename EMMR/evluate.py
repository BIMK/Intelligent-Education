import numpy as np


def precision(rank, ground_truth):
    hits = 0
    for item in rank:
        if item in ground_truth:
            hits += 1
    result = hits/len(ground_truth)
    return result

def hit(rank, ground_truth):
    hits = 0
    for item in rank:
        if item in ground_truth:
            hits += 1
    result = hits/len(rank)
    return result

def novelty(rank, nov_list):
    nov = 0
    for item in rank:
        nov += nov_list[item]
    result = nov / len(rank)
    return result/nov_list.max()


# def diversity(rank, div_list):
#     div = 0
#     for item1 in rank:
#         for item2 in rank:
#             if item1 == item2:
#                 continue
#             try:
#                 div += div_list['VALUES'][item1][str(item2)]
#             except:
#                 pass
#     result = div / 2 * (len(rank) * (len(rank) - 1))
#     return result
def diversity(rank, div_list, max, st):
    div = ""
    for item in rank:
        try:
            div = div + st + div_list['genre'][item]
        except:
            pass
    result = np.unique(div.split(st))
    return (len(result) - 1)/max





