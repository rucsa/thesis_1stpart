# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:04:43 2018

@author: Ruxandra
"""

def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)


if __name__ == '__main__':
    print(f1(['A', 'B', 'C'], ['D']))