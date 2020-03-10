# -*- coding: utf-8 -*-
# @Time    : 19/12/11 15:57
# @Author  : Jay Lam
# @File    : trend2boolean.py
# @Software: PyCharm

import os
import pandas as pd
import pymysql
import numpy as np
from scipy.special import ndtri, ndtr
import sys

pathDir = os.path.abspath('.') + '//'


def sqlExtra(path):
    idCode = ['PG-40']
    indexCode = ['K+']

    for x in idCode:
        for y in indexCode:
            writer = pd.ExcelWriter(path + x + "井的" + y + "指标筛选结果.xlsx")
            db = pymysql.connect("localhost", "root", "sustc@10", "pggroundwater")
            cursor = db.cursor()
            sqlStr = "SELECT `" + y + "` FROM `pg1718` WHERE `统一编号` = \'" + x + "\'"
            cursor.execute(sqlStr)
            sqlResults = cursor.fetchall()

            columnDes = cursor.description
            columnNames = [columnDes[i][0] for i in range(len(columnDes))]
            df = pd.DataFrame([list(i) for i in sqlResults], columns=columnNames)
            df.fillna("0", inplace=True)

            result = df.values.tolist()
        return result


def boolConverter(narray):
    newArray = []
    for i in range(len(narray) - 1):
        if narray[i] > narray[i + 1]:
            newArray.append(1)
        elif narray[i] <= narray[i + 1]:
            newArray.append(0)
    print(newArray)


def mannkendall(t,x,eps,alpha,Ha):
    # assert a least count for the measurements x
    assert eps, "Please provide least count error for measurements 'x'"
    assert alpha, "Please provide significance level 'alpha' for the test"
    assert Ha, "Please provide the alternative hypothesis 'Ha'"

    # estimate sign of all possible (n(n-1)) / 2 differences
    n = len(t)
    sgn = np.zeros((n, n), dtype="int")
    for i in range(n):
        tmp = x - x[i]
        tmp[np.where(np.fabs(tmp) <= eps)] = 0.
        sgn[i] = np.sign(tmp)

    # estimate mean of the sign of all possible differences
    S = sgn[np.triu_indices(n, k=1)].sum()

    # estimate variance of the sign of all possible differences
    # 1. Determine no. of tie groups 'p' and no. of ties in each group 'q'
    np.fill_diagonal(sgn, eps * 1E6)
    i, j = np.where(sgn == 0.)
    ties = np.unique(x[i])
    p = len(ties)
    q = np.zeros(len(ties), dtype="int")
    for k in range(p):
        idx = np.where(np.fabs(x - ties[k]) < eps)[0]
        q[k] = len(idx)
    # 2. Determine the two terms in the variance calculation
    term1 = n * (n - 1) * (2 * n + 5)
    term2 = (q * (q - 1) * (2 * q + 5)).sum()
    # 3. estimate variance
    varS = float(term1 - term2) / 18.

    # Compute the Z-score based on above estimated mean and variance
    if S > eps:
        Zmk = (S - 1) / np.sqrt(varS)
    elif np.fabs(S) <= eps:
        Zmk = 0.
    elif S < -eps:
        Zmk = (S + 1) / np.sqrt(varS)

    # compute test based on given 'alpha' and alternative hypothesis
    # note: for all the following cases, the null hypothesis Ho is:
    # Ho := there is no monotonic trend
    #
    # Ha := There is an upward monotonic trend
    if Ha == "up":
        Z_ = ndtri(1. - alpha)
        if Zmk >= Z_:
            MK = "accept Ha := upward trend"
        else:
            MK = "reject Ha := upward trend"
    # Ha := There is a downward monotonic trend
    elif Ha == "down":
        Z_ = ndtri(1. - alpha)
        if Zmk <= -Z_:
            MK = "accept Ha := downward trend"
        else:
            MK = "reject Ha := downward trend"
    # Ha := There is an upward OR downward monotonic trend
    elif Ha == "upordown":
        Z_ = ndtri(1. - alpha / 2.)
        if np.fabs(Zmk) >= Z_:
            MK = "accept Ha := upward OR downward trend"
        else:
            MK = "reject Ha := upward OR downward trend"

    # ----------
    # AS A BONUS
    # ----------
    # estimate the slope and intercept of the line
    m = np.corrcoef(t, x)[0, 1] * (np.std(x) / np.std(t))
    c = np.mean(x) - m * np.mean(t)

    # ----------
    # AS A BONUS
    # ----------
    # estimate the p-value for the obtained Z-score Zmk
    if S > eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (1. - ndtr(Zmk))
    elif np.fabs(S) <= eps:
        p = 0.5
    elif S < -eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (ndtr(Zmk))

    return MK, m, c, p


if __name__ == '__main__':
    #sqlExtra(pathDir)
    boolConverter(sqlExtra(pathDir))
