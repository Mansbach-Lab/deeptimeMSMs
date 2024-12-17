# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:10:03 2024

@author: raman
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

def LMethodGapID(values,plot=False):
    """
    Use the L Method (https://ieeexplore-ieee-org.lib-ezproxy.concordia.ca/document/1374239) to determine
    optimal place to identify a spectral gap.

    ----------
    Parameters
    ----------
    values : numpy array
        the spectrum to find a gap in
    plot : bool, default = False
        whether to plot the spectrum with the fitted lines
    -------
    Returns
    -------
    optimal_split : tuple of (int, ndarray, ndarray, float)
        contains the index to perform the split, the array of 
        fits to the first line, the array of fits to the second line
        and the minimum RMSE
    """
    optimal_split = None
    minR = np.inf
    for i in range(3,len(values)-2):
        vals1 = values[0:i]
        vals2 = values[i:]
        xs = np.arange(len(values))
        x1 = xs[0:i]
        x2 = xs[i:]
        A1 = np.vstack([x1,np.ones(len(x1))]).T
        A2 = np.vstack([x2,np.ones(len(x2))]).T
        fit1,sse1,rank1,s1 = lstsq(A1,vals1,rcond=None)
        fit2,sse2,rank2,s2 = lstsq(A2,vals2,rcond=None)
        rmse1 = np.sqrt(sse1)[0]
        rmse2 = np.sqrt(sse2)[0]

        totalR = (len(x1)/len(values))*rmse1 + (len(x2)/len(values))*rmse2

        if totalR < minR:
            minR = totalR
            optimal_split = (i,fit1,fit2,totalR)

    if plot:
        l1 = optimal_split[1]
        l2 = optimal_split[2]
        i = optimal_split[0]
        plt.bar(xs,values)
        plt.plot(xs[0:i],l1[0]*xs[0:i]+l1[1],'--')
        plt.plot(xs[i:],l2[0]*xs[i:]+l2[1],'--')

    return optimal_split