# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:10:03 2024

@author: raman
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
from scipy.stats import gaussian_kde

def visFES2D(pcs,ax,cmap=matplotlib.cm.plasma,levels=10,thresh=1e-6,nbins=300,kbT=1):
    """
    visualizes a two-dimensional free energy surface, smoothed with a Gaussian kernel
    density estimation

    ----------
    Parameters
    ----------
    pcs : 2 x N numpy array
        the two variables in which to construct the FES
    ax : matplotlib axis object
        which axis to put the thing on
    cmap : matplotlib colormap, default = plasma
        consider carefully what colormap to use
        the default is a perceptually uniform colormap
    levels : int, default 10
        the number of contour levels to plot on the contour plot
    thresh : float, default 1e-6
        values of smaller than this threshold will be set to zero
    nbins : int, default = 300
        how finely to divide up the mesh for the final contour plot
    kbT : float, default = 1.0
        unit of energy, temperature-dependent. have a care -- this is reduced unit dependent

    -------
    Returns
    -------
    contour : matplotlib contourf object
    """
    x = pcs[:,0]
    y = pcs[:,1]
    xi,yi = np.mgrid[x.min():x.max():nbins*1j,y.min():y.max():nbins*1j]
    k = gaussian_kde(pcs[:,0:2].T)
    zi = k(np.vstack([xi.flatten(),yi.flatten()]))
    zi[np.abs(zi)<=thresh] = 0.0
    F = -kbT*np.log(zi)
    contour = ax.contourf(xi,yi,F.reshape(xi.shape),levels=levels,cmap=cmap)
    #ax.colorbar()
    return contour

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