# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:05:00 2024

@author: raman
"""

import mdtraj as mdt, numpy as np
import itertools

def local_dihedral_featurizer(trajlist,periodic):
    """
    Computes all possible dihedral angles in a list of trajectories
    Returns the sin/cos functions of those angles
    ----------
    Parameters
    ----------
    trajlist : list of mdtraj trajectories
    periodic : bool 
        determine whether unit cell wrapping needs to be applied
        see mdtraj.compute_phi for details
    -------
    Returns
    -------
    dihslist = list of numpy arrays
    """
    dihslist = []
    for traj in trajlist:
        phis = mdt.compute_phi(traj,periodic=periodic)[1]
        psis = mdt.compute_psi(traj,periodic=periodic)[1]
        chi1s = mdt.compute_chi1(traj,periodic=periodic)[1]
        chi2s = mdt.compute_chi2(traj,periodic=periodic)[1]
        chi3s = mdt.compute_chi3(traj,periodic=periodic)[1]
        chi4s = mdt.compute_chi4(traj,periodic=periodic)[1]
        omegas = mdt.compute_omega(traj,periodic=periodic)[1]
        cphis = np.hstack([np.sin(phis),np.cos(phis)])
        cpsis = np.hstack([np.sin(psis),np.cos(psis)])
        cchi1s = np.hstack([np.sin(chi1s),np.cos(chi1s)])
        cchi2s = np.hstack([np.sin(chi2s),np.cos(chi2s)])
        cchi3s = np.hstack([np.sin(chi3s),np.cos(chi3s)])
        cchi4s = np.hstack([np.sin(chi4s),np.cos(chi4s)])
        comegas = np.hstack([np.sin(omegas),np.cos(omegas)])
        dihs = np.hstack([cphis,cpsis,cchi1s,cchi2s,cchi3s,cchi4s,comegas])
        dihslist.append(dihs)
    return dihslist

def local_ca_distances_featurizer(trajlist,periodic):
    """
    compute the distances between alpha carbons in a series of trajectories

    ----------
    Parameters
    ----------
    trajlist : list of mdtraj trajectories
    periodic: bool
        whether to take PBCs into account (cf mdtraj.compute_distances documentation)

    -------
    Returns
    -------
    distances_feats : list of numpy arrays
    """
    distance_feats = []
    for traj in trajlist:
        cais = traj.topology.select("name CA")
        cai2s = np.array(list(itertools.combinations(cais,2)))
        ca_dists = mdt.compute_distances(traj,cai2s,periodic=periodic)
        distance_feats.append(ca_dists)
    return distance_feats

def local_contacts(traj,selection,threshold=0.3,periodic=True):
    """
    For a given trajectory and selection, returns the matrix of contacts
    between all pairs of selected atoms

    ----------
    Parameters
    ----------
    traj : mdtraj trajectory
    selection : list of indices
    threshold : cutoff distance in nm (default 0.3)
    periodic : bool, default True
        see mdtraj.compute_distances for more information

    -------
    Returns
    -------
    cmatrix : numpy array of 0s and 1s
    """
    pairs = np.array(list(itertools.combinations(selection,2)))
    dists = mdt.compute_distances(traj,pairs,periodic=periodic)
    dists_thresh = dists.copy()
    dists_thresh[dists<=threshold] = 1
    dists_thresh[dists>threshold] = 0
    return dists_thresh

def local_heavy_contacts_featurizer(trajlist,periodic):
    """
    computes contacts between heavy atoms in a list of trajectories

    ----------
    Parameters
    ----------
    trajlist : list of mdtraj trajectories
    periodic : bool
        for more information see mdtraj.compute_distances

    -------
    Returns
    -------
    cmatrices : list of numpy arrays
    
    """
    cmatrices = []
    for traj in trajlist:
        heavy = traj.topology.select("protein and not type H")
        cmatrix = local_contacts(traj,heavy)
        cmatrices.append(cmatrix)
    return cmatrices