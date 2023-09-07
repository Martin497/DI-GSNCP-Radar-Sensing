# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:37:13 2023

@author: Martin Voigt Vejling
Email: mvv@math.aau.dk

Track changes:
    v1.0 - Implemented a toy example for the doubly inhomogeneous
           generalized shot noise Cox process model.
           Added docstrings. Included an Inverse-Wishart
           extent model. Included a sensor resolution function on the
           spatial domain. (22/08/2023)
"""


import numpy as np

from scipy.stats import invwishart
from scipy.linalg import cholesky
from scipy.interpolate import RegularGridInterpolator

from PP_Sim import PoissonPointProcess, CoxProcess

def extent_model(extent_par, Sigma, rho, sensor_resolution, extent_prior):
    """
    Parameters
    ----------
    extent_par : float or ndarray, size=(6,)
        The extent paramter.
    Sigma : ndarray, size=(K, 3, 3) or size=(3, 3)
        Sensor noise covariance.
    rho : ndarray, size=(K,) or float
        Detection probability.
    sensor_resolution : ndarray, size=(K,) or float
        The sensor resolution factor.

    Returns
    -------
    Sigma : ndarray, size=(K, 3, 3) or size=(3, 3)
        The extent-updated cluster spread.
    rho : ndarray, size=(K,) or float
        The extent-updated cluster sizes.
    """
    d = 3
    if extent_prior == "1d_uniform":
        E_mat = extent_par * np.eye(d)
    elif extent_prior == "inverse-wishart" or extent_prior == "6d_uniform":
        E_mat = np.zeros((d, d))
        tril_idx = np.tril_indices(d, k=0)
        E_mat[tril_idx] = extent_par

    if len(Sigma.shape) == 2:
        Sigma = Sigma + E_mat @ E_mat.T
    elif len(Sigma.shape) > 2:
        Sigma = Sigma + (E_mat @ E_mat.T)[None, :, :]

    rho = rho * np.maximum(np.ones(rho.shape), np.linalg.det(E_mat) * sensor_resolution)
    return Sigma, rho

def setup_functions(window, grid_dim, K):
    """
    Parameters
    ----------
    window : list, len=2*(d+1)
        The rectangular boundaries for the parameter space.
    grid_dim : list, len=d+1
        The number of grid points.
    K : int
        The number of sensors.

    Returns
    -------
    rho_interp : RegularGridInterpolator
        The cluster size function.
    Sigma_interp : RegularGridInterpolator
        cluster size function.
    """
    # Partitioning
    x_dist = (window[1]-window[0])/(grid_dim[0])
    y_dist = (window[3]-window[2])/(grid_dim[1])
    z_dist = (window[5]-window[4])/(grid_dim[2])
    # partition_dists = [x_dist, y_dist, z_dist]

    ext_x_points = np.linspace(window[0]-x_dist/2, window[1]+x_dist/2, grid_dim[0]+2)
    ext_y_points = np.linspace(window[2]-y_dist/2, window[3]+y_dist/2, grid_dim[1]+2)
    ext_z_points = np.linspace(window[4]-z_dist/2, window[5]+z_dist/2, grid_dim[2]+2)
    ext_partition_points = (ext_x_points, ext_y_points, ext_z_points)

    C1, C2, C3 = np.meshgrid(ext_x_points, ext_y_points, ext_z_points, indexing="ij")
    partition_flat = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)

    # Construct interpolator functions
    ext_partition_rho = rho_example_func(partition_flat)
    ext_partition_Sigma = Sigma_example_func(partition_flat)

    rho_interp = RegularGridInterpolator(ext_partition_points, ext_partition_rho.reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2)),
                                         method="linear", bounds_error=False, fill_value=None)
    Sigma_interp = RegularGridInterpolator(ext_partition_points, ext_partition_Sigma.reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2, 3, 3)),
                                           method="linear", bounds_error=False, fill_value=None)

    rho_interp = [rho_interp for k in range(K)]
    Sigma_interp = [Sigma_interp for k in range(K)]
    return rho_interp, Sigma_interp

def coord_rotation_matrix(self, yaw=np.pi/2, pitch=0, roll=0):
    """
    Three-dimensional rotation matrix defined as the negative of the rotation
    matrix rotating yaw radians around the z-axis, pitch radians
    around the y-axis, and roll radians round the x-axis.
    """
    Rz = np.array([[np.cos(yaw), - np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    O = -(Rz @ Ry @ Rx)
    return O

def sensor_res_fun(p, sU):
    """
    Sensor resolution function taking a spatial position p \in \R^d and
    a sensor state sU \in \R^d \times [0, 2\pi]^d.
    """
    c = 299792458
    d = 3
    W = 2e08
    Naz, Nel = 26, 26

    ### Input preparation ###
    pU, oU = sU[:, :d], sU[:, d:]
    K = sU.shape[0]

    ### Find channel parameters ###
    difference = p[None, :] - pU
    distance = np.linalg.norm(difference, axis=-1) # size=(K,)
    delay = distance/c # size=(K,)

    rotation_matrix = np.zeros((K, 3, 3)) # size=(K, 3, 3)
    for k in range(K):
        rotation_matrix[k] = coord_rotation_matrix(*oU[k, :])
    p_SU = np.einsum("kij,ki->kj", rotation_matrix, difference) # size=(K, 3)

    theta_az = np.arctan2(p_SU[:, 1], p_SU[:, 0]) # size=(K,)
    theta_el = np.arccos(p_SU[:, 2]/np.linalg.norm(p_SU, axis=-1)) # size=(K,)

    ### Construct resolution region ###
    delay_resolution = 1/(2*W)
    az_resolution = 2/((Naz-1)*np.abs(np.sin(theta_az)))
    el_resolution = 2/((Nel-1)*np.abs(np.sin(theta_el)))
    delay_interval = [delay-delay_resolution/2, delay+delay_resolution/2]
    az_interval = [theta_az-az_resolution/2, theta_az+az_resolution/2]
    el_interval = [theta_el-el_resolution/2, theta_el+el_resolution/2]

    ### Compute volume of resolution region ###
    volume = (az_interval[1] - az_interval[0]) \
             * (np.cos(el_interval[0]) - np.cos(el_interval[1])) \
             * 1/3 * ((delay_interval[1]*c)**3 - (delay_interval[0]*c)**3)

    ### The sensor resolution factor is the reciprocal of the volume ###
    r = np.minimum(np.ones(volume.shape)*5, 1/volume)
    return r

def rho_example_func(p):
    return (p[:, 0] + 3 * p[:, 1] + 2 * p[:, 2])/5 * 5

def Sigma_example_func(p):
    return 1/8 * np.ones(p.shape[0])[:, None, None] \
        * np.array([[1, 0.2, 0.1],
                    [0.2, 0.9, 0.05],
                    [0.1, 0.05, 0.5]])[None, :, :]

def DI_GSNCP(lambda_, R, rho_interp, Sigma_interp, lambda_c, K, window,
             sensor_resolution_function, sU, extent_prior, sigma_extent_max=None,
             sigma_extent_min=None, extent_IW_df=None, extent_IW_scale=None,
             cor_extent_min=None, cor_extent_max=None):
    """
    Parameters
    ----------
    lambda_ : float
        The ground driving process intensity.
    R : float
        The hard core radius for the ground driving process.
    rho_interp : RegularGridInterpolator
        The cluster size function.
    Sigma_interp : RegularGridInterpolator
        cluster size function.
    lambda_c : float
        The clutter intensity.
    K : int
        The number of sensors.
    window : list, len=2*(d+1)
        The rectangular boundaries for the parameter space.
    sensor_resolution_function : function
        The sensor resolution function mapping from \R^d \times \mathcal{S}.
    sU : ndarray, size=(K, 2d)
        The sensor positions and orientations.

    Returns
    -------
    Psi_da : list, len=K
        List of length K of lists of length L of ndarrays of size=(M_kl, d).
        This is the observed data with known origin and data association.
    Psi_c : list, len=K
        List of length K of ndarrays of size(M_c, d).
    Phi : ndarray, size=(L, d)
        The ground driving process.
    Phi_E : ndarray, size=(L,)
        The mark driving process.
    """
    d = 3
    Psi_c = list()
    Psi_da = list()
    Cox = CoxProcess(window)
    Phi = Cox.simulate_driving_process(lambda_, type_="MaternTypeII", HardCoreRadius=R)
    L = Phi.shape[0]
    if extent_prior == "1d_uniform":
        Phi_E = np.random.uniform(sigma_extent_min, sigma_extent_max, L)
        Tilde_Phi = np.concatenate((Phi, np.expand_dims(Phi_E, axis=-1)), axis=1)
    elif extent_prior == "inverse-wishart":
        Sigma_E = invwishart.rvs(extent_IW_df, extent_IW_scale, size=L).reshape((L, 3, 3)) # size=(partition_extent, 3, 3)
        tril_idx = np.tril_indices(d, k=0)
        Phi_E = np.zeros((L, 6))
        for i in range(L):
            E_mat = cholesky(Sigma_E[i], lower=True)
            Phi_E[i] = E_mat[tril_idx]
        Tilde_Phi = np.concatenate((Phi, Phi_E), axis=1)
    elif extent_prior == "6d_uniform":
        low_tril_idx = np.tril_indices(3, k=-1)
        tril_idx = np.tril_indices(3, k=0)
        Phi_E = np.zeros((L, 6))
        for l in range(L):
            std = np.random.uniform(sigma_extent_min, sigma_extent_max, size=3)
            cov = np.random.uniform(cor_extent_min, cor_extent_max, size=3)
            E_mat = std * np.eye(3)
            E_mat[low_tril_idx] = cov
            Phi_E[l] = E_mat[tril_idx]
        Tilde_Phi = np.concatenate((Phi, Phi_E), axis=1)

    for k in range(K):
        Psi_c.append(PoissonPointProcess(window, lambda_c).points)
        daughter_points = list()
        for C in Tilde_Phi:
            c, extent_par = C[:d], C[d:]
            Sigma = Sigma_interp[k](c)[0]
            rho = rho_interp[k](c)[0]
            sensor_resolution = sensor_resolution_function(c, sU)[k]
            Sigma_tilde, rho_tilde = extent_model(extent_par, Sigma, rho, sensor_resolution, extent_prior)
            daughter_points.append(np.random.multivariate_normal(c, Sigma_tilde, size=np.random.poisson(rho_tilde)))
        Psi_da.append(daughter_points)
    return Psi_da, Psi_c, Phi, Phi_E


if __name__ == "__main__":
    pass