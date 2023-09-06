# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:35:31 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Include oracle sensing model for 3D data with known covariance.
           Also implemented two object extent algorithms: a fast ad hoc
           approximation and a method based on numerical likelihood
           optimization. (04/08/2023)
    v1.1 - Altered the sensing_oracle class to support various inputs
           and to handle different extent priors.  (24/08/2023)
"""


import sys
import os
if os.path.abspath("..")+"/Modules" not in sys.path:
    sys.path.append(os.path.abspath("..")+"/Modules")

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.linalg import cholesky

from toy_simulation import DI_GSNCP, setup_functions, sensor_res_fun
from SensingMetrics import GOSPA, OSPA, OSPA_extent


class sensing_oracle(object):
    """
    The sensing oracle algorithm.
    """
    def __init__(self, **kwargs):
        """
        Keyword arguments
        -----------------
        extent : bool, optional
            If True estimate the object extent model parameters.
            The default is False.
        extent_optimize : bool, optional
            If True use the optimization algorithm to estimate object
            extent model parameters. The default is False.
        """
        super(sensing_oracle, self).__init__()
        self.d = 3
        self.max_cov = 1000

        self.extent = kwargs.get("extent", False)
        self.extent_prior = kwargs.get("extent_prior", "1d_uniform")
        self.extent_optimize = kwargs.get("extent_optimize", False)
        self.sigma_E_min = kwargs.get("sigma_E_min", 0.5)
        self.sigma_E_max = kwargs.get("sigma_E_max", 1.5)
        self.extent_IW_df = kwargs.get("extent_IW_df", 5)
        self.extent_IW_scale = kwargs.get("extent_IW_scale", 3*(self.extent_IW_df-self.d-1)*np.eye(self.d))
        self.cor_E_min = kwargs.get("cor_E_min", -0.5)
        self.cor_E_max = kwargs.get("cor_E_max", 0.5)

        self.tril_idx = np.tril_indices(self.d)
        assert self.extent_optimize is False, "This optimization does not work!"

    def sensing_oracle(self, points, covariance):
        """
        Oracle sensing algorithm: This assumes knowledge of the data association
        and then computes the maximum likelihood estimates given the data points
        and associated covariances.
        """
        K, L, M_kl, d = points.shape
        Phi_est = np.zeros((0, self.d))
        Sigma_est = np.zeros((0, self.d, self.d))
        rho_est = np.zeros(0)
        for l in range(L):
            points_temp = points[:, l, :, :].reshape(-1, self.d)
            covariance_temp = covariance[:, l, :, :, :].reshape(-1, self.d, self.d)

            covariance_temp[(covariance_temp > self.max_cov).any(axis=(1,2))] = None
            covariance_target_l = covariance_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]
            points_temp[(covariance_temp > self.max_cov).any(axis=(1,2))] = None
            points_target_l = points_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]

            M = points_target_l.shape[0]
            if M > 0:
                numerator = np.zeros(3)
                denominator = np.zeros((3, 3))
                cov_inv = np.linalg.inv(covariance_target_l)
                numerator = np.sum(np.einsum("mj,mji->mi", points_target_l, cov_inv), axis=0)
                denominator = np.sum(cov_inv, axis=0)
                est = np.expand_dims(np.dot(numerator, np.linalg.inv(denominator)), axis=0)
                var_est = np.expand_dims(np.linalg.inv(np.sum(cov_inv, axis=0)), axis=0)
                Phi_est = np.concatenate((Phi_est, est), axis=0)
                Sigma_est = np.concatenate((Sigma_est, var_est), axis=0)
                rho_est = np.concatenate((rho_est, np.array([M])), axis=0)
        return Phi_est, Sigma_est, rho_est

    # def sensing_oracle_extent(self, points, covariance, init_):
    #     """
    #     Oracle sensing algorithm: This assumes knowledge of the data association
    #     and then computes the maximum likelihood estimates given the data points
    #     and associated covariances.
    #     """
    #     K, L, M_kl, d = points.shape
    #     Tilde_Phi_est = np.zeros((0, d+1))
    #     Sigma_est = np.zeros((0, d+1, d+1))
    #     for l in range(L):
    #         points_temp = points[:, l, :, :].reshape(-1, 3)
    #         covariance_temp = covariance[:, l, :, :, :].reshape(-1, 3, 3)
    
    #         covariance_temp[(covariance_temp > 1000).any(axis=(1,2))] = None
    #         covariance_target_l = covariance_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]
    #         points_temp[(covariance_temp > 1000).any(axis=(1,2))] = None
    #         points_target_l = points_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]
    
    #         M = points_target_l.shape[0]
    #         if M > 0:
    #             x0 = init_[l, :]
    #             res = minimize(self.extent_fun, x0, args=(points_target_l, covariance_target_l), jac=self.extent_jac)
    #             est, var_est = np.expand_dims(res.x, axis=0), np.expand_dims(res.hess_inv, axis=0)
    #             Tilde_Phi_est = np.concatenate((Tilde_Phi_est, est), axis=0)
    #             Sigma_est = np.concatenate((Sigma_est, var_est), axis=0)
    #     return Tilde_Phi_est, Sigma_est

    # def extent_fun(self, x, Psi_l, Sigma):
    #     """
    #     Computing the negative log-likelihood function
    #     for the object extent model.
    #     """
    #     M_l, dim = Psi_l.shape
    #     c, sigma_E = x[:dim], x[dim]
    #     fun = 0
    #     for m in range(M_l):
    #         cov = Sigma[m]+sigma_E**2*np.eye(dim)
    #         inv_cov = np.linalg.inv(cov)
    #         det_cov = np.linalg.det(cov)
    #         if det_cov <= 0:
    #             det_cov = 1e-10
    #         log_term = np.log((2*np.pi)**dim * det_cov)
    #         prod_term = np.dot(Psi_l[m] - c, np.dot(inv_cov, Psi_l[m] - c))
    #         fun += log_term + prod_term
    #     return fun

    # def extent_jac(self, x, Psi_l, Sigma):
    #     """
    #     Compute the Jacobian for the object extent model.
    #     """
    #     M_l, dim = Psi_l.shape
    #     c, sigma_E = x[:dim], x[dim]
    #     dc = np.zeros(dim)
    #     dE = 0
    #     for m in range(M_l):
    #         inv_cov = np.linalg.inv(Sigma[m] + sigma_E**2*np.eye(dim))

    #         # position
    #         dc += np.dot(c, inv_cov) - np.dot(Psi_l[m], inv_cov)

    #         # extent
    #         dE += np.trace(Sigma[m]) + dim*sigma_E**2 - np.dot(Psi_l[m] - c, np.dot(np.dot(inv_cov, inv_cov), Psi_l[m] - c))
    #     dc = 2*dc
    #     jac = np.hstack((dc, dE))
    #     return jac

    def extent_sample_cov(self, points, covariance, Phi):
        """
        Estimate the object extent assuming a known object position by
        using a sample covariance approximation.
        """
        K, L, M_kl, d = points.shape
        if self.extent_prior == "1d_uniform":
            Phi_E_est = np.zeros(0)
            Sigma_Phi_E_est = np.zeros(0)
        elif self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform":
            Phi_E_est = np.zeros((0, 6))
            Sigma_Phi_E_est = np.zeros((0, 6, 6))
        counter = 0
        for l in range(L):
            points_temp = points[:, l, :, :].reshape(-1, self.d)
            covariance_temp = covariance[:, l, :, :, :].reshape(-1, self.d, self.d)
    
            covariance_temp[(covariance_temp > self.max_cov).any(axis=(1,2))] = None
            covariance_target_l = covariance_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]
            points_temp[(covariance_temp > self.max_cov).any(axis=(1,2))] = None
            points_target_l = points_temp[~np.isnan(covariance_temp).any(axis=(1, 2))]

            M = points_target_l.shape[0]
            if M > 1:
                Sigma_W = np.einsum("mj,mi->mji", points_target_l - Phi[counter], points_target_l - Phi[counter])
                Sigma_W_est = 1/(M-1) * np.sum(Sigma_W, axis=0)
                Sigma_E_est = Sigma_W_est - 1/(M-1) * np.sum(covariance_target_l, axis=0)

                if self.extent_prior == "1d_uniform":
                    Phi_E_l_est = np.mean(np.sqrt(np.maximum(np.diag(Sigma_E_est), np.ones(Sigma_E_est.shape[0])*1e-06)))
                    Phi_E_l_est = min(self.sigma_E_max, max(self.sigma_E_min, Phi_E_l_est))
                    Phi_E_est = np.hstack((Phi_E_est, Phi_E_l_est))
                    Sigma_Phi_E_l_est = np.mean(2/(M-1) * np.diag(Sigma_W_est)**2)
                    Sigma_Phi_E_est = np.hstack((Sigma_Phi_E_est, Sigma_Phi_E_l_est))
                elif self.extent_prior == "inverse-wishart":
                    try:
                        Phi_E_l_est = cholesky(Sigma_E_est)[self.tril_idx]
                    except np.linalg.LinAlgError:
                        Phi_E_l_est = cholesky(self.extent_IW_scale/(self.extent_IW_df - d- 1))[self.tril_idx]
                    Phi_E_est = np.concatenate((Phi_E_est, np.expand_dims(Phi_E_l_est, axis=0)), axis=0)
                    Sigma_Phi_E_l_est = np.zeros((1, 6, 6))
                    Sigma_Phi_E_l_est[:] = None
                    Sigma_Phi_E_est = np.concatenate((Sigma_Phi_E_est, Sigma_Phi_E_l_est), axis=0)
                elif self.extent_prior == "6d_uniform":
                    try:
                        Phi_E_l_est = cholesky(Sigma_E_est)[self.tril_idx]
                    except np.linalg.LinAlgError:
                        E_mat = np.zeros((3, 3))
                        E_mat = (self.sigma_E_min+self.sigma_E_max)/2 * np.eye(3)
                        Phi_E_l_est = E_mat[self.tril_idx]
                    Phi_E_est = np.concatenate((Phi_E_est, np.expand_dims(Phi_E_l_est, axis=0)), axis=0)
                    Sigma_Phi_E_l_est = np.zeros((1, 6, 6))
                    Sigma_Phi_E_l_est[:] = None
                    Sigma_Phi_E_est = np.concatenate((Sigma_Phi_E_est, Sigma_Phi_E_l_est), axis=0)
                counter += 1
            elif M > 0:
                if self.extent_prior == "1d_uniform":
                    Phi_E_est = np.hstack((Phi_E_est, (self.sigma_E_min+self.sigma_E_max)/2))
                    Sigma_Phi_E_est = np.hstack((Sigma_Phi_E_est, (self.sigma_E_min-self.sigma_E_max)**2/12))
                elif self.extent_prior == "inverse-wishart":
                    Phi_E_l_est = cholesky(self.extent_IW_scale/(self.extent_IW_df - d- 1))[self.tril_idx]
                    Phi_E_est = np.concatenate((Phi_E_est, np.expand_dims(Phi_E_l_est, axis=0)), axis=0)
                    Sigma_Phi_E_l_est = np.zeros((1, 6, 6))
                    Sigma_Phi_E_l_est[:] = None
                    Sigma_Phi_E_est = np.concatenate((Sigma_Phi_E_est, Sigma_Phi_E_l_est), axis=0)
                elif self.extent_prior == "6d_uniform":
                    E_mat = np.zeros((3, 3))
                    E_mat = (self.sigma_E_min+self.sigma_E_max)/2 * np.eye(3)
                    Phi_E_l_est = E_mat[self.tril_idx]
                    Phi_E_est = np.concatenate((Phi_E_est, np.expand_dims(Phi_E_l_est, axis=0)), axis=0)
                    Sigma_Phi_E_l_est = np.zeros((1, 6, 6))
                    Sigma_Phi_E_l_est[:] = None
                    Sigma_Phi_E_est = np.concatenate((Sigma_Phi_E_est, Sigma_Phi_E_l_est), axis=0)
                counter += 1
        return Phi_E_est, Sigma_Phi_E_est

    def prepare_data(self, Psi_da, Sigma_interp, Phi_true, Phi_E_true,
                     true_marks, true_extent):
        """
        Parameters
        ----------
        Psi_da : list
            List of length K of lists of length L of ndarrays of size=(M_kl, d).
            This is the observed data with known origin and data association.
        Sigma_interp : fun
            Function taking a spatial position and returning the
            related covariance.

        See run_oracle docstring for description of the remaining inputs.

        Returns
        -------
        oracle_Psi : ndarray, size=(K, L, max_M, d)
        oracle_Sigma : ndarray, size=(K, L, max_M, d, d)
        """
        # Rearrange measurements: list to ndarray - fill with nans
        K, L, d = len(Psi_da), len(Psi_da[0]), 3
        nr_meas_per_target_per_sensor = [[Psi_da[k][l].shape[0] for l in range(L)] for k in range(K)]
        max_nr_meas_per_sensor = [max(nr_meas_per_target_per_sensor[k], default=0) for k in range(K)]
        max_nr_meas = max(max_nr_meas_per_sensor)
        Psi = np.zeros((K, L, max_nr_meas, d))
        Psi[:] = None
        Sigma = np.zeros((K, L, max_nr_meas, d, d))
        Sigma[:] = None
        Tilde_Sigma = np.zeros((K, L, max_nr_meas, d, d))
        Tilde_Sigma[:] = None
        for k in range(K):
            for l in range(L):
                nr_meas = nr_meas_per_target_per_sensor[k][l]
                Psi[k, l, :nr_meas, :] = Psi_da[k][l]
                if true_marks is False:
                    Sigma_kl = Sigma_interp[k](Psi_da[k][l])
                elif true_marks is True:
                    Sigma_kl = (Sigma_interp[k](Phi_true[l])[0])[None, :, :] * np.ones(nr_meas)[:, None, None]
                if true_extent is False:
                    if self.extent_prior == "1d_uniform" or self.extent_prior == "6d_uniform":
                        Sigma_E = (self.sigma_E_min+self.sigma_E_max/2)**2 * np.eye(d)
                    elif self.extent_prior == "inverse-wishart":
                        Sigma_E = self.extent_IW_scale/(self.extent_IW_df - d- 1)
                elif true_extent is True:
                    if self.extent_prior == "1d_uniform":
                        Sigma_E = Phi_E_true[l]**2 * np.eye(d)
                    elif self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform":
                        E_mat = np.zeros((d, d))
                        E_mat[self.tril_idx] = Phi_E_true[l]
                        Sigma_E = E_mat @ E_mat.T                        
                    else:
                        Sigma_E = np.zeros((d, d))
                Tilde_Sigma_kl = Sigma_kl + Sigma_E[None, :, :]
                Sigma[k, l, :nr_meas, :, :] = Sigma_kl
                Tilde_Sigma[k, l, :nr_meas, :, :] = Tilde_Sigma_kl
        return Psi, Sigma, Tilde_Sigma

    def run_oracle(self, Psi, Sigma, **kwargs):
        """
        Main class call. Takes the observed data with known origin and
        data association and the known covariance function and returns the
        estimated driving process and the cluster size and spread estimates.

        Parameters
        ----------
        Psi : 
            Observed data. Data type required depends on prepare_data.
            If False, input ndarray, size=(K, L, M_kl, d).
            If True, input list of length K of lists of length L of
            ndarrays of size=(M_kl, d).
        Sigma : 
            Covariance. Data type required depends on prepared_data.
            If False, input ndarray, size=(K, L, M_kl, d, d).
            If True, input function taking a spatial position and returning
            the related covariance.

        Keyword arguments
        -----------------
        prepare_data : bool
            See oracle_Psi and oracle_Sigma description. The default is False.
        Phi_true : ndarray, size=(L, d)
            The true locations of the targets. The default is None.
        Phi_E_true : ndarray, size=(L, extent_dim)
            The true object extents. The default is None.
        true_marks : bool
            If True, use the covariance in the true location. If False, use
            the covariance in the observed location.
            Use this if prepare_data is True. The default is True.
        true_extent : bool
            If True, use the true extent in the covariance model. If False,
            assume zero extent. The default is True.
        true_loc : bool
            If True, use the true target positions for estimating the
            object extent.

        Returns
        -------
        Phi_oracle : ndarray, size=(L, d)
            The estimated driving process.
        Sigma_oracle : ndarray, size=(L, d, d)
            The covariance of the driving process estimates.
        rho_oracle : ndarray, size=(L,)
            The number of observed measurements per cluster center.
        """
        # =============================================================================
        # Keyword argument handling
        # =============================================================================
        prepare_data = kwargs.get("prepare_data", False)
        Phi_true = kwargs.get("Phi_true", None)
        Phi_E_true = kwargs.get("Phi_E_true", None)
        true_marks = kwargs.get("true_marks", True)
        true_extent = kwargs.get("true_extent", True)
        true_loc = kwargs.get("true_loc", True)

        # =============================================================================
        # Prepare data
        # =============================================================================
        if prepare_data is True:
            Psi, Sigma, Tilde_Sigma = self.prepare_data(Psi, Sigma, Phi_true, Phi_E_true,
                                                        true_marks, true_extent)            
        K, L, meas, dim = Psi.shape

        # =============================================================================
        # Estimate ground process
        # =============================================================================
        Phi_est, Sigma_Phi_est, rho_est = self.sensing_oracle(Psi, Sigma)
        if self.extent is False:
            return Phi_est, Sigma_Phi_est, rho_est

        # =============================================================================
        # Estimate mark process
        # =============================================================================
        L_est, _ = Phi_est.shape
        if self.extent is True:
            if true_loc is False:
                Phi_E_est, Sigma_Phi_E_est = self.extent_sample_cov(Psi, Sigma, Phi_est)
            elif true_loc is True:
                Phi_E_est, Sigma_Phi_E_est = self.extent_sample_cov(Psi, Sigma, Phi_true)
            if self.extent_prior == "1d_uniform":
                Tilde_Phi_est = np.concatenate((Phi_est, np.expand_dims(Phi_E_est, axis=1)), axis=1)
                Sigma_Tilde_Phi_est = np.zeros((L_est, 4, 4))
                Sigma_Tilde_Phi_est[:, :3, :3] = Sigma_Phi_est
                Sigma_Tilde_Phi_est[:, 3, 3] = Sigma_Phi_E_est
            elif self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform":
                Tilde_Phi_est = np.concatenate((Phi_est, Phi_E_est), axis=1)
                Sigma_Tilde_Phi_est = np.zeros((L_est, 9, 9))
                Sigma_Tilde_Phi_est[:, :3, :3] = Sigma_Phi_est
                Sigma_Tilde_Phi_est[:, 3:, 3:] = Sigma_Phi_E_est
            # if self.extent_optimize is True:
            #     Tilde_Phi_est, Sigma_Tilde_Phi_est = self.sensing_oracle_extent(oracle_Psi, oracle_Sigma, Tilde_Phi_est)
            return Tilde_Phi_est, Sigma_Tilde_Phi_est, rho_est

    def __call__(self, Psi, Sigma, **kwargs):
        """
        Run the oracle algorithm.
        """
        return self.run_oracle(Psi, Sigma, **kwargs)


def main():
    np.random.seed(10)

    # =============================================================================
    # Inputs
    # =============================================================================
    d = 3
    K = 2
    lambda_c = 3e-03
    lambda_ = 2e-03
    R = 6
    true_marks = True
    true_extent = True
    true_loc = True
    extent = True

    extent_prior = "inverse-wishart"
    sigma_E_min = 0.5
    sigma_E_max = 2.5
    extent_IW_df = 50
    extent_IW_var = 10
    extent_IW_scale = extent_IW_var*(extent_IW_df-d-1)*np.eye(d)

    sU = np.array([[-1, -1, -1, np.pi/2, np.pi/2, 0],
                   [51, 21, 11, np.pi/2, np.pi/2, 0]])
    window = [0, 50, 0, 20, 0, 10]
    grid_dim = [30, 12, 6]

    # =============================================================================
    # Simulate data
    # =============================================================================
    rho_interp, Sigma_interp = setup_functions(window, grid_dim, K)
    Psi_da, Psi_c, Phi, Phi_E = DI_GSNCP(lambda_, R, rho_interp, Sigma_interp, lambda_c, K, window,
                                         sensor_res_fun, sU, extent_prior,
                                         sigma_E_max, sigma_E_min,
                                         extent_IW_df, extent_IW_scale)

    if extent_prior == "1d_uniform":
        Tilde_Phi = np.concatenate((Phi, np.expand_dims(Phi_E, axis=-1)), axis=1)
    elif extent_prior == "inverse-wishart":
        Tilde_Phi = np.concatenate((Phi, Phi_E), axis=1)

    # Psi = [np.concatenate((Psi_c[k], np.concatenate(Psi_da[k])), axis=0) for k in range(K)]
    # L = Phi.shape[0]

    # =============================================================================
    # Setup input
    # =============================================================================
    init_kwargs = {"extent": extent,
                   "extent_prior": extent_prior,
                   "sigma_E_min": sigma_E_min,
                   "sigma_E_max": sigma_E_max,
                   "extent_IW_df": extent_IW_df,
                   "extent_IW_scale": extent_IW_scale}

    call_kwargs = {"prepare_data": True,
                   "true_marks": true_marks,
                   "true_extent": true_extent,
                   "true_loc": true_loc,
                   "Phi_true": Phi,
                   "Phi_E_true": Phi_E}

    # =============================================================================
    # Run oracle sensing algorithm
    # =============================================================================
    oracle_alg = sensing_oracle(**init_kwargs)
    Tilde_Phi_oracle, Sigma_est, rho_est = oracle_alg(Psi_da, Sigma_interp, **call_kwargs)
    Phi_oracle, Phi_E_oracle = Tilde_Phi_oracle[:, :d], Tilde_Phi_oracle[:, d:]

    print("Oracle   GOSPA         :", GOSPA(Phi, Phi_oracle, c=3))
    print("Oracle   OSPA          :", OSPA(Phi, Phi_oracle, c=3))
    print("Oracle   Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_oracle, c=3))

if __name__ == "__main__":
    main()

