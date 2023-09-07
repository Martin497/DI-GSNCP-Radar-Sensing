# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:00:23 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implement DBSCAN sensing algorithm using the oracle_sensing class
           as inheritance. Include hyperparameter optimization based on
           the negative log posterior. (25/08/2023)
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.stats import invwishart
from scipy.linalg import cholesky

from digsncp import DIGSNCP_base
from toy_simulation import DI_GSNCP, setup_functions, sensor_res_fun
from SensingMetrics import GOSPA, OSPA, OSPA_extent
from oracle import sensing_oracle


class sensing_DBSCAN(sensing_oracle, DIGSNCP_base):
    """
    The DBSCAN sensing algorithm.
    """
    def __init__(self, rho_interp, Sigma_interp, sensor_resolution_function,
                 sU, window, p_FA=1e-03, optimize=False, **kwargs):
        """
        Initilize the class from inheritance. See sensing_oracle docstring
        for description of keyword arguments.

        Parameters
        ----------
        rho_interp : fun
            Input function taking a spatial position and returning
            the related detection probability.
        Sigma_interp : fun
            Input function taking a spatial position and returning
            the related covariance.
        sensor_resolution_function : fun
            Input function taking a spatial position and a user state
            and returning the related radar resolution factor.
        sU : ndarray, size=(K, 6)
            The user states.
        window : list, len=6
            The boundaries of the rectancular spatial domain of interest.
        p_FA : float
            The false alarm probability. Used as a lower bound on interpolation
            of detection probability. The default is 1e-03.
        optimize : bool
            If False, use profile likelihood to estimate first
            spatial positions then extent. The default is False.
        kwargs : dict
            Input to initialize oracle class from inheritance.
        """
        super(sensing_DBSCAN, self).__init__(**kwargs)
        self.Sigma = Sigma_interp
        self.rho = rho_interp
        self.sensor_resolution_function = sensor_resolution_function

        self.window = window
        self.area = np.prod([self.window[2*i+1]-self.window[2*i] for i in range(self.d)]) # Area of observation window

        self.sU = sU
        self.K = sU.shape[0]
        self.p_FA = p_FA
        self.optimize = optimize

        self.tril_idx = np.tril_indices(self.d)

    def intensity_value(self, p, Phi, rho_k_Phi, Sigma_k_Phi):
        """
        Parameters
        ----------
        p : ndarray, size=(d,)
            The d-dimensional point to compute intensity value in.
        Phi : ndarray, size=(L, d)
            Ground driving process.
        rho_k_Phi : ndarray, size=(L,)
            Size for driving process.
        Sigma_k_Phi : ndarray, size=(L, d, d)
            Covariance for driving process.

        Returns
        -------
        res : ndarray, size=(L,)
            The intensity value for each point in the driving process.
        """
        in_ = p[None, :] - Phi # size=(L, d)

        with np.errstate(invalid="ignore"):
            det = np.maximum(np.linalg.det(Sigma_k_Phi), np.ones(Sigma_k_Phi.shape[0])*1e-02)
        inv = np.linalg.inv(Sigma_k_Phi)
        prod = np.einsum("lc,lc->l", in_, np.einsum("lcd,ld->lc", inv, in_))
        res = rho_k_Phi * 1/(np.sqrt(2*np.pi)**self.d*np.sqrt(det))*np.exp(-1/2*prod)
        return res

    def evaluate_size_and_spread(self, Phi_est, Phi_E_est):
        """
        Parameters
        ----------
        Phi_est : ndarray, size=(L, d)
            The estimate ground process.
        Phi_E_est : ndarray, size=(L, de)
            The estimated extent process.

        Returns
        -------
        rho_Phi : ndarray, size=(K, L)
            The cluster sizes.
        Sigma_Phi : ndarray, size=(K, L, d, d)
            The cluster spreads.
        """
        L = Phi_est.shape[0]
        Sigma_Phi = np.zeros((self.K, L, self.d, self.d))
        rho_Phi = np.zeros((self.K, L))
        for l in range(L):
            xi = Phi_est[l]
            sensor_resolution_xi = self.sensor_resolution_function(xi, self.sU)
            rho_xi, Sigma_xi = self.compute_rho_and_Sigma(xi)
            Sigma_xi, rho_xi = self.extent_model(Phi_E_est[l], Sigma_xi, rho_xi, sensor_resolution_xi)
            Sigma_Phi[:, l], rho_Phi[:, l] = Sigma_xi, rho_xi
        return rho_Phi, Sigma_Phi

    def negative_log_posterior_loss(self, Psi, Phi_est, Phi_E_est, lambda_est, lambda_c_est):
        """
        Parameters
        ----------
        Psi : list, len=K
            List of length K of list of lengths M_k of ndarray of size=(d,).
            The observed measurements.
        Phi_est : ndarray, size=(L, d)
            The estimate ground process.
        Phi_E_est : ndarray, size=(L, de)
            The estimated extent process.
        lamdba_est : float
            The estimated cluster center intensity.
        lambda_c_est : float
            The estimated clutter intensity.

        Returns
        -------
        ell : float
            The negative log posterior.
        """
        rho_Phi, Sigma_Phi = self.evaluate_size_and_spread(Phi_est, Phi_E_est)
        ell = -self.log_prior(Phi_est, Phi_E_est, lambda_est)
        for k, Psi_k in enumerate(Psi):
            ell -= (1-lambda_c_est)*self.area - np.sum(rho_Phi[k])
            for p in Psi_k:
                eta_k_p = self.intensity_value(p, Phi_est, rho_Phi[k], Sigma_Phi[k])
                Z_Phi_est_p = lambda_c_est + np.sum(eta_k_p)
                ell -= np.log(Z_Phi_est_p)
        return ell

    def __call__(self, Psi, eps, min_samples, **kwargs):
        """
        Parameters
        ----------
        Psi : list, len=K
            List of length K of ndarrays of size=(M_k, d) consisting of the
            measurements.
        eps : float or 1D ndarray
            DBSCAN neighborhood distance parameter.
        min_samples : float or 1D ndarray
            DBSCAN minimum weight to form cluter parameter.

        See run_oracle docstring for description of the keyword arguments.
        """
        K = len(Psi)
        M_k = [Psi_k.shape[0] for Psi_k in Psi]
        Psi_DBSCAN = np.concatenate(Psi)

        # Do data association with DBSCAN
        if self.optimize is False:
            DBSCAN_Cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(Psi_DBSCAN, sample_weight=None)
            DBSCAN_labels = DBSCAN_Cluster.labels_
            unique_labels = set(DBSCAN_labels)
            n_clusters = len(unique_labels) - (1 if -1 in DBSCAN_labels else 0)
    
            # Use data association to prepare data for target estimation
            A_0 = DBSCAN_labels + 1
            A_0_k = [A_0[sum(M_k[:i]):sum(M_k[:i+1])] for i in range(K)]
            Psi_da = list()
            for k in range(K):
                Psi_k = list()
                for l in range(1, n_clusters+1):
                    Psi_kl = Psi[k][A_0_k[k] == l]
                    Psi_k.append(Psi_kl)
                Psi_da.append(Psi_k)
    
            # Run sensing algorithm with the estimated data association
            Tilde_Phi_est, Sigma_Tilde_Phi_est, rho_est = self.run_oracle(Psi_da, self.Sigma, **kwargs)
            Phi_est, Phi_E_est = Tilde_Phi_est[:, :self.d], Tilde_Phi_est[:, self.d:]

        elif self.optimize is True:
            print("\nRun DBSCAN hyperparameter optimization!")
            eps_list, min_samples_list = eps, min_samples
            NLP = list()
            Tilde_Phi_est_list = list()
            Sigma_Tilde_Phi_est_list = list()
            rho_est_list = list()
            pars_list = list()
            OSPA_list = list()
            OSPA_extent_list = list()
            partition_extent = 1
            for i, eps in enumerate(eps_list):
                for j, min_samples in enumerate(min_samples_list):
                    pars_list.append([eps, min_samples])
                    DBSCAN_Cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(Psi_DBSCAN, sample_weight=None)
                    DBSCAN_labels = DBSCAN_Cluster.labels_
                    unique_labels = set(DBSCAN_labels)
                    n_clusters = len(unique_labels) - (1 if -1 in DBSCAN_labels else 0)
            
                    # Use data association to prepare data for target estimation
                    A_0 = DBSCAN_labels + 1
                    A_0_k = [A_0[sum(M_k[:i]):sum(M_k[:i+1])] for i in range(K)]
                    Psi_da = list()
                    for k in range(K):
                        Psi_k = list()
                        for l in range(1, n_clusters+1):
                            Psi_kl = Psi[k][A_0_k[k] == l]
                            Psi_k.append(Psi_kl)
                        Psi_da.append(Psi_k)
            
                    # Run sensing algorithm with the estimated data association
                    for k in range(partition_extent):
                        if partition_extent > 1:
                            if self.extent_prior == "inverse-wishart":
                                Sigma_E_star = invwishart.rvs(self.extent_IW_df, self.extent_IW_scale, size=(n_clusters))
                                Phi_E_star = np.zeros((n_clusters, 6))
                                for n in range(n_clusters):
                                    E_mat_star = cholesky(Sigma_E_star[n], lower=True)
                                    Phi_E_star[n] = E_mat_star[self.tril_idx]
                            kwargs["true_extent"] = True
                            kwargs["Phi_E_true"] = Phi_E_star
                        Tilde_Phi_est, Sigma_Tilde_Phi_est, rho_est = self.run_oracle(Psi_da, self.Sigma, **kwargs)
                        Phi_est, Phi_E_est = Tilde_Phi_est[:, :self.d], Tilde_Phi_est[:, self.d:]
                        Tilde_Phi_est_list.append(Tilde_Phi_est)
                        Sigma_Tilde_Phi_est_list.append(Sigma_Tilde_Phi_est)
                        rho_est_list.append(rho_est)

                        Phi = kwargs["Phi_true"]
                        Tilde_Phi = np.concatenate((Phi, kwargs["Phi_E_true"]), axis=-1)
                        OSPA_list.append(OSPA(kwargs["Phi_true"], Phi_est, c=10))
                        OSPA_extent_list.append(OSPA_extent(Tilde_Phi, Tilde_Phi_est, c=10))

                        lambda_est = Phi_est.shape[0]/self.area
                        lambda_c_est = np.sum(A_0==0)/(K*self.area)
                        NLP.append(self.negative_log_posterior_loss(Psi, Phi_est, Phi_E_est, lambda_est, lambda_c_est))

            argmin = np.argmin(NLP)
            Tilde_Phi_est = Tilde_Phi_est_list[argmin]
            Sigma_Tilde_Phi_est = Sigma_Tilde_Phi_est_list[argmin]
            rho_est = rho_est_list[argmin]
            pars_est = pars_list[argmin]
            print(f"Optimized hyperparameters     : eps = {pars_est[0]}, min-samples = {pars_est[1]}")

            OSPA_min = np.min(OSPA_list)
            OSPA_est = OSPA_list[argmin]
            Tilde_Phi_opt = Tilde_Phi_est_list[np.argmin(OSPA_list)]
            print(f"OSPA distance                 : True optimum = {OSPA_min:.3f}, Estimated optimum = {OSPA_est:.3f}")

            OSPA_extent_min = np.min(OSPA_extent_list)
            OSPA_extent_est = OSPA_extent_list[argmin]
            Tilde_Phi_extent_opt = Tilde_Phi_est_list[np.argmin(OSPA_extent_list)]
            print(f"Gaussian-Wasserstein distance : True optimum = {OSPA_extent_min:.3f}, Estimated optimum = {OSPA_extent_est:.3f}\n")
        return Tilde_Phi_est, Sigma_Tilde_Phi_est, rho_est, pars_est, Tilde_Phi_opt, Tilde_Phi_extent_opt


def main():
    np.random.seed(10)
    plt.style.use("ggplot")
    plt.rcParams["axes.grid"] = True
    verbose = True

    # =============================================================================
    # Inputs
    # =============================================================================
    d = 3
    K = 2
    lambda_c = 3e-03
    lambda_ = 2e-03
    R = 6
    extent = True
    optimize = True

    eps = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    min_samples = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    extent_prior = "inverse-wishart"
    sigma_E_min = 1
    sigma_E_max = 1.5
    extent_IW_df = 5
    extent_IW_var = 2
    extent_IW_scale = extent_IW_var*(extent_IW_df-d-1)*np.eye(d)

    p_FA = 1e-03

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
    Psi = [np.concatenate((Psi_c[k], np.concatenate(Psi_da[k])), axis=0) for k in range(K)]

    if extent_prior == "1d_uniform":
        Tilde_Phi = np.concatenate((Phi, np.expand_dims(Phi_E, axis=-1)), axis=1)
    elif extent_prior == "inverse-wishart":
        Tilde_Phi = np.concatenate((Phi, Phi_E), axis=1)

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
                   "true_marks": False,
                   "true_extent": False,
                   "true_loc": False,
                   "Phi_true": Phi,
                   "Phi_E_true": Phi_E}

    # =============================================================================
    # Run DBSCAN sensing algorithm
    # =============================================================================
    DBSCAN_alg = sensing_DBSCAN(rho_interp, Sigma_interp, sensor_res_fun,
                                sU, window, p_FA, optimize, **init_kwargs)
    Tilde_Phi_DBSCAN, Sigma_est, rho_est, _, _, _ = DBSCAN_alg(Psi, min_samples, eps,**call_kwargs)
    Phi_DBSCAN, Phi_E_DBSCAN = Tilde_Phi_DBSCAN[:, :d], Tilde_Phi_DBSCAN[:, d:]

    if verbose is True:
        ### Sensing plot ###
        plt.plot(Psi[0][:, 0], Psi[0][:, 1],  color="purple", marker="x", markersize=2.5, linestyle="", alpha=0.75, label="Measurements")
        for k in range(1, K): # Measurements
            plt.plot(Psi[k][:, 0], Psi[k][:, 1],  color="purple", marker="x", markersize=2.5, linestyle="", alpha=0.75)
        plt.plot(Phi[0, 0], Phi[0, 1], color="tab:red", marker="o", markersize=4, linestyle="", label="Targets")
        for l in range(1, Phi.shape[0]): # Target
            plt.plot(Phi[l, 0], Phi[l, 1], color="tab:red", marker="o", markersize=4, linestyle="")
        plt.plot(Phi_DBSCAN[0, 0], Phi_DBSCAN[0, 1],  color="tab:blue", marker="o", markersize=4, linestyle="", label="DBSCAN")
        for l in range(1, Phi_DBSCAN.shape[0]): # DBSCAN
            plt.plot(Phi_DBSCAN[l, 0], Phi_DBSCAN[l, 1],  color="tab:blue", marker="o", markersize=4, linestyle="")
        plt.xlim(window[0]-2, window[1]+2)
        plt.ylim(window[2]-2, window[3]+2)
        plt.legend()
        plt.show()

        print("DBSCAN   GOSPA         :", GOSPA(Phi, Phi_DBSCAN, c=3))
        print("DBSCAN   OSPA          :", OSPA(Phi, Phi_DBSCAN, c=3))
        print("DBSCAN   Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_DBSCAN, c=3))


if __name__ == "__main__":
    main()



