# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:29:58 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implement the multi-scan multi-sensor multiple extended target
           sensing with oracle, DBSCAN, and MCMC information fusion methods.
           The filter iterates the belief through time as additional scans
           are made, thereby collecting measurements and
           performing processing. (25/08/2023)
    v1.1 - Adapted to v1.4 MCMC.py, v1.2 oracle.py, and v1.0 clustering.py.
           Changed inheritance to Fisher_Information instead
           of Measurement_Model. Update to sensor_resolution taking
           multiple channel uses into account. CRLB exception handling.
           Zero measurements exception handling. (07/09/2023)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pickle
import toml
from scipy.stats import ncx2, invwishart
from scipy.linalg import cholesky
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

import warnings

from fisher_information import Fisher_Information
from oracle import sensing_oracle
from clustering import sensing_DBSCAN
from MCMC import RJMCMC_DoublyNeymanScott
from PP_Sim import PoissonPointProcess, CoxProcess
from SensingMetrics import GOSPA, OSPA, OSPA_extent


class main_filter(Fisher_Information):
    """
    In this class sensing maps are constructed through time.
    """
    def __init__(self, sensing_alg=["oracle"], oracle_hparams=dict(), DBSCAN_hparams=dict(), MCMC_hparams=dict(), verbose=False, **kwargs):
        """
        Initialize experiment scenario.
        """
        super(main_filter, self).__init__(None, False, False, **kwargs)
        self.sensing_alg = sensing_alg
        self.Q = len(self.signal_types)
        self.oracle_hparams = oracle_hparams
        self.DBSCAN_hparams = DBSCAN_hparams
        self.MCMC_hparams = MCMC_hparams
        self.verbose = verbose

        area = np.prod([self.window[2*i+1]-self.window[2*i] for i in range(3)])
        self.lambda_c = self.lambda_c_Poi/area
        self.lambda_Phi = self.lambda_Poi/area

        self.max_cov = 20*np.eye(3)
        self.tril_idx = np.tril_indices(3, k=0)

        if self.resource_division == "time_division":
            self.time_division = True
            self.frequency_division = False
            self.T = self.T//2
            self.T1 = self.T1//2
            self.T2 = self.T//2 - self.T1
            assert self.K == 2, "Time division only meant for two users!"
        elif self.resource_division == "frequency_division":
            self.time_division = False
            self.frequency_division = True
            self.N = self.N//self.K
            assert self.K == 2, "Frequency division only meant for two users!"
        else:
            self.time_division = False
            self.frequency_division = False
            assert self.K == 1, "Assuming one user!"

        self.MCMC_init_kwargs = {"p_m": self.MCMC_hparams["p_m"],
                                 "p_em": self.MCMC_hparams["p_em"],
                                 "vartheta": self.MCMC_hparams["vartheta"],
                                 "method": self.MCMC_hparams["method"],
                                 "clutter_std": self.MCMC_hparams["clutter_std"],
                                 "object_extent": self.extent,
                                 "partition_extent": self.MCMC_hparams["partition_extent"],
                                 "extent_move_std": self.MCMC_hparams["extent_move_std"],
                                 "acceptance_type": self.MCMC_hparams["acceptance_type"],
                                 "p_FA": self.p_FA,
                                 "extent_prior": self.MCMC_hparams["extent_prior"],
                                 "sigma_extent_max": self.MCMC_hparams["sigma_extent_max"],
                                 "sigma_extent_min": self.MCMC_hparams["sigma_extent_min"],
                                 "extent_IW_df": self.MCMC_hparams["extent_IW_df"],
                                 "extent_IW_scale": self.MCMC_hparams["extent_IW_scale"],
                                 "cor_extent_min": self.MCMC_hparams["cor_extent_min"],
                                 "cor_extent_max": self.MCMC_hparams["cor_extent_max"],
                                 "sensor_resolution_function": self.sensor_resolution_function,
                                 "birth_density_v1": self.MCMC_hparams["birth_density_v1"],
                                 "birth_shift": self.MCMC_hparams["birth_shift"],
                                 "birth_clutter_thin_var": self.MCMC_hparams["birth_clutter_thin_var"],
                                 "data association_sim": self.MCMC_hparams["data_association_sim"],
                                 "patience": self.MCMC_hparams["patience"],
                                 "verbose": self.verbose}

        self.MCMC_call_kwargs = {"burn_in": self.MCMC_hparams["burn_in"],
                                 "move_prob_inc_interval": self.MCMC_hparams["move_prob_inc_interval"],
                                 "move_prob_inc": self.MCMC_hparams["move_prob_inc"],
                                 "move_prob_max": self.MCMC_hparams["move_prob_max"],
                                 "move_trace_interval": self.MCMC_hparams["move_trace_interval"],
                                 "move_trace_min": self.MCMC_hparams["move_trace_min"],
                                 "move_trace_max": self.MCMC_hparams["move_trace_max"],
                                 "move_trace_decrease": self.MCMC_hparams["move_trace_decrease"],
                                 "move_trace_increase": self.MCMC_hparams["move_trace_increase"]}

        self.oracle_init_kwargs = {"extent": True,
                                   "sigma_E_min": self.MCMC_hparams["sigma_extent_min"],
                                   "sigma_E_max": self.MCMC_hparams["sigma_extent_max"],
                                   "extent_prior": self.MCMC_hparams["extent_prior"],
                                   "extent_IW_df": self.MCMC_hparams["extent_IW_df"],
                                   "extent_IW_scale": self.MCMC_hparams["extent_IW_scale"],
                                   "cor_E_min": self.MCMC_hparams["cor_extent_min"],
                                   "cor_E_max": self.MCMC_hparams["cor_extent_max"]}

        self.oracle_call_kwargs = {"prepare_data": False,
                                   "true_marks": self.oracle_hparams["true_marks"],
                                   "true_extent": self.oracle_hparams["true_extent"],
                                   "true_loc": self.oracle_hparams["true_loc"]}
        self.oracle_alg = sensing_oracle(**self.oracle_init_kwargs)

        self.DBSCAN_init_kwargs = {"extent": True,
                                   "sigma_E_min": self.MCMC_hparams["sigma_extent_min"],
                                   "sigma_E_max": self.MCMC_hparams["sigma_extent_max"],
                                   "extent_prior": self.MCMC_hparams["extent_prior"],
                                   "extent_IW_df": self.MCMC_hparams["extent_IW_df"],
                                   "extent_IW_scale": self.MCMC_hparams["extent_IW_scale"],
                                   "cor_E_min": self.MCMC_hparams["cor_extent_min"],
                                   "cor_E_max": self.MCMC_hparams["cor_extent_max"]}

        self.DBSCAN_call_kwargs = {"prepare_data": True,
                                   "true_marks": False,
                                   "true_extent": False,
                                   "true_loc": False}

    def lambda_N(self, pars, f_tilde, aU_thetal):
        """
        Compute non-centrality parameter non-RIS.
        """
        inside_term = np.abs(np.einsum("keli,tkei->tkel", aU_thetal, f_tilde))**2
        sum_term = np.prod(self.N_U)*(self.N+1)*np.sum(inside_term, axis=0)
        return 4*(np.abs(pars["betal"])**2)/self.p_noise_sc * sum_term

    def lambda_D(self, pars, W_mat, aR_phi0, ar_phil, aU_thetal, nu_tilde):
        """
        Compute non-centrality parameter directional.
        """
        factor = np.prod(self.N_U)*(self.N+1)*4*(np.abs(pars["alphal"])**2)/self.p_noise_sc
        time_sum = np.sum(np.abs(nu_tilde)**2, axis=0)
        combiner_term = np.sum(np.abs(np.einsum("keli,keij->kelj", np.conjugate(aU_thetal), W_mat[:, :, :, 1:]))**2, axis=-1)
        return factor*time_sum*combiner_term

    def lambda_O(self, pars, f_tilde, nu_tilde, aU_thetal):
        """
        Compute non-centrality parameter orthogonal.
        """
        inside_term = np.abs(nu_tilde)**2*np.abs(np.einsum("keli,tkei->tkel", aU_thetal, f_tilde))**2
        sum_term = np.prod(self.N_U)*(self.N+1)*np.sum(inside_term, axis=0)
        return 4*(np.abs(pars["alphal"])**2)/self.p_noise_sc * sum_term

    def simulate_UE_positions(self, K, E, heading, speed, epoch_time, UE_in=None):
        """
        Simulate position of two UEs as they move through space in time.
        """
        if self.time_division is True:
            epoch_time = epoch_time*2

        if UE_in is not None:
            p = UE_in
        else:
            p = np.zeros((K, E, 3))
            if K == 1:
                p[:, 0, :] = np.array(
                    [[self.window[0], self.window[2]-4, (self.window[5]+self.window[4])/2]])
                v = np.array([[speed, 0, 0]])
            elif K == 2:
                p[:, 0, :] = np.array([[self.window[0], self.window[2]-4, (self.window[5]+self.window[4])/2],
                                       [self.window[1], self.window[3]+4, (self.window[5]+self.window[4])/2]])
                v = np.array([[speed, 0, 0],
                              [-speed, 0, 0]])
            else:
                assert False, "Undefined user position definition"

            for tau in range(1, E):
                p[:, tau, :] = p[:, tau-1, :] + v*epoch_time
        return p

    def sensor_resolution_function(self, p, sU):
        """
        Sensor resolution function taking a spatial position p \in \R^d and
        a sensor state sU \in \R^d \times [0, 2\pi]^d.
        """    
        ### Input preparation ###
        pU, oU = sU[:, :self.d], sU[:, self.d:]
        K = sU.shape[0]
        Naz, Nel = self.N_U

        ### Find channel parameters ###
        difference = p[None, :] - pU
        distance = np.linalg.norm(difference, axis=-1) # size=(K,)
        delay = distance/self.c # size=(K,)

        rotation_matrix = np.zeros((K, 3, 3)) # size=(K, 3, 3)
        for k in range(K):
            rotation_matrix[k] = self.coord_rotation_matrix(*oU[k, :])
        p_SU = np.einsum("kij,ki->kj", rotation_matrix, difference) # size=(K, 3)

        theta_az = np.arctan2(p_SU[:, 1], p_SU[:, 0]) # size=(K,)
        theta_el = np.arccos(p_SU[:, 2]/np.linalg.norm(p_SU, axis=-1)) # size=(K,)

        # Find the two closest integers which, when multiplied, equals T
        N = np.ceil(np.sqrt(self.T))
        mod_ = -1
        N -= 1
        while mod_ != 0:
            N += 1
            mod_ = self.T % N
        az_fac = N
        el_fac = self.T/N
        assert az_fac*el_fac == self.T, "The number of channel uses must have a decomposition into two integers!"

        ### Construct resolution region ###
        delay_resolution = 1/(2*self.W)
        az_resolution = 2/(az_fac*(Naz-1)*np.abs(np.sin(theta_az)))
        el_resolution = 2/(el_fac*(Nel-1)*np.abs(np.sin(theta_el)))
        delay_interval = [delay-delay_resolution/2, delay+delay_resolution/2]
        az_interval = [theta_az-az_resolution/2, theta_az+az_resolution/2]
        el_interval = [theta_el-el_resolution/2, theta_el+el_resolution/2]

        ### Compute volume of resolution region ###
        volume = (az_interval[1] - az_interval[0]) \
                 * (np.cos(el_interval[0]) - np.cos(el_interval[1])) \
                 * 1/3 * ((delay_interval[1]*self.c)**3 - (delay_interval[0]*self.c)**3)

        ### The sensor resolution factor is the reciprocal of the volume ###
        if self.real_resolution is True:
            r = np.maximum(np.minimum(np.ones(volume.shape)*50, 1/volume), np.ones(volume.shape)*1)
        else:
            r = np.maximum(np.minimum(np.ones(volume.shape)*8, 1/volume), np.ones(volume.shape)*4)
        return r

    def plot_sensor_resolution(self, sU):
        """
        Parameters
        ----------
        sU : ndarray, size=(K, 2*d)
            The user states.
        """
        grid_dim = [50, 20, 10]
        window = self.window

        # Partitioning
        x_dist = (window[1]-window[0])/(grid_dim[0])
        y_dist = (window[3]-window[2])/(grid_dim[1])
        z_dist = (window[5]-window[4])/(grid_dim[2])
    
        x_points = np.linspace(window[0]-x_dist/2, window[1]+x_dist/2, grid_dim[0]+2)
        y_points = np.linspace(window[2]-y_dist/2, window[3]+y_dist/2, grid_dim[1]+2)
        z_points = np.linspace(window[4]-z_dist/2, window[5]+z_dist/2, grid_dim[2]+2)
    
        C1, C2, C3 = np.meshgrid(x_points, y_points, z_points, indexing="ij")
        partition_flat = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)

        r = np.zeros((self.K, partition_flat.shape[0]))
        for i, p in enumerate(partition_flat):
            r[:, i] = self.sensor_resolution_function(p, sU)

        plt.style.use("ggplot")
        plt.rcParams["axes.grid"] = False
        for k in range(self.K):
            spec = plt.pcolormesh(x_points, y_points, r[k].reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2))[:, :, 5].T,
                                  cmap="viridis_r", shading="auto", norm=LogNorm(vmin=np.min(r[k]), vmax=np.max(r[k])))
            cb = plt.colorbar(spec)
            cb.set_label(label="Sensor resolution factor")
            plt.show()

    def rho_func(self, pos, s_U, f_tilde, omega_tilde, W_mat, signal_type):
        """
        Parameters
        ----------
        pos : ndarray, size=(L, d)
            The positions in which to evaluate.
        s_U : ndarray, size=(d+2,)
            The UE state.
        f_tilde : ndarray, size=(T/2, N_U)
            Precoder.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile.
        W_mat : ndarray, size=(N_U, N_U)
            Combiner.
        signal_type : str
            The type of signal.

        Returns
        -------
        detection_probability : ndarray, size=(L,)
            The detection probability.
        """
        L, d = pos.shape
        f_tilde = np.expand_dims(f_tilde, axis=(1, 2))
        W_mat = np.expand_dims(W_mat, axis=(0, 1))
        s_U = np.expand_dims(s_U, axis=(0, 1))

        pars = self.initialise_channel_params(pos, s_U)
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)
        if signal_type == "directional" or signal_type == "orthogonal":
            aR_phi0 = self.UPA_array_response_vector(pars["phi0"], self.N_R, self.d_R) # size=(K, E, N_R)
            aR_phil = self.UPA_array_response_vector(pars["phil"], self.N_R, self.d_R) # size=(L, N_R)
            aR_phi0_prod = omega_tilde[:, None, None, :]*aR_phi0[None, :, :, :]
        gamma = -2*np.log(self.p_FA)

        if signal_type == "nonRIS":
            noncentrality = self.lambda_N(pars, f_tilde, aU_thetal)[0, 0]
        elif signal_type == "directional":
            nu_tilde_D = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod[:self.T1, :, :, :]) # size=(T, K, E, L)
            noncentrality = self.lambda_D(pars, W_mat, aR_phi0, aR_phil, aU_thetal, nu_tilde_D)[0, 0]
        elif signal_type == "orthogonal":
            nu_tilde_O = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod[self.T1:, :, :, :]) # size=(T, K, E, L)
            noncentrality = self.lambda_O(pars, f_tilde[self.T1:, :, :, :], nu_tilde_O, aU_thetal)[0, 0]

        detection_probability = np.zeros(L)
        for l in range(L):
            detection_probability[l] = 1 - ncx2.cdf(gamma, 2, noncentrality[l])
        return detection_probability

    def Sigma_func(self, pos, s_U, f_tilde, omega_tilde, W_mat, signal_type):
        """
        Parameters
        ----------
        pos : ndarray, size=(L, d)
            The positions in which to evaluate.
        s_U : ndarray, size=(d+2,)
            The UE state.
        f_tilde : ndarray, size=(T/2, N_U)
            Precoder.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile.
        W_mat : ndarray, size=(N_U, N_U)
            Combiner.
        signal_type : str
            The type of signal.

        Returns
        -------
        CRLB : ndarray, size=(L, d, d)
            The covariance.
        """
        L, d = pos.shape
        f_tilde = np.expand_dims(f_tilde, axis=(1, 2))
        W_mat = np.expand_dims(W_mat, axis=(0, 1))
        s_U = np.expand_dims(s_U, axis=(0, 1))

        sims = 1
        for sim in range(sims):
            CRLB_sims = np.zeros((sims, L, d, d))
            for l in range(L):
                p = np.expand_dims(pos[l], axis=0)
                pars = self.initialise_channel_params(p, s_U)
                if signal_type == "nonRIS":
                    FIM = self.Fisher_information_matrix_nonRIS_simplified(pars, f_tilde)
                elif signal_type == "directional":
                    FIM = self.Fisher_information_matrix_directional_simplified(pars, omega_tilde, W_mat)
                elif signal_type == "orthogonal":
                    FIM = self.Fisher_information_matrix_orthogonal_simplified(pars, f_tilde, omega_tilde, W_mat)
                CRLB_sims[sim, l] = self.FIM_processing(FIM, pars, pos, s_U[:, :, :3], s_U[:, :, 3:])[0, 0, 0, :, :] # size=(3, 3)
                if np.any(np.isnan(CRLB_sims[sim, l])) or np.any(np.linalg.eigvals(CRLB_sims[sim, l]) <= 0):
                    CRLB_sims[sim, l] = self.max_cov
        CRLB = np.mean(CRLB_sims, axis=0)
        return CRLB

    def boundary_points(self, window):
        """
        Make an array of size=(8, 3) with the boundary points of the window.
        """
        points = np.array([[window[0], window[2], window[4]],
                           [window[0], window[2], window[5]],
                           [window[0], window[3], window[4]],
                           [window[0], window[3], window[5]],
                           [window[1], window[2], window[4]],
                           [window[1], window[2], window[5]],
                           [window[1], window[3], window[4]],
                           [window[1], window[3], window[5]]])
        return points

    def construct_interpolations(self, window, grid_dim, sU, f_tilde, omega_tilde, W_mat,
                                 points=None):
        """
        Parameters
        ----------
        window : list, len = 2*d
            The boundary of the assumed rectangular domain. In 3D space the
            list should be specified as
                [x_min, x_max, y_min, y_max, z_min, z_max]
        grid_dim : list, len = d
            The number of grid points. In 3D space the
            list should be specified as
                [nr_x, nr_y, nr_z]
        s_U : ndarray, size=(K, 1, d+2)
            The UE state.
        f_tilde : ndarray, size=(T/2, K, 1, N_U)
            Precoder for times t_tilde = 1,\dots,T/2.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile for times t_tilde = 1,\dots,T/2
        W_mat : ndarray, size=(K, 1, N_U, N_U)
            Combiner matrix.

        Returns
        -------
        rho_interp : RegularGridInterpolator
            Detection probability interpolator.
        Sigma_interp : RegularGridInterpolator
            Sensor noise covariance interpolator.
        """
        self.regular_grid_interpolator = True
        if self.regular_grid_interpolator is True:
            # Partitioning
            x_dist = (window[1]-window[0])/(grid_dim[0])
            y_dist = (window[3]-window[2])/(grid_dim[1])
            z_dist = (window[5]-window[4])/(grid_dim[2])
    
            ext_x_points = np.linspace(window[0]-x_dist/2, window[1]+x_dist/2, grid_dim[0]+2)
            ext_y_points = np.linspace(window[2]-y_dist/2, window[3]+y_dist/2, grid_dim[1]+2)
            ext_z_points = np.linspace(window[4]-z_dist/2, window[5]+z_dist/2, grid_dim[2]+2)
            ext_partition_points = (ext_x_points, ext_y_points, ext_z_points)
    
            C1, C2, C3 = np.meshgrid(ext_x_points, ext_y_points, ext_z_points, indexing="ij")
            partition_flat = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)
    
            # Construct interpolator functions
            ext_partition_rho = [self.rho_func(partition_flat, sU[k, 0],
                                               f_tilde[:, k, 0, :], omega_tilde, W_mat[k, 0, :, :],
                                               type_) for type_ in self.signal_types for k in range(self.K)]
            ext_partition_Sigma = [self.Sigma_func(partition_flat, sU[k, 0],
                                                   f_tilde[:, k, 0, :], omega_tilde, W_mat[k, 0, :, :],
                                                   type_) for type_ in self.signal_types for k in range(self.K)]

            rho_interp = [RegularGridInterpolator(ext_partition_points,
                                    ext_partition_rho[k].reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2)),
                                    method="linear", bounds_error=False, fill_value=self.p_FA) for k in range(self.Q*self.K)]
            Sigma_interp = [RegularGridInterpolator(ext_partition_points,
                                    ext_partition_Sigma[k].reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2, 3, 3)),
                                    method="linear", bounds_error=False, fill_value=self.max_cov) for k in range(self.Q*self.K)]
        elif self.regular_grid_interpolator is False:
            L, max_measurements = points.shape[2], points.shape[3]
            covariance_marks = np.zeros((self.Q, self.K, L+1, max_measurements, self.d, self.d))
            covariance_marks[:] = None
            points_boundary = self.boundary_points(self.window)[None, None, :, :] * np.ones((self.Q, self.K))[:, :, None, None] # size=(Q, K, 8, d)
            covariance_boundary = np.zeros((self.Q, self.K, 8, self.d, self.d)) # size=(Q, K, 8, d, d)
            for q in range(self.Q):
                for k in range(self.K):
                    f_temp = np.expand_dims(f_tilde[:, k, :, :], axis=1)
                    W_temp = np.expand_dims(W_mat[k], axis=0)
                    sU_k = np.expand_dims(sU[k], axis=0)
                    for l in range(L+1):
                        M = np.sum(~np.isnan(points[q, k, l, :, 0]))
                        covariance_marks[q, k, -1, :M, :, :] = self.compute_cramer_rao(points[q, k, l, :M, :],
                                                                        sU_k, f_temp, omega_tilde, W_temp, self.signal_types[q])
                        covariance_boundary[q, k, : :, :] = self.compute_cramer_rao(points_boundary[q, k, :, :],
                                                                        sU_k, f_temp, omega_tilde, W_temp, self.signal_types[q])

            points_flat = points.reshape((self.Q, self.K, -1, self.d))
            ground_points = np.concatenate((points_flat[~np.isnan(points_flat).all(axis=-1)], points_boundary), axis=2)
            covariance_flat = covariance_marks.reshape((self.Q, self.K, -1, self.d, self.d))
            covariance_points = np.concatenate((covariance_flat[~np.isnan(covariance_flat).all(axis=(-2, -1))], covariance_boundary), axis=2)

            Sigma_interp = [LinearNDInterpolator(ground_points[q, k], covariance_points[q, k], fill_value=self.max_cov) for q in range(self.Q) for k in range(self.K)]

        if self.verbose is True:
            grid_dim = [50, 20, 10]
    
            # Partitioning
            x_dist = (window[1]-window[0])/(grid_dim[0])
            y_dist = (window[3]-window[2])/(grid_dim[1])
            z_dist = (window[5]-window[4])/(grid_dim[2])
        
            x_points = np.linspace(window[0]-x_dist/2, window[1]+x_dist/2, grid_dim[0]+2)
            y_points = np.linspace(window[2]-y_dist/2, window[3]+y_dist/2, grid_dim[1]+2)
            z_points = np.linspace(window[4]-z_dist/2, window[5]+z_dist/2, grid_dim[2]+2)
        
            C1, C2, C3 = np.meshgrid(x_points, y_points, z_points, indexing="ij")
            partition_flat = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)
    
            rho_eval = [rho(partition_flat) for rho in rho_interp]
            Sigma_eval = [Sigma(partition_flat) for Sigma in Sigma_interp]
            PEB_eval = [np.sqrt(np.trace(Sigma, axis1=1, axis2=2)) for Sigma in Sigma_eval]

            plt.style.use("ggplot")
            plt.rcParams["axes.grid"] = False
            for k in range(len(PEB_eval)):
                spec = plt.pcolormesh(x_points, y_points, PEB_eval[k].reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2))[:, :, 5].T,
                                      cmap="viridis_r", shading="auto", norm=LogNorm(vmin=np.min(PEB_eval[k]), vmax=np.max(PEB_eval[k])))
                cb = plt.colorbar(spec)
                cb.set_label(label="Position Error Bound")
                plt.show()

            for k in range(len(rho_eval)):
                spec = plt.pcolormesh(x_points, y_points, rho_eval[k].reshape((grid_dim[0]+2, grid_dim[1]+2, grid_dim[2]+2))[:, :, 5].T,
                                      cmap="viridis", shading="auto", vmin=0, vmax=1)
                cb = plt.colorbar(spec)
                cb.set_label(label="Detection Probability")
                plt.show()

        return rho_interp, Sigma_interp

    def compute_cramer_rao(self, p, s_U_k, f_tilde, omega_tilde, W_mat, signal_type):
        """
        Parameters
        ----------
        p : ndarray, size=(L, d)
            Target positions.
        s_U_k : ndarray, size=(1, 1, 2d)
            The user state.
        f_tilde : ndarray, size=(T/2, K, E, N_U)
            Precoder for times t_tilde = 1,\dots,T/2.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile for times t_tilde = 1,\dots,T/2
        W_mat : ndarray, size=(K, E, N_U, N_U)
            Combiner matrix.
        signal_type : str
            The type of signal. Options: 'nonRIS', 'directional', 'orthogonal'.

        Returns
        -------
        CRLB : ndarray, size=(L, d, d)
            The Cramer-Rao lower bound.
        """
        pars = self.initialise_channel_params(p, s_U_k)
        if signal_type == "nonRIS":
            FIM_nonRIS = self.Fisher_information_matrix_nonRIS_simplified(pars, f_tilde)
            CRLB = self.FIM_processing(FIM_nonRIS, pars, p, s_U_k[:, :, :3], s_U_k[:, :, 3:]) # size=(K, 1, L, 3, 3)
        if signal_type == "directional":
            FIM_directional = self.Fisher_information_matrix_directional_simplified(pars, omega_tilde, W_mat)
            CRLB = self.FIM_processing(FIM_directional, pars, p, s_U_k[:, :, :3], s_U_k[:, :, 3:])
        if signal_type == "orthogonal":
            FIM_orthogonal = self.Fisher_information_matrix_orthogonal_simplified(pars, f_tilde, omega_tilde, W_mat)
            CRLB = self.FIM_processing(FIM_orthogonal, pars, p, s_U_k[:, :, :3], s_U_k[:, :, 3:])
        return CRLB[0, 0]

    def compute_detection_probability(self, pars, f_tilde, omega_tilde, W_mat):
        """
        Parameters
        ----------
        pars : dict
            Dictionary of channel parameters.
        f_tilde : ndarray, size=(T/2, K, E, N_U)
            Precoder for times t_tilde = 1,\dots,T/2.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile for times t_tilde = 1,\dots,T/2
        W_mat : ndarray, size=(K, E, N_U, N_U)
            Combiner matrix.

        Returns
        -------
        detection_probability : ndarray, size=(Q, K, 1, L)
            Detection probabilities.
        """
        K, _, L = pars["alphal"].shape
        gamma = -2*np.log(self.p_FA)

        # Do preliminary computations
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)
        if "directional" or "orthogonal" in self.signal_types:
            aR_phi0 = self.UPA_array_response_vector(pars["phi0"], self.N_R, self.d_R) # size=(K, E, N_R)
            aR_phil = self.UPA_array_response_vector(pars["phil"], self.N_R, self.d_R) # size=(L, N_R)
            aR_phi0_prod = omega_tilde[:, None, None, :]*aR_phi0[None, :, :, :]

        # Compute detection probability
        detection_probability = np.zeros((0, K, 1, L))
        for signal_type in self.signal_types:
            if signal_type == "nonRIS":
                lambda_N = self.lambda_N(pars, f_tilde, aU_thetal)
                p_D = np.zeros((K, 1, L))
                for k in range(K):
                    for l in range(L):
                        p_D[k, 0, l] = 1 - ncx2.cdf(gamma, 2, lambda_N[k, 0, l])
            if signal_type == "directional":
                nu_tilde_D = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod[:self.T1, :, :, :]) # size=(T, K, E, L)
                lambda_D = self.lambda_D(pars, W_mat, aR_phi0, aR_phil, aU_thetal, nu_tilde_D)
                p_D = np.zeros((K, 1, L))
                for k in range(K):
                    for l in range(L):
                        p_D[k, 0, l] = 1 - ncx2.cdf(gamma, 2, lambda_D[k, 0, l])
            if signal_type == "orthogonal":
                nu_tilde_O = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod[self.T1:, :, :, :]) # size=(T, K, E, L)
                lambda_O = self.lambda_O(pars, f_tilde[self.T1:, :, :, :], nu_tilde_O, aU_thetal)
                p_D = np.zeros((K, 1, L))
                for k in range(K):
                    for l in range(L):
                        p_D[k, 0, l] = 1 - ncx2.cdf(gamma, 2, lambda_O[k, 0, l])
            detection_probability = np.concatenate((detection_probability, np.expand_dims(p_D, axis=0)), axis=0)
        return detection_probability

    def get_extent_measurements(self, pars, Tilde_Phi, sU, f_tilde, omega_tilde, W_mat):
        """
        Parameters
        ----------
        pars : dict
            Dictionary of channel parameters.
        Tilde_Phi : ndarray, size=(L, d+extent_dim)
            Target states.
        sU : ndarray, size=(K, 1, 2d)
            The user states.
        f_tilde : ndarray, size=(T/2, K, 1, N_U)
            Precoder for times t_tilde = 1,\dots,T/2.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile for times t_tilde = 1,\dots,T/2
        W_mat : ndarray, size=(K, 1, N_U, N_U)
            Combiner matrix.

        Returns
        -------
        points : ndarray, size=(Q, K, L+1, M_max, d)
            The observed measurements.
        covariance_marks : ndarray, size=(Q, K, L+1, M_max, d, d)
            The observed measurement marks.
        """
        Phi, Phi_E = Tilde_Phi[:, :self.d], Tilde_Phi[:, self.d]
        K, _, L = pars["alphal"].shape

        # Compute detection probability
        detection_probability = self.compute_detection_probability(pars, f_tilde, omega_tilde, W_mat)

        # Compute measurement rates and simulate measurement counts
        sensor_resolution = np.zeros((K, L))
        for l in range(L):
            sensor_resolution[:, l] = self.sensor_resolution_function(Phi[l], sU[:, 0, :])

        object_size = np.zeros(L)
        object_spread = np.zeros((L, self.d, self.d))
        if self.MCMC_hparams["extent_prior"] == "1d_uniform":
            for l in range(L):
                E_mat = Phi_E[l] * np.eye(self.d)
                object_size[l] = np.linalg.det(E_mat)
                object_spread[l] = E_mat @ E_mat.T
        elif self.MCMC_hparams["extent_prior"] == "inverse-wishart" or self.MCMC_hparams["extent_prior"] == "6d_uniform":
            for l in range(L):
                E_mat = np.zeros((self.d, self.d))
                E_mat[self.tril_idx] = Phi_E[l]
                object_size[l] = np.linalg.det(E_mat)
                object_spread[l] = E_mat @ E_mat.T

        measurement_rate = detection_probability * np.maximum(np.ones(detection_probability.shape), object_size[None, None, None, :] * sensor_resolution[None, :, None, :])
        measurement_counts = np.zeros((self.Q, K, 1, L), dtype=np.int16)
        for q in range(self.Q):
            for k in  range(K):
                for l in range(L):
                    measurement_counts[q, k, 0, l] = np.random.poisson(measurement_rate[q, k, 0, l])
        if self.verbose is True:
            print("Measurement counts:", measurement_counts[:, :, 0, :])

        # Simulate clutter measurement counts
        clutter = [[PoissonPointProcess(self.window, self.lambda_c).points for k in range(K)] for q in range(self.Q)]
        M_clutter = [[cl.shape[0] for cl in clut] for clut in clutter]

        # Find the maximum number of measurements
        max_measurements = max(np.max(measurement_counts), max([max(M_clut) for M_clut in M_clutter]))

        # Simulate measurements
        source_points = np.zeros((self.Q, K, 1, L, max_measurements, self.d))
        source_points[:] = None
        location_covariance = np.zeros((self.Q, K, 1, L, max_measurements, self.d, self.d))
        location_covariance[:] = None
        measurements = np.zeros((self.Q, K, 1, L, max_measurements, self.d))
        measurements[:] = None
        for q in range(self.Q):
            for k in  range(K):
                f_temp = np.expand_dims(f_tilde[:, k, :, :], axis=1)
                W_temp = np.expand_dims(W_mat[k], axis=0)
                sU_k = np.expand_dims(sU[0], axis=0)
                for l in range(L):
                    M = measurement_counts[q, k, 0, l]

                    if M > 0:
                        # Simulate source points
                        source_points[q, k, 0, l, :M, :] = np.random.multivariate_normal(Phi[l, :], object_spread[l], size=M)
                        p = source_points[q, k, 0, l, :M, :]
    
                        # Compute Cramer-Rao lower bound
                        location_covariance[q, k, 0, l, :M, :, :] = self.compute_cramer_rao(p, sU_k, f_temp, omega_tilde, W_temp, self.signal_types[q])
    
                        # Simulate measurements
                        for m in range(M):
                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')
                                try:
                                    measurements[q, k, 0, l, m, :] = np.random.multivariate_normal(source_points[q, k, 0, l, m, :], location_covariance[q, k, 0, l, m, :, :])
                                except RuntimeWarning:
                                    # Sometimes the CRLB is non-invertible
                                    measurements[q, k, 0, l, m, :] = None
                                    location_covariance[q, k, 0, l, :, :, :] = None

        # Simulate clutter
        clutter_points = np.zeros((self.Q, K, 1, 1, max_measurements, 3))
        clutter_points[:] = None
        for q in range(self.Q):
            for k in range(K):
                clutter_points[q, k, 0, 0, :M_clutter[q][k], :] = clutter[q][k]
        points = np.concatenate((measurements, clutter_points), axis=3)

        # Compute marks
        if self.oracle_hparams["true_marks"] is True:
            covariance_marks = np.zeros((self.Q, K, 1, L+1, max_measurements, 3, 3))
            covariance_marks[:] = None
            covariance_marks[:, :, :, :L, :, :, :] = location_covariance
            for q in range(self.Q):
                for k in range(K):
                    f_temp = np.expand_dims(f_tilde[:, k, :, :], axis=1)
                    W_temp = np.expand_dims(W_mat[k], axis=0)
                    sU_k = np.expand_dims(sU[k], axis=0)

                    M = M_clutter[q][k]
                    if M > 0:
                        p = points[q, k, 0, -1, :M, :]
                        # Compute Cramer-Rao lower bound
                        covariance_marks[q, k, 0, -1, :M, :, :] = self.compute_cramer_rao(p, sU_k, f_temp, omega_tilde, W_temp, self.signal_types[q])
        elif self.oracle_hparams["true_marks"] is False:
            covariance_marks = np.zeros((self.Q, K, 1, L+1, max_measurements, 3, 3))
            covariance_marks[:] = None
            for q in range(self.Q):
                for k in range(K):
                    f_temp = np.expand_dims(f_tilde[:, k, :, :], axis=1)
                    W_temp = np.expand_dims(W_mat[k], axis=0)
                    sU_k = np.expand_dims(sU[k], axis=0)
                    for l in range(L+1):
                        if l < L:
                            M = measurement_counts[q, k, 0, l]
                        elif l == L:
                            M = M_clutter[q][k]

                        if M > 0:
                            p = points[q, k, 0, l, :M, :]
                            # Compute Cramer-Rao lower bound
                            covariance_marks[q, k, 0, l, :M, :, :] = self.compute_cramer_rao(p, sU_k, f_temp, omega_tilde, W_temp, self.signal_types[q])
        elif self.oracle_hparams["true_marks"] is None:
            covariance_marks = np.zeros((self.Q, K, 1, L+1, max_measurements, 3, 3))
            covariance_marks[:] = None
            for q in range(self.Q):
                for k in range(K):
                    for l in range(L+1):
                        if l < L:
                            M = measurement_counts[q, k, 0, l]
                        elif l == L:
                            M = M_clutter[q][k]
                        for m in range(M):
                            covariance_marks[q, k, 0, l, m, :, :] = np.eye(3)                           
        return points[:, :, 0, :, :, :], covariance_marks[:, :, 0, :, :, :, :]

    def plot_and_print(self, log_posterior_list, acceptance_history, Psi_tot, points_tot,
                       Phi, Phi_DBSCAN, Phi_oracle, Phi_est, Phi_est_v2, ti, 
                       Tilde_Phi, Tilde_Phi_est, Tilde_Phi_est_v2, Tilde_Phi_DBSCAN, Tilde_Phi_oracle):
        """
        Plot and print recursion results.
        """
        if self.verbose is True:
            plt.rcParams["axes.grid"] = True

            ### Trace plot ###
            plt.plot(log_posterior_list, color="tab:green")
            plt.ylabel("log posterior")
            # plt.savefig("Results/MCMC/log_posterior.png", dpi=500, bbox_inches="tight")
            plt.show()

            ### Acceptance probability history ###
            data = [acceptance_history["move"], acceptance_history["extent_move"], acceptance_history["birth"], acceptance_history["death"]]
            n_bins = 10
            fig, ax0 = plt.subplots()
            colors = ['tab:purple', 'tab:olive', 'tab:orange', 'tab:cyan']
            ax0.hist(data, n_bins, histtype='bar', color=colors, label=["move", "extent-move", "birth", "death"])
            ax0.legend(prop={'size': 10})
            # plt.savefig("Results/MCMC/acceptance_probability.png", dpi=500, bbox_inches="tight")
            plt.show()

            ### Sensing plot ###
            plt.plot(Psi_tot[0][:, 0], Psi_tot[0][:, 1],  color="purple", marker="x", markersize=2.5, linestyle="", alpha=0.75, label="Measurements")
            for k in range(1, len(points_tot)): # Measurements
                plt.plot(Psi_tot[k][:, 0], Psi_tot[k][:, 1],  color="purple", marker="x", markersize=2.5, linestyle="", alpha=0.75)
            plt.plot(Phi[0, 0], Phi[0, 1], color="tab:red", marker="o", markersize=4, linestyle="", label="Targets")
            for l in range(1, Phi.shape[0]): # Target
                plt.plot(Phi[l, 0], Phi[l, 1], color="tab:red", marker="o", markersize=4, linestyle="")
            try:
                plt.plot(Phi_DBSCAN[0, 0], Phi_DBSCAN[0, 1],  color="tab:blue", marker="o", markersize=4, linestyle="", label="DBSCAN")
                for l in range(1, Phi_DBSCAN.shape[0]): # DBSCAN
                    plt.plot(Phi_DBSCAN[l, 0], Phi_DBSCAN[l, 1],  color="tab:blue", marker="o", markersize=4, linestyle="")
            except IndexError:
                pass
            plt.plot(Phi_oracle[0, 0], Phi_oracle[0, 1],  color="tab:orange", marker="o", markersize=4, linestyle="", label="Oracle")
            for l in range(1, Phi_oracle.shape[0]): # Oracle
                plt.plot(Phi_oracle[l, 0], Phi_oracle[l, 1],  color="tab:orange", marker="o", markersize=4, linestyle="")
            plt.plot(Phi_est[0, 0], Phi_est[0, 1],  color="tab:green", marker="o", markersize=4, linestyle="", label="MCMC")
            for l in range(1, Phi_est.shape[0]): # MCMC
                plt.plot(Phi_est[l, 0], Phi_est[l, 1],  color="tab:green", marker="o", markersize=4, linestyle="")
            plt.plot(Phi_est_v2[0, 0], Phi_est_v2[0, 1],  color="tab:olive", marker="o", markersize=4, linestyle="", label="MCMC_v2")
            for l in range(1, Phi_est_v2.shape[0]): # MCMC
                plt.plot(Phi_est_v2[l, 0], Phi_est_v2[l, 1],  color="tab:olive", marker="o", markersize=4, linestyle="")
            plt.xlim(self.window[0]-2, self.window[1]+2)
            plt.ylim(self.window[2]-2, self.window[3]+2)
            plt.legend()
            plt.show()

        print("")
        print("True   Cardinality :", Phi.shape[0])
        print("DBSCAN Cardinality :", Phi_DBSCAN.shape[0])
        print("MCMC   Cardinality :", Phi_est.shape[0])
        print("Oracle Cardinality :", Phi_oracle.shape[0])
        print("")
        print("DBSCAN  GOSPA :", GOSPA(Phi, Phi_DBSCAN, c=10))
        print("MCMC    GOSPA :", GOSPA(Phi, Phi_est, c=10))
        print("MCMCv2  GOSPA :", GOSPA(Phi, Phi_est_v2, c=10))
        print("Oracle  GOSPA :", GOSPA(Phi, Phi_oracle, c=10))
        print("")
        print("DBSCAN  OSPA :", OSPA(Phi, Phi_DBSCAN, c=10))
        print("MCMC    OSPA :", OSPA(Phi, Phi_est, c=10))
        print("MCMC v2 OSPA :", OSPA(Phi, Phi_est_v2, c=10))
        print("Oracle  OSPA :", OSPA(Phi, Phi_oracle, c=10))
        print("")
        print("DBSCAN  Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_DBSCAN, c=10))
        print("MCMC    Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_est, c=10))
        print("MCMC v2 Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_est_v2, c=10))
        print("Oracle  Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_oracle, c=10))
        print("")

    def __call__(self, Tilde_Phi):
        """
        Run filter recursion.
        """
        assert (self.extent is True) and (self.use_directional_RIS is False) and (self.oracle_hparams["true_marks"] is True), \
            "The method is not prepared for all possible settings yet!"

        pred_Map = dict()
        pred_Map["prior"] = None
        Phi, Phi_E = Tilde_Phi[:, :3], Tilde_Phi[:, 3:]


        pU = self.simulate_UE_positions(self.K, self.E, self.heading, self.speed,
                                         self.epoch_time, None) # size=(K, E, 3)
        sU_all = np.concatenate((pU, np.ones(pU.shape)*np.array(self.oU)[None, None, :]), axis=-1)

        L_true, d = Phi.shape
        self.L_true, self.d = L_true, d

        res_dict = dict()
        res_dict["Phi"] = Phi
        res_dict["Phi_E"] = Phi_E

        points_tot = list()
        covariance_tot = list()
        Psi_da_tot = list()
        Psi_tot = list()

        # if self.sensing_alg == "MCMC":
        rho_interp_all = list()
        Sigma_interp_all = list()

        for ti in range(self.E):
            print(f"\nTime epoch {ti+1}:\n-------------")
            # Initialize states and channel parameters
            sU = np.expand_dims(sU_all[:, ti, :], axis=1)
            pars = self.initialise_channel_params(Phi, sU)

            # Design precoder, combiner, RIS phase profile
            f_tilde, omega_tilde, W_mat = self.construct_RIS_phase_precoder_combiner(pars["theta0"], sU[:, :, :3], pred_Map)

            print("Simulate measurement process!")
            points, covariance_marks = self.get_extent_measurements(pars, Tilde_Phi, sU, f_tilde, omega_tilde, W_mat)
            points_tot.append(points)
            covariance_tot.append(covariance_marks)

            Psi_da, Psi_c = list(), list()
            for q in range(self.Q):
                for k in range(self.K):
                    sublist = list()
                    for l in range(self.L_true):
                        inner_points = points[q, k, l]
                        sublist.append(inner_points[~np.isnan(inner_points).any(axis=1)])
                    Psi_da.append(sublist)
                    inner_points = points[q, k, -1]
                    Psi_c.append(inner_points[~np.isnan(inner_points).any(axis=1)])
            Psi = [np.concatenate((Psi_c[k], np.concatenate(Psi_da[k])), axis=0) for k in range(self.Q*self.K)]
            Psi_da_tot += Psi_da
            Psi_tot += Psi

            # Create detection probability and CRLB mapping interpolators
            if "DBSCAN" in self.sensing_alg or "MCMC" in self.sensing_alg:
                print("Construct interpolations!")
                rho_interp, Sigma_interp = self.construct_interpolations(self.window, self.grid_dim, sU,
                                                                         f_tilde, omega_tilde, W_mat, points)
                rho_interp_all += rho_interp
                Sigma_interp_all += Sigma_interp
                if self.verbose is True:
                    self.plot_sensor_resolution(sU_all[:, ti, :])

            # Run oracle for comparison
            if "oracle" in self.sensing_alg:
                self.oracle_call_kwargs["Phi_true"] = Phi
                self.oracle_call_kwargs["Phi_E_true"] = Phi_E
                M_k = [p.shape[3] for p in points_tot]
                M_max = max(M_k)
                points_oracle_in = np.zeros((ti+1, self.K, L_true, M_max, d))
                points_oracle_in[:] = None
                covariance_oracle_in = np.zeros((ti+1, self.K, L_true, M_max, d, d))
                covariance_oracle_in[:] = None
                for i in range(ti+1):
                    points_oracle_in[i, :, :, :M_k[i], :] = points_tot[i][:, :, :-1, :M_k[i], :].reshape((-1, L_true, M_k[i], d))
                    covariance_oracle_in[i, :, :, :M_k[i], :, :] = covariance_tot[i][:, :, :-1, :M_k[i], :, :].reshape((-1, L_true, M_k[i], d, d))
                points_oracle_in = points_oracle_in.reshape((-1, L_true, M_max, d))
                covariance_oracle_in = covariance_oracle_in.reshape((-1, L_true, M_max, d, d))
                for l in range(L_true):
                    if self.MCMC_hparams["extent_prior"] == "1d_uniform":
                        Sigma_E = Phi_E[l]**2 * np.eye(d)
                        covariance_oracle_in[:, l, :, :, :] += Sigma_E[None, None, :, :] * np.ones((self.K*(ti+1), M_max))[:, :, None, None]
                    elif self.MCMC_hparams["extent_prior"] == "inverse-wishart" or self.MCMC_hparams["extent_prior"] == "6d_uniform":
                        E_mat = np.zeros((d, d))
                        E_mat[self.tril_idx] = Phi_E[l]
                        Sigma_E = E_mat @ E_mat.T
                        covariance_oracle_in[:, l, :, :, :] += Sigma_E[None, None, :, :] * np.ones((self.K*(ti+1), M_max))[:, :, None, None]
                Tilde_Phi_oracle, _, _ = self.oracle_alg(points_oracle_in, covariance_oracle_in, **self.oracle_call_kwargs)
                Phi_oracle, Phi_E_oracle = Tilde_Phi_oracle[:, :d], Tilde_Phi_oracle[:, d:]

            # Run DBSCAN for comparison
            if "DBSCAN" in self.sensing_alg:
                self.DBSCAN_call_kwargs["Phi_true"] = Phi
                self.DBSCAN_call_kwargs["Phi_E_true"] = Phi_E
                DBSCAN_alg = sensing_DBSCAN(rho_interp_all, Sigma_interp_all, self.sensor_resolution_function,
                                            sU_all[:, :ti+1, :].reshape((self.K*(ti+1), 6), order="F"),
                                            self.window, self.p_FA, True, **self.DBSCAN_init_kwargs)
                Tilde_Phi_DBSCAN, _, _, DBSCAN_pars_est, Tilde_Phi_DBSCAN_opt, Tilde_Phi_DBSCAN_extent_opt \
                    = DBSCAN_alg(Psi_tot, self.DBSCAN_hparams["eps"], np.array(self.DBSCAN_hparams["min_samples"])*(ti+1), **self.DBSCAN_call_kwargs)
                Phi_DBSCAN, Phi_E_DBSCAN = Tilde_Phi_DBSCAN[:, :d], Tilde_Phi_DBSCAN[:, d:]
                Phi_DBSCAN_opt, Phi_E_DBSCAN_opt = Tilde_Phi_DBSCAN_opt[:, :d], Tilde_Phi_DBSCAN_opt[:, d:]
                Phi_DBSCAN_extent_opt, Phi_E_DBSCAN_extent_opt = Tilde_Phi_DBSCAN_extent_opt[:, :d], Tilde_Phi_DBSCAN_extent_opt[:, d:]

            # Run MCMC sensing algorithm
            if "MCMC" in self.sensing_alg:
                self.MCMC_init_kwargs["sU"] = sU_all[:, :ti+1, :].reshape((self.K*(ti+1), 6), order="F")
                mod = RJMCMC_DoublyNeymanScott(Psi_tot, self.window, self.MCMC_hparams["grid_dim"],
                                               rho_interp_all, Sigma_interp_all, self.MCMC_hparams["hard_core_radius"],
                                               **self.MCMC_init_kwargs)
                Phi_list, Extent_list, A_list, _, _, lambda_list, lambda_c_list, log_posterior_list, init_ = \
                    mod(self.MCMC_hparams["MCMC_runs"], DBSCAN_pars_est[1], DBSCAN_pars_est[0], pred_Map, **self.MCMC_call_kwargs)

                acceptance_history = mod.acceptance_history
                Phi_est = np.mean(np.array(Phi_list), axis=0)
                Extent_est = np.mean(np.array(Extent_list), axis=0)
                A_est = list()
                for k in range(self.K):
                    A_k = np.zeros((len(A_list), A_list[0][k].shape[0]))
                    for i in range(len(A_list)):
                        A_k[i] = A_list[i][k]
                    A_k_est = np.median(A_k, axis=0)
                    A_est.append(A_k_est)
                lambda_est = np.mean(lambda_list)
                lambda_c_est = np.mean(lambda_c_list)

                Phi_est_v2 = Phi_list[-1]
                Extent_est_v2 = Extent_list[-1]
                A_est_v2 = list()
                for k in range(self.K):
                    A_k_est_v2 = A_list[-1][k]
                    A_est_v2.append(A_k_est_v2)
                lambda_est_v2 = lambda_list[-1]
                lambda_c_est_v2 = lambda_c_list[-1]

                # Target estimates
                if self.MCMC_hparams["extent_prior"] == "1d_uniform":
                    Tilde_Phi_est = np.concatenate((Phi_est, np.expand_dims(Extent_est, axis=-1)), axis=-1)
                    Tilde_Phi_est_v2 = np.concatenate((Phi_est_v2, np.expand_dims(Extent_est_v2, axis=-1)), axis=-1)
                elif self.MCMC_hparams["extent_prior"] == "inverse-wishart" or self.MCMC_hparams["extent_prior"] == "6d_uniform":
                    Tilde_Phi_est = np.concatenate((Phi_est, Extent_est), axis=-1)
                    Tilde_Phi_est_v2 = np.concatenate((Phi_est_v2, Extent_est_v2), axis=-1)

            # Verbose
            if "oracle" in self.sensing_alg and "MCMC" in self.sensing_alg and "DBSCAN" in self.sensing_alg:
                self.plot_and_print(log_posterior_list, acceptance_history, Psi_tot, points_tot, Phi, Phi_DBSCAN, Phi_oracle, Phi_est, Phi_est_v2, ti,
                                    Tilde_Phi, Tilde_Phi_est, Tilde_Phi_est_v2, Tilde_Phi_DBSCAN, Tilde_Phi_oracle)

            # Update prediction map
            if self.use_directional_RIS is True:
                pred_Map["prior"] = True
            else:
                pred_Map["prior"] = False

            if "MCMC" in self.sensing_alg:
                pred_Map["Phi_0"] = Phi_est
                pred_Map["Phi_E_0"] = Extent_est
                pred_Map["A_0"] = A_est
                pred_Map["lambda_0"] = lambda_est
                pred_Map["lambda_c_0"] = lambda_c_est

            ### Save estimation results ###
            res_dict[f"{ti}"] = dict()
            if "MCMC" in self.sensing_alg:
                res_dict[f"{ti}"]["Phi_est"] = Phi_est
                res_dict[f"{ti}"]["Phi_est_v2"] = Phi_est_v2
            if "DBSCAN" in self.sensing_alg:
                res_dict[f"{ti}"]["Phi_DBSCAN"] = Phi_DBSCAN
                res_dict[f"{ti}"]["Phi_DBSCAN_opt"] = Phi_DBSCAN_opt
                res_dict[f"{ti}"]["Phi_DBSCAN_extent_opt"] = Phi_DBSCAN_extent_opt
            if "oracle" in self.sensing_alg:
                res_dict[f"{ti}"]["Phi_oracle"] = Phi_oracle
            if self.extent is True:
                if "MCMC" in self.sensing_alg:
                    res_dict[f"{ti}"]["Phi_E_est"] = Extent_est
                    res_dict[f"{ti}"]["Phi_E_est_v2"] = Extent_est_v2
                if "DBSCAN" in self.sensing_alg:
                    res_dict[f"{ti}"]["Phi_E_DBSCAN"] = Phi_E_DBSCAN
                    res_dict[f"{ti}"]["Phi_E_DBSCAN_opt"] = Phi_E_DBSCAN_opt
                    res_dict[f"{ti}"]["Phi_E_DBSCAN_extent_opt"] = Phi_E_DBSCAN_extent_opt
                if "oracle" in self.sensing_alg:
                    res_dict[f"{ti}"]["Phi_E_oracle"] = Phi_E_oracle

            ### Save algorithm interpretability results ###
            if "DBSCAN" in self.sensing_alg:
                res_dict[f"{ti}"]["DBSCAN_parameters"] = DBSCAN_pars_est
            if "MCMC" in self.sensing_alg:
                res_dict[f"{ti}"]["log_posterior"] = log_posterior_list
                res_dict[f"{ti}"]["acceptance"] = acceptance_history
                res_dict[f"{ti}"]["lambda"] = lambda_est
                res_dict[f"{ti}"]["lambda_c"] = lambda_c_est

            ### Save observed data process ###
            # res_dict[f"{ti}"]["rho_interp"] = rho_interp
            # res_dict[f"{ti}"]["Sigma_interp"] = Sigma_interp
            res_dict[f"{ti}"]["Psi_da"] = Psi_da
            res_dict[f"{ti}"]["Psi_c"] = Psi_c            

            with open("sensing_filter_temp.pickle", "wb") as file:
                pickle.dump(res_dict, file, pickle.HIGHEST_PROTOCOL)
        return res_dict


def simulate_driving_process(lambda_SP, hard_core_radius, window, extent_prior,
                             sigma_extent_min, sigma_extent_max, extent_IW_df, extent_IW_scale,
                             cor_extent_min, cor_extent_max):
    """
    Parameters
    ----------
    lambda_SP : float
        The parent process intensity.
    hard_core_radius : float
        The hard core radius.
    window : list, len=(6,)
        The rectangular boundaries of the point process domain.

    Returns
    -------
    Tilde_Phi : ndarray, size=(L, d+e)
        The driving process consisting of L parent points with a spatial position
        and an object extent.
    """
    d = 3
    Cox = CoxProcess(window)
    Phi = Cox.simulate_driving_process(lambda_SP, type_="MaternTypeII", HardCoreRadius=hard_core_radius)
    L = Phi.shape[0]
    if extent_prior == "1d_uniform":
        Phi_E = np.random.uniform(sigma_extent_min, sigma_extent_max, L)
        Tilde_Phi = np.concatenate((Phi, np.expand_dims(Phi_E, axis=-1)), axis=1)
    elif extent_prior == "inverse-wishart":
        tril_idx = np.tril_indices(d, k=0)
        Phi_E = np.zeros((L, 6))
        for l in range(L):
            det = 100
            while det > 3.375 or det < 1:
                Sigma_E = invwishart.rvs(extent_IW_df, extent_IW_scale) # size=(partition_extent, 3, 3)
                E_mat = cholesky(Sigma_E, lower=True)
                det = np.linalg.det(E_mat)
            Phi_E[l] = E_mat[tril_idx]
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
    return Tilde_Phi

def main():
    np.random.seed(8)
    plt.style.use("ggplot")
    verbose = True

    toml_in = toml.load("sensing_filter.toml")
    toml_settings = toml_in["channel_settings"]
    toml_scenario = toml_in["scenario_settings"]
    toml_booleans = toml_in["boolean_specifications"]
    toml_experiment = toml_in["experiment_settings"]
    toml_oracle = toml_in["oracle_hparams"]
    toml_dbscan = toml_in["DBSCAN_hparams"]
    toml_MCMC = toml_in["MCMC_hparams"]

    # =============================================================================
    # Prepare input data structure
    # =============================================================================
    experiment_kwargs = {"f_c": toml_settings["f_c"],
                         "p_tx_hz_dbm": toml_settings["p_tx_hz_dbm"],
                         "p_noise_hz_dbm": toml_settings["p_noise_hz_dbm"],
                         "N_U": toml_settings["N_U"],
                         "N_R": toml_settings["N_R"],
                         "p_R": toml_settings["p_R"],
                         "n_R": toml_settings["n_R"],
                         "delta_f": toml_settings["delta_f"],
                         "N": toml_settings["N"],
                         "c": toml_settings["c"],
                         "T": toml_settings["T"],
                         "T1": toml_settings["T1"],
                         "window": toml_scenario["window"],
                         "grid_dim": toml_scenario["grid_dim"],
                         "speed": toml_scenario["speed"],
                         "heading" : None,
                         "epoch_time": toml_scenario["epoch_time"],
                         "E": toml_scenario["E"],
                         "K": toml_scenario["K"],
                         "lambda_c_Poi": toml_scenario["lambda_c_Poi"],
                         "lambda_Poi": toml_scenario["lambda_Poi"],
                         "use_directional_RIS": toml_booleans["use_directional_RIS"],
                         "hard_core_radius": toml_scenario["R"],
                         "extent": toml_booleans["object_extent"],
                         "real_resolution": toml_booleans["real_resolution"],
                         "p_FA": toml_scenario["p_FA"]}

    sigma_extent_min = toml_MCMC["sigma_extent_min"]
    sigma_extent_max = toml_MCMC["sigma_extent_max"]
    cor_extent_min = toml_MCMC["cor_extent_min"]
    cor_extent_max = toml_MCMC["cor_extent_max"]
    extent_IW_df = toml_MCMC["extent_IW_df"]
    extent_IW_scale = toml_MCMC["extent_IW_scale"]*(extent_IW_df-3-1)*np.eye(3)
    toml_MCMC["extent_IW_scale"] = extent_IW_scale
    extent_prior = toml_MCMC["extent_prior"]

    window = toml_scenario["window"]
    area = np.prod([window[2*i+1]-window[2*i] for i in range(3)])
    Tilde_Phi = simulate_driving_process(toml_scenario["lambda_Poi"]/area, toml_scenario["R"], window, extent_prior,
                                         sigma_extent_min, sigma_extent_max, extent_IW_df, extent_IW_scale, cor_extent_min, cor_extent_max)

    # =============================================================================
    # Run algorithm
    # =============================================================================
    resource_division = toml_experiment["data_types_list"][0]
    signal_types_in = toml_experiment["signal_types_list"]
    experiment_kwargs.update({"resource_division": resource_division,
                              "signal_types": signal_types_in})

    # =============================================================================
    # Sensing
    # =============================================================================
    alg = main_filter(toml_experiment["sensing_alg"], toml_oracle, toml_dbscan, toml_MCMC, verbose, **experiment_kwargs)
    oracle_res_dict = alg(Tilde_Phi)
    # alg.run_profile()

if __name__ == "__main__":
    main()


