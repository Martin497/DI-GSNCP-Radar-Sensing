# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:27:14 2023

@author: Martin Voigt Vejling
Email: mvv@math.aau.dk

Track changes:
    v1.0 - Initial implementation of the RJMCMC_DoublyNeymanScott() class.
           This algorithm implementation contains three strategies for
           computing acceptance probabilities. The model assumes naively
           a known target extent. (31/07/2023)
    v1.1 - Introduced a sensor resolution parameter. 
           More intelligent initialization: use covariance for DBSCAN sample
           weights and for estimating cluster centers. (14/08/2023)
    v1.2 - Made the sensor resolution into a function on the spatial domain.
           Included an Inverse-Wishart extent model option.
           Now includes an extent-move step, thereby, transitioning
           the spatial parameters and the object extent parameters
           independently. Included early stopping with a patience
           parameter. Included a second birth density that only proposes
           birth on points labelled as clutter. (23/08/2023)
    v1.3 - Move some methods to an inheritance class. Extent prior bugfix.
           Include init_ input to __call__ method. (25/08/2023)
    v1.4 - Optimize code: thinning of points for birth,
           simulate clutter/non-clutter option, only compute eta_partition_sum
           when doing birth/death.
           Functionality: Include birth shift parameter, introduce
           initialization from a birth-step, include 6d_uniform extent model.
           Bugfix: no death when one target, correct prior handling for extent
           model. (07/09/2023)
"""


import numpy as np
import matplotlib.pyplot as plt

from random import choices
from scipy.stats import invwishart
from scipy.linalg import cholesky
from sklearn.cluster import DBSCAN

from digsncp import DIGSNCP_base
from oracle import sensing_oracle
from clustering import sensing_DBSCAN
from toy_simulation import DI_GSNCP, setup_functions, sensor_res_fun
from SensingMetrics import GOSPA, OSPA, OSPA_extent

import line_profiler
profile = line_profiler.LineProfiler()


class RJMCMC_DoublyNeymanScott(DIGSNCP_base):
    """
    Reversible Markov chain Monte Carlo algorithm for cluster point process.

    This class represents the proposed Bayesian sensing/tracking framework
    using Markov chain Monte Carlo methods, i.e., Birth-Death-Move MCMC.

    In each iteration we can propose a state based on 3 transitions:
        Birth: A cluster center is birthed according to density on a discrete
               partition space.
        Death: A cluster center is killed off from a discrete density.
        Move: A cluster center is moved a small step defined as sampling
              from a Gaussian density centred at the previous state and
              moved according to a specified move covariance.

    Accept the transition with some acceptance probability computed according
    to the specified algorithm. Options are "Metropolis-Hastings" and
    "Stochastic Marginal Maximum A Posteriori".

    The class can handle both point targets and extended targets according to
    the covariance matrix parameterization of extended targets.
    """
    @profile
    def __init__(self, Psi, window, grid_dim, rho, Sigma, R, **kwargs):
        """
        Initialize class settings.

        Inputs:
        -------
            Psi : list, len=K
                The k-th entry in the list is an ndarray of size=(M_k, d)
            window : list, len = 2*(d+1)
                The boundary of the assumed rectangular domain. In 3D space the
                list should be specified as
                    [x_min, x_max, y_min, y_max, z_min, z_max]
            grid_dim : list, len = d+1
                The number of grid points. In 3D space the
                list should be specified as
                    [nr_x, nr_y, nr_z]
            rho : list, len=K
                The k-th entry in the list is a function \rho_k on \R^d
                which defines the measurement rates.
            Sigma : list, len=K
                The k-th entry in the list is a function \Sigma_k on the
                space of positive definite matrices in \R^{d \times d}
                which defines the sensor covariances.
            R : float
                Hard core prior radius.

        Keyword arguments:
        ------------------
            p_m : float \in [0, 1]
                The probability of proposing a move step.
            vartheta : float > 0
                The move covariance scaling parameter.
            intensity_std : float > 0
                The cluster center intensity transition density variance.
            clutter_std : float > 0
                The clutter intensity transition density variance.
            method : str
                The chosen method for updating the prior intensities.
                Options: "pure", "modified".
            acceptance_type : str
                The chosen strategy for computing acceptance probabilities.
                Options:
                    "Metropolis-Hastings"
                    "Stochastic Marginal Maximum A Posteriori".
            object_extent : bool
                If True use extended object model. The default is False.
            sigma_extent_max : float
                The maximum value for the extent standard deviation for the
                '1d_uniform' extent prior model. The default is 0.
            sigma_extent_min : float
                The minimum value for the extent standard deviation for the
                '1d_uniform' extent prior model. The default is 0.
            extent_IW_df : float
                The degrees of freedom for the extent prior 'inverse-wishart'
                model. The default is 5.
            extent_IW_scale : ndarray, size=(d, d)
                The scale matrix for the extent prior 'inverse-wishart'
                model. The default is 3*(df-d-1) * np.eye(d).
            extent_move_std : float
                The object extent parameter move step
                standard deviation. The default is 0.5.
            extent_joint_update : bool
                If False, update the ground and mark processes independently
                in the move step.
            extent_prior : str
                The prior on the object extent. Options are '1d_uniform' and
                'inverse-wishart'. The default is '1d_uniform'.
            p_FA : float
                The probability of false alarm. This provides the minimum
                for the detection probability. The default is 1e-03.
            sensor_resolution_function : fun
                Function on R^d \times \mathcal{S} which defines the number
                of measurements per volume.
                The default is the equal to 1 function.
            sU : ndarray, size=(K, 2d):
                The sensor position and orientation. The default is None.
            birth_density_v1 : bool
                If True, use the first birth density proposal, otherwise
                use the second. The default is False.
            birth_shift : list
                If birth_density_v1 is False, then point are birthed around
                clutter points with uniform shifts in x, y, and z directions
                given by this list.
            birth_clutter_thin_var : int
                The posible birth points are thinned by sampling a Bernoulli
                process with retention probability p[k] for the k-th sensor
                state. The retention probability p[k] is computed as
                p[k] = 1/divmod(M_k, birth_clutter_thin_var)[0]. Hence, it
                is the expected number of measurements from each sensor which
                can be used to birth points. The default is 50.
            data_association_sim : bool
                If True, data association is simulated. If False, the data
                association is chosen as the one maximizing the conditional
                density. The False options is significantly faster
                due to inefficient random number sampling in python.
                The default is False.
            patience : int
                The patience parameter for early stopping. The default is 200.
            DBSCAN_init : bool
                If True, initialize with DBSCAN. The default is False.
            verbose : bool
                If True, print progress, else mute. The default is False.
        """
        super(RJMCMC_DoublyNeymanScott, self).__init__()            

        # Assign attributes from input
        self.Psi = Psi
        self.window = window
        self.grid_dim = grid_dim
        self.rho = rho
        self.Sigma = Sigma
        self.R = R

        # Assign additional attributes
        self.d = 3
        self.area = np.prod([self.window[2*i+1]-self.window[2*i] for i in range(self.d)]) # Area of observation window
        self.K = len(self.Psi)
        self.M_k = [self.Psi[k].shape[0] for k in range(self.K)]
        self.M = sum(self.M_k)
        print("Number of datapoints:", self.M)
        # self.rho_Psi = [self.rho[k](Psi[k]) for k in range(len(Psi))]
        self.Sigma_Psi = [self.Sigma[k](Psi[k]) for k in range(len(Psi))]

        self.tril_idx = np.tril_indices(self.d, k=0)
        self.low_tril_idx = np.tril_indices(self.d, k=-1)

        self.profile = profile

        # Keyword argument handling
        self.p_m = kwargs.get("p_m", 0.5)
        self.p_em = kwargs.get("p_em", 0.5)
        self.vartheta = kwargs.get("vartheta", 1)
        self.intensity_std = kwargs.get("intensity_std", 1e-03)
        self.clutter_std = kwargs.get("clutter_std", 1e-03)
        self.extent_move_std = kwargs.get("extent_move_std", 5e-01)
        self.method = kwargs.get("method", "pure")
        self.acceptance_type = kwargs.get("acceptance_type", "Metropolis-Hastings")
        self.object_extent = kwargs.get("object_extent", False)
        self.sigma_extent_max = kwargs.get("sigma_extent_max", 0)
        self.sigma_extent_min = kwargs.get("sigma_extent_min", 0)
        self.extent_IW_df = kwargs.get("extent_IW_df", 5)
        self.extent_IW_scale = kwargs.get("extent_IW_scale", 3*(self.extent_IW_df-self.d-1)*np.eye(self.d))
        self.cor_extent_max = kwargs.get("cor_extent_max", 0)
        self.cor_extent_min = kwargs.get("cor_extent_min", 0)
        self.partition_extent = kwargs.get("partition_extent", 1)
        self.extent_prior = kwargs.get("extent_prior", "1d_uniform")
        self.p_FA = kwargs.get("p_FA", 1e-03)
        self.sensor_resolution_function = kwargs.get("sensor_resolution_function", lambda x, y: np.ones(len(self.Psi)))
        self.sU = kwargs.get("sU", None)
        self.birth_density_v1 = kwargs.get("birth_density_v1", False)
        self.birth_shift = kwargs.get("birth_shift", [0.2, 0.2, 0.2])
        self.birth_clutter_thin_var = kwargs.get("birth_clutter_thin_var", 50)
        self.data_association_sim = kwargs.get("data_association_sim", False)
        self.patience = kwargs.get("patience", 200)
        self.DBSCAN_init = kwargs.get("DBSCAN_init", False)
        self.verbose = kwargs.get("verbose", False)

        # Check input is allowed
        assert len(window) == 6, "Class only compatible with 3D spatial data."
        assert (self.p_m >= 0) & (self.p_m <= 1), "p_m is a probability parameter."
        assert self.vartheta > 0, "The move covariance scaling parameter must be positive."
        assert self.intensity_std > 0, "Standard deviation must be positive."
        assert self.clutter_std > 0, "Standard deviation must be positive."
        assert self.extent_move_std > 0, "Standard deviation must be positive."
        if (self.object_extent is True) and (self.extent_model == "1d_uniform"):
            assert (self.sigma_extent_max > self.sigma_extent_min) and (self.sigma_extent_min > 0) and (self.partition_extent > 1), \
                "The extent model must be well-defined."

        # Keep track of transitions
        self.acceptance_history = dict()
        self.acceptance_history["move"] = list()
        self.acceptance_history["birth"] = list()
        self.acceptance_history["death"] = list()
        self.acceptance_history["extent_move"] = list()

        # Prepare birth density
        if self.verbose is True:
            print("Prepare birth density!")
        if self.birth_density_v1 is True:
            partition_spatial = self.prepare_spatial_birth_partition()
        else:
            dm = [divmod(self.M_k[k], self.birth_clutter_thin_var)[0] for k in range(self.K)]
            thin_p = [1/mult if mult > 0 else 1 for mult in dm]
            self.thin_birth_mask = [np.random.binomial(1, thin_p[k], size=self.M_k[k]).astype(bool) for k in range(self.K)]
            # self.thin_birth_mask = [np.ones(self.M_k[k]).astype(bool) for k in range(self.K)]
            partition_spatial = np.concatenate([self.Psi[k][self.thin_birth_mask[k]] for k in range(self.K)], axis=0)
        self.compute_birth_intensity_values(partition_spatial)

    def prepare_spatial_birth_partition(self):
        """
        Prepare the birth density. Assign the relevant variables to attributes.
        """
        # Prepare birth density partition
        x_dist = (self.window[1]-self.window[0])/(self.grid_dim[0])
        y_dist = (self.window[3]-self.window[2])/(self.grid_dim[1])
        z_dist = (self.window[5]-self.window[4])/(self.grid_dim[2])
        self.partition_dists = [x_dist, y_dist, z_dist]
        ext_x_points = np.linspace(self.window[0]-x_dist/2, self.window[1]+x_dist/2, self.grid_dim[0]+2)
        ext_y_points = np.linspace(self.window[2]-y_dist/2, self.window[3]+y_dist/2, self.grid_dim[1]+2)
        ext_z_points = np.linspace(self.window[4]-z_dist/2, self.window[5]+z_dist/2, self.grid_dim[2]+2)
        x_points = ext_x_points[1:-1]
        y_points = ext_y_points[1:-1]
        z_points = ext_z_points[1:-1]
        C1, C2, C3 = np.meshgrid(x_points, y_points, z_points, indexing="ij")
        partition_spatial = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)
        return partition_spatial

    def compute_birth_intensity_values(self, partition_spatial):
        """
        Compute the birth rate on the spatial-extent partition.
        """
        # Compute birth intensity values
        sensor_resolution_partition_spatial = np.zeros((self.K, partition_spatial.shape[0]))
        for i, p in enumerate(partition_spatial):
            sensor_resolution_partition_spatial[:, i] = self.sensor_resolution_function(p, self.sU)

        if self.object_extent is True and self.extent_prior == "1d_uniform":
            self.partition_extent_dist = (self.sigma_extent_max - self.sigma_extent_min)/(self.partition_extent-1)
            self.partition_extent_par = np.linspace(self.sigma_extent_min, self.sigma_extent_max, self.partition_extent)
            self.eta_partition = self.construct_eta_partition(partition_spatial, self.partition_extent_par, sensor_resolution_partition_spatial)
            self.eta_partition_sum = sum([np.sum(eta_partition_k, axis=0).flatten() for eta_partition_k in self.eta_partition])
            partition = np.concatenate((partition_spatial[:, None, :]+np.zeros((1, self.partition_extent, 1)),
                                        self.partition_extent_par[None, :, None]+np.zeros((partition_spatial.shape[0], 1, 1))), axis=2)
            self.partition_flat = partition.reshape((-1, self.d+1))
        elif self.object_extent is True and self.extent_prior == "inverse-wishart":
            # self.pdf_E = np.zeros(self.partition_extent)
            # for i in range(self.partition_extent):
            #     self.pdf_E[i] = invwishart.pdf(Sigma_E[i], self.extent_IW_df, self.extent_IW_scale)
            # self.pdf_E = (self.pdf_E[None, None, None, :] * np.ones(self.grid_dim)[:, :, :, None]).flatten()
            self.partition_extent_par = np.zeros((self.partition_extent, 6))
            for i in range(self.partition_extent):
                det = 100
                while det <= 0.8 or det >= 4:
                    Sigma_E = invwishart.rvs(self.extent_IW_df, self.extent_IW_scale) # size=(partition_extent, 3, 3)
                    E_mat = cholesky(Sigma_E, lower=True)
                    det = np.linalg.det(E_mat)
                self.partition_extent_par[i] = E_mat[self.tril_idx]
            self.eta_partition = self.construct_eta_partition(partition_spatial, self.partition_extent_par, sensor_resolution_partition_spatial)
            self.eta_partition_sum = sum([np.sum(eta_partition_k, axis=0).flatten() for eta_partition_k in self.eta_partition])
            partition = np.concatenate((partition_spatial[:, None, :]+np.zeros((1, self.partition_extent, 1)),
                                        self.partition_extent_par[None, :, :]+np.zeros((partition_spatial.shape[0], 1, 1))), axis=2)
            self.partition_flat = partition.reshape((-1, self.d+6))
        elif self.object_extent is True and self.extent_prior == "6d_uniform":
            self.partition_extent_par = np.zeros((self.partition_extent, 6))
            for i in range(self.partition_extent):
                std = np.random.uniform(self.sigma_extent_min, self.sigma_extent_max, size=3)
                cov = np.random.uniform(self.cor_extent_min, self.cor_extent_max, size=3)
                E_mat = std * np.eye(3)
                E_mat[self.low_tril_idx] = cov
                self.partition_extent_par[i] = E_mat[self.tril_idx]
            self.eta_partition = self.construct_eta_partition(partition_spatial, self.partition_extent_par, sensor_resolution_partition_spatial)
            self.eta_partition_sum = sum([np.sum(eta_partition_k, axis=0).flatten() for k, eta_partition_k in enumerate(self.eta_partition)])
            partition = np.concatenate((partition_spatial[:, None, :]+np.zeros((1, self.partition_extent, 1)),
                                        self.partition_extent_par[None, :, :]+np.zeros((partition_spatial.shape[0], 1, 1))), axis=2)
            self.partition_flat = partition.reshape((-1, self.d+6))
        else:
            self.partition_extent_dist = None
            self.eta_partition = list()
            for k in range(self.K):
                eta_temp = np.zeros((self.Psi[k].shape[0], self.partition_spatial.shape[0]))
                rho_partition_k = self.rho[k](partition_spatial)
                Sigma_partition_k = self.Sigma[k](partition_spatial)
                eta_temp = self.cluster_intensity(partition_spatial, self.Psi[k], rho_partition_k, Sigma_partition_k)
                self.eta_partition.append(eta_temp)
            self.eta_partition_sum = sum([np.sum(eta_partition_k, axis=0).flatten() for eta_partition_k in self.eta_partition])
            self.partition_flat = self.partition_spatial

    def construct_eta_partition(self, partition_spatial, extent_par, sensor_resolution_partition_spatial):
        """
        Parameters
        ----------
        extent_par : float or ndarray, size=(6, )
            The extent model parameters.
        sensor_resolution_partition_spatial : ndarray, size=(K, spatial_partition_size)
            The sensor resolution for each sensor in each spatial partition point.

        Returns
        -------
        eta_partition : list, len=K
            List of length K of ndarrays of size=(M_k, spatial_partition_size, extent_partition_size).
            Each entry is the evaluation of the eta-function.
        """
        eta_partition = list()
        for k in range(self.K):
            eta_temp = np.zeros((self.Psi[k].shape[0], partition_spatial.shape[0], self.partition_extent))
            rho_partition_k = self.rho[k](partition_spatial)
            Sigma_partition_k = self.Sigma[k](partition_spatial)
            for i in range(self.partition_extent):
                Sigma_tilde, rho_tilde = self.extent_model(extent_par[i], Sigma_partition_k,
                                                           rho_partition_k, sensor_resolution_partition_spatial[k])
                eta_temp[:, :, i] = self.cluster_intensity(partition_spatial, self.Psi[k], rho_tilde, Sigma_tilde)
            eta_partition.append(eta_temp)
        return eta_partition

    def run_profile(self):
        """
        Run line profiler.
        """
        self.profile.print_stats()

    def estimate_intensity(self, rho_i, L_i, M_c_i):
        """
        Compute estimate of the scalar intensity of the parent process.
        """
        lambda_hat = max(L_i*(self.M - M_c_i)/(self.area * np.sum(rho_i)), 1e-05)
        return lambda_hat

    def prior_check(self, extent_par):
        """
        Check that the proposal fulfils the prior assumptions:
            1) Check extent model.
        Return 0 if the prior assumptions are broken.
    
        Parameters
        ----------
        extent_par : float or ndarray, size=(6,)
            The extent parameters.
        """
        if self.extent_prior == "1d_uniform":
            if extent_par < self.sigma_extent_min or extent_par > self.sigma_extent_max:
                return 0
        elif self.extent_prior == "inverse-wishart":
            E_mat = np.zeros((self.d, self.d))
            E_mat[self.tril_idx] = extent_par
            det = np.linalg.det(E_mat)
            if det <= 0.8 or det >= 4 or np.any(np.diag(E_mat) <= 0):
                return 0
        elif self.extent_prior == "6d_uniform":
            E_mat = np.zeros((self.d, self.d))
            E_mat[self.tril_idx] = extent_par
            if np.any(np.diag(E_mat) < self.sigma_extent_min) or np.any(np.diag(E_mat) > self.sigma_extent_max) or \
               np.any(E_mat[self.low_tril_idx] < self.cor_extent_min) or np.any(E_mat[self.low_tril_idx] > self.cor_extent_max):
                return 0
        return 1

    def inverse_wishart_density_ratio(self, extent_par_old, extent_par_star):
        """
        Parameters
        ----------
        extent_par_old : ndarray, size=(6, )
            The old extent parameter vector.
        extent_par_star : ndarray, size=(6, )
            The proposed extent parameter vector.

        Returns
        -------
        density_ratio : float
            The value of the density ratio.
        """
        # Convert extent parameter vectors to covariance matrices
        E_mat_old = np.zeros((self.d, self.d))
        E_mat_old[self.tril_idx] = extent_par_old
        Sigma_E_old = E_mat_old @ E_mat_old.T
        E_mat_star = np.zeros((self.d, self.d))
        E_mat_star[self.tril_idx] = extent_par_star
        Sigma_E_star = E_mat_star @ E_mat_star.T

        # Compute density ratio
        det_old = np.linalg.det(Sigma_E_old)
        det_star = np.linalg.det(Sigma_E_star)
        det_term = (det_star/det_old)**(-(self.extent_IW_df+self.d+1))

        inv_old = np.linalg.inv(Sigma_E_old)
        inv_star = np.linalg.inv(Sigma_E_star)
        inv_diff = inv_star - inv_old

        density_ratio = det_term * np.exp(-1/2*np.trace(self.extent_IW_scale @ inv_diff))
        return density_ratio

    def birth_death_Metropolis_ratio(self, lambda_old, lambda_c_old, eta_old,
                                     rho_xi, eta_xi, birth_death_ratio):
        """
        Compute the Metropolis ratio for the birth-death transition.
        """
        Metropolis_ratio = lambda_old * np.exp(- np.sum(rho_xi)) * birth_death_ratio \
            * np.prod([np.prod(1 + eta_xi[k]/(lambda_c_old + np.sum(eta_old[k], axis=1))) for k in range(self.K)])
        return Metropolis_ratio

    def birth_ratio_MH(self, Phi_old, lambda_old, lambda_c_old, eta_old,
                       Phi_star, rho_star, eta_star, xi_partition_idx, extent_par_xi,
                       birth_density, birth_cdf, death_cdf_old):
        """
        Computing the birth acceptance probability when transitioning from
        Phi_old to Phi_old + delta_xi.
        """
        xi_idx = -1

        # Check hard core radius
        dist = np.linalg.norm(Phi_old - (Phi_star[xi_idx])[None, :], axis=1)
        if np.any(dist < self.R):
            return 0
        if self.prior_check(extent_par_xi) == 0:
            return 0

        if self.extent_prior == "inverse-wishart":
            pdf_E = self.inverse_wishart_density(extent_par_xi)
        else:
            pdf_E = 1

        # Compute birth-death probability ratio
        if death_cdf_old == None:
            death_cdf_old = 0
        death_density_star, death_cdf_star = self.compute_death_density(rho_star, lambda_c_old, eta_star)
        p_b_old = birth_cdf/(death_cdf_old + birth_cdf)
        p_b_star = birth_cdf/(death_cdf_star + birth_cdf)
        p_b_ratio = (1 - p_b_star)/p_b_old
        birth_death_ratio = p_b_ratio * death_density_star[xi_idx]/birth_density[xi_partition_idx]

        eta_xi = [eta_star[k][:, xi_idx] for k in range(self.K)]
        Metropolis_ratio = pdf_E * self.birth_death_Metropolis_ratio(lambda_old, lambda_c_old, eta_old,
                                                                     rho_star[:, xi_idx], eta_xi, birth_death_ratio)
        accept_prob = min(1, Metropolis_ratio)
        return accept_prob

    def death_ratio_MH(self, Phi_old, Extent_old, rho_old, lambda_old, lambda_c_old, eta_old,
                       Phi_star, rho_star, eta_star, c_idx,
                       birth_density, birth_cdf, death_density_old, death_cdf_old):
        """
        Computing the death acceptance probability when transitioning from
        Phi_old = Phi_star + delta_xi to Phi_star.
        """
        if Phi_star.shape[0] == 0:
            return 0

        if self.extent_prior == "inverse-wishart":
            pdf_E = self.inverse_wishart_density(Extent_old[c_idx])
        else:
            pdf_E = 1

        death_density_star, death_cdf_star = self.compute_death_density(rho_star, lambda_c_old, eta_star)
        p_b_old = birth_cdf/(death_cdf_old + birth_cdf)
        p_b_star = birth_cdf/(death_cdf_star + birth_cdf)
        p_b_ratio = (1 - p_b_old)/(p_b_star)
        if self.object_extent is True and self.extent_prior == "1d_uniform":
            Tilde_Phi_old = np.concatenate((Phi_old, np.expand_dims(Extent_old, axis=-1)), axis=-1)
        elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
            Tilde_Phi_old = np.concatenate((Phi_old, Extent_old), axis=-1)
        else:
            Tilde_Phi_old = Phi_old
        c_partition_idx = np.argmin(np.linalg.norm((Tilde_Phi_old[c_idx])[None, :] - self.partition_flat, axis=1))
        birth_death_ratio = p_b_ratio * death_density_old[c_idx]/birth_density[c_partition_idx]

        eta_c = [eta_old[k][:, c_idx] for k in range(self.K)]
        Metropolis_ratio = pdf_E * self.birth_death_Metropolis_ratio(lambda_old, lambda_c_old, eta_star,
                                                                     rho_old[:, c_idx], eta_c, birth_death_ratio)
        accept_prob = min(1, 1/Metropolis_ratio)
        return accept_prob

    def move_ratio_MH(self, Phi_old, rho_old, Sigma_old, lambda_c_old, eta_old,
                      c_idx, xi, rho_xi, Sigma_xi, eta_xi):
        """
        Computing the move acceptance probability when transitioning from
        Phi_old to Phi_old - delta_c + delta_xi.
        """
        # Check hard core radius
        L_old = Phi_old.shape[0]
        dist = np.linalg.norm(Phi_old[~np.in1d(np.arange(L_old), c_idx)] - xi[None, :], axis=1)
        if np.any(dist < self.R):
            return 0

        ### speed this up! ###
        move_covariance_c = self.vartheta*np.linalg.inv(np.sum(np.linalg.inv(Sigma_old[:, c_idx, :, :]), axis=0))
        move_covariance_xi = self.vartheta*np.linalg.inv(np.sum(np.linalg.inv(Sigma_xi), axis=0))
        kernel_ratio = self.kernel(xi - Phi_old[c_idx], 1, move_covariance_c)/self.kernel(Phi_old[c_idx] - xi, 1, move_covariance_xi)
        ######################

        Metropolis_ratio = np.exp(np.sum(rho_old[:, c_idx] - rho_xi)) * kernel_ratio \
            * np.prod([np.prod(1 + (eta_xi[k] - eta_old[k][:, c_idx])/(lambda_c_old + np.sum(eta_old[k], axis=1))) for k in range(self.K)])
        accept_prob = min(1, Metropolis_ratio)
        return accept_prob

    def birth_ratio_MarginalMAP(self, Phi_old, lambda_old, lambda_c_old, eta_old,
                                Phi_star, rho_star, eta_star, xi_partition_idx, extent_par_xi):
        """
        Compute birth acceptance probability for the Stochastic
        Marginal Maximum A Posteriori algorithm.
        """
        xi_idx = -1

        # Check hard core radius
        dist = np.linalg.norm(Phi_old - (Phi_star[xi_idx])[None, :], axis=1)
        if np.any(dist < self.R):
            return 0
        if self.prior_check(extent_par_xi) == 0:
            return 0

        if self.extent_prior == "inverse-wishart":
            pdf_E = self.inverse_wishart_density(extent_par_xi)
        else:
            pdf_E = 1

        eta_xi = [eta_star[k][:, xi_idx] for k in range(self.K)]
        Metropolis_ratio = pdf_E * self.birth_death_Metropolis_ratio(lambda_old, lambda_c_old, eta_old,
                                                                     rho_star[:, xi_idx], eta_xi, 1)
        accept_prob = min(1, Metropolis_ratio)
        return accept_prob

    def death_ratio_MarginalMAP(self, Extent_old, rho_old, lambda_old, lambda_c_old, eta_old,
                                Phi_star, eta_star, c_idx):
        """
        Compute death acceptance probability for the Stochastic
        Marginal Maximum A Posteriori algorithm.
        """
        if Phi_star.shape[0] == 0:
            return 0

        if self.extent_prior == "inverse-wishart":
            pdf_E = self.inverse_wishart_density(Extent_old[c_idx])
        else:
            pdf_E = 1

        eta_c = [eta_old[k][:, c_idx] for k in range(self.K)]
        Metropolis_ratio = pdf_E * self.birth_death_Metropolis_ratio(lambda_old, lambda_c_old, eta_star,
                                                                     rho_old[:, c_idx], eta_c, 1)
        accept_prob = min(1, 1/Metropolis_ratio)
        return accept_prob

    def move_ratio_MarginalMAP(self, Phi_old, rho_old, lambda_c_old, eta_old,
                               c_idx, xi, rho_xi, eta_xi):
        """
        Compute move acceptance probability for the Stochastic
        Marginal Maximum A Posteriori algorithm.
        """
        # Check hard core radius
        L_old = Phi_old.shape[0]
        dist = np.linalg.norm(Phi_old[~np.in1d(np.arange(L_old), c_idx)] - xi[None, :], axis=1)
        if np.any(dist < self.R):
            return 0

        Metropolis_ratio = np.exp(np.sum(rho_old[:, c_idx] - rho_xi)) \
            * np.prod([np.prod(1 + (eta_xi[k] - eta_old[k][:, c_idx])/(lambda_c_old + np.sum(eta_old[k], axis=1))) for k in range(self.K)])
        accept_prob = min(1, Metropolis_ratio)
        return accept_prob

    def extent_move_ratio(self, Extent_old, rho_old, lambda_c_old, eta_old,
                          c_idx, xi, rho_xi, eta_xi, extent_par_xi):
        """
        Computing the move acceptance probability when transitioning from
        Phi_old to Phi_old - delta_c + delta_xi.
        """
        if self.prior_check(extent_par_xi) == 0:
            return 0
        if self.extent_prior == "inverse-wishart":
            extent_prior_ratio = self.inverse_wishart_density_ratio(Extent_old[c_idx], extent_par_xi)
        else:
            extent_prior_ratio = 1
        Metropolis_ratio = extent_prior_ratio * np.exp(np.sum(rho_old[:, c_idx] - rho_xi)) \
            * np.prod([np.prod(1 + (eta_xi[k] - eta_old[k][:, c_idx])/(lambda_c_old + np.sum(eta_old[k], axis=1))) for k in range(self.K)])
        accept_prob = min(1, Metropolis_ratio)
        return accept_prob

    @profile
    def cluster_transition(self, Phi_old, Extent_old, rho_old, Sigma_old, lambda_old,
                           lambda_c_old, eta_old, transition_type,
                           birth_density, birth_cdf, death_density_old, death_cdf_old):
        """
        Simulate a proposal transition Phi_star. Then compute acceptance
        probabilities for the proposed transition. Finally, Simulate acceptance
        and return the new state of the Markov chain.
        """
        L_old = Phi_old.shape[0]
        ### Make a birth-death-move proposal and compute the Metropolis ratio ###
        if transition_type == "move":
            c_idx = np.random.randint(low=0, high=L_old)
            move_covariance = self.vartheta*np.linalg.inv(np.sum(np.linalg.inv(Sigma_old[:, c_idx, :, :]), axis=0))
            with np.errstate(invalid="ignore"):
                move_det = np.linalg.det(move_covariance)
            if move_det > 0:
                xi = np.random.multivariate_normal(Phi_old[c_idx, :], move_covariance)
            else:
                xi = np.random.multivariate_normal(Phi_old[c_idx, :], move_covariance*np.eye(self.d))
            rho_xi, Sigma_xi = self.compute_rho_and_Sigma(xi)

            if self.object_extent is True:
                sensor_resolution_xi = self.sensor_resolution_function(xi, self.sU)
                Sigma_xi, rho_xi = self.extent_model(Extent_old[c_idx], Sigma_xi, rho_xi, sensor_resolution_xi)

            eta_xi = [self.cluster_intensity(xi, self.Psi[k], rho_xi[k], Sigma_xi[k]) for k in range(self.K)] # len=K of size=(M_k,)

            Phi_star = np.delete(np.insert(Phi_old, c_idx, xi, axis=0), c_idx+1, axis=0)
            Extent_star = Extent_old
            rho_star = np.delete(np.insert(rho_old, c_idx, rho_xi, axis=1), c_idx+1, axis=1)
            Sigma_star = np.delete(np.insert(Sigma_old, c_idx, Sigma_xi, axis=1), c_idx+1, axis=1)
            eta_star = [np.delete(np.insert(eta_old[k], c_idx, eta_xi[k], axis=1), c_idx+1, axis=1) for k in range(self.K)]

            if self.acceptance_type == "Metropolis-Hastings":
                accept_prob = self.move_ratio_MH(Phi_old, rho_old, Sigma_old, lambda_c_old, eta_old,
                                                 c_idx, xi, rho_xi, Sigma_xi, eta_xi)
            elif self.acceptance_type == "Stochastic Marginal Maximum A Posteriori":
                accept_prob = self.move_ratio_MarginalMAP(Phi_old, rho_old, lambda_c_old, eta_old,
                                                          c_idx, xi, rho_xi, eta_xi)
        elif transition_type == "extent_move":
            assert self.object_extent is True, "Cannot make extent move if there is no object extent..."
            c_idx = np.random.randint(low=0, high=L_old)
            xi = Phi_old[c_idx]

            if self.object_extent is True and self.extent_prior == "1d_uniform":
                extent_par_xi = np.random.normal(Extent_old[c_idx], self.extent_move_std*Extent_old[c_idx])
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                extent_move_covariance = (self.extent_move_std*Extent_old[c_idx])**2*np.eye(6)
                extent_par_xi = np.random.multivariate_normal(Extent_old[c_idx], extent_move_covariance)

            rho_xi, Sigma_xi = self.compute_rho_and_Sigma(xi)
            sensor_resolution_xi = self.sensor_resolution_function(xi, self.sU)
            Sigma_xi, rho_xi = self.extent_model(extent_par_xi, Sigma_xi, rho_xi, sensor_resolution_xi)

            eta_xi = [self.cluster_intensity(xi, self.Psi[k], rho_xi[k], Sigma_xi[k]) for k in range(self.K)] # len=K of size=(M_k,)

            Phi_star = Phi_old
            if self.object_extent is True and self.extent_prior == "1d_uniform":
                Extent_star = np.delete(np.insert(Extent_old, c_idx, extent_par_xi), c_idx+1)
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                Extent_star = np.delete(np.insert(Extent_old, c_idx, extent_par_xi, axis=0), c_idx+1, axis=0)
            else:
                Extent_star = np.delete(np.insert(Extent_old, c_idx, extent_par_xi), c_idx+1)
            rho_star = np.delete(np.insert(rho_old, c_idx, rho_xi, axis=1), c_idx+1, axis=1)
            Sigma_star = np.delete(np.insert(Sigma_old, c_idx, Sigma_xi, axis=1), c_idx+1, axis=1)
            eta_star = [np.delete(np.insert(eta_old[k], c_idx, eta_xi[k], axis=1), c_idx+1, axis=1) for k in range(self.K)]

            accept_prob = self.extent_move_ratio(Extent_old, rho_old, lambda_c_old, eta_old,
                                                 c_idx, xi, rho_xi, eta_xi, extent_par_xi)
        elif transition_type == "birth":
            xi_partition_idx = np.random.choice(np.arange(self.partition_flat.shape[0]), p=birth_density)
            if self.birth_density_v1 is True:
                partition_midpoint_shift = np.array([np.random.uniform(-self.partition_dists[0]/2, self.partition_dists[0]/2),
                                                     np.random.uniform(-self.partition_dists[1]/2, self.partition_dists[1]/2),
                                                     np.random.uniform(-self.partition_dists[2]/2, self.partition_dists[2]/2)])
            else:
                partition_midpoint_shift = np.array([np.random.uniform(-self.birth_shift[0], self.birth_shift[0]),
                                                     np.random.uniform(-self.birth_shift[1], self.birth_shift[1]),
                                                     np.random.uniform(-self.birth_shift[2], self.birth_shift[2])])

            if self.object_extent is True and self.extent_prior == "1d_uniform":
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, np.random.uniform(-self.partition_extent_dist/2, self.partition_extent_dist/2)))
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, np.zeros(6)))
            else:
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, 0))
            proposal = self.partition_flat[xi_partition_idx]+partition_midpoint_shift

            xi, extent_par_xi = proposal[:self.d], proposal[self.d:]
            rho_xi, Sigma_xi = self.compute_rho_and_Sigma(xi)
            if self.object_extent is True:
                sensor_resolution_xi = self.sensor_resolution_function(xi, self.sU)
                Sigma_xi, rho_xi = self.extent_model(extent_par_xi, Sigma_xi, rho_xi, sensor_resolution_xi)
            eta_xi = [self.cluster_intensity(xi, self.Psi[k], rho_xi[k], Sigma_xi[k]) for k in range(self.K)] # len=K of size=(M_k,)

            if self.object_extent is True and self.extent_prior == "1d_uniform":
                Extent_star = np.hstack((Extent_old, extent_par_xi))
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                Extent_star = np.concatenate((Extent_old, np.expand_dims(extent_par_xi, axis=0)), axis=0)
            else:
                Extent_star = np.hstack((Extent_old, extent_par_xi))

            Phi_star = np.concatenate((Phi_old, np.expand_dims(xi, axis=0)), axis=0)
            rho_star = np.concatenate((rho_old, np.expand_dims(rho_xi, axis=1)), axis=1)
            Sigma_star = np.concatenate((Sigma_old, np.expand_dims(Sigma_xi, axis=1)), axis=1)
            eta_star = [np.concatenate((eta_old[k], np.expand_dims(eta_xi[k], axis=1)), axis=1) for k in range(self.K)]

            if self.acceptance_type == "Metropolis-Hastings":
                accept_prob = self.birth_ratio_MH(Phi_old, lambda_old, lambda_c_old, eta_old,
                                                  Phi_star, rho_star, eta_star, xi_partition_idx, extent_par_xi,
                                                  birth_density, birth_cdf, death_cdf_old)
            elif self.acceptance_type == "Stochastic Marginal Maximum A Posteriori":
                accept_prob = self.birth_ratio_MarginalMAP(Phi_old, lambda_old, lambda_c_old, eta_old,
                                                           Phi_star, rho_star, eta_star, xi_partition_idx, extent_par_xi)
        elif transition_type == "death":
            c_idx = np.random.randint(low=0, high=L_old)

            if self.object_extent is True and self.extent_prior == "1d_uniform":
                Extent_star = np.delete(Extent_old, c_idx)
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                Extent_star = np.delete(Extent_old, c_idx, axis=0)
            else:
                Extent_star = np.delete(Extent_old, c_idx)

            Phi_star = np.delete(Phi_old, c_idx, axis=0)
            rho_star = np.delete(rho_old, c_idx, axis=1)
            Sigma_star = np.delete(Sigma_old, c_idx, axis=1)
            eta_star = [np.delete(eta_old[k], c_idx, axis=1) for k in range(self.K)]

            if self.acceptance_type == "Metropolis-Hastings":
                accept_prob = self.death_ratio_MH(Phi_old, Extent_old, rho_old, lambda_old, lambda_c_old, eta_old,
                                                  Phi_star, rho_star, eta_star, c_idx,
                                                  birth_density, birth_cdf, death_density_old, death_cdf_old)
            elif self.acceptance_type == "Stochastic Marginal Maximum A Posteriori":
                accept_prob = self.death_ratio_MarginalMAP(Extent_old, rho_old, lambda_old, lambda_c_old, eta_old,
                                                           Phi_star, eta_star, c_idx)
        else:
            assert False, "Non-supported transition type chosen!"

        ### Simulate acceptance ###
        self.acceptance_history[f"{transition_type}"].append(accept_prob)
        U = np.random.rand()
        if U < accept_prob:
            return Phi_star, Extent_star, rho_star, Sigma_star, eta_star
        else:
            return Phi_old, Extent_old, rho_old, Sigma_old, eta_old

    def compute_birth_density(self, lambda_, lambda_c):
        """
        Compute the discrete birth density on the partition given
        the intensity parameters.
        """
        birth_rate = lambda_*(1 + self.eta_partition_sum/lambda_c)
        birth_cdf = np.sum(birth_rate)
        birth_density = birth_rate/birth_cdf
        return birth_density, birth_cdf

    @profile
    def compute_death_density(self, rho, lambda_c, eta):
        """
        Compute the discrete death density.
        """
        L = eta[0].shape[1]
        death_rate = np.zeros(L)
        eta_m_sum = sum([np.sum(eta_k, axis=0) for eta_k in eta])
        eta_l_sum = [lambda_c + np.sum(eta[k], axis=1) for k in range(self.K)]
        for l in range(L):
            Z_Phi_minus_c = [eta_l_sum[k] - eta[k][:, l] for k in range(self.K)]
            prod = np.prod([np.prod(1 + eta[k][:, l]/Z_Phi_minus_c[k]) for k in range(self.K)])
            death_rate[l] = np.exp(np.sum(rho[:, l]))/prod * (1 + eta_m_sum[l]/lambda_c)
        death_cdf = np.sum(death_rate)
        death_density = death_rate/death_cdf
        return death_density, death_cdf

    @profile
    def simulate_cluster_transition_type(self, Phi_old, A_old, lambda_old, lambda_c_old, rho_old, eta_old):
        """
        Simulate which transition to do: move, birth, death.
        """
        u = np.random.rand()
        if u < self.p_m:
            if self.object_extent is True:
                ue = np.random.rand()
                if ue < self.p_em:
                    return "extent_move", None, None, None, None
                else:
                    return "move", None, None, None, None
            else:
                return "move", None, None, None, None
        else:
            if self.birth_density_v1 is False:
                A_c_old = [A_old[k] == 0 for k in range(self.K)]
                # Psi_c = np.concatenate([self.Psi[k][A_c_old[k]] for k in range(self.K)])
                thin_mask = [np.logical_and(A_c_old[k], self.thin_birth_mask[k]) for k in range(self.K)]
                Psi_c_thin = np.concatenate([self.Psi[k][thin_mask[k]] for k in range(self.K)])
                thin_mask_small = [A_c_old[k][self.thin_birth_mask[k]] for k in range(self.K)]
                clutter_mask = np.concatenate([thin_mask_small[k] for k in range(self.K)], axis=0)
                num_clut = sum(clutter_mask)
                eta_partition_c = [self.eta_partition[k][:, clutter_mask, :] for k in range(self.K)]
                self.eta_partition_sum = sum([np.sum(epc, axis=0).flatten() for epc in eta_partition_c])
                if self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                    self.partition_flat = np.concatenate((Psi_c_thin[:, None, :]+np.zeros((1, self.partition_extent, 1)),
                                                          self.partition_extent_par[None, :, :]+np.zeros((Psi_c_thin.shape[0], 1, 1))),
                                                          axis=2).reshape((-1, self.d+6))
                elif self.object_extent is True and self.extent_prior == "1d_uniform":
                    self.partition_flat = np.concatenate((Psi_c_thin[:, None, :]+np.zeros((1, self.partition_extent, 1)),
                                                          self.partition_extent_par[None, :, None]+np.zeros((Psi_c_thin.shape[0], 1, 1))),
                                                          axis=2).reshape((-1, self.d+1))
    
            birth_density, birth_cdf = self.compute_birth_density(lambda_old, lambda_c_old)
            if Phi_old.shape[0] <= 1: # No death when only 1 target exists
                p_b, death_density, death_cdf = 1, None, None
            elif num_clut == 0:
                p_b, death_density, death_cdf = 0, None, None
            elif Phi_old.shape[0] > 1 and num_clut > 0:
                death_density, death_cdf = self.compute_death_density(rho_old, lambda_c_old, eta_old)
                p_b = birth_cdf/(death_cdf + birth_cdf) # Birth probability
            else:
                return "move", None, None, None, None

            ubd = np.random.rand()
            if ubd < p_b: # Birth
                return "birth", birth_density, birth_cdf, death_density, death_cdf
            else: # Death
                return "death", birth_density, birth_cdf, death_density, death_cdf

    @profile
    def birth_death_move_step(self, Phi_old, Extent_old, A_old, rho_old, Sigma_old, lambda_old, lambda_c_old, eta_old):
        """
        Updating the cluster centers, cluster sizes, cluster spreads, and eta.

        Input:
        ------
            Phi_old : ndarray, size=(L_old, d)
                Cluster centers.
            Extent_old : ndarray, size=(L_old, )
                The standard deviation of the object extent.
            A_old : list
                List of ndarrays of size=(M_k,) for the k-th entry containing
                data association labels.
            rho_old : ndarray, size=(K, L_old)
                Measurement rates.
            Sigma_old : ndarray, size=(K, L_old, d, d)
                Sensor covariances.
            lambda_old : float
                Cluster center intensity.
            lambda_c_old : float
                Clutter intensity.
            eta_old : list
                List of intensity values with size=(M_k, L_old) in the k-th
                entry corresponding to the intensity value for the k-th
                sensor and the l-th cluster center in the m_k-th measurement.

        Output:
        -------
            Phi_new : ndarray, size=(L_new, d)
                Cluster centers.
            rho_new : ndarray, size=(K, L_old)
                Measurement rates.
            Sigma_new : ndarray, size=(K, L_old, d, d)
                Sensor covariances.
            eta_new : list
                List of intensity values with size=(M_k, L_old) in the k-th
                entry corresponding to the intensity value for the k-th
                sensor and the l-th cluster center in the m_k-th measurement.
        """
        # Transition
        transition_type, birth_density, birth_cdf, death_density, death_cdf = \
            self.simulate_cluster_transition_type(Phi_old, A_old, lambda_old, lambda_c_old, rho_old, eta_old)
        Phi_new, Extent_new, rho_new, Sigma_new, eta_new = \
            self.cluster_transition(Phi_old, Extent_old, rho_old, Sigma_old, lambda_old,
                                    lambda_c_old, eta_old, transition_type,
                                    birth_density, birth_cdf, death_density, death_cdf)
        return Phi_new, Extent_new, rho_new, Sigma_new, eta_new

    @profile
    def simulate_association(self, lambda_c_old, eta_new):
        """
        Simulate independently data association given the new state.
        """
        # Speed this method up!
        L_new = eta_new[0].shape[1]
        association = list()
        for k in range(self.K):
            eta_sum = np.sum(eta_new[k], axis=1)
            if self.data_association_sim is True:
                association_rate = np.insert(eta_new[k], 0, lambda_c_old, axis=1)
                normalization_fator = lambda_c_old + eta_sum
                prob = association_rate/normalization_fator[:, None] # size=(M_k, L_new+1)
                a_k = np.zeros(self.M_k[k], dtype=np.int16)
                for m_k in range(self.M_k[k]):
                    a_k[m_k] = choices(np.arange(L_new+1), weights=prob[m_k, :])[0]
            elif self.data_association_sim is False:        
                association_rate = np.insert(eta_new[k], 0, lambda_c_old, axis=1)
                normalization_fator = lambda_c_old + eta_sum
                prob = association_rate/normalization_fator[:, None] # size=(M_k, L_new+1)
                a_k = np.argmax(prob, axis=1)
            elif self.data_association_sim == "clutter":
                association_rate = np.insert(np.expand_dims(eta_sum, axis=1), 0, lambda_c_old, axis=1)
                normalization_fator = lambda_c_old + eta_sum
                prob = association_rate/normalization_fator[:, None] # size=(M_k, 2)
                U = np.random.rand(self.M_k[k])
                a_k = np.ones(self.M_k[k], dtype=np.int16)
                a_k[U < prob[:, 0]] = 0
            association.append(a_k)
        return association

    def metropolis_hastings_step(self, Phi_new, A_new, lambda_old, lambda_c_old, eta_new):
        """
        Updating the cluster center and clutter intensity parameters.
        """
        ### Cluster center intensity ###
        lambda_star = np.random.normal(loc=lambda_old, scale=self.intensity_std)
        if lambda_star <= 0:
            alpha_move = 0
        else:
            Metropolis_ratio = np.exp(self.area*(lambda_old-lambda_star)) \
                * (lambda_star/lambda_old)**(Phi_new.shape[0])
            alpha_move = min(1, Metropolis_ratio)

        R = np.random.rand()
        if R < alpha_move:
            lambda_new = lambda_star
        else:
            lambda_new = lambda_old

        ### Clutter intensity ###
        M_c_new = np.sum([np.sum(A_new[k] == 0) for k in range(self.K)])
        lambda_c_star = np.random.normal(loc=lambda_c_old, scale=self.clutter_std)
        if lambda_c_star <= 0:
            alpha_move = 0
        else:
            Metropolis_ratio = np.exp(self.K*self.area*(lambda_c_old-lambda_c_star)) \
                * (lambda_c_star/lambda_c_old)**(M_c_new/self.K)
            alpha_move = min(1, Metropolis_ratio)

        R = np.random.rand()
        if R <= alpha_move and lambda_c_star > 0:
            lambda_c_new = lambda_c_star
        else:
            lambda_c_new = lambda_c_old
        return lambda_new, lambda_c_new

    def initialize(self, min_samples, eps):
        """
        Initialize the Markov chain using DBSCAN.
        """
        points = np.concatenate(self.Psi)
        cov = np.concatenate(self.Sigma_Psi)
        # sample_weights = 1/np.linalg.det(cov)
        DBSCAN_Cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(points, sample_weight=None)
        DBSCAN_labels = DBSCAN_Cluster.labels_
        A_0_arr = DBSCAN_labels + 1

        unique_labels = set(DBSCAN_labels)
        n_clusters = len(unique_labels) - (1 if -1 in DBSCAN_labels else 0)
        Phi_0 = np.zeros((n_clusters, 3))
        if self.object_extent is True and self.extent_prior == "1d_uniform":
            extent_par_0 = np.zeros(n_clusters)
        elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
            extent_par_0 = np.zeros((n_clusters, 6))
        else:
            extent_par_0 = np.zeros(n_clusters)
        for l in unique_labels:
            if l != -1:
                cov_target_l = cov[DBSCAN_labels == l]
                cov_inv = np.linalg.inv(cov_target_l)
                points_target_l = points[DBSCAN_labels == l]
                numerator = np.sum(np.einsum("mj,mji->mi", points_target_l, cov_inv), axis=0)
                denominator = np.sum(cov_inv, axis=0)
                Phi_0[l] = np.dot(numerator, np.linalg.inv(denominator))

                if self.object_extent is True:
                    M = points_target_l.shape[0]
                    if M > 1:
                        Sigma_W = np.einsum("mj,mi->mji", points_target_l - Phi_0[l], points_target_l - Phi_0[l])
                        Sigma_W_est = 1/(M-1) * np.sum(Sigma_W, axis=0)
                        Sigma_E = Sigma_W_est - 1/(M-1) * np.sum(cov_target_l, axis=0)
                        if self.extent_prior == "1d_uniform":
                            extent_par_0[l] = min(self.sigma_extent_max, np.sqrt(max(self.sigma_extent_min**2, np.mean(np.diag(Sigma_E)))))
                        elif self.extent_prior == "inverse-wishart":
                            try:
                                extent_par_0[l] = cholesky(Sigma_E)[self.tril_idx]
                            except np.linalg.LinAlgError:
                                extent_par_0[l] = cholesky(self.extent_IW_scale/(self.extent_IW_df-self.d-1))[self.tril_idx]
                        elif self.extent_prior == "6d_uniform":
                            try:
                                extent_par_0[l] = cholesky(Sigma_E)[self.tril_idx]
                            except np.linalg.LinAlgError:
                                E_mat = np.zeros((3, 3))
                                E_mat = self.sigma_extent_min * np.eye(3)
                                extent_par_0[l] = E_mat[self.tril_idx]
                    else:
                        if self.extent_prior == "1d_uniform":
                            extent_par_0[l] = self.sigma_extent_min
                        elif self.extent_prior == "inverse-wishart":
                            extent_par_0[l] = cholesky(self.extent_IW_scale/(self.extent_IW_df-self.d-1))[self.tril_idx]
                        elif self.extent_prior == "6d_uniform":
                            E_mat = np.zeros((3, 3))
                            E_mat = self.sigma_extent_min * np.eye(3)
                            extent_par_0[l] = E_mat[self.tril_idx]

        rho_0, Sigma_0 = self.compute_rho_and_Sigma(Phi_0)
        for idx_c, c in enumerate(Phi_0):
            sensor_resolution = self.sensor_resolution_function(c, self.sU)
            Sigma_0[:, idx_c], rho_0[:, idx_c] = self.extent_model(extent_par_0[idx_c], Sigma_0[:, idx_c, :, :], rho_0[:, idx_c], sensor_resolution)
        lambda_0 = Phi_0.shape[0]/self.area
        lambda_c_0 = np.sum(A_0_arr==0)/(self.K*self.area)
        A_0 = list()
        for k in range(self.K):
            A_0.append(A_0_arr[sum(self.M_k[:k]):sum(self.M_k[:k+1])])

        init_ = {"Phi": Phi_0, "extent_parameter": extent_par_0, "A": A_0, "rho": rho_0, "Sigma": Sigma_0, "lambda": lambda_0, "lambda_c": lambda_c_0}

        # Make a check that the initial parameter configuration has probability > 0...
        # That is, check that Phi_0 is a realization of hard core process...
        Phi_0_age = np.random.rand(Phi_0.shape[0])
        dist = np.linalg.norm(Phi_0[:, None, :] - Phi_0[None, :, :], axis=-1)
        bool_keep = np.zeros(Phi_0.shape[0], dtype=bool)
        for i in range(Phi_0.shape[0]):
            bool_in_disk  = (dist[i, :] < self.R) & (dist[i, :] > 0)
            if np.sum(bool_in_disk) == 0:
                bool_keep[i] = True
            else:
                bool_keep[i] = all(Phi_0_age[i] < Phi_0_age[bool_in_disk])
        if np.sum(np.invert(bool_keep)) != 0 and self.verbose is True:
            print(f"Removing {np.sum(np.invert(bool_keep))} point(s) due to hard core radius!")

        # Update the initial parameter estimates
        Phi_0 = Phi_0[bool_keep]
        extent_par_0 = extent_par_0[bool_keep]
        rho_0 = rho_0[:, bool_keep]
        Sigma_0 = Sigma_0[:, bool_keep]

        eta_0 = list()
        for k in range(self.K):
            eta_temp = np.zeros((self.Psi[k].shape[0], Phi_0.shape[0]))
            for idx_c, c in enumerate(Phi_0):
                eta_temp[:, idx_c] = self.cluster_intensity(c, self.Psi[k], rho_0[k, idx_c], Sigma_0[k, idx_c])
            eta_0.append(eta_temp)
        A_0 = self.simulate_association(lambda_c_0, eta_0)

        M_c_0 = np.sum([np.sum(A_0[k] == 0) for k in range(self.K)])
        lambda_0 = self.estimate_intensity(rho_0, Phi_0.shape[0], M_c_0)
        lambda_c_0 = max(M_c_0/(self.K*self.area), 1e-05)

        init_adjusted = {"Phi": Phi_0, "extent_parameter": extent_par_0, "A": A_0, "rho": rho_0, "Sigma": Sigma_0, "lambda": lambda_0, "lambda_c": lambda_c_0}
        return init_adjusted, init_, eta_0

    def initialize_from_birth(self):
        """
        Initialize the Markov chain by birthing a point from the birth density.

        Returns
        -------
        init_ : dict
            The initial Markov chain configuration.
        eta_0 : list, len=K
            List of ndarrays of size=(M_k, 1) of cluster intensity values in
            each observed point.
        """
        lambda_0 = 1/self.area
        lambda_c_0 = self.M/(self.K*self.area)

        birth_density, _ = self.compute_birth_density(lambda_0, lambda_c_0)
        xi_partition_idx = np.random.choice(np.arange(self.partition_flat.shape[0]), p=birth_density)
        if self.birth_density_v1 is True:
            partition_midpoint_shift = np.array([np.random.uniform(-self.partition_dists[0]/2, self.partition_dists[0]/2),
                                                 np.random.uniform(-self.partition_dists[1]/2, self.partition_dists[1]/2),
                                                 np.random.uniform(-self.partition_dists[2]/2, self.partition_dists[2]/2)])
            if self.object_extent is True and self.extent_prior == "1d_uniform":
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, np.random.uniform(-self.partition_extent_dist/2, self.partition_extent_dist/2)))
            elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, np.zeros(6)))
            else:
                partition_midpoint_shift = np.hstack((partition_midpoint_shift, 0))
            proposal = self.partition_flat[xi_partition_idx]+partition_midpoint_shift
        else:
            proposal = self.partition_flat[xi_partition_idx]
        xi, extent_par_xi = proposal[:self.d], proposal[self.d:]
        rho_xi, Sigma_xi = self.compute_rho_and_Sigma(xi)
        if self.object_extent is True:
            sensor_resolution_xi = self.sensor_resolution_function(xi, self.sU)
            Sigma_xi, rho_xi = self.extent_model(extent_par_xi, Sigma_xi, rho_xi, sensor_resolution_xi)
        eta_xi = [self.cluster_intensity(xi, self.Psi[k], rho_xi[k], Sigma_xi[k]) for k in range(self.K)] # len=K of size=(M_k,)

        if self.object_extent is True and self.extent_prior == "1d_uniform":
            Phi_E_0 = extent_par_xi
        elif self.object_extent is True and (self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform"):
            Phi_E_0 = np.expand_dims(extent_par_xi, axis=0)
        else:
            Phi_E_0 = extent_par_xi

        Phi_0 = np.expand_dims(xi, axis=0)
        rho_0 = np.expand_dims(rho_xi, axis=1)
        Sigma_0 = np.expand_dims(Sigma_xi, axis=1)
        eta_0 = [np.expand_dims(eta_xi[k], axis=1) for k in range(self.K)]
        A_0 = self.simulate_association(lambda_c_0, eta_0)

        init_ = {"Phi": Phi_0, "extent_parameter": Phi_E_0, "A": A_0, "rho": rho_0, "Sigma": Sigma_0, "lambda": lambda_0, "lambda_c": lambda_c_0}
        return init_, init_, eta_0

    def initialize_from_init_(self, init_):
        """
        Parameters
        ----------
        init_ : TYPE
            DESCRIPTION.

        Returns
        -------
        init_adjusted : TYPE
            DESCRIPTION.
        """
        Phi_0 = init_["Phi_0"]
        Phi_E_0 = init_["Phi_E_0"]
        lambda_c_0 = init_["lambda_c_0"]

        rho_0, Sigma_0 = self.compute_rho_and_Sigma(Phi_0)
        for idx_c, c in enumerate(Phi_0):
            sensor_resolution = self.sensor_resolution_function(c, self.sU)
            Sigma_0[:, idx_c], rho_0[:, idx_c] = self.extent_model(Phi_E_0[idx_c], Sigma_0[:, idx_c, :, :], rho_0[:, idx_c], sensor_resolution)

        # Make a check that the initial parameter configuration has probability > 0...
        # That is, check that Phi_0 is a realization of hard core process...
        Phi_0_age = np.random.rand(Phi_0.shape[0])
        dist = np.linalg.norm(Phi_0[:, None, :] - Phi_0[None, :, :], axis=-1)
        bool_keep = np.zeros(Phi_0.shape[0], dtype=bool)
        for i in range(Phi_0.shape[0]):
            bool_in_disk  = (dist[i, :] < self.R) & (dist[i, :] > 0)
            if np.sum(bool_in_disk) == 0:
                bool_keep[i] = True
            else:
                bool_keep[i] = all(Phi_0_age[i] < Phi_0_age[bool_in_disk])
        if np.sum(np.invert(bool_keep)) != 0 and self.verbose is True:
            print(f"Removing {np.sum(np.invert(bool_keep))} point(s) due to hard core radius!")

        # Update the initial parameter estimates
        Phi_0 = Phi_0[bool_keep]
        Phi_E_0 = Phi_E_0[bool_keep]
        rho_0 = rho_0[:, bool_keep]
        Sigma_0 = Sigma_0[:, bool_keep]

        eta_0 = list()
        for k in range(self.K):
            eta_temp = np.zeros((self.Psi[k].shape[0], Phi_0.shape[0]))
            for idx_c, c in enumerate(Phi_0):
                eta_temp[:, idx_c] = self.cluster_intensity(c, self.Psi[k], rho_0[k, idx_c], Sigma_0[k, idx_c])
            eta_0.append(eta_temp)
        A_0 = self.simulate_association(lambda_c_0, eta_0)

        M_c_0 = np.sum([np.sum(A_0[k] == 0) for k in range(self.K)])
        lambda_0 = self.estimate_intensity(rho_0, Phi_0.shape[0], M_c_0)
        lambda_c_0 = max(M_c_0/(self.K*self.area), 1e-05)

        init_adjusted = {"Phi": Phi_0, "extent_parameter": Phi_E_0, "A": A_0, "rho": rho_0, "Sigma": Sigma_0, "lambda": lambda_0, "lambda_c": lambda_c_0}
        return init_adjusted, eta_0

    @profile
    def __call__(self, runs, min_samples=None, eps=None, init_={"prior": None}, **kwargs):
        """
        Main class method, running the Markov chain Monte Carlo algorithm.

        Input:
        ------
            runs : int
                Number of MCMC iterations.
            min_samples : int
                Number of samples that need to be shared as neighborts for two
                points being part of the same cluster. Used for pre-processing
                clustering algorithm for initializing the cluster center states.
            eps : float
                Maximum distance between two samples for one to be considered
                as in the neighborhood of the other. Used for pre-processing
                clustering algorithm for initializing the cluster center states.
            kwargs : dict, optional
                Additional hyper-parameters of the algorithm.

        Output:
        -------
            Phi_list : list, len=(end_runs,)
                Posterior samples for the cluster centers.
            A_list : list, len=(end_runs,)
                Posterior samples for the data association.
            rho_list : list, len=(end_runs,)
                Posterior samples for the cluster sizes.
            Sigma_list : list, len=(end_runs,)
                Posterior samples for the cluster spreads.
            lambda_list : list, len=(end_runs,)
                Posterior samples for the cluster center intensity.
            lambda_c_list : list, len=(end_runs,)
                Posterior samples for the clutter intensity.
            log_posterior_list : list, len=(runs,)
                Log of the posterior of each Markov chain state.
            init_ : dict
                Dictionary of parameters values from initialization.
        """
        # =============================================================================
        # Keyword argument handling
        # =============================================================================
        burn_in = int(kwargs.get("burn_in", 0))
        move_prob_inc_interval = int(kwargs.get("move_prob_inc_interval", 100))
        move_prob_inc = kwargs.get("move_prob_inc", 0.1)
        move_prob_max = kwargs.get("move_prob_max", 0.95)
        move_trace_interval = int(kwargs.get("move_trace_interval", 20))
        move_trace_min = kwargs.get("move_trace_min", 0.6)
        move_trace_max = kwargs.get("move_trace_max", 0.9)
        move_trace_decrease = kwargs.get("move_trace_decrease", 0.75)
        move_trace_increase = kwargs.get("move_trace_increase", 1.15)

        assert burn_in >= 0, "Burn-in must be non-negative."
        assert move_prob_inc_interval >= 0, "Increment intervals must be non-negative."
        assert move_prob_inc >= 0, "The move probability is assumed to be non-negative."
        assert move_prob_max >= 0 and move_prob_max <= 1, "The move probability must be between 0 and 1."
        assert move_trace_interval > 0, "The interval for the move trace computation must be positive."
        assert move_trace_min < 1 and move_trace_min > 0, "The lower bound on the move acceptance probability must be between 0 and 1."
        assert move_trace_max < 1 and move_trace_max > 0, "The lower bound on the move acceptance probability must be between 0 and 1."
        assert move_trace_min < move_trace_max, "The lower bound must be smaller than the upper bound."
        assert move_trace_decrease < 1 and move_trace_decrease > 0, "The multiplicative factor for decreasing the move covariance scaling parameter must be between 0 and 1."
        assert move_trace_increase > 1, "The multiplicative factor for increasing the move covariance scaling parameter must be greater than 1."

        # =============================================================================
        # Initialize algorithm
        # =============================================================================
        print("Initialize Markov chain!")

        if init_["prior"] is None:
            if self.DBSCAN_init is True:
                init_adjusted = {"Phi": np.zeros(0)}
                while init_adjusted["Phi"].shape[0] == 0:
                    init_adjusted, init_, eta_adjusted = self.initialize(min_samples, eps)
                    min_samples -= 1
            elif self.DBSCAN_init is False:
                init_adjusted, init_, eta_adjusted = self.initialize_from_birth()
        else:
            init_adjusted, eta_adjusted = self.initialize_from_init_(init_)

        if self.extent_prior == "1d_uniform":
            assert np.all(init_adjusted["extent_parameter"] >= self.sigma_extent_min) and np.all(init_adjusted["extent_parameter"] <= self.sigma_extent_max), \
                "The initial model extent is outside the specified prior!"
        elif self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform":
            for l in range(init_adjusted["Phi"].shape[0]):
                E_mat = np.zeros((self.d, self.d))
                E_mat[self.tril_idx] = init_adjusted["extent_parameter"][l]
                assert np.linalg.det(E_mat) > 0, "The initial model extent is not positive definite!"

        Phi_list = list()
        Extent_list = list()
        A_list = list()
        rho_list = list()
        Sigma_list = list()
        lambda_list = list()
        lambda_c_list = list()
        log_posterior_list = list()

        Phi_old = init_adjusted["Phi"]
        Extent_old = init_adjusted["extent_parameter"]
        A_old = init_adjusted["A"]
        lambda_old = init_adjusted["lambda"]
        lambda_c_old = init_adjusted["lambda_c"]
        rho_old = init_adjusted["rho"]
        Sigma_old = init_adjusted["Sigma"]
        eta_old = eta_adjusted

        # =============================================================================
        # Being MCMC loop
        # =============================================================================
        log_posterior_max = 0
        patience_iter = 0

        end_bool = False
        print("Begin MCMC loop!")
        iteration = 0
        while iteration < runs:
            Phi_new, Extent_new, rho_new, Sigma_new, eta_new = self.birth_death_move_step(Phi_old, Extent_old, A_old, rho_old, Sigma_old, lambda_old, lambda_c_old, eta_old)
            A_new = self.simulate_association(lambda_c_old, eta_new)

            # for k in range(self.K):
            #     plt.plot((self.Psi[k][A_new[k] != 0])[:, 0], (self.Psi[k][A_new[k] != 0])[:, 1], "x", color="tab:red", markersize=2)
            #     plt.plot((self.Psi[k][A_new[k] == 0])[:, 0], (self.Psi[k][A_new[k] == 0])[:, 1], "x", color="tab:green", markersize=2)
            #     plt.plot(Phi_new[:, 0], Phi_new[:, 1], "o", color="tab:purple", markersize=3)
            # plt.show()

            if self.method == "pure":
                lambda_new, lambda_c_new = self.metropolis_hastings_step(Phi_new, A_new, lambda_old, lambda_c_old, eta_new)
            elif self.method == "modified":
                # lambda_new = max(Phi_new.shape[0]/self.area, 1e-05)
                M_c_i = np.sum([np.sum(A_new[k] == 0) for k in range(self.K)])
                lambda_new = self.estimate_intensity(rho_new, Phi_new.shape[0], M_c_i)
                lambda_c_new =  max(M_c_i/(self.K*self.area), 1e-05)

            log_posterior = self.log_posterior(Phi_new, Extent_new, rho_new, lambda_new, lambda_c_new, eta_new)
            log_posterior_list.append(log_posterior)
            if log_posterior > log_posterior_max:
                log_posterior_max = log_posterior
                patience_iter = 0
                Phi_max, Extent_max, A_max, rho_max, Sigma_max, eta_max = Phi_new, Extent_new, A_new, rho_new, Sigma_new, eta_new
            else:
                patience_iter += 1

            if iteration > 0 and (iteration == burn_in or iteration == runs-1) and end_bool is False:
                end_bool = True
                print("Fine-tune result!")
            elif iteration > 0:
                # Adaptively scale move covariance
                if len(self.acceptance_history["move"]) > 0:
                    move_trace = np.mean(self.acceptance_history["move"][-move_trace_interval:])
                    if move_trace < move_trace_min and self.vartheta > 1e-03:
                        self.vartheta = self.vartheta*move_trace_decrease
                    elif move_trace > move_trace_max and self.vartheta < 2:
                        self.vartheta = self.vartheta*move_trace_increase

                if len(self.acceptance_history["extent_move"]) > 0:
                    extent_move_trace = np.mean(self.acceptance_history["extent_move"][-move_trace_interval:])
                    if extent_move_trace < move_trace_min and self.extent_move_std > 1e-04:
                        self.extent_move_std = self.extent_move_std*move_trace_decrease
                    elif extent_move_trace > move_trace_max and self.extent_move_std < 1e-01:
                        self.extent_move_std = self.extent_move_std*move_trace_increase

                # Increase probability of move transition over time
                if iteration % move_prob_inc_interval == 0:
                    if self.p_m < move_prob_max:
                        self.p_m += move_prob_inc
                    else:
                        if burn_in == 0:
                            end_bool = True
                            print("Fine-tune result!")
                        else:
                            self.p_m = move_prob_max

            if end_bool is False and patience_iter == self.patience:
                end_bool = True
                print("Markov chain Monte Carlo loop stopped by early stopping at iteration: ", iteration)
                print("Fine-tune result!")
                Phi_new, Extent_new, A_new, rho_new, Sigma_new, eta_new = Phi_max, Extent_max, A_max, rho_max, Sigma_max, eta_max
                iteration = burn_in

            if end_bool is True:
                self.p_m = 1 # After burn-in only do move steps
                Phi_list.append(Phi_new)
                Extent_list.append(Extent_new)
                A_list.append(A_new)
                rho_list.append(rho_new)
                Sigma_list.append(Sigma_new)
                lambda_list.append(lambda_new)
                lambda_c_list.append(lambda_c_new)

            Phi_old = Phi_new
            Extent_old = Extent_new
            A_old = A_new
            rho_old = rho_new
            Sigma_old = Sigma_new
            lambda_old = lambda_new
            lambda_c_old = lambda_c_new
            eta_old = eta_new

            iteration += 1
        return Phi_list, Extent_list, A_list, rho_list, Sigma_list, lambda_list, lambda_c_list, log_posterior_list, init_


def main():
    """
    Markov chain Monte Carlo for a doubly inhomogeneous Neyman-Scott process.
    """
    np.random.seed(41)
    plt.style.use("ggplot")

    # =============================================================================
    # Unknown scenario parameters
    # =============================================================================
    lambda_c = 70*1e-04
    lambda_ = 20*1e-04

    # =============================================================================
    # Known scenario parameters
    # =============================================================================
    d = 3

    sigma_extent_min = 1
    sigma_extent_max = 1.5
    cor_extent_min = -0.5
    cor_extent_max = 0.5
    extent_IW_df = 10
    extent_IW_scale = 0.9*(extent_IW_df-d-1)*np.eye(d)
    list_extent = ["1d_uniform", "inverse-wishart", "6d_uniform"]
    extent_prior = list_extent[2]

    R = 5

    K = 2
    sU = np.array([[-1, -1, -1, np.pi/2, np.pi/2, 0],
                   [51, 21, 11, np.pi/2, np.pi/2, 0]])
    window = [0, 50, 0, 20, 0, 10]

    # =============================================================================
    # Simulate data
    # =============================================================================
    grid_dim = [30, 12, 6]

    rho_interp, Sigma_interp = setup_functions(window, grid_dim, K)
    Psi_da, Psi_c, Phi, Phi_E = DI_GSNCP(lambda_, R, rho_interp, Sigma_interp, lambda_c, K, window,
                                         sensor_res_fun, sU, extent_prior,
                                         sigma_extent_max, sigma_extent_min,
                                         extent_IW_df, extent_IW_scale, cor_extent_min, cor_extent_max)

    if extent_prior == "1d_uniform":
        Tilde_Phi = np.concatenate((Phi, np.expand_dims(Phi_E, axis=-1)), axis=1)
    elif extent_prior == "inverse-wishart" or extent_prior == "6d_uniform":
        Tilde_Phi = np.concatenate((Phi, Phi_E), axis=1)
    L_true = Tilde_Phi.shape[0]
    Psi = [np.concatenate((Psi_c[k], np.concatenate(Psi_da[k])), axis=0) for k in range(K)]

    # =============================================================================
    # MCMC hyperparameters
    # =============================================================================
    MCMC_runs = 10000

    list_ = ["Stochastic Marginal Maximum A Posteriori", "Metropolis-Hastings"]
    acceptance_type = list_[0]

    hparams = {"burn_in": 9000,
               "move_prob_inc_interval": 10,
               "move_prob_inc": 0.0,
               "move_prob_max": 0.9,
               "move_trace_interval": 20,
               "move_trace_min": 0.6,
               "move_trace_max": 0.9,
               "move_trace_decrease": 0.75,
               "move_trace_increase": 1.15}

    kwargs = {"p_m": 0.9,
              "p_em": 0.5,
              "vartheta": 0.5,
              "method": "modified",
              "object_extent": True,
              "partition_extent": 20,
              "extent_move_std": 1e-02,
              "acceptance_type": acceptance_type,
              "p_FA": 1e-03,
              "extent_prior": extent_prior,
              "sigma_extent_max": sigma_extent_max,
              "sigma_extent_min": sigma_extent_min,
              "extent_IW_df": extent_IW_df,
              "extent_IW_scale": extent_IW_scale,
              "cor_extent_min": cor_extent_min,
              "cor_extent_max": cor_extent_max,
              "sensor_resolution_function": sensor_res_fun,
              "sU": sU,
              "birth_density_v1": False,
              "patience": 500,
              "data_association_sim": "clutter",
              "verbose": True}

    # =============================================================================
    # Plot data
    # =============================================================================
    # Plot the observed data together with the true states
    color_list = ["tab:green", "tab:purple", "tab:orange", "tab:blue", "tab:red"]
    plt.plot(Phi[0, 0], Phi[0, 1], color="tab:red", marker="o", linestyle="", markersize=4, label="Targets")
    for l in range(1, Phi.shape[0]):
        plt.plot(Phi[l, 0], Phi[l, 1], color="tab:red", marker="o", linestyle="", markersize=4)
    for k in range(K):
        plt.plot(Psi[k][:, 0], Psi[k][:, 1],  color=color_list[k], marker="x", markersize=2.5, linestyle="", alpha=0.75, label=f"Sensor {k+1}")
    plt.xlim(window[0]-2, window[1]+2)
    plt.ylim(window[2]-2, window[3]+2)
    plt.legend()
    # plt.savefig("Results/MCMC/observed_data.png", dpi=500, bbox_inches="tight")
    plt.show()

    # =============================================================================
    # Run oracle
    # =============================================================================
    init_kwargs = {"extent": True,
                   "extent_prior": extent_prior,
                   "sigma_E_min": sigma_extent_min,
                   "sigma_E_max": sigma_extent_max,
                   "extent_IW_df": extent_IW_df,
                   "extent_IW_scale": extent_IW_scale,
                   "cor_E_min": cor_extent_min,
                   "cor_E_max": cor_extent_max}

    call_kwargs = {"prepare_data": True,
                   "true_marks": True,
                   "true_loc": True,
                   "Phi_true": Phi,
                   "Phi_E_true": Phi_E}

    oracle_alg = sensing_oracle(**init_kwargs)
    Tilde_Phi_oracle, Sigma_est, rho_est = oracle_alg(Psi_da, Sigma_interp, **call_kwargs)
    Phi_oracle, Phi_E_oracle = Tilde_Phi_oracle[:, :d], Tilde_Phi_oracle[:, d:]

    # =============================================================================
    # Run DBSCAN
    # =============================================================================
    init_kwargs = {"extent": True,
                   "extent_prior": extent_prior,
                   "sigma_E_min": sigma_extent_min,
                   "sigma_E_max": sigma_extent_max,
                   "extent_IW_df": extent_IW_df,
                   "extent_IW_scale": extent_IW_scale,
                   "cor_E_min": cor_extent_min,
                   "cor_E_max": cor_extent_max}

    call_kwargs = {"prepare_data": True,
                   "true_marks": False,
                   "true_extent": False,
                   "true_loc": False,
                   "Phi_true": Phi,
                   "Phi_E_true": Phi_E}

    p_FA = 1e-03
    optimize = True
    eps = [1.0, 1.5, 2.0, 2.5, 3.0]
    min_samples = [2,3,4,5,6,7,8]

    DBSCAN_alg = sensing_DBSCAN(rho_interp, Sigma_interp, sensor_res_fun,
                                sU, window, p_FA, optimize, **init_kwargs)
    Tilde_Phi_DBSCAN, Sigma_est, rho_est, _, _, _ = DBSCAN_alg(Psi, min_samples, eps, **call_kwargs)
    Phi_DBSCAN, Phi_E_DBSCAN = Tilde_Phi_DBSCAN[:, :d], Tilde_Phi_DBSCAN[:, d:]

    # =============================================================================
    # Run MCMC
    # =============================================================================
    mod = RJMCMC_DoublyNeymanScott(Psi, window, grid_dim, rho_interp, Sigma_interp, R, **kwargs)
    Phi_list, Extent_list, A_list, rho_list, Sigma_list, lambda_list, lambda_c_list, log_posterior_list, init_ = mod(MCMC_runs, **hparams)
    mod.run_profile()

    A_est = list()
    for k in range(K):
        A_k = np.zeros((len(A_list), A_list[0][0].shape[0]))
        for i in range(len(A_list)):
            A_k[i] = A_list[i][0]
        A_k_est = np.median(A_k, axis=0)
        A_est.append(A_k_est)

    ### Trace plot ###
    plt.plot(log_posterior_list, color="tab:green")
    plt.ylabel("log posterior")
    # plt.savefig("Results/MCMC/log_posterior.png", dpi=500, bbox_inches="tight")
    plt.show()

    ### Parameter estimation plots ###
    # Phi_card = [Phi_est.shape[0] for Phi_est in Phi_list]
    # plt.plot(Phi_card, color="tab:red")
    # plt.axhline(Phi.shape[0], color="tab:green")
    # plt.axhline(init_["Phi"].shape[0], color="tab:blue")
    # plt.ylabel("Estimated L")
    # plt.show()

    plt.hist(lambda_list, color="tab:red", bins=30, density=True)
    plt.axvline(lambda_, color="tab:green")
    plt.axvline(init_["lambda"], color="tab:blue")
    plt.title("Cluster center intensity")
    plt.show()

    # plt.plot(lambda_list, color="tab:red")
    # plt.axhline(lambda_, color="tab:green")
    # plt.axhline(init_["lambda"], color="tab:blue")
    # plt.ylabel("Cluster center intensity")
    # plt.show()

    plt.hist(lambda_c_list, color="tab:red", bins=30, density=True)
    plt.axvline(lambda_c, color="tab:green")
    plt.axvline(init_["lambda_c"], color="tab:blue")
    plt.title("Clutter intensity")
    plt.show()

    # plt.plot(lambda_c_list, color="tab:red")
    # plt.axhline(lambda_c, color="tab:green")
    # plt.axhline(init_["lambda_c"], color="tab:blue")
    # plt.ylabel("Clutter intensity")
    # plt.show()

    ### Acceptance probability history ###
    acceptance_history = mod.acceptance_history
    data = [acceptance_history["move"], acceptance_history["extent_move"], acceptance_history["birth"], acceptance_history["death"]]
    n_bins = 10
    fig, ax0 = plt.subplots()
    colors = ['tab:purple', 'tab:olive', 'tab:orange', 'tab:cyan']
    ax0.hist(data, n_bins, histtype='bar', color=colors, label=["move", "extent-move", "birth", "death"])
    ax0.legend(prop={'size': 10})
    # plt.savefig("Results/MCMC/acceptance_probability.png", dpi=500, bbox_inches="tight")
    plt.show()

    ### Target state estimation ###
    posterior_samples = np.array(Phi_list)

    Phi_est = np.mean(posterior_samples, axis=0)
    Extent_est = np.mean(np.array(Extent_list), axis=0)

    Phi_est_v2 = Phi_list[-1]
    Extent_est_v2 = Extent_list[-1]

    # plt.plot(Phi[:, 0], Phi[:, 1], color="tab:red", marker="o", markersize=6, linestyle="")
    # plt.plot(posterior_samples[:, :, 0].flatten(), posterior_samples[:, :, 1].flatten(),  color="tab:green", marker="x", markersize=2.5, linestyle="")
    # plt.xlim(window[0], window[1])
    # plt.ylim(window[2], window[3])
    # plt.show()

    plt.plot(Phi[0, 0], Phi[0, 1], color="tab:red", marker="o", linestyle="", markersize=4, label="Targets")
    plt.plot(Phi_DBSCAN[0, 0], Phi_DBSCAN[0, 1],  color="tab:green", marker="o", linestyle="", markersize=4, label="DBSCAN")
    for l in range(1, Phi.shape[0]):
        plt.plot(Phi[l, 0], Phi[l, 1], color="tab:red", marker="o", linestyle="", markersize=4)
    for l in range(1, Phi_DBSCAN.shape[0]):
        plt.plot(Phi_DBSCAN[l, 0], Phi_DBSCAN[l, 1],  color="tab:green", marker="o", linestyle="", markersize=4)
    plt.xlim(window[0]-2, window[1]+2)
    plt.ylim(window[2]-2, window[3]+2)
    # plt.savefig("Results/MCMC/DBSCAN_estimate.png", dpi=500, bbox_inches="tight")
    plt.show()

    plt.plot(Phi[0, 0], Phi[0, 1], color="tab:red", marker="o", linestyle="", markersize=4, label="Targets")
    plt.plot(Phi_est_v2[0, 0], Phi_est_v2[0, 1],  color="tab:green", marker="o", linestyle="", markersize=4, label="MCMC")
    for l in range(1, Phi.shape[0]):
        plt.plot(Phi[l, 0], Phi[l, 1], color="tab:red", marker="o", linestyle="", markersize=4)
    for l in range(1, Phi_est_v2.shape[0]):
        plt.plot(Phi_est_v2[l, 0], Phi_est_v2[l, 1],  color="tab:green", marker="o", linestyle="", markersize=4)
    plt.xlim(window[0]-2, window[1]+2)
    plt.ylim(window[2]-2, window[3]+2)
    # plt.savefig("Results/MCMC/MCMC_estimate.png", dpi=500, bbox_inches="tight")
    plt.show()

    if extent_prior == "1d_uniform":
        Tilde_Phi_est = np.concatenate((Phi_est, np.expand_dims(Extent_est, axis=-1)), axis=-1)
        Tilde_Phi_est_v2 = np.concatenate((Phi_est_v2, np.expand_dims(Extent_est_v2, axis=-1)), axis=-1)
    elif extent_prior == "inverse-wishart" or extent_prior == "6d_uniform":
        Tilde_Phi_est = np.concatenate((Phi_est, Extent_est), axis=-1)
        Tilde_Phi_est_v2 = np.concatenate((Phi_est_v2, Extent_est_v2), axis=-1)

    print("True   Cardinality :", Phi.shape[0])
    print("DBSCAN Cardinality :", Phi_DBSCAN.shape[0])
    print("MCMC   Cardinality :", Phi_est.shape[0])
    print("Oracle Cardinality :", Phi_oracle.shape[0])
    print("")
    print("DBSCAN  GOSPA :", GOSPA(Phi, Phi_DBSCAN, c=10))
    print("MCMC    GOSPA :", GOSPA(Phi, Phi_est, c=10))
    print("MCMC_v2 GOSPA :", GOSPA(Phi, Phi_est_v2, c=10))
    print("Oracle  GOSPA :", GOSPA(Phi, Phi_oracle, c=10))
    print("")
    print("DBSCAN  OSPA :", OSPA(Phi, Phi_DBSCAN, c=10))
    print("MCMC    OSPA :", OSPA(Phi, Phi_est, c=10))
    print("MCMC_v2 OSPA :", OSPA(Phi, Phi_est_v2, c=10))
    print("Oracle  OSPA :", OSPA(Phi, Phi_oracle, c=10))
    print("")
    print("DBSCAN  Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_DBSCAN, c=10))
    print("MCMC    Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_est, c=10))
    print("MCMC_v2 Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_est_v2, c=10))
    print("Oracle  Gaussian-Wasserstein :", OSPA_extent(Tilde_Phi, Tilde_Phi_oracle, c=10))
    print("")


if __name__ == "__main__":
    main()

