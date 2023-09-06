# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:00:13 2023

@author: Martin Voigt Vejling
Email: mvv@math.aau.dk

Track changes:
    v1.0 - Copy class methods from MCMC.py.
"""


import numpy as np
from scipy.stats import invwishart


class DIGSNCP_base(object):
    """
    Base class for doubly inhomogeneous generalized shot noise Cox processes.
    This class provides functionality to evaluate the cluster size and
    spread functions, the intensity, the log-likelihood, the log-prior,
    and the unnormalized log-posterior.
    """
    def __init__(self):
        """
        """
        super(DIGSNCP_base, self).__init__()

    def compute_rho(self, Phi):
        """
        Estimator of the scale parameter (measurement rate) in
        cluster center positions in Phi.
        """
        if len(Phi.shape) == 1:
            rho_i = np.zeros(self.K)
            for k in range(self.K):
                rho_i[k] = np.clip(self.rho[k](Phi)[0], self.p_FA, None)
        elif len(Phi.shape) == 2:
            rho_i = np.zeros((self.K, Phi.shape[0]))
            for k in range(self.K):
                rho_i[k, :] = np.clip(self.rho[k](Phi), self.p_FA, None)
        return rho_i

    def compute_Sigma(self, Phi):
        """
        Estimator of the bandwidth (cluster spread) in
        cluster center positions in Phi.
        """
        if len(Phi.shape) == 1:
            Sigma = np.zeros((self.K, self.d, self.d))
            for k in range(self.K):
                Sigma[k, :, :] = self.Sigma[k](Phi)[0]
        elif len(Phi.shape) == 2:
            Sigma = np.zeros((self.K, Phi.shape[0], self.d, self.d))
            for k in range(self.K):
                Sigma[k, :, :, :] = self.Sigma[k](Phi)
        return Sigma

    def compute_rho_and_Sigma(self, Phi):
        """
        Wrapper method for compute_rho() and compute_Sigma().
        """
        rho = self.compute_rho(Phi)
        Sigma = self.compute_Sigma(Phi)
        return rho, Sigma

    def log_likelihood(self, rho, lambda_c, eta):
        """
        Computing the log-likelihood.
        """
        ell = 0
        for k in range(self.K):
            ell += (1-lambda_c)*self.area-np.sum(rho[k]) + np.sum(np.log(lambda_c + np.sum(eta[k], axis=1)))
        return ell

    def log_prior(self, Phi, Extent, lambda_):
        """
        Computing the logarithm of the prior probability.
        """
        L = Phi.shape[0]
        ground_process_prior = L*np.log(lambda_) if L > 0 else -1e50
        if self.extent_prior == "inverse-wishart":
            extent_process_prior = np.sum([np.log(max(self.inverse_wishart_density(Extent[l]), 1e-150)) for l in range(L)])
            return ground_process_prior + extent_process_prior
        else:
            return ground_process_prior

    def log_posterior(self, Phi, Extent, rho, lambda_, lambda_c, eta):
        """
        Computing the logarithm of the posterior probability
        """
        if Phi.shape[0] == 0:
            return -1e50
        else:
            return self.log_likelihood(rho, lambda_c, eta) + self.log_prior(Phi, Extent, lambda_)

    def kernel(self, in_, rho_k_Phi, Sigma_k_Phi):
        """
        Evaluate the Gaussian kernel function in each of the points in_
        with size parameters rho and bandwidths Sigma.

        Input:
        ------
            in_ : ndarray, size=(d,) or size=(M_k, d) or size=(M_k, L, d)
                Point to evaluate the kernel in.
            rho_k_Phi : float or float or ndarray of size=(L,)
                Measurement rate for the k-th sensor in point(s) Phi.
            Sigma_k_Phi : ndarray, size=(d, d) or size=(d, d) or size=(L, d, d)
                Sensor covariance for the k-th sensor in point(s) Phi.

        Output:
        -------
            res : float or ndarray of size=(M_k,) or size=(M_k, L)
                Kernel evaluated in M points.
        """
        inv = np.linalg.inv(Sigma_k_Phi)
        if len(in_.shape) == 1:
            with np.errstate(invalid="ignore"):
                det = max(np.linalg.det(Sigma_k_Phi), 1e-02)
            prod = np.einsum("c,c->", in_, np.einsum("cd,d->c", inv, in_))
            # prod = np.dot(in_, np.linalg.solve(Sigma_k_Phi, in_))
            res = rho_k_Phi * 1/(np.sqrt(2*np.pi)**self.d*np.sqrt(det))*np.exp(-1/2*prod)
        elif len(in_.shape) == 2:
            with np.errstate(invalid="ignore"):
                det = max(np.linalg.det(Sigma_k_Phi), 1e-02)
            prod = np.einsum("mc,mc->m", in_, np.einsum("cd,md->mc", inv, in_))
            # prod = np.dot(in_, np.linalg.solve(Sigma_k_Phi, in_.T).T)
            res = rho_k_Phi * 1/(np.sqrt(2*np.pi)**self.d*np.sqrt(det))*np.exp(-1/2*prod)
        elif len(in_.shape) == 3:
            with np.errstate(invalid="ignore"):
                det = np.maximum(np.linalg.det(Sigma_k_Phi), np.ones(Sigma_k_Phi.shape[0])*1e-02)
            prod = np.einsum("mgc,mgc->mg", in_, np.einsum("gcd,mgd->mgc", inv, in_))
            # prod = np.einsum("mgc,gcm->mg", in_, np.linalg.solve(Sigma_k_Phi, np.moveaxis(in_, 0, -1)))
            res = rho_k_Phi[None, :] * 1/(np.sqrt(2*np.pi)**self.d*np.sqrt(det)[None, :])*np.exp(-1/2*prod)
        return res

    def cluster_intensity(self, Phi, Psi_k, rho_k_Phi, Sigma_k_Phi):
        """
        Compute the cluster intensity in each of the points Psi-Phi.

        Input:
        ------
            Phi : ndarray, size=(d,) or size=(L, d)
                Point(s) to evaluate intensity.
            Psi_k : ndarray, size=(M_k, d)
                The k-th sensor measurements.
            rho_k_Phi : ndarray, size=(,) or size=(L,)
                Measurement rate for the k-th sensor in point(s) Phi.
            Sigma_k_Phi : ndarray, size=(d, d) or size=(L, d, d)
                Sensor covariance function for the k-th sensor in point(s) Phi.

        Output:
            res : ndarray, size=(M_k,) or size=(M_k, L)
                Evaluated cluster intensity function in each of
                the M (or M x L) points.
        """
        if len(Phi.shape) == 1:
            in_ = Psi_k-Phi[None,:] # size=(M_k, d)
        elif len(Phi.shape) == 2:
            in_ = Psi_k[:, None, :]-Phi[None, :, :] # size=(M_k, L, d)
        res = self.kernel(in_, rho_k_Phi, Sigma_k_Phi)
        return res

    def inverse_wishart_density(self, extent_par):
        """
        Parameters
        ----------
        extent_par : ndarray, size=(6, )
            The extent parameter vector.

        Returns
        -------
        pdf_E : float
            The value of the density.
        """
        # Convert extent parameter vectors to covariance matrices
        E_mat = np.zeros((self.d, self.d))
        E_mat[self.tril_idx] = extent_par
        Sigma_E = E_mat @ E_mat.T
        pdf_E = invwishart.pdf(Sigma_E, self.extent_IW_df, self.extent_IW_scale)
        return pdf_E

    def extent_model(self, extent_par, Sigma, rho, sensor_resolution):
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
        if self.extent_prior == "1d_uniform":
            E_mat = extent_par * np.eye(self.d)
        elif self.extent_prior == "inverse-wishart" or self.extent_prior == "6d_uniform":
            E_mat = np.zeros((3, 3))
            E_mat[self.tril_idx] = extent_par

        if len(Sigma.shape) == 2:
            Sigma = Sigma + E_mat @ E_mat.T
        elif len(Sigma.shape) > 2:
            Sigma = Sigma + (E_mat @ E_mat.T)[None, :, :]

        rho = rho * np.maximum(np.ones(rho.shape), np.linalg.det(E_mat) * sensor_resolution)
        return Sigma, rho