# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:23:59 2022

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implemented the computation of the Fisher information matrix for
           the three refined signals. (21/12/2022)
    v1.1 - Implemented equivalent Fisher information computation as well as
           transformation to location parameters. Adopted units in
           nanoseconds, GHz, and nanowatts instead of seconds, Hz and watts.
           Simplified computations of the Fisher information matrix is in
           progress. (12/01/2022)
    v1.2 - Fixed the mistake on v1.1. Now the units are in seconds for delay
           and watts for path coefficient. However, angles are in nanoradians.
    v1.3 - Edited the Fisher information class to accomodate an additional
           axis for the time epochs. (19/01/2023)
    v1.4 - Bugfixing Fisher information computations. Improving computation
           efficiency for computing Fisher information. Changing position
           transformation to fit the spherical coordinate system convention
           used. (15/03/2023)
    v1.5 - Memory and computationally efficient implementations of the
           Fisher information matrix computations. (23/03/2023)
    v1.6 - Restructuring. Now main functionality is done in __call__() rather
           than __init__(). (04/04/2023)
    v2.0 - Refactoring: Methods do not rely on channel parameters, precoder,
           combiner, RIS phase profiles, and SPs and UEs positions to be
           provided as attributes from inheritance. Instead, these are
           inputs to the relevant methods. (08/05/2023)
    v2.1 - New compute_EFIM() method. Bugfix old
           compute_EFIM() method. (09/05/2023)
    v2.2 - Fixed array response definition -> changes to derivatives. (31/05/2023)
    v2.3 - Now supports odd AND even number of subcarriers. (09/06/2023)
"""

import sys
import os
if os.path.abspath("..")+"/Modules" not in sys.path:
    sys.path.append(os.path.abspath("..")+"/Modules")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import line_profiler
profile = line_profiler.LineProfiler()

import warnings

from channel import channel


class Fisher_Information(channel):
    """
    Compute the Fisher information matrix for the channel model in the
    inheritance class "channel". Then compute the equivalent Fisher
    for the localization relevant channel parameters, transform this into
    the spatial domain and compute the Cramer-Rao lower bound for position.
    This is for the scattering point positions (p_x, p_y, p_z).
    Also computes the position error bound (PEB).

    Methods:
    --------
        __init__ : Initialize settings.
        __call__ : Main function call to compute the Cramer-Rao lower bound.
        FIM_main : Compute the CRLB using Fisher analysis.
        FIM_processing : Map the Fisher information matrix to the Cramer-Rao lower bound.
        array_weighted_norm_diag :
            Used in Fisher_information_matrix_nonRIS_simplified.
        array_weighted_norm :
            Used in Fisher_information_matrix_directional_simplified.
        array_weighted_inner_prod_diag :
            Used in Fisher_information_matrix_orthogonal_simplified.
        Fisher_information_matrix_nonRIS_simplified :
            Compute the Fisher information matrix for the non-RIS signal.
        Fisher_information_matrix_directional_simplified :
            Compute the Fisher information matrix for the directional signal.
        Fisher_information_matrix_orthogonal_simplified :
            Compute the Fisher information matrix for the orthogonal signal.
        rearrange_FIM : Rearrange the dimensions of the FIM.
        compute_EFIM : Compute the equivalent Fisher information.
        compute_EFIM_transformation :
            Transform the equivalent Fisher information for the channel
            parameters to the Fisher information for the location parameters.
        compute_CRLB : Compute the Cramér-Rao lower bound for position.
        compute_PEB : Compute the position error bound.
        FIM_colorplot : Make an image plot of a Fisher information matrix.

    Attributes:
    -----------
        delta_theta : Conversion from radians to nanoradians.
    """
    def __init__(self, config_file=None, verbose=False, save_verbose=False, **kwargs):
        """
        Initializes channel model from inheritance.
        """
        super(Fisher_Information, self).__init__(config_file, verbose, save_verbose, **kwargs)

        # FIM attributes
        # self.delta_theta = 1e09 # rad -> nanorad conversion
        self.delta_theta = 1e00 #
        self.delta_tau = 1e-09 # s -> mus conversion
        self.delta_alpha = self.delta_tau # amp conversion

    @profile
    def FIM_main(self, pars, f_tilde, omega_tilde, W_mat, p, s_U, type_="nonRIS"):
        """
        Main method. Simulates the channel model, computes the Fisher
        information matrix and the Cramer-Rao lower bound.

        Input
        -----
        pars : dict
            Parameter dictionary.
        f_tilde : ndarray, size=(T/2, K, E, N_U)
            Precoder.
        omega_tilde : ndarray, size=(T/2, N_R)
            RIS phase profile.
        W_mat : ndarray, size=(K, E, N_U, N_U)
            Combiner.
        p : ndarray, size=(L, 3)
            SP positions.
        s_U : ndarray, size=(K, E, 5)
            UE state.
        type_ : str
            The type of signal. Options are 'nonRIS', 'directional',
            and 'orthogonal'. The default is 'nonRIS'.

        Returns
        -------
        CRLB : ndarray, size=(K, E, L, 3, 3), dtype=np.float64
            Cramer-Rao lower bound.
        """
        # Compute Fisher information
        if type_ == "nonRIS":
            FIM = self.Fisher_information_matrix_nonRIS_simplified(pars, f_tilde)
        elif type_ == "directional":
            FIM = self.Fisher_information_matrix_directional_simplified(pars, omega_tilde, W_mat)
        elif type_ == "orthogonal":
            FIM = self.Fisher_information_matrix_orthogonal_simplified(pars, f_tilde, omega_tilde, W_mat)

        if self.verbose is True:
            self.FIM_colorplot(FIM, type_=type_)
            print(f"\n{type_}")
            for k in range(self.K):
                for tau in range(self.E):
                    eigs = np.linalg.eigvals(FIM[k, tau, :, :])
                    eigs_abs = np.abs(eigs)
                    print(f"Condition Number k={k}, tau={tau}: {np.max(eigs_abs)/np.min(eigs_abs)}")
                    print(f"Eigenvalues k={k}, tau={tau}: {eigs}")

        CRLB = self.FIM_processing(FIM, pars, p, s_U[:, :, :3], s_U[:, :, 3:])
        return CRLB

    def __call__(self, SP_in=None, type_="nonRIS"):
        """
        Class call method. Return the result of FIM_main method.
        """
        # Initialize channel model
        p, s_U = self.initialise_target_and_UE_states(SP_in=SP_in)
        pars = self.initialise_channel_params(p, s_U)
        f_tilde, omega_tilde, W_mat = self.construct_RIS_phase_precoder_combiner(pars["theta0"])
        return self.FIM_main(pars, f_tilde, omega_tilde, W_mat, p, s_U, type_)

    # def compute_FIM(self):
    #     """
    #     Compute the Fisher information matrices for the three refined signals.

    #     Returns
    #     -------
    #     FIM_nonRIS : ndarray, size=(K, E, 5L, 5L), dtype=np.float64
    #         Fisher information matrix non-RIS.
    #     FIM_directional : ndarray, size=(K, E, 7L, 7L), dtype=np.float64
    #         Fisher information matrix directional.
    #     FIM_orthogonal : ndarray, size=(K, E, 7L, 7L), dtype=np.float64
    #         Fisher information matrix orthogonal.
    #     """
    #     if self.simplified is False:
    #         FIM_nonRIS = self.Fisher_information_matrix_nonRIS()
    #         FIM_directional = self.Fisher_information_matrix_directional()
    #         FIM_orthogonal = self.Fisher_information_matrix_orthogonal()
    #     elif self.simplified is True:
    #         FIM_nonRIS = self.Fisher_information_matrix_nonRIS_simplified()
    #         FIM_directional = self.Fisher_information_matrix_directional_simplified()
    #         FIM_orthogonal = self.Fisher_information_matrix_orthogonal_simplified()
    #     return FIM_nonRIS, FIM_directional, FIM_orthogonal

    @profile
    def FIM_processing(self, FIM, pars, p, p_U, o_U):
        """
        Assigning attributes by running the methods in this class.
        """
        EFIM = self.compute_EFIM(FIM, pars)
        # if EFIM.shape[-1] == 3:
        #     print(EFIM[0, 0, 0])
        Jx = self.compute_EFIM_transformation(EFIM, p, p_U, o_U)
        CRLB = self.compute_CRLB(Jx)
        return CRLB

    # def Fisher_information_matrix_nonRIS(self):
    #     """
    #     Compute the Fisher information matrix without using simplified
    #     expressions for the non-RIS path.
    #     """
    #     aU_thetal = self.UPA_array_response_vector(self.thetal, self.N_U, self.d_U) # size=(K, E, L, N_U)
    #     dn_overline_taul = self.delay_steering_vector(self.overline_taul, self.n) # size=(N+1, K, E, L)
    
    #     aU_pre_prod = np.einsum("keli,tkei->tkel", aU_thetal, self.f_tilde) # size=(T, K, E, L)

    #     # Coefficients and delay
    #     pdv_Re_betal = dn_overline_taul[None, :, :, :, :, None]*aU_thetal[None, None, :, :, :, :] \
    #                     * aU_pre_prod[:, None, :, :, :, None] # size=(T/2, N+1, K, E, L, N_U)
    #     pdv_Im_betal = 1j*pdv_Re_betal # size=(T/2, N+1, K, E, L, N_U)
    #     pdv_overline_taul = -1j*2*np.pi*self.n[None, :, None, None, None, None]*self.delta_f \
    #                         * self.betal[None, None, :, :, :, None] * pdv_Re_betal # size=(T/2, N+1, K, E, L, N_U)

    #     # Azimuth
    #     diag_az = np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_az = -1j*4*np.pi*self.d_U/self.lambda_*np.sin(self.thetal[:, :, :, 0, None]) \
    #                         * diag_az[None, None, None, :] * aU_thetal # size=(K, E, L, N_U)
    #     pdv_thetal_az = self.delta_theta*self.betal[None, None, :, :, :, None] \
    #                     * pdv_aU_thetal_az[None, None, :, :, :, :] * aU_pre_prod[:, None, :, :, :, None] \
    #                     * dn_overline_taul[None, :, :, :, :, None]

    #     # Elevation
    #     diag_el = np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_el = 1j*4*np.pi*self.d_U/self.lambda_*np.cos(self.thetal[:, :, :, 1, None]) \
    #                         * diag_el[None, None, None, :] * aU_thetal # size=(K, E, L, N_U)
    #     pdv_thetal_el = self.delta_theta*self.betal[None, None, :, :, :, None] \
    #                     * pdv_aU_thetal_el[None, None, :, :, :, :] * aU_pre_prod[:, None, :, :, :, None] \
    #                     * dn_overline_taul[None, :, :, :, :, None]

    #     # Collect partial derivatives in an array
    #     pdv_x = np.concatenate((pdv_Re_betal, pdv_Im_betal, pdv_overline_taul,
    #                 pdv_thetal_az, pdv_thetal_el), axis=4) # size=(T/2, N+1, K, E, 5*L, N_U)

    #     # Compute the FIM
    #     JN = 4/self.p_noise_sc * np.sum(np.real(np.einsum("tnkeli,tnkepi->tnkelp",
    #                         np.conjugate(pdv_x), pdv_x)), axis=(0, 1)) # size=(K, E, 5*L, 5*L)
    #     return JN

    def array_weighted_norm_diag(self, a, b):
        """
        Compute the weighted inner product
                a^H B a
        assuming that B = diag(b) as
                a^H (b * a)
        where * denotes element-wise multiplication along the last axis.
        Compute this inner product for every combination along the second to
        last axes of a.

        Input
        -----
            a : ndarray, size=(K, E, L, N_U)
            b : ndarray, size=(N_U)

        Output
        ------
            out : ndarray, size=(K, E, L, L)
        """
        out = np.einsum("keli,kepi->kelp", np.conjugate(a), b[None, None, None, :]*a)
        return out

    def Fisher_information_matrix_nonRIS_simplified(self, pars, f_tilde):
        """
        Simplified Fisher information matrix for the non-RIS signal.
        This method is memory and computational efficient.
        """
        # Frequency terms
        tau_diff = pars["overline_taul"][:, :, None, :] - pars["overline_taul"][:, :, :, None] # size=(K, E, L, L)
        if self.N % 2 == 0: # even number of subcarriers
            npos = np.arange(1, self.N//2) # size=(N/2-1)
            p_freq = -self.N//2
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) \
                + np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) \
                + p_freq**2 * np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) \
                + -p_freq*np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
        else: # odd number of subcarriers
            npos = np.arange(1, (self.N-1)//2+1) # size=(N/2)
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) # size=(K, E, L, L)

        # Array vector inner products
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)
        array_inner_prod = self.array_weighted_norm_diag(aU_thetal, np.ones(np.prod(self.N_U))) # size=(K, E, L, L)
        array_inner_prod_NUaz = self.array_weighted_norm_diag(aU_thetal, np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0]))) # size=(K, E, L, L)
        array_inner_prod_NUazSquared = self.array_weighted_norm_diag(aU_thetal, np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])**2)) # size=(K, E, L, L)
        array_inner_prod_NUel = self.array_weighted_norm_diag(aU_thetal, np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0]))) # size=(K, E, L, L)
        array_inner_prod_NUelSquared = self.array_weighted_norm_diag(aU_thetal, np.kron(np.arange(self.N_U[1])**2, np.ones(self.N_U[0]))) # size=(K, E, L, L)
        array_inner_prod_NUelNUaz = self.array_weighted_norm_diag(aU_thetal, np.kron(np.arange(self.N_U[1]), np.arange(self.N_U[0]))) # size=(K, E, L, L)
        array_inner_prod_NUazNUel = self.array_weighted_norm_diag(aU_thetal, np.kron(np.arange(self.N_U[0]), np.arange(self.N_U[1]))) # size=(K, E, L, L)

        # Angle terms
        az_term = np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUaz - np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUel
        el_term = np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUaz + np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUel
        az_az_term = np.sin(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazSquared \
                     + np.cos(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelSquared \
                     - np.sin(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelNUaz \
                     - np.cos(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazNUel
        az_el_term = np.sin(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazSquared \
                     - np.cos(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelSquared \
                     + np.sin(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelNUaz \
                     - np.cos(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazNUel
        el_el_term = np.cos(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazSquared \
                     + np.sin(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelSquared \
                     + np.cos(pars["thetal"][:, :, :, None, 0]) * np.sin(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUelNUaz \
                     + np.sin(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 0]) * array_inner_prod_NUazNUel

        # Temporal terms
        z_tilde_factor = np.einsum("keli,tkei->tkel", aU_thetal, f_tilde) # size=(T/2, K, E, L)
        z_tilde = np.conjugate(z_tilde_factor)[:, :, :, :, None]*z_tilde_factor[:, :, :, None, :] # size=(T/2, K, E, L, L)
        real_z_tilde_temp = np.sum(np.real(z_tilde),axis=0) # size=(K, E, L, L)
        real_z_tilde = np.triu(real_z_tilde_temp, k=0) + np.transpose(np.triu(real_z_tilde_temp, k=1), (0, 1, 3, 2)) # size=(K, E, L, L)
        imag_z_tilde_temp = np.sum(np.imag(z_tilde),axis=0) # size=(K, E, L, L)
        imag_z_tilde = np.triu(imag_z_tilde_temp, k=1) - np.transpose(np.triu(imag_z_tilde_temp, k=1), (0, 1, 3, 2)) # size=(K, E, L, L)

        # Compute sub-matrices
        real_real = self.delta_alpha**2 * np.real(delay_cos*array_inner_prod) * real_z_tilde - np.imag(delay_cos*array_inner_prod) * imag_z_tilde
        real_imaginary = - self.delta_alpha**2 * np.imag(delay_cos*array_inner_prod) * real_z_tilde + np.real(delay_cos*array_inner_prod) * imag_z_tilde
        real_delay = -self.delta_alpha * self.delta_tau*2*np.pi*self.delta_f * (np.real(delay_nsin * pars["betal"][:, :, None, :] * array_inner_prod) \
                                * real_z_tilde - np.imag(delay_nsin * pars["betal"][:, :, None, :] * array_inner_prod) * imag_z_tilde)
        real_azimuth = -self.delta_alpha * self.delta_theta*4*np.pi*np.sin(pars["thetal"][:, :, None, :, 1]) \
                       * self.d_U/self.lambda_ * (np.imag(delay_cos * pars["betal"][:, :, None, :] * az_term) * real_z_tilde \
                                                  + np.real(delay_cos * pars["betal"][:, :, None, :] * az_term) * imag_z_tilde)
        real_elevation = self.delta_alpha * self.delta_theta*4*np.pi*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ * (np.imag(delay_cos * pars["betal"][:, :, None, :] * el_term) * real_z_tilde \
                                       + np.real(delay_cos * pars["betal"][:, :, None, :] * el_term) * imag_z_tilde)

        imaginary_imaginary = real_real
        imaginary_delay = -self.delta_alpha * self.delta_tau*2*np.pi*self.delta_f * (np.imag(delay_nsin * pars["betal"][:, :, None, :] * array_inner_prod) * real_z_tilde \
                                                              + np.real(delay_nsin * pars["betal"][:, :, None, :] * array_inner_prod) * imag_z_tilde)
        imaginary_azimuth = self.delta_alpha * self.delta_theta*4*np.pi*np.sin(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ * (np.real(delay_cos * pars["betal"][:, :, None, :] * az_term) * real_z_tilde \
                                       - np.imag(delay_cos * pars["betal"][:, :, None, :] * az_term) * imag_z_tilde)
        imaginary_elevation = -self.delta_alpha * self.delta_theta*4*np.pi*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ * (np.real(delay_cos * pars["betal"][:, :, None, :] * el_term) * real_z_tilde \
                                       - np.imag(delay_cos * pars["betal"][:, :, None, :] * el_term) * imag_z_tilde)

        delay_delay = self.delta_tau**2 * (2*np.pi*self.delta_f)**2 \
            * (np.real(delay_nncos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * array_inner_prod) * real_z_tilde \
                - np.imag(delay_nncos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * array_inner_prod) * imag_z_tilde)
        delay_azimuth = -self.delta_tau*self.delta_theta*8*np.pi**2*self.delta_f*np.sin(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * (np.imag(delay_nsin * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_term) * real_z_tilde \
                + np.real(delay_nsin * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_term) * imag_z_tilde)
        delay_elevation = self.delta_tau*self.delta_theta*8*np.pi**2*self.delta_f*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * (np.imag(delay_nsin * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * el_term) * real_z_tilde \
                + np.real(delay_nsin * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * el_term) * imag_z_tilde)

        azimuth_azimuth = self.delta_theta**2*(4*np.pi*self.d_U/self.lambda_)**2 * np.sin(pars["thetal"][:, :, :, None, 1]) * np.sin(pars["thetal"][:, :, None, :, 1]) \
            * (np.real(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_az_term) * real_z_tilde \
                - np.imag(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_az_term) * imag_z_tilde)
        azimuth_elevation = -self.delta_theta**2*(4*np.pi*self.d_U/self.lambda_)**2 * np.sin(pars["thetal"][:, :, :, None, 0]) * np.cos(pars["thetal"][:, :, None, :, 1]) \
            * (np.real(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_el_term) * real_z_tilde \
                - np.imag(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * az_el_term) * imag_z_tilde)

        elevation_elevation = self.delta_theta**2*(4*np.pi*self.d_U/self.lambda_)**2 * np.cos(pars["thetal"][:, :, :, None, 1]) * np.cos(pars["thetal"][:, :, None, :, 1]) \
            * (np.real(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * el_el_term) * real_z_tilde \
                - np.imag(delay_cos * np.conjugate(pars["betal"][:, :, :, None])*pars["betal"][:, :, None, :] * el_el_term) * imag_z_tilde)

        # Combine sub-matrices
        combined_mat_col1 = np.concatenate((real_real, np.transpose(real_imaginary, (0, 1, 3, 2)), np.transpose(real_delay, (0, 1, 3, 2)), np.transpose(real_azimuth, (0, 1, 3, 2)), np.transpose(real_elevation, (0, 1, 3, 2))), axis=2)
        combined_mat_col2 = np.concatenate((real_imaginary, imaginary_imaginary, np.transpose(imaginary_delay, (0, 1, 3, 2)), np.transpose(imaginary_azimuth, (0, 1, 3, 2)), np.transpose(imaginary_elevation, (0, 1, 3, 2))), axis=2)
        combined_mat_col3 = np.concatenate((real_delay, imaginary_delay, delay_delay, np.transpose(delay_azimuth, (0, 1, 3, 2)), np.transpose(delay_elevation, (0, 1, 3, 2))), axis=2)
        combined_mat_col4 = np.concatenate((real_azimuth, imaginary_azimuth, delay_azimuth, azimuth_azimuth, np.transpose(azimuth_elevation, (0, 1, 3, 2))), axis=2)
        combined_mat_col5 = np.concatenate((real_elevation, imaginary_elevation, delay_elevation, azimuth_elevation, elevation_elevation), axis=2)
        combined_mat = np.concatenate((combined_mat_col1, combined_mat_col2, combined_mat_col3, combined_mat_col4, combined_mat_col5), axis=3)

        # FIM
        JN = 4/self.p_noise_sc * combined_mat
        return JN

    # def Fisher_information_matrix_directional(self):
    #     """
    #     Compute the Fisher information matrix without using simplified
    #     expressions for the directional signal.
    #     """
    #     aU_thetal = self.UPA_array_response_vector(self.thetal, self.N_U, self.d_U) # size=(K, E, L, N_U)
    #     aR_phi0 = self.UPA_array_response_vector(self.phi0, self.N_R, self.d_R) # size=(K, E, N_R)
    #     aR_phil = self.UPA_array_response_vector(self.phil, self.N_R, self.d_R) # size=(L, N_R)
    #     dn_taul = self.delay_steering_vector(self.taul, self.n) # size=(N+1, K, E, L)
    #     W_perp = self.W_mat[:, :, :, 1:] # size=(K, E, N_U, N_U-1)
    #     # nu_tilde_T1 = np.einsum("li,tkei->tkel", aR_phil,
    #     #         np.einsum("tij,kej -> tkei", self.Omega_tilde, aR_phi0))[:self.T1, :, :, :] # size=(T1, K, E, L)
    #     aR_phi0_prod = self.omega_tilde[:self.T1, None, None, :]*aR_phi0[None, :, :, :]
    #     nu_tilde_T1 = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod) # size=(T1, K, E, L)

    #     comb_aU_prod = np.einsum("keij,keli->kelj", np.conjugate(W_perp), aU_thetal) # size=(K, E, L, N_U-1)

    #     # Coefficients and delay
    #     pdv_Re_alphal = np.sqrt(np.prod(self.N_U)) * nu_tilde_T1[:, None, :, :, :, None] \
    #         * comb_aU_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_U-1)
    #     pdv_Im_alphal = 1j*pdv_Re_alphal # size=(T1, N+1, K, E, L, N_U)
    #     pdv_taul = -1j*2*np.pi*self.delta_f*np.sqrt(np.prod(self.N_U))*self.n[None, :, None, None, None, None] \
    #              * self.alphal[None, None, :, :, :, None] * nu_tilde_T1[:, None, :, :, :, None] \
    #              * comb_aU_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_U-1)

    #     # Azimuth at RIS
    #     diag_az_RIS = np.kron(np.ones(self.N_R[1]), np.arange(self.N_R[0])) # size=(N_R)
    #     pdv_nu_tilde_T1_phil_az = np.einsum("li,tkei->tkel", aR_phil, diag_az_RIS[None, None, None, :]*aR_phi0_prod) # size=(T1, K, E, L)
    #     pdv_phil_az = -self.delta_theta*1j*2*np.pi*np.sin(self.phil[:, 0])[None, None, None, None, :, None]*self.d_R/self.lambda_*np.sqrt(np.prod(self.N_U)) \
    #         * self.alphal[None, None, :, :, :, None] * pdv_nu_tilde_T1_phil_az[:, None, :, :, :, None] \
    #         * comb_aU_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_R-1)

    #     # Elevation at RIS
    #     diag_el_RIS = np.kron(np.arange(self.N_R[1]), np.ones(self.N_R[0])) # size=(N_R)
    #     pdv_nu_tilde_T1_phil_el = np.einsum("li,tkei->tkel", aR_phil, diag_el_RIS[None, None, None, :]*aR_phi0_prod) # size=(T1, K, E, L)
    #     pdv_phil_el = self.delta_theta*1j*2*np.pi*np.cos(self.phil[:, 1])[None, None, None, None, :, None]*self.d_R/self.lambda_*np.sqrt(np.prod(self.N_U)) \
    #         * self.alphal[None, None, :, :, :, None] * pdv_nu_tilde_T1_phil_el[:, None, :, :, :, None] \
    #         * comb_aU_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_R-1)

    #     # Azimuth at UE
    #     diag_az = np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_az = -1j*2*np.pi*self.d_U/self.lambda_*np.sin(self.thetal[:, :, :, 0, None]) \
    #                         * diag_az[None, None, None, :] * aU_thetal # size=(K, E, L, N_U)
    #     comb_pdv_aU_thetal_az_prod = np.einsum("keij,keli->kelj", np.conjugate(W_perp), pdv_aU_thetal_az) # size=(K, E, L, N_U-1)
    #     pdv_thetal_az = self.delta_theta*np.sqrt(np.prod(self.N_U)) * self.alphal[None, None, :, :, :, None] * nu_tilde_T1[:, None, :, :, :, None] \
    #         * comb_pdv_aU_thetal_az_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_U-1)

    #     # Elevation at UE
    #     diag_el = np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_el = 1j*2*np.pi*self.d_U/self.lambda_*np.cos(self.thetal[:, :, :, 1, None]) \
    #                         * diag_el[None, None, None, :] * aU_thetal # size=(K, E, L, N_U)
    #     comb_pdv_aU_thetal_el_prod = np.einsum("keij,keli->kelj", np.conjugate(W_perp), pdv_aU_thetal_el) # size=(K, E, L, N_U-1)
    #     pdv_thetal_el = self.delta_theta*np.sqrt(np.prod(self.N_U)) * self.alphal[None, None, :, :, :, None] * nu_tilde_T1[:, None, :, :, :, None] \
    #         * comb_pdv_aU_thetal_el_prod[None, None, :, :, :, :] * dn_taul[None, :, :, :, :, None] # size=(T1, N+1, K, E, L, N_U-1)

    #     # Collect partial derivatives in an array
    #     pdv_x = np.concatenate((pdv_Re_alphal, pdv_Im_alphal, pdv_taul, pdv_thetal_az,
    #                         pdv_thetal_el, pdv_phil_az, pdv_phil_el), axis=4) # size=(T1, N+1, K, E, 7*L, N_U-1)

    #     # Compute the FIM
    #     JD = 4/self.p_noise_sc * np.sum(np.real(np.einsum("tnkeli,tnkepi->tnkelp",
    #                         np.conjugate(pdv_x), pdv_x)), axis=(0, 1)) # size=(K, E, 7*L, 7*L)
    #     return JD

    def array_weighted_norm(self, a, B):
        """
        Compute the weighted inner product
                a^H B a
        Compute this inner product for every combination along the second to
        last axes of a.

        Input
        -----
            a : ndarray, size=(K, E, L, N_U)
            B : ndarray, size=(K, E, N_U, N_U)

        Output
        ------
            out : ndarray, size=(K, E, L, L)
        """
        right = np.einsum("keij,kepj->kepi", B, a)
        out = np.einsum("keli,kepi->kelp", np.conjugate(a), right)
        return out

    def Fisher_information_matrix_directional_simplified(self, pars, omega_tilde, W_mat):
        """
        Simplified Fisher information matrix for the directional signal.
        This method is memory and computational efficient.
        """
        # Frequency terms
        tau_diff = pars["taul"][:, :, None, :] - pars["taul"][:, :, :, None] # size=(K, E, L, L)
        if self.N % 2 == 0: # even number of subcarriers
            npos = np.arange(1, self.N//2) # size=(N/2-1)
            p_freq = -self.N//2
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) \
                + np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) \
                + p_freq**2 * np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) \
                + -p_freq*np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
        else: # odd number of subcarriers
            npos = np.arange(1, (self.N-1)//2+1) # size=(N/2)
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) # size=(K, E, L, L)

        # Array vector inner products
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)
        W_term = np.einsum("keij,kezj->keiz", W_mat[:, :, :, 1:], np.conjugate(W_mat[:, :, :, 1:])) # size=(K, E, N_U, N_U)
        az_vec = np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])) # size=(N_U)
        el_vec = np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0])) # size=(N_U)
        inner_prod = self.array_weighted_norm(aU_thetal, W_term) # size=(K, E, L, L)
        inner_prod_0az = self.array_weighted_norm(aU_thetal, W_term*az_vec[None, None, None, :]) # size=(K, E, L, L)
        inner_prod_0el = self.array_weighted_norm(aU_thetal, W_term*el_vec[None, None, None, :]) # size=(K, E, L, L)
        inner_prod_az0 = self.array_weighted_norm(aU_thetal, W_term*az_vec[None, None, :, None]) # size=(K, E, L, L)
        inner_prod_el0 = self.array_weighted_norm(aU_thetal, W_term*el_vec[None, None, :, None]) # size=(K, E, L, L)
        inner_prod_azSquared = self.array_weighted_norm(aU_thetal, W_term*az_vec[None, None, :, None]*az_vec[None, None, None, :]) # size=(K, E, L, L)
        inner_prod_elSquared = self.array_weighted_norm(aU_thetal, W_term*el_vec[None, None, :, None]*el_vec[None, None, None, :]) # size=(K, E, L, L)
        inner_prod_azel = self.array_weighted_norm(aU_thetal, W_term*az_vec[None, None, :, None]*el_vec[None, None, None, :]) # size=(K, E, L, L)

        # RIS/Temporal terms
        aR_phi0 = self.UPA_array_response_vector(pars["phi0"], self.N_R, self.d_R) # size=(K, E, N_R)
        aR_phil = self.UPA_array_response_vector(pars["phil"], self.N_R, self.d_R) # size=(L, N_R)
        az_vec_R = np.kron(np.ones(self.N_R[1]), np.arange(self.N_R[0])) # size=(N_R)
        el_vec_R = np.kron(np.arange(self.N_R[1]), np.ones(self.N_R[0])) # size=(N_R)
        aR_phi0_prod = omega_tilde[:self.T1, None, None, :]*aR_phi0[None, :, :, :] # size=(T1, K, E, N_R)
        nu_tilde = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod) # size=(T1, K, E, L)
        nu_tilde_az = np.einsum("li,tkei->tkel", aR_phil*az_vec_R, aR_phi0_prod) # size=(T1, K, E, L)
        nu_tilde_el = np.einsum("li,tkei->tkel", aR_phil*el_vec_R, aR_phi0_prod) # size=(T1, K, E, L)
        nu_tilde_term = np.sum(np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde[:, :, :, None, :], axis=0) # size=(K, E, L, L)
        nu_tilde_az_term = np.sum(np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde_az[:, :, :, None, :], axis=0) # size=(K, E, L, L)
        nu_tilde_el_term = np.sum(np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :], axis=0) # size=(K, E, L, L)
        nu_tilde_azaz_term = np.sum(np.conjugate(nu_tilde_az)[:, :, :, :, None] * nu_tilde_az[:, :, :, None, :], axis=0) # size=(K, E, L, L)
        nu_tilde_azel_term = np.sum(np.conjugate(nu_tilde_az)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :], axis=0) # size=(K, E, L, L)
        nu_tilde_elel_term = np.sum(np.conjugate(nu_tilde_el)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :], axis=0) # size=(K, E, L, L)

        # Compute sub-matrices
        real_real = np.real(delay_cos*nu_tilde_term*inner_prod)
        real_imag = - np.imag(delay_cos*nu_tilde_term*inner_prod)
        real_delay = -2*np.pi*self.delta_f * np.real(delay_nsin * pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod)
        real_theta_az = -self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0az)
        real_theta_el = self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0el)
        real_phi_az = -self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_az_term * inner_prod)
        real_phi_el = self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_el_term * inner_prod)

        imag_imag = real_real
        imag_delay = -2*np.pi*self.delta_f * np.imag(delay_nsin*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod)
        imag_theta_az = self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0az)
        imag_theta_el = -self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0el)
        imag_phi_az = self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_az_term * inner_prod)
        imag_phi_el = -self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * nu_tilde_el_term * inner_prod)

        delay_delay = 4*np.pi**2*self.delta_f**2 \
            * np.real(delay_nncos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod)
        delay_theta_az = -self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0az)
        delay_theta_el = self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_0el)
        delay_phi_az = -self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
            * self.d_R/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_az_term * inner_prod)
        delay_phi_el = self.delta_theta*4*np.pi**2*self.delta_f*np.cos(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
            * self.d_R/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_el_term * inner_prod)

        theta_az_theta_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["thetal"][:, :, None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["thetal"][:, :, None, :, 1])*self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_azSquared)
        theta_az_theta_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["thetal"][:, :, None, :, 0])*self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_azel)
        theta_az_phi_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 1])*self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_az_term * inner_prod_az0)
        theta_az_phi_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 0])*self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_el_term * inner_prod_az0)

        theta_el_theta_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["thetal"][:, :, None, :, 0])*self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_term * inner_prod_elSquared)
        theta_el_phi_az = -self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 1])*self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_az_term * inner_prod_el0)
        theta_el_phi_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 0])*self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_el_term * inner_prod_el0)

        phi_az_phi_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["phil"][:, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["phil"][:, None, 1])*np.sin(pars["phil"][None, :, 1])*self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_azaz_term * inner_prod)
        phi_az_phi_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["phil"][:, None, 0])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["phil"][:, None, 1])*np.sin(pars["phil"][None, :, 0])*self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_azel_term * inner_prod)

        phi_el_phi_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["phil"][:, None, 1])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["phil"][:, None, 0])*np.sin(pars["phil"][None, :, 0])*self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * nu_tilde_elel_term * inner_prod)

        # Combine sub-matrices
        combined_mat_col1 = np.concatenate((real_real, np.transpose(real_imag, (0, 1, 3, 2)), np.transpose(real_delay, (0, 1, 3, 2)),
                                            np.transpose(real_theta_az, (0, 1, 3, 2)), np.transpose(real_theta_el, (0, 1, 3, 2)),
                                            np.transpose(real_phi_az, (0, 1, 3, 2)), np.transpose(real_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col2 = np.concatenate((real_imag, imag_imag, np.transpose(imag_delay, (0, 1, 3, 2)),
                                            np.transpose(imag_theta_az, (0, 1, 3, 2)), np.transpose(imag_theta_el, (0, 1, 3, 2)),
                                            np.transpose(imag_phi_az, (0, 1, 3, 2)), np.transpose(imag_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col3 = np.concatenate((real_delay, imag_delay, delay_delay, np.transpose(delay_theta_az, (0, 1, 3, 2)),
                                            np.transpose(delay_theta_el, (0, 1, 3, 2)), np.transpose(delay_phi_az, (0, 1, 3, 2)),
                                            np.transpose(delay_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col4 = np.concatenate((real_theta_az, imag_theta_az, delay_theta_az, theta_az_theta_az,
                                            np.transpose(theta_az_theta_el, (0, 1, 3, 2)), np.transpose(theta_az_phi_az, (0, 1, 3, 2)),
                                            np.transpose(theta_az_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col5 = np.concatenate((real_theta_el, imag_theta_el, delay_theta_el, theta_az_theta_el, theta_el_theta_el,
                                            np.transpose(theta_el_phi_az, (0, 1, 3, 2)), np.transpose(theta_el_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col6 = np.concatenate((real_phi_az, imag_phi_az, delay_phi_az, theta_az_phi_az, theta_el_phi_az,
                                            phi_az_phi_az, np.transpose(phi_az_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col7 = np.concatenate((real_phi_el, imag_phi_el, delay_phi_el, theta_az_phi_el, theta_el_phi_el, phi_az_phi_el, phi_el_phi_el), axis=2)
        combined_mat = np.concatenate((combined_mat_col1, combined_mat_col2, combined_mat_col3, combined_mat_col4,
                                       combined_mat_col5, combined_mat_col6, combined_mat_col7), axis=3)

        JD = 4/self.p_noise_sc * np.prod(self.N_U) * combined_mat
        return JD

    # def Fisher_information_matrix_orthogonal(self):
    #     """
    #     Compute the Fisher information matrix without using simplified
    #     expressions for the orthogonal signal.
    #     """
    #     aU_thetal = self.UPA_array_response_vector(self.thetal, self.N_U, self.d_U) # size=(K, E, L, N_U)
    #     aR_phi0 = self.UPA_array_response_vector(self.phi0, self.N_R, self.d_R) # size=(K, E, N_R)
    #     aR_phil = self.UPA_array_response_vector(self.phil, self.N_R, self.d_R) # size=(L, N_R)
    #     dn_taul = self.delay_steering_vector(self.taul, self.n) # size=(N+1, K, E, L)
    #     aR_phi0_prod = self.omega_tilde[self.T1:, None, None, :]*aR_phi0[None, :, :, :]
    #     nu_tilde_T2 = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod) # size=(T2, K, E, L)
    #     f_tilde_T2 = self.f_tilde[self.T1:, :, :, :] # size=(T2, K, E, N_U)

    #     aU_pre_prod = np.einsum("keli,tkei->tkel", aU_thetal, f_tilde_T2) # size=(T2, K, E, L)

    #     # Coefficients and delay
    #     pdv_Re_alphal = np.sqrt(np.prod(self.N_U)) * nu_tilde_T2[:, None, :, :, :] \
    #         * aU_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)
    #     pdv_Im_alphal = 1j*pdv_Re_alphal # size=(T2, N+1, K, E, L)
    #     pdv_taul = -1j*2*np.pi*self.delta_f*np.sqrt(np.prod(self.N_U))*self.n[None, :, None, None, None] \
    #               * self.alphal[None, None, :, :, :] * nu_tilde_T2[:, None, :, :, :] \
    #               * aU_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)

    #     # Azimuth at RIS
    #     diag_az_RIS = np.kron(np.ones(self.N_R[1]), np.arange(self.N_R[0])) # size=(N_R)
    #     pdv_nu_tilde_T2_phil_az = np.einsum("li,tkei->tkel", aR_phil, diag_az_RIS[None, None, None, :]*aR_phi0_prod) # size=(T2, K, E, L)
    #     pdv_phil_az = -self.delta_theta*1j*2*np.pi*np.sin(self.phil[:, 0])[None, None, None, None, :]*self.d_R/self.lambda_*np.sqrt(np.prod(self.N_U)) \
    #         * self.alphal[None, None, :, :, :] * pdv_nu_tilde_T2_phil_az[:, None, :, :, :] \
    #         * aU_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)

    #     # Elevation at RIS
    #     diag_el_RIS = np.kron(np.arange(self.N_R[1]), np.ones(self.N_R[0])) # size=(N_R)
    #     pdv_nu_tilde_T2_phil_el = np.einsum("li,tkei->tkel", aR_phil, diag_el_RIS[None, None, None, :]*aR_phi0_prod) # size=(T2, K, E, L)
    #     pdv_phil_el = self.delta_theta*1j*2*np.pi*np.cos(self.phil[:, 1])[None, None, None, None, :]*self.d_R/self.lambda_*np.sqrt(np.prod(self.N_U)) \
    #         * self.alphal[None, None, :, :, :] * pdv_nu_tilde_T2_phil_el[:, None, :, :, :] \
    #         * aU_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)

    #     # Azimuth at UE
    #     diag_az = np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_az = -1j*2*np.pi*self.d_U/self.lambda_*np.sin(self.thetal[:, :, :, 0, None]) \
    #                         * aU_thetal * diag_az[None, None, None, :]
    #     pdv_aU_az_pre_prod = np.einsum("keli,tkei->tkel", pdv_aU_thetal_az, f_tilde_T2) # size=(T2, K, E, L)
    #     pdv_thetal_az = self.delta_theta*np.sqrt(np.prod(self.N_U)) * self.alphal[None, None, :, :, :] * nu_tilde_T2[:, None, :, :, :] \
    #         * pdv_aU_az_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)

    #     # Elevation at UE
    #     diag_el = np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0])) # size=(N_U)
    #     pdv_aU_thetal_el = 1j*2*np.pi*self.d_U/self.lambda_*np.cos(self.thetal[:, :, :, 1, None]) \
    #                         * diag_el[None, None, None, :] * aU_thetal # size=(K, E, L, N_U)
    #     pdv_aU_el_pre_prod = np.einsum("keli,tkei->tkel", pdv_aU_thetal_el, f_tilde_T2) # size=(T2, K, E, L)
    #     pdv_thetal_el = self.delta_theta*np.sqrt(np.prod(self.N_U)) * self.alphal[None, None, :, :, :] * nu_tilde_T2[:, None, :, :, :] \
    #         * pdv_aU_el_pre_prod[:, None, :, :, :] * dn_taul[None, :, :, :, :] # size=(T2, N+1, K, E, L)

    #     # Collect partial derivatives in an array
    #     pdv_x = np.concatenate((pdv_Re_alphal, pdv_Im_alphal, pdv_taul, pdv_thetal_az,
    #                         pdv_thetal_el, pdv_phil_az, pdv_phil_el), axis=4) # size=(T2, N+1, K, E, 7*L)

    #     # Compute the FIM
    #     JO = 4/self.p_noise_sc * np.sum(np.real(np.einsum("tnkel,tnkep->tnkelp",
    #                         np.conjugate(pdv_x), pdv_x)), axis=(0, 1)) # size=(K, E, 7*L, 7*L)
    #     return JO

    def array_weighted_inner_prod_diag(self, a, b, c):
        """
        Compute the weighted inner product
                a^T B c
        assuming that B = diag(b) as
                a^T (b * c)
        where * denotes element-wise multiplication along the last axis.

        Input
        -----
            a : ndarray, size=(K, E, L, N_U)
            b : ndarray, size=(N_U)
            c : ndarray, size=(T, K, E, N_U)

        Output
        ------
            out : ndarray, size=(T, K, E, L)
        """
        out = np.einsum("keli,tkei->tkel", a, b[None, None, None, :]*c)
        return out

    def Fisher_information_matrix_orthogonal_simplified(self, pars, f_tilde, omega_tilde, W_mat):
        """
        Simplified Fisher information matrix for the orthogonal signal.
        This method is memory and computational efficient.
        """
        # Frequency terms
        tau_diff = pars["taul"][:, :, None, :] - pars["taul"][:, :, :, None] # size=(K, E, L, L)
        if self.N % 2 == 0: # even number of subcarriers
            npos = np.arange(1, self.N//2) # size=(N/2-1)
            p_freq = -self.N//2
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) \
                + np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) \
                + p_freq**2 * np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) \
                + -p_freq*np.exp(-1j*2*np.pi*p_freq*self.delta_f*tau_diff) # size=(K, E, L, L)
        else: # odd number of subcarriers
            npos = np.arange(1, (self.N-1)//2+1) # size=(N/2)
            delay_cos = 1 + 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1), axis=0) # size=(K, E, L, L)
            delay_nncos = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.cos, c=1)*(npos**2)[:, None, None, None, None], axis=0) # size=(K, E, L, L)
            delay_nsin = 2*np.sum(self.delay_steering_vector(tau_diff, npos, fun=np.sin, c=1)*npos[:, None, None, None, None], axis=0) # size=(K, E, L, L)

        # RIS terms
        aR_phi0 = self.UPA_array_response_vector(pars["phi0"], self.N_R, self.d_R) # size=(K, E, N_R)
        aR_phil = self.UPA_array_response_vector(pars["phil"], self.N_R, self.d_R) # size=(L, N_R)
        az_vec_R = np.kron(np.ones(self.N_R[1]), np.arange(self.N_R[0])) # size=(N_R)
        el_vec_R = np.kron(np.arange(self.N_R[1]), np.ones(self.N_R[0])) # size=(N_R)
        aR_phi0_prod = omega_tilde[self.T1:, None, None, :]*aR_phi0[None, :, :, :] # size=(T2, K, E, N_R)
        nu_tilde = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod) # size=(T2, K, E, L)
        nu_tilde_az = np.einsum("li,tkei->tkel", aR_phil*az_vec_R, aR_phi0_prod) # size=(T2, K, E, L)
        nu_tilde_el = np.einsum("li,tkei->tkel", aR_phil*el_vec_R, aR_phi0_prod) # size=(T2, K, E, L)
        nu_tilde_term = np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde[:, :, :, None, :] # size=(T2, K, E, L, L)
        nu_tilde_az_term = np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde_az[:, :, :, None, :] # size=(T2, K, E, L, L)
        nu_tilde_el_term = np.conjugate(nu_tilde)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :] # size=(T2, K, E, L, L)
        nu_tilde_azaz_term = np.conjugate(nu_tilde_az)[:, :, :, :, None] * nu_tilde_az[:, :, :, None, :] # size=(T2, K, E, L, L)
        nu_tilde_azel_term = np.conjugate(nu_tilde_az)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :] # size=(T2, K, E, L, L)
        nu_tilde_elel_term = np.conjugate(nu_tilde_el)[:, :, :, :, None] * nu_tilde_el[:, :, :, None, :] # size=(T2, K, E, L, L)

        # Precoder terms
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)
        az_vec = np.kron(np.ones(self.N_U[1]), np.arange(self.N_U[0])) # size=(N_U)
        el_vec = np.kron(np.arange(self.N_U[1]), np.ones(self.N_U[0])) # size=(N_U)
        z_tilde_factor = self.array_weighted_inner_prod_diag(aU_thetal, np.ones(np.prod(self.N_U)), f_tilde[self.T1:, :, :, :]) # size=(T2, K, E, L)
        z_tilde_az_factor = self.array_weighted_inner_prod_diag(aU_thetal, az_vec, f_tilde[self.T1:, :, :, :]) # size=(T2, K, E, L)
        z_tilde_el_factor = self.array_weighted_inner_prod_diag(aU_thetal, el_vec, f_tilde[self.T1:, :, :, :]) # size=(T2, K, E, L)
        z_tilde = np.conjugate(z_tilde_factor)[:, :, :, :, None]*z_tilde_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_0az = np.conjugate(z_tilde_factor)[:, :, :, :, None]*z_tilde_az_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_az0 = np.conjugate(z_tilde_az_factor)[:, :, :, :, None]*z_tilde_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_0el = np.conjugate(z_tilde_factor)[:, :, :, :, None]*z_tilde_el_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_el0 = np.conjugate(z_tilde_el_factor)[:, :, :, :, None]*z_tilde_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_azaz = np.conjugate(z_tilde_az_factor)[:, :, :, :, None]*z_tilde_az_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_azel = np.conjugate(z_tilde_az_factor)[:, :, :, :, None]*z_tilde_el_factor[:, :, :, None, :] # size=(T2, K, E, L, L)
        z_tilde_elel = np.conjugate(z_tilde_el_factor)[:, :, :, :, None]*z_tilde_el_factor[:, :, :, None, :] # size=(T2, K, E, L, L)

        # Temporal terms
        temporal = np.sum(z_tilde*nu_tilde_term, axis=0) # size=(K, E, L, L)

        # Compute sub-matrices
        real_real = np.real(delay_cos*temporal)
        real_imag = - np.imag(delay_cos*temporal)
        real_delay = -2*np.pi*self.delta_f * np.real(delay_nsin * pars["alphal"][:, :, None, :] * temporal)
        real_theta_az = -self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.imag(delay_cos * pars["alphal"][:, :, None, :] * np.sum(z_tilde_0az*nu_tilde_term, axis=0))
        real_theta_el = self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde_0el*nu_tilde_term, axis=0))
        real_phi_az = -self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_az_term, axis=0))
        real_phi_el = self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.imag(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_el_term, axis=0))

        imag_imag = real_real
        imag_delay = -2*np.pi*self.delta_f * np.imag(delay_nsin*pars["alphal"][:, :, None, :] * temporal)
        imag_theta_az = self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde_0az*nu_tilde_term, axis=0))
        imag_theta_el = -self.delta_theta*2*np.pi*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
                        * self.d_U/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde_0el*nu_tilde_term, axis=0))
        imag_phi_az = self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_az_term, axis=0))
        imag_phi_el = -self.delta_theta*2*np.pi*np.sin(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
                        * self.d_R/self.lambda_ \
                        * np.real(delay_cos*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_el_term, axis=0))

        delay_delay = 4*np.pi**2*self.delta_f**2 \
            * np.real(delay_nncos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * temporal)
        delay_theta_az = -self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["thetal"][:, :, None, :, 0])*np.sin(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_0az*nu_tilde_term, axis=0))
        delay_theta_el = self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["thetal"][:, :, None, :, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_0el*nu_tilde_term, axis=0))
        delay_phi_az = -self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["phil"][None, :, 0])*np.sin(pars["phil"][None, :, 1]) \
            * self.d_R/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_az_term, axis=0))
        delay_phi_el = self.delta_theta*4*np.pi**2*self.delta_f*np.sin(pars["phil"][None, :, 0])*np.cos(pars["phil"][None, :, 1]) \
            * self.d_R/self.lambda_ \
            * np.imag(delay_nsin*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_el_term, axis=0))

        theta_az_theta_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["thetal"][:, :, None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["thetal"][:, :, None, :, 1]) \
            * self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_azaz*nu_tilde_term, axis=0))
        theta_az_theta_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["thetal"][:, :, None, :, 0]) \
            * self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_azel*nu_tilde_term, axis=0))
        theta_az_phi_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 1]) \
            * self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_az0*nu_tilde_az_term, axis=0))
        theta_az_phi_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["thetal"][:, :, :, None, 0])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 0]) \
            * self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_az0*nu_tilde_el_term, axis=0))

        theta_el_theta_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.cos(pars["thetal"][:, :, None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["thetal"][:, :, None, :, 0]) \
            * self.d_U**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_elel*nu_tilde_term, axis=0))
        theta_el_phi_az = -self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 1]) \
            * self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_el0*nu_tilde_az_term, axis=0))
        theta_el_phi_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["thetal"][:, :, :, None, 1])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["thetal"][:, :, :, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * self.d_U*self.d_R/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde_el0*nu_tilde_el_term, axis=0))

        phi_az_phi_az = self.delta_theta**2*4*np.pi**2*np.sin(pars["phil"][:, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * np.sin(pars["phil"][:, None, 1])*np.sin(pars["phil"][None, :, 1]) \
            * self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_azaz_term, axis=0))
        phi_az_phi_el = -self.delta_theta**2*4*np.pi**2*np.sin(pars["phil"][:, None, 0])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["phil"][:, None, 1])*np.sin(pars["phil"][None, :, 0]) \
            * self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_azel_term, axis=0))

        phi_el_phi_el = self.delta_theta**2*4*np.pi**2*np.cos(pars["phil"][:, None, 1])*np.cos(pars["phil"][None, :, 1]) \
            * np.sin(pars["phil"][:, None, 0])*np.sin(pars["phil"][None, :, 0]) \
            * self.d_R**2/self.lambda_**2 \
            * np.real(delay_cos*np.conjugate(pars["alphal"])[:, :, :, None]*pars["alphal"][:, :, None, :] * np.sum(z_tilde*nu_tilde_elel_term, axis=0))

        # Combine sub-matrices
        combined_mat_col1 = np.concatenate((real_real, np.transpose(real_imag, (0, 1, 3, 2)), np.transpose(real_delay, (0, 1, 3, 2)),
                                            np.transpose(real_theta_az, (0, 1, 3, 2)), np.transpose(real_theta_el, (0, 1, 3, 2)),
                                            np.transpose(real_phi_az, (0, 1, 3, 2)), np.transpose(real_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col2 = np.concatenate((real_imag, imag_imag, np.transpose(imag_delay, (0, 1, 3, 2)),
                                            np.transpose(imag_theta_az, (0, 1, 3, 2)), np.transpose(imag_theta_el, (0, 1, 3, 2)),
                                            np.transpose(imag_phi_az, (0, 1, 3, 2)), np.transpose(imag_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col3 = np.concatenate((real_delay, imag_delay, delay_delay, np.transpose(delay_theta_az, (0, 1, 3, 2)),
                                            np.transpose(delay_theta_el, (0, 1, 3, 2)), np.transpose(delay_phi_az, (0, 1, 3, 2)),
                                            np.transpose(delay_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col4 = np.concatenate((real_theta_az, imag_theta_az, delay_theta_az, theta_az_theta_az,
                                            np.transpose(theta_az_theta_el, (0, 1, 3, 2)), np.transpose(theta_az_phi_az, (0, 1, 3, 2)),
                                            np.transpose(theta_az_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col5 = np.concatenate((real_theta_el, imag_theta_el, delay_theta_el, theta_az_theta_el, theta_el_theta_el,
                                            np.transpose(theta_el_phi_az, (0, 1, 3, 2)), np.transpose(theta_el_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col6 = np.concatenate((real_phi_az, imag_phi_az, delay_phi_az, theta_az_phi_az, theta_el_phi_az,
                                            phi_az_phi_az, np.transpose(phi_az_phi_el, (0, 1, 3, 2))), axis=2)
        combined_mat_col7 = np.concatenate((real_phi_el, imag_phi_el, delay_phi_el, theta_az_phi_el, theta_el_phi_el, phi_az_phi_el, phi_el_phi_el), axis=2)
        combined_mat = np.concatenate((combined_mat_col1, combined_mat_col2, combined_mat_col3, combined_mat_col4,
                                       combined_mat_col5, combined_mat_col6, combined_mat_col7), axis=3)

        JO = 4/self.p_noise_sc * np.prod(self.N_U) * combined_mat
        return JO

    # def compute_EFIM(self, FIM, pars):
    #     """
    #     Compute the equivalent Fisher information for the localization relevant
    #     parameters for each scatter point. This is done by rearranging the
    #     rows and columns of the Fisher information matrix such that the terms
    #     upper right sub-matrix expresses the Fisher information for the
    #     localization relevant parameters. Then the theorem for the inverse
    #     of a block matrix is exploited.
    #     """
    #     K, E, L = pars["alphal"].shape
    #     dim = FIM.shape[-1]//L
    #     use_dim = dim-2
    #     EFIM = np.zeros((K, E, L, use_dim, use_dim))
    #     for l in range(L):
    #         # Rearrange order
    #         rearranged_FIM = np.zeros(FIM.shape)
    #         list_ = [i for i in range(dim*L)]
    #         interest_list = [i*L+l for i in range(2, dim)]
    #         rearrange_list = interest_list + list(set(list_) - set(interest_list))
    #         rearranged_FIM[:, :, list_, :] = FIM[:, :, rearrange_list, :]
    #         rearranged_FIM[:, :, :, list_] = rearranged_FIM[:, :, :, rearrange_list]

    #         # Remove resolvable paths
    #         # if use_dim == 3:
    #         #     nonresolvable_paths = np.abs(self.overline_taul[:, :, :]-self.overline_taul[:, :, l, None]) <= self.BW_T
    #         # elif use_dim == 5:
    #         #     nonresolvable_paths = np.abs(self.taul[:, :, :]-self.taul[:, :, l, None]) <= self.BW_T
    #         if use_dim == 3:
    #             nonresolvable_paths = np.abs(pars["overline_taul"][:, :, :]-pars["overline_taul"][:, :, l, None]) <= 0
    #         elif use_dim == 5:
    #             nonresolvable_paths = np.abs(pars["taul"][:, :, :]-pars["taul"][:, :, l, None]) <= 0

    #         for k in range(K):
    #             for tau in range(E):
    #                 nonresolvable_list = list(np.argwhere(nonresolvable_paths[k, tau, :])[:, 0])
    #                 nonresolvable_list_without_l = list(set(nonresolvable_list) - set([l]))
    #                 if len(nonresolvable_list_without_l) > 0:
    #                     nonresolvable_arr = np.array([i for i in range(use_dim)]+[use_dim+i*L+j for i in range(2) for j in nonresolvable_list]\
    #                         +[use_dim+2*L+i*(L-1)+(j-1) if j > l else use_dim+2*L+i*(L-1)+j for i in range(use_dim) for j in nonresolvable_list_without_l])
    #                 else:
    #                     nonresolvable_arr = np.array([i for i in range(use_dim)]+[use_dim+i*L+j for i in range(2) for j in nonresolvable_list])
    #                 nonresolvable_FIM = rearranged_FIM[k, tau, nonresolvable_arr, :][:, nonresolvable_arr]
    #                 EFIM[k, tau, l, :, :] = nonresolvable_FIM[:use_dim, :use_dim] - np.dot(nonresolvable_FIM[:use_dim, use_dim:],
    #                                         np.linalg.solve(nonresolvable_FIM[use_dim:, use_dim:], nonresolvable_FIM[use_dim:, :use_dim]))
    #     return EFIM

    def rearrange_FIM(self, FIM, pars):
        """
        Rearrange the rows and columns of the Fisher information matrix to
        give a precision matrix independently for each point.
        """
        K, E, L = pars["alphal"].shape
        dim = FIM.shape[-1]//L
        reFIM = np.zeros((K, E, L, dim, dim))
        for l in range(L):
            for k in range(K):
                for tau in range(E):
                    idx_arr = np.array([i*L+l for i in range(dim)])
                    reFIM[k, tau, l, :, :] = FIM[k, tau, idx_arr, :][:, idx_arr]
        return reFIM

    def compute_EFIM(self, FIM, pars):
        """
        Compute the equivalent Fisher information for the localization relevant
        parameters for each scatter point. First rearrange the Fisher information
        matrix. Then exploit the theorem for the inverse of a block matrix.
        """
        K, E, L = pars["alphal"].shape
        dim = FIM.shape[-1]//L
        use_dim = dim-2
        reFIM = self.rearrange_FIM(FIM, pars)
        EFIM = np.zeros((K, E, L, use_dim, use_dim))
        for l in range(L):
            for k in range(K):
                for tau in range(E):
                    information_loss = np.dot(reFIM[k, tau, l, 2:, :2], np.linalg.solve(reFIM[k, tau, l, :2, :2], reFIM[k, tau, l, :2, 2:]))
                    # information_loss = np.dot(reFIM[k, tau, l, 2:, :2], reFIM[k, tau, l, :2, 2:]/np.diag(reFIM[k, tau, l, :2, :2])[:, None])
                    symmetric_information_loss = 1/2 * (information_loss + information_loss.T)
                    EFIM[k, tau, l, :, :] = reFIM[k, tau, l, 2:, 2:] - symmetric_information_loss
                    if np.any(np.linalg.eigvals(EFIM[k, tau, l, :, :]) <= 0):
                        EFIM[k, tau, l, :, :] *= np.eye(use_dim)
                    if np.any(np.diag(EFIM[k, tau, l, :, :]) <= 0):
                        EFIM[k, tau, l, :, :] += (np.min(np.diag(EFIM[k, tau, l, :, :])) + 1e-03)*np.eye(use_dim)
        # assert np.all(np.linalg.eigvals(EFIM[0, 0, 0, :, :]) > 0), "Equivalent Fisher information must be positive definite!"
        return EFIM

    def compute_EFIM_transformation(self, EFIM, p, p_U, o_U):
        """
        Based on the equivalent Fisher information matrix, compute the equivalent
        Fisher information matrix for the position. This is done using the
        Jacobian matrix.
        """
        K, E = p_U.shape[:2]
        use_dim = EFIM.shape[-1]
        distance_diff = p[None, None, :, :] - p_U[:, :, None, :] # size=(K, E, L, 3)
        O_U = np.zeros((K, E, 3, 3))
        for k in range(K):
            for tau in range(E):
                O_U[k, tau, :, :] = self.coord_rotation_matrix(*self.oU)

        p_SU = np.einsum("keij,keli->kelj", O_U, distance_diff) # size=(K, E, L, 3)
        p_SU_norm = np.linalg.norm(p_SU, axis=-1)

        dtheta_az = 1/self.delta_theta*(O_U[:, :, None, :, 1] * p_SU[:, :, :, 0, None] - O_U[:, :, None, :, 0] * p_SU[:, :, :, 1, None]) \
                    / ((p_SU**2)[:, :, :, 0, None] + (p_SU**2)[:, :, :, 1, None]) # size=(K, E, L, 3)
        dtheta_el = -1/self.delta_theta*(O_U[:, :, None, :, 2] * (p_SU_norm**2)[:, :, :, None] - p_SU[:, :, :, 2, None] * distance_diff) \
                / ((p_SU_norm**3)[:, :, :, None] * np.sqrt(1 - (p_SU**2)[:, :, :, 2, None]/(p_SU_norm**2)[:, :, :, None])) # size=(K, E, L, 3)
        if use_dim == 3:
            dtau_overline = 1/self.delta_tau * 2*distance_diff/(self.c*np.linalg.norm(distance_diff, axis=-1)[:, :, :, None]) # size=(K, E, L, 3)
            T = np.concatenate((dtau_overline[:, :, :, None, :], dtheta_az[:, :, :, None, :], dtheta_el[:, :, :, None, :]), axis=3) # size=(K, E, L, zdim, pdim)
        elif use_dim == 5:
            RIS_distance_diff = p - self.p_R[None, :] # size=(L, 3)
            O_R = self.coord_rotation_matrix(*self.oR) # size=(3, 3)
            # O_R = self.coord_rotation_matrix() # size=(3, 3)
            p_SR = np.einsum("ij,ki->kj", O_R, p - self.p_R[None, :]) # size=(L, 3)
            p_SR_norm = np.linalg.norm(p_SR, axis=1)

            dtau = RIS_distance_diff[None, None, :, :]/(self.c*np.linalg.norm(RIS_distance_diff, axis=1)[None, None, :, None]) \
                  + distance_diff/(self.c*np.linalg.norm(distance_diff, axis=-1)[:, :, :, None]) # size=(K, E, L, 3)
            dphi_az = 1/self.delta_theta*(O_R[None, :, 1] * p_SR[:, 0, None] - O_R[None, :, 0] * p_SR[:, 1, None]) \
                        / ((p_SR**2)[:, 0, None] + (p_SR**2)[:, 1, None]) # size=(L, 3)
            dphi_el = -1/self.delta_theta*(O_R[None, :, 2] * (p_SR_norm**2)[:, None] - p_SR[:, 2, None] * RIS_distance_diff) \
                    / ((p_SR_norm**3)[:, None] * np.sqrt(1 - (p_SR**2)[:, 2, None]/(p_SR_norm**2)[:, None])) # size=(L, 3)
            T = np.concatenate((dtau[:, :, :, None, :], dtheta_az[:, :, :, None, :], dtheta_el[:, :, :, None, :],
                                dphi_az[None, None, :, None, :]*np.ones((K, E))[:, :, None, None, None],
                                dphi_el[None, None, :, None, :]*np.ones((K, E))[:, :, None, None, None]), axis=3) # size=(K, E, L, zdim, pdim)

        Jx = np.einsum("kelin,kelim->kelnm", T, np.einsum("kelij,keljm->kelim", EFIM, T))
        # Jx = np.triu(Jx, k=0) + np.transpose(np.triu(Jx, k=1), (0, 1, 3, 2))
        return Jx

    @profile
    def compute_CRLB(self, Jx):
        """
        Compute the Cramér-Rao lower bound for the position estimator by taking
        the inverse of the equivalent Fisher information of the position.
        """
        CRLB = np.zeros_like(Jx)
        CRLB[:] = None
        K, E, L, _, _ = Jx.shape
        # for k in range(K):
        #     for kappa in range(E):
        #         for l in range(L):
        #             try:
        #                 CRLB[k, kappa, l, :, :] = np.linalg.inv(Jx[k, kappa, l, :, :])
        #             except np.linalg.LinAlgError:
        #                 print("Hej")
        #                 CRLB[k, kappa, l, :, :] = None

        # det = np.linalg.det(Jx)
        # CRLB[det>0] = np.linalg.inv(Jx[det>0])

        CRLB = np.linalg.pinv(Jx, hermitian=True)

        # CRLB = np.triu(CRLB, k=0) + np.transpose(np.triu(CRLB, k=1), (0, 1, 3, 2))
        return CRLB

    def compute_PEB(self, CRLB):
        """
        Compute the position error bound given the Cramér-Rao lower bound on the
        position estimator.
        """
        K, E, L, _, _ = CRLB.shape
        PEB = np.zeros((K, E, L))
        for k in range(K):
            for kappa in range(E):
                for l in range(L):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            PEB[k, kappa, l] = np.sqrt(np.trace(CRLB[k, kappa, l, :, :]))
                        except RuntimeWarning:
                            PEB[k, kappa, l] = None
        # PEB = np.sqrt(np.trace(CRLB, axis1=3, axis2=4))
        return PEB

    def FIM_colorplot(self, FIM, k=0, tau=0, savenames=["log_amp", "sign"], type_="nonRIS", norm=LogNorm()):
        """
        Plot the Fisher information matrix FIM at position index k.
        """
        L = FIM.shape[-1]
        x = np.arange(L)
        y = np.arange(L)

        plt.style.use("seaborn-v0_8")
        plt.rcParams["axes.grid"] = False

        spec = plt.pcolormesh(x, y, np.abs(FIM[k, tau, ::-1, :])+1e-01, cmap="cool",
                              shading="auto", norm=norm)
        cb = plt.colorbar(spec)
        cb.set_label(label="log(|FIM|+eps)")
        if type_ == "nonRIS":
            plt.title("$\eta=(\Re\{\\beta\}, \Im\{\\beta\}, \overline{\\tau}, \\theta_{az}, \\theta_{el})$")
        elif type_ == "directional" or type_ == "orthogonal":
            plt.title("$\eta=(\Re\{\\beta\}, \Im\{\\beta\}, \overline{\\tau}, \\phi_{az}, \\phi_{el}, \\theta_{az}, \\theta_{el})$")
        plt.savefig(f"Plots/FIM_plots/{savenames[0]}.png", dpi=500, bbox_inches="tight")
        plt.show()

        spec = plt.pcolormesh(x, y, np.sign(FIM[k, tau, ::-1, :]), cmap="cool",
                              shading="auto")
        cb = plt.colorbar(spec)
        cb.set_label(label="sign(FIM)")
        if type_ == "nonRIS":
            plt.title("$\eta=(\Re\{\\beta\}, \Im\{\\beta\}, \overline{\\tau}, \\theta_{az}, \\theta_{el})$")
        elif type_ == "directional" or type_ == "orthogonal":
            plt.title("$\eta=(\Re\{\\beta\}, \Im\{\\beta\}, \overline{\\tau}, \\phi_{az}, \\phi_{el}, \\theta_{az}, \\theta_{el})$")
        plt.savefig(f"Plots/FIM_plots/{savenames[1]}.png", dpi=500, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    np.random.seed(11)
    chn = Fisher_Information(config_file="channel_config.toml", verbose=True, save_verbose=False)
    CRLB_nonRIS, CRLB_directional, CRLB_orthogonal \
        = chn(type_="nonRIS"), chn(type_="directional"), chn(type_="orthogonal")

    profile.print_stats()
