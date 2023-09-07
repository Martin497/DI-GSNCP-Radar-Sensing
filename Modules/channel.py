# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:34:20 2022

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Initial construction of the signal and channel model. (16/12/2022)
    v1.1 - Fixed bug with ULA method. (20/12/2022)
    v1.2 - Re-organized such that RIS phase profile is saved as an
           attribute. (21/12/2022)
    v1.3 - Added verbose input parameter. Changed units from seconds to
           nanoseconds. Added more functionality to the method
           delay_steering_vector(). Corrected the energy levels of the
           received signals. Made prettier plots in plot_response()
           method. (12/01/2022)
    v1.4 - Fixed a mistake with energy levels introduced in v1.3.
           Corrected sign of rotation matrix. Changed how orientations of UEs
           are simulated. Now they all point in the same direction; towards
           the RIS. Also corrected the RIS orientation. The elevation angle
           uses arcsin instead of arccos due to how the geometry is made; the
           elevation angle I imagine is not the conventional one.
    v1.5 - Edited the channel class to accomodate an additional axis for the
           time epochs. (19/01/2023)
    v1.6 - Reduce memory demands by removing H and H_list from attributes. Add
           different options to simulate SP positions. Split the __init__() 
           functionality in to multiple methods to allow for more customizability
           in inheritance classes. Plot of beampatterns added.
    v1.7 - Change to construction of precoder and combiner after email
           exchange with Hyowon Kim. (06/03/2023)
    v1.8 - Fixed bug in 1.7 precoder. Also changed the spherical coordinate
           system convention which changes elevation angle to be computed
           by arccos instead of arcsin. With these changes the rotation matrix
           should be correctly defined. (14/03/2023)
    v1.9 - Fixed the coordinatsystem change bug.
           Now rotations to local coordinatesystems are properly defined and
           the sign of the rotation matrix is correct. (26/04/2023)
    v2.0 - Refactoring: Now the channel parameters and locations of SPs and
           UEs are no longer attributes. Instead, they are treated as local
           variables. This is done to avoid complex inheritance and
           in-place changes of attributes in daughter classes. Instead,
           these variables should be given as input to the relevant
           methods. (08/05/2023)
    v2.1 - Fix error in UPA response. There is a bug in the rotation between local and global
           coordinates. I'm ignoring this and only doing the special case with a fixed rotation.
    v2.2 - Refactoring of config file. Added directional RIS phase profiles. (09/06/2023)
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import toml

from matplotlib.colors import LogNorm

import line_profiler
profile = line_profiler.LineProfiler()

from PP_Sim import PoissonPointProcess


class channel(object):
    """
    Simulate the signal and channel model. In __init__ all attributes are
    assigned. The signal and channel model is described in

                "RIS-Aided Radar Sensing and Object Detection
                 with Single and Double Bounce Multipath" (2022)
                by Hyowon Kim, Alessio Fascista, Hui Chen,
                    Yu Ge, George C. Alexandropoulos,
                    Gonzalo Seco-Granados, Henk Wymeersch

    Methods:
        __init__ : Initialize settings and simulate received signal
        __str__ : Print class specifications
        __call__ : Main class method.
        load_toml_params : Load parameters from config.toml file.
        initialise_target_and_UE_states : Initialise the geometry.
        initialise_channel_params : Compute channel parameters.
        simulate_node_positions :
            Simulates a number of node positions random uniformly in the
            observation window
        delays_and_channel_coefficients : Compute delays and channel coefficients.
        angles : Compute azimuth and elevation angles.
        coord_rotation_matrix :
            Compute a 3D rotation matrix to shift from global to local
            coordinates given an azimuth and an elevation.
        channel_matrix : Compute the channel matrices.
        ULA_array_response_vector :
            Computes the array response vector for a uniform linear array.
        UPA_array_response_vector :
            Computes the array response vector for a uniform planar array.
        delay_steering_vector :
            Computes the delay steering vector.
        orthogonal_RIS : Construct the orthogonal RIS phase profile time sequence.
        RIS_random_codebook : Simulate random RIS phase profiles.
        signal_separation_and_refinement :
            Separate the signals originating from the different paths.
        simulate_received_signal : Simulate the received signal.
        precoder_time_sequence : Construct the precoder time sequence.
        construct_precoder : Construct the precoders.
        construct_combiner : Construct the combiner.
        plot_configuration : Plot the environment in 2D.
        plot_configuration_3D : Plot the environment in 3D.
        plot_response : Plot the frequency and impulse responses.
        plot_beampattern : Plot the directivity.
        beampattern_polar : Used in plot_beampattern.
        plot_beampattern_map : Plot the directivity in both azimuth and elevation.
        beampattern_map : Used in plot_beampattern_map.

    Attributes
    ----------
        toml_in : dict
            Dictionary containing input from config.toml.
        verbose : bool
            If true show plots.
        save_verbose : bool
            If true save plots.
        f_c : float
            Carrier frequency.
        p_tx_dbm : float
            Transmission power in dBm.
        p_tx : float
            Transmission power in W.
        p_tx_sc : float
            Transmission power per subcarrier.
        p_tx_sc_dbm : float
            Transmission power per subcarrier in dBm.
        p_noise_dbm : float
            Noise power in dBm.
        p_noise : float
            Noise power in W.
        p_noise_sc : float
            Noise power per subcarrier in W.
        p_noise_sc_dbm : float
            Noise power per subcarrier in dBm.
        N_U : list, len=(2,)
            List of number of antennas at the UEs.
        N_R : list, len=(2,)
            List of number of RIS elements.
        p_R : ndarray, size=(3,)
            RIS position.
        o_R_txt : list, len=(2,)
            RIS orientation as strings.
        o_R : ndarray, size=(2,)
            RIS orientation.
        n_R : ndarray, size=(3,)
            RIS normal vector.
        p : ndarray, size=(L, 3)
            SP positions.
        p_U : ndarray, size=(K, E, 3)
            UE positions.
        o_U : ndarray, size=(K, E, 2)
            UE orientations.
        W : float
            Bandwidth.
        N : int
            Number of subcarriers.
        n : ndarray, size=(N+1,)
            Array of subcarrier indices.
        c : float
            Speed of light.
        T : int
            Number of temporal samples.
        T1 : int
            Number of temporal samples towards the RIS.
        T2 : int
            Number of temporal samples with null towards the RIS.
        window : list, len=(6,)
            List of rectangular window bounds.
        K : int
            Number of UEs.
        L : int
            Number of SPs.
        seed : int
            Random seed.
        lambda_ : float
            Wavelength.
        delta_f : float
            Subcarrier frequency spacing.
        d_R : float
            Spacing between RIS elements.
        d_U : float
            Spacing between UE antennas.
    """
    def __init__(self, config_file=None, verbose=False, save_verbose=False, **kwargs):
        """
        Initializes channel.
        Parameters for the channel are loaded from the config.toml file.
        """
        # config_file = "Process_Data/config.toml"
        # load parameters from config file
        if config_file is not None:
            self.load_toml_params(config_file)
        else:
            for k in kwargs.keys():
                self.__setattr__(k, kwargs[k])
            self.p_R = np.array(self.p_R)
            self.n_R = np.array(self.n_R)
            self.T2 = int(self.T/2) - self.T1

        try:
            self.setup_attributes()
        except AttributeError:
            print("Input attributes are missing!")
            sys.exit()

        self.verbose = verbose
        self.save_verbose = save_verbose
        plt.style.use("ggplot")

        self.oU = (np.pi/2, np.pi/2, 0)
        self.oR = (np.pi/2, -np.pi/2, 0)

    def __call__(self):
        """
        Main method. Simulates the channel and signal model.
        """
        # compute channel parameters
        p, s_U = self.initialise_target_and_UE_states()
        pars = self.initialise_channel_params(p, s_U)
        f_tilde, omega_tilde, W_mat = self.construct_RIS_phase_precoder_combiner(pars["theta0"])

        if self.verbose is True:
            #self.plot_configuration()
            self.plot_configuration_3D(p, s_U[:, :, :3])

        if self.E==1 and self.K == 1:
            y, y_tilde_N, y_overline_D, y_overline_O = self.simulate_data(pars, W_mat, omega_tilde, f_tilde)
    
            if self.verbose is True:
                self.plot_response(y_tilde_N)
                self.plot_beampattern(pars["theta0"][0, 0, :], W_mat, f_tilde)
                self.plot_beampattern_map(pars["theta0"][0, 0, :], W_mat, f_tilde)

    def load_toml_params(self, config_file):
        """
        Load config.toml file and assign to attributes.
        """
        ### Load and interpret .toml file ###
        self.toml_in = toml.load(config_file)
        toml_settings = self.toml_in["settings"]
        for key, value in toml_settings.items():
            setattr(self, key, value)
        self.p_R = np.array(self.p_R)
        # self.n_R = np.array(self.n_R)
        # self.o_R = np.array([np.arctan2(self.n_R[1], self.n_R[0]), np.arccos(self.n_R[2]/np.linalg.norm(self.n_R))])
        #print(self.o_R)
        self.T2 = int(self.T/2) - self.T1
        assert self.T % 2 == 0, "T must be even"
        assert self.T1 < int(self.T/2), "T1 must be less than T/2."
        self.heading = eval(self.heading)

    def setup_attributes(self):
        """
        Define additional attributes from input attributes.
        """
        #### Frequency parameters ###
        self.W = self.delta_f*self.N # Bandwidth
        if self.N % 2 == 0:
            self.n = np.arange(-self.N//2, self.N//2) # n=-N/2,...,N/2-1
        else:
            self.n = np.arange(-(self.N-1)//2, (self.N-1)//2+1) # n=-(N-1)/2,...,(N-1)/2

        ### Power ###
        self.p_tx_hz = 10**(self.p_tx_hz_dbm/10 - 3) # transmit power per Hz in W
        self.p_tx_sc = self.p_tx_hz*self.delta_f # transmit power per sub-carrier in W
        self.p_tx_sc_dbm = 10*np.log10(self.p_tx_sc) + 30 # in dBm
        self.p_tx = self.p_tx_sc*self.N # total transmit power in W
        self.p_tx_dbm = 10*np.log10(self.p_tx) + 30 # in dBm
        tot_p_tx_dbm = 10*np.log10(self.p_tx*self.T*np.prod(self.N_U)) + 30
        print(f"Transmit power: {tot_p_tx_dbm:.2f} dBm")

        self.p_noise_hz = 10**(self.p_noise_hz_dbm/10 - 3) # noise power per Hz in W
        self.p_noise_sc = self.p_noise_hz*self.delta_f # noise power per Hz in W
        self.p_noise_sc_dbm = 10*np.log10(self.p_noise_sc) + 30 # in dBm
        self.p_noise = self.p_noise_sc*self.N # total noise power in W
        self.p_noise_dbm = 10*np.log10(self.p_noise) + 30 # in dBm

        self.SNR = self.p_tx_sc/self.p_noise_sc # signal-to-noise-ratio in W
        self.SNR_db = 10*np.log10(self.SNR)  # signal-to-noise-ratio in dB

        ### Wavelength and antenna array spacing ###
        self.lambda_ = self.c/self.f_c # Wavelength
        self.d_R = self.c/(self.f_c+self.W/2)/4 # Grating lobes at the RIS are avoided with antenna spacing lambda/4.
        self.d_U = self.c/(self.f_c+self.W/2)/2

    def initialise_target_and_UE_states(self, SP_in=None, UE_in=None):
        """
        Initialize the scatter point and user equipment positions.

        Inputs:
        -------
            SP_in : ndarray, size=(L, d), optional
                The scatter points positions. The default is None.
            UE_in : ndarray, size=(K, E, d), optional
                The user equipment positions. The default is None.

        Outputs:
        --------
            p : ndarray, size=(L, d)
                The scatter point positions.
            s_U : ndarray, size=(K, E, d)
                The user equipment states.
        """
        p, self.L = self.simulate_SP_positions(SP_in) # size=(L, 3)

        # Simulate UE positions
        p_U = self.simulate_UE_positions(self.K, self.E, self.heading, self.speed,
                                         self.epoch_time, UE_in) # size=(K, E, 3)
        self.K, self.E = p_U.shape[0:2]

        n_U = [0, 1, 0]
        o_U = np.repeat(np.repeat(np.array([
                            np.arctan2(n_U[1], n_U[0]),
                            np.arccos(n_U[2]/np.linalg.norm(n_U))
                                ])[None, :], self.K, axis=0)[:, None, :], self.E, axis=1)

        s_U = np.concatenate((p_U, o_U), axis=-1)
        return p, s_U

    def initialise_channel_params(self, p, s_U):
        """
        Initialize channel parameters.

        Inputs:
        -------
            p : ndarray, size=(L, d)
                The scatter point positions.
            s_U : ndarray, size=(K, E, d+2)
                The UE state given as (x, y, z, azimuth, elevation).

        Outputs:
        --------
            pars : dict
                Dictionary containing the channel parameters.
        """
        p_U = s_U[:, :, :3]
        o_U = s_U[:, :, 3:]

        # Compute the channel parameters
        tau0, taul, overline_taul, alpha0, alphal, betal = self.delays_and_channel_coefficients(p, p_U)
        theta0, thetal, phi0, phil = self.angles(p, p_U, o_U)
        pars = {"tau0": tau0,
                "taul": taul,
                "overline_taul": overline_taul,
                "alpha0": alpha0,
                "alphal": alphal,
                "betal": betal,
                "theta0": theta0,
                "thetal": thetal,
                "phi0": phi0,
                "phil": phil}
        return pars

    def construct_RIS_phase_precoder_combiner(self, theta0, p_U=None, pred_Map=None):
        """
        Construct the RIS phase profile, precoder, and combiner.

        Inputs:
        -------
            theta0 : ndarray, size=(K, 1, 2)
                The UE-RIS angle.
            p_U : ndarray, size=(K, 1, 3)
                The UE positions.
            pred_Map : dict
                Use as input to directional RIS phase profile.

        Outputs:
        --------
            f_tilde : ndarray, size=(T/2, K, E, N_U)
                Precoder for times t_tilde = 1,\dots,T/2.
            omega_tilde : ndarray, size=(T/2, N_R)
                RIS phase profile for times t_tilde = 1,\dots,T/2
            W_mat : ndarray, size=(K, E, N_U, N_U)
                Combiner matrix.
        """
        # Construct precoder
        f_tilde = self.construct_precoder(theta0) # size=(T/2, K, E, N_U)

        # Construct RIS phase profile
        if self.use_directional_RIS is True:
            omega_tilde = self.RIS_directional(p_U, pred_Map) # size=(T/2, N_R)
        else:
            omega_tilde = self.RIS_random_codebook()

        # Construct combiner
        W_mat = self.construct_combiner(theta0) # size=(K, E, N_U, N_U)
        return f_tilde, omega_tilde, W_mat

    def simulate_data(self, pars, W_mat, omega_tilde, f_tilde):
        """
        Construct the channel matrix and simulate received signals.
        """
        # Construct channel
        H = self.channel_matrix(pars, omega_tilde) # H size=(T, N+1, K, E, N_U, N_U)

        # Construct precoder
        f = self.precoder_time_sequence(f_tilde) # size=(T, K, E, N_U)

        # Simulate received signal
        y, eps = self.simulate_received_signal(H, f) # size=(T, N+1, K, E, N_U)

        # Separate the signals
        y_tilde_N, y_overline_D, y_overline_O = self.signal_separation_and_refinement(W_mat, H, f_tilde, eps)
        return y, y_tilde_N, y_overline_D, y_overline_O

    def __str__(self):
        """
        Gets information of channel.
        """
        s = f"""Cell in the area {self.window} m^3 with RIS at {self.p_R}.

Channel settings
----------------
Total transmit power             : {self.p_tx_dbm:10.1f} dBm
Transmit power per Hz            : {self.p_tx_hz_dbm:10.2f} dBm/Hz
Centre frequeny                  : {self.f_c:10.2e} Hz
Wavelength                       : {self.lambda_:10.2e} m
Bandwidth                        : {self.W:10.2e} Hz
Number of subcarriers            : {self.N:10d}
Frequency spacing                : {self.delta_f:10.2e} Hz
UE antennas                      : \t ({self.N_U[0]}, {self.N_U[1]})
RIS elements                     : \t ({self.N_R[0]}, {self.N_R[1]})

Environment
-----------
Total noise power                : {self.p_noise_dbm:10.1f} dBm
Noise power spectral density     : {self.p_noise_hz_dbm:10.1f} dBm/Hz
Transmit signal-to-noise-ratio   : {self.SNR_db:10.1f} dB
Number of scatter points         : {self.L:10d}

Measurement settings
--------------------
Number of users measurements     : {self.K:10d}
Number of time epochs            : {self.E:10d}
Number of temporal measurements  : {self.T:10d}
"""
        return s

    def simulate_SP_positions(self, SP_in=None):
        """
        Simulate position of scatter points.

        If SP_in is not None, then the scatterers are positions as specified by SP_in.

        Else if lambda_SP is not 0, then the SP points are simulated by a
        homogeneous Poisson point process with intensity lambda_SP.

        Else, L points are simulated uniformly in the window.
        """
        if SP_in is not None:
            p = SP_in
            L = SP_in.shape[0]
        elif self.lambda_SP != 0:
            p = PoissonPointProcess(self.window, self.lambda_SP).points
            L = p.shape[0]
        else:
            p = np.zeros((self.L, 3))
            for dim in range(3):
                p[:, dim] = np.random.uniform(self.window[dim*2], self.window[dim*2+1], size=self.L)
            L = self.L
        assert L > 0, "The modules assumes at least one scatter point!"
        return p, L

    def simulate_UE_positions(self, K, E, heading, speed, epoch_time, UE_in=None):
        """
        Simulate initial position of num nodes uniformly in the window. Then
        move each node in the heading direction in the xy-plane with the
        given speed for E time steps of length epoch_time.
        """
        if UE_in is not None:
            p = UE_in
        else:
            p = np.zeros((K, E, 3))
            for dim in range(3):
                p[:, 0, dim] = np.random.uniform(self.window[dim*2], self.window[dim*2+1], size=K)
            v = np.array([speed*np.cos(heading), speed*np.sin(heading), 0])
            for tau in range(1, E):
                p[:, tau, :] = p[:, tau-1, :] + v[None, :]*epoch_time
        return p

    def delays_and_channel_coefficients(self, p, p_U, q0=0.285, S_RCS=50, active_RIS_factor=1):
        """
        Computes the channel delays and simulates the channel coefficients for
        each of the paths.

        Inputs:
        -------
            p : 
            p_U : 
            q0 : 
            S_RCS : 

        Outputs:
        --------
            tau0 : ndarray, size=(K, E)
                Delay of UE-RIS-UE path.
            taul : ndarray, size=(K, E, L)
                Delay of UE-RIS-SP-UE/UE-SP-RIS-UE paths.
            overline_taul : ndarray, size=(K, E, L)
                Delay of UE-SP-UE paths.
            alpha0 : ndarray, size=(K, E)
                Channel coefficient of UE-RIS-UE path.
            alphal : ndarray, size=(K, E, L)
                Channel coefficients of UE-RIS-SP-UE/UE-SP-RIS-UE paths.
            betal : ndarray, size=(K, E, L)
                Channel coefficients of UE-SP-UE paths.
        """
        K, E = p_U.shape[:2]
        L = p.shape[0]

        d_UR = np.linalg.norm(p_U - self.p_R[None, None, :], axis=-1) # size=(K, E)
        d_SR = np.linalg.norm(p - self.p_R[None, :], axis=-1) # size=(L,)
        d_SU = np.linalg.norm(p[None, None, :, :] - p_U[:, :, None, :], axis=-1) # size=(K, E, L)

        tau0 = 2*d_UR/self.c # size=(K, E)
        taul = (d_UR[:, :, None] + d_SR[None, None, :] + d_SU)/self.c # size=(K, E, L)
        overline_taul = 2*d_SU/self.c # size=(K, E, L)

        g_UR = np.dot(p_U - self.p_R[None, None, :], self.n_R)/d_UR # size=(K, E)
        g_SR = np.abs(np.dot(p - self.p_R[None, :], self.n_R)/d_SR) # size=(L,)

        alpha0_amp = np.sqrt(active_RIS_factor*self.p_tx_sc*self.lambda_**4 * g_UR**(4*q0) / ((4*np.pi)**3 * d_UR**4)) # size=(K, E)
        alphal_amp = np.sqrt(active_RIS_factor*self.p_tx_sc*self.lambda_**4 * g_UR[:, :, None]**(2*q0) * g_SR[None, None, :]**(2*q0) * S_RCS \
            / ((4*np.pi)**4 * d_UR[:, :, None]**2 * d_SR[None, None, :]**2 * d_SU**2)) # size=(K, E, L)
        betal_amp = np.sqrt(self.p_tx_sc*self.lambda_**2 * S_RCS / ((4*np.pi)**3 * d_SU**4)) # size=(K, E, L)
        # print(alpha0_amp)
        # print(alphal_amp)
        # print(betal_amp)

        nu_G = np.random.uniform(low=0, high=2*np.pi, size=(K, E, 2*L+1))

        alpha0 = alpha0_amp * np.exp(-1j*(2*np.pi*self.f_c*tau0 + nu_G[:, :, -1])) # size=(K, E)
        alphal = alphal_amp * np.exp(-1j*(2*np.pi*self.f_c*taul + nu_G[:, :, :L])) # size=(K, E, L)
        betal = betal_amp * np.exp(-1j*(2*np.pi*self.f_c*overline_taul + nu_G[:, :, L:2*L])) # size=(K, E, L)
        return tau0, taul, overline_taul, alpha0, alphal, betal

    def angles(self, p, p_U, o_U):
        """
        Compute the angles.

        Inputs:
        -------
            p : ndarray, size=(L, d)
                The scatter point locations.
            p_U : ndarray, size=(K, E, d)
                The UE locations.
            o_U : ndarray, size=(K, E, 2)
                The UE orientations.

        Outputs:
        --------
            theta0 : ndarray, size=(K, E, 2)
                AOA Azimuth and elevation at the UE from the RIS.
            thetal : ndarray, size=(K, E, L, 2)
                AOA Azimuth and elevation at the UE from the SPs.
            phi0 : ndarray, size=(K, E, 2)
                AOA Azimuth and elevation at the RIS from the UE.
            phil : ndarray, size=(L, 2)
                AOA Azimuth and elevation at the RIS from the SPs.
        """
        K, E = p_U.shape[:2]
        L = p.shape[0]

        # Generate rotation matrices (parallelise?)
        O_R = self.coord_rotation_matrix(*self.oR) # size=(3, 3)
        # O_R = self.coord_rotation_matrix()
        O_U = np.zeros((K, E, 3, 3)) # size=(K, E, 3, 3)
        for k in range(K):
            for tau in range(E):
                O_U[k, tau, :, :] = self.coord_rotation_matrix(*self.oU)

        # Compute difference vectors in local coordinates
        p_UR = np.einsum("ij,kei->kej", O_R, p_U - self.p_R[None, None, :]) # size=(K, E, 3)
        p_SR = np.einsum("ij,ki->kj", O_R, p - self.p_R[None, :]) # size=(L, 3)
        p_RU = np.einsum("keij,kei->kej", O_U, self.p_R[None, None, :] - p_U) # size=(K, E, 3)
        p_SU = np.einsum("keij,keli->kelj", O_U, p[None, None, :, :] - p_U[:, :, None, :]) # size=(K, E, L, 3)

        # Compute the azimuth and elevation angles
        theta0_az = np.arctan2(p_RU[:, :, 1], p_RU[:, :, 0]) # size=(K, E)
        theta0_el = np.arccos(p_RU[:, :, 2]/np.linalg.norm(p_RU, axis=-1)) # size=(K, E)
        theta0 = np.concatenate((theta0_az[:, :, None], theta0_el[:, :, None]), axis=-1) # size=(K, E, 2)

        thetal_az = np.arctan2(p_SU[:, :, :, 1], p_SU[:, :, :, 0]) # size=(K, E, L)
        thetal_el = np.arccos(p_SU[:, :, :, 2]/np.linalg.norm(p_SU, axis=-1)) # size=(K, E, L)
        thetal = np.concatenate((thetal_az[:, :, :, None], thetal_el[:, :, :, None]), axis=-1) # size=(K, E, L, 2)

        phi0_az = np.arctan2(p_UR[:, :, 1], p_UR[:, :, 0]) # size=(K, E)
        phi0_el = np.arccos(p_UR[:, :, 2]/np.linalg.norm(p_UR, axis=-1)) # size=(K, E)
        phi0 = np.concatenate((phi0_az[:, :, None], phi0_el[:, :, None]), axis=-1) # size=(K, E, 2)

        phil_az = np.arctan2(p_SR[:, 1], p_SR[:, 0]) # size=(L,)
        phil_el = np.arccos(p_SR[:, 2]/np.linalg.norm(p_SR, axis=-1)) # size=(L,)
        phil = np.concatenate((phil_az[:, None], phil_el[:, None]), axis=-1) # size=(L, 2)
        return theta0, thetal, phi0, phil

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

    @profile
    def channel_matrix(self, pars, omega_tilde):
        """
        Compute the channel matrix components.
        """
        # Compute UE array response
        aU_theta0 = self.UPA_array_response_vector(pars["theta0"], self.N_U, self.d_U) # size=(K, E, N_U)
        aU_thetal = self.UPA_array_response_vector(pars["thetal"], self.N_U, self.d_U) # size=(K, E, L, N_U)

        # Compute RIS response
        aR_phi0 = self.UPA_array_response_vector(pars["phi0"], self.N_R, self.d_R) # size=(K, E, N_R)
        aR_phil = self.UPA_array_response_vector(pars["phil"], self.N_R, self.d_R) # size=(L, N_R)

        #nu0 = np.einsum("kei,tkei->tke", aR_phi0, np.einsum("tij,kej -> tkei", self.Omega, aR_phi0)) # size=(T, K, E)
        #nu = np.einsum("li,tkei->tkel", aR_phil, np.einsum("tij,kej -> tkei", self.Omega, aR_phi0)) # size=(T, K, E, L)
        omega = self.orthogonal_RIS(omega_tilde) # size=(T, N_R)

        aR_phi0_prod = omega[:, None, None, :]*aR_phi0[None, :, :, :]
        nu0 = np.einsum("kei,tkei->tke", aR_phi0, aR_phi0_prod) # size=(T, K, E)
        nu = np.einsum("li,tkei->tkel", aR_phil, aR_phi0_prod) # size=(T, K, E, L)

        # Compute delay steering vectors
        dn_tau0 = self.delay_steering_vector(pars["tau0"], self.n) # size=(N+1, K, E)
        dn_taul = self.delay_steering_vector(pars["taul"], self.n) # size=(N+1, K, E, L)
        dn_overline_taul = self.delay_steering_vector(pars["overline_taul"], self.n) # size=(N+1, K, E, L)

        # Compute channel matrix components
        outer_prod0 = np.einsum("kei,kej->keij", aU_theta0, aU_theta0) # size=(K, E, N_U, N_U)
        H0 = pars["alpha0"][None, None, :, :, None, None] * nu0[:, None, :, :, None, None] * \
            outer_prod0[None, None, :, :, :, :] * dn_tau0[None, :, :, :, None, None] # size=(T, N+1, K, E, N_U, N_U)

        outer_prodl0 = np.einsum("keli,kej->kelij", aU_thetal, aU_theta0) # size=(K, E, L, N_U, N_U)
        H1 = np.sum(pars["alphal"][None, None, :, :, :, None, None] * nu[:, None, :, :, :, None, None] \
            * outer_prodl0[None, None, :, :, :, :, :] * dn_taul[None, :, :, :, :, None, None], axis=4) # size=(T, N+1, K, E, N_U, N_U)

        outer_prod0l = np.einsum("kei,kelj->kelij", aU_theta0, aU_thetal) # size=(K, E, L, N_U, N_U)
        H2 = np.sum(pars["alphal"][None, None, :, :, :, None, None] * nu[:, None, :, :, :, None, None] \
            * outer_prod0l[None, None, :, :, :, :, :] * dn_taul[None, :, :, :, :, None, None], axis=4) # size=(T, N+1, K, E, N_U, N_U)

        outer_prodll = np.einsum("keli,kelj->kelij", aU_thetal, aU_thetal) # size=(K, E, L, N_U, N_U)
        H3 = np.sum(pars["betal"][None, None, :, :, :, None, None] * np.ones(self.T)[:, None, None, None, None, None, None] \
            * outer_prodll[None, None, :, :, :, :, :] * dn_overline_taul[None, :, :, :, :, None, None], axis=4) # size=(T, N+1, K, E, N_U, N_U)

        # Aggregate the results
        H = sum([H0, H1, H2, H3])
        return H

    def UPA_array_response_vector(self, phi, N_Q, d):
        """
        Compute the array response vector of a uniform planar array at the given
        angle phi_vec[i, j, :] = [azimuth, elevation] (size=(K, 2) or (K, L, 2) or (K, E, L, 2))
        for a given number of antennas N_Q=[N_Q^{az}, N_Q^{el}] spaced by distance d.
        """
        if phi.ndim == 1:
            beta = np.exp(-1j*2*np.pi*np.cos(phi[0, None])*np.sin(phi[1, None])*d/self.lambda_)**np.arange(N_Q[0]) # size=(N_az)
            gamma = np.exp(-1j*2*np.pi*np.sin(phi[0, None])*np.sin(phi[1, None])*d/self.lambda_)**np.arange(N_Q[1]) # size=(N_el)
            # beta = self.ULA_array_response_vector(np.array([phi_vec[0]]), N_Q[0], d, np.cos)[0, :]
            # gamma = self.ULA_array_response_vector(np.array([phi_vec[1]]), N_Q[1], d, np.sin)[0, :]
            a = (beta[:,None]*gamma[None,:]).flatten() # Kronecker product; size=(N_az*N_el)
        elif phi.ndim == 2:
            beta = np.exp(-1j*2*np.pi*np.cos(phi[:, 0, None])*np.sin(phi[:, 1, None])*d/self.lambda_)**np.arange(N_Q[0])[None, :] # size=(K, N_az)
            gamma = np.exp(-1j*2*np.pi*np.sin(phi[:, 0, None])*np.sin(phi[:, 1, None])*d/self.lambda_)**np.arange(N_Q[1])[None, :] # size=(K, N_el)
            # beta = self.ULA_array_response_vector(phi_vec[:, 0], N_Q[0], d, np.cos)
            # gamma = self.ULA_array_response_vector(phi_vec[:, 1], N_Q[1], d, np.sin)
            a = (beta[:,:,None]*gamma[:,None,:]).reshape(beta.shape[0],-1) # Kronecker product; size=(K, N_az*N_el)
        elif phi.ndim == 3:
            beta = np.exp(-1j*2*np.pi*np.cos(phi[:, :, 0, None])*np.sin(phi[:, :, 1, None])*d/self.lambda_)**np.arange(N_Q[0])[None, None, :] # size=(K, E, N_az)
            gamma = np.exp(-1j*2*np.pi*np.sin(phi[:, :, 0, None])*np.sin(phi[:, :, 1, None])*d/self.lambda_)**np.arange(N_Q[1])[None, None, :] # size=(K, E, N_el)
            # beta = self.ULA_array_response_vector(phi_vec[:, :, 0], N_Q[0], d, np.cos)
            # gamma = self.ULA_array_response_vector(phi_vec[:, :, 1], N_Q[1], d, np.sin)
            a = (beta[:,:,:,None]*gamma[:,:,None,:]).reshape(beta.shape[0],beta.shape[1],-1) # Kronecker product; size=(K, E, N_az*N_el)
        elif phi.ndim == 4:
            beta = np.exp(-1j*2*np.pi*np.cos(phi[:, :, :, 0, None])*np.sin(phi[:, :, :, 1, None])*d/self.lambda_)**np.arange(N_Q[0])[None, None, None, :] # size=(K, E, L, N_az)
            gamma = np.exp(-1j*2*np.pi*np.sin(phi[:, :, :, 0, None])*np.sin(phi[:, :, :, 1, None])*d/self.lambda_)**np.arange(N_Q[1])[None, None, None, :] # size=(K, E, L, N_el)
            # beta = self.ULA_array_response_vector(phi_vec[:, :, :, 0], N_Q[0], d, np.cos)
            # gamma = self.ULA_array_response_vector(phi_vec[:, :, :, 1], N_Q[1], d, np.sin)
            a = (beta[:,:,:,:,None]*gamma[:,:,:,None,:]).reshape(beta.shape[0],beta.shape[1],beta.shape[2],-1) # Kronecker product; size=(K, E, L, N_az*N_el)
        else:
            raise ValueError("Input phi_vec must have 1 or 2 or 3 or 4 dimensions.")
        return a

    def delay_steering_vector(self, tau, n, fun=np.exp, c=-1j):
        """
        Compute steering vector given a delay tau
        (size=(K,) or (K, L) or (K, E, L) or (K, L, L) or (K, E, L, L))
        and subcarrier index n (size=(N,)).
        """
        if tau.ndim == 1 and n.ndim == 1:
            dn_tau = fun(c*2*np.pi*n[:, None]*self.delta_f*tau[None, :])
        elif tau.ndim == 2 and n.ndim == 1:
            dn_tau = fun(c*2*np.pi*n[:, None, None]*self.delta_f*tau[None, :, :])
        elif tau.ndim == 3 and n.ndim == 1:
            dn_tau = fun(c*2*np.pi*n[:, None, None, None]*self.delta_f*tau[None, :, :, :])
        elif tau.ndim == 4 and n.ndim == 1:
            dn_tau = fun(c*2*np.pi*n[:, None, None, None, None]*self.delta_f*tau[None, :, :, :, :])
        else:
            raise ValueError("Input tau must have 1 or 2 or 3 or 4 dimensions and input n must have 1 dimension.")
        return dn_tau

    def orthogonal_RIS(self, omega_tilde):
        """
        Construct the orthogonal RIS phase profile time sequence from the
        existing omega_tilde RIS phase profiles.
        """
        omega = np.zeros((self.T, np.prod(self.N_R)), dtype=np.complex128)
        for t_tilde in range(0, int(self.T/2)):
            omega[t_tilde*2, :] = omega_tilde[t_tilde, :]
            omega[t_tilde*2+1, :] = -omega_tilde[t_tilde, :]
        return omega

    def RIS_random_codebook(self):
        """
        Construct a sequence of random RIS phase profiles with phases in the
        interval [-\pi, \pi].
        """
        random_phases = np.random.uniform(low=-np.pi, high=np.pi, size=(int(self.T/2), np.prod(self.N_R)))
        omega_tilde = np.exp(1j*random_phases)
        return omega_tilde

    def RIS_directional(self, p_U, pred_Map, N_pos=100, PE=0.99):
        """
        Directional RIS phase profiles based on prior information.

        Inputs:
        -------
            p_U : ndarray, size=(K, 1, 3)
                The UE positions.
            prior : dict
                Has keys "prior" and "Phi_est".
            N_pos : int, optional
                Number of target points in the uncertainty region. The default is 100.
            PE : float, optional
                Probability defining the boundary of the angular uncertainty region.
                The default is 0.99.
        """
        if self.use_directional_RIS is True and pred_Map["prior"] is True and pred_Map["Phi_est"].shape[0] > 0:
            TE = -2*np.log(PE)

            # Simulate UE positions
            p_U = p_U[:, 0, :]
            # UE_idx = np.random.randint(self.K, size=self.T//2)
            UE_idx = np.random.randint(self.K)*np.ones(self.T//2, dtype=np.int16)
            UE_points = p_U[UE_idx, :]

            # Choose UE and SP(s)
            # prec = np.sqrt(np.trace(np.linalg.inv(pred_Map["omega_est"]), axis1=1, axis2=2))
            prec = np.sqrt(np.trace(pred_Map["omega_est"], axis1=1, axis2=2))
            cluster_weights = prec/np.sum(prec)
            cluster_idx = np.random.choice(np.arange(pred_Map["Phi_est"].shape[0]), size=(self.T//2), replace=True, p=cluster_weights)
            cluster_choice = pred_Map["Phi_est"][cluster_idx, :]
            cluster_choice_covariance = pred_Map["omega_est"][cluster_idx, :, :]

            # Create spatial uncertainty region
            target_points = np.zeros((self.T//2, N_pos, 3))
            for t in range(self.T//2):        
                i = 0
                k = 0
                while i < N_pos:
                    propose = np.random.multivariate_normal(cluster_choice[t, :], cluster_choice_covariance[t, :, :], size=2*N_pos)
                    mu_propose = cluster_choice[t, None, :] - propose
                    criterion = np.einsum("ni,ni->n", mu_propose, np.einsum("ij,nj->ni", cluster_choice_covariance[t, :, :], mu_propose))
                    cond = (criterion <= TE)
                    chosen = propose[cond][:N_pos-i, :]
                    card = chosen.shape[0]
                    target_points[t, i:i+card, :] = chosen
                    i += card
                    k += 1
                    if k > 50:
                        print("Ups, I'm stuck!")
                        random_phases = np.random.uniform(low=-np.pi, high=np.pi, size=(int(self.T/2), np.prod(self.N_R)))
                        omega_tilde = np.exp(1j*random_phases)
                        return omega_tilde

            # Find UE-RIS angle
            O_R = self.coord_rotation_matrix(*self.oR) # size=(3, 3)
            p_UR = np.einsum("ij,ti->tj", O_R, UE_points - self.p_R[None, :]) # size=(T/2, 3)
            phi0_az = np.arctan2(p_UR[:, 1], p_UR[:, 0]) # size=(T/2)
            phi0_el = np.arccos(p_UR[:, 2]/np.linalg.norm(p_UR, axis=-1)) # size=(T/2)
            phi0 = np.concatenate((phi0_az[:, None], phi0_el[:, None]), axis=-1) # size=(T/2, 2)

            # Find RIS-target angle
            O_R = self.coord_rotation_matrix(*self.oR) # size=(3, 3)
            p_SR = np.einsum("ij,tni->tnj", O_R, target_points - self.p_R[None, None, :]) # size=(T/2, N_pos, 3)
            phil_az = np.arctan2(p_SR[:, :, 1], p_SR[:, :, 0]) # size=(T/2, N_pos)
            phil_el = np.arccos(p_SR[:, :, 2]/np.linalg.norm(p_SR, axis=-1)) # size=(T/2, N_pos)

            # Create angular uncertainty region
            phi_az_min = np.min(phil_az, axis=1)
            phi_az_max = np.max(phil_az, axis=1)
            phi_el_min = np.min(phil_el, axis=1)
            phi_el_max = np.max(phil_el, axis=1)
            zeta_az = phi_az_max - phi_az_min
            zeta_el = phi_el_max - phi_el_min

            # Create array element indices for subbox
            N_RIS = np.prod(self.N_R)
            N_az, N_el = self.N_R
            n = np.arange(1, N_RIS+1)
            n_el = n - N_az*(np.ceil(n/N_az).astype(np.int16) - 1)
            n_az = np.ceil(n/N_el).astype(np.int16)

            # Define target angles
            phi_az = phi_az_min[:, None] + zeta_az[:, None]*(n_az[None, :]-1)/(N_az-1)
            phi_el = phi_el_min[:, None] + zeta_el[:, None]*(n_el[None, :]-1)/(N_el-1)

            # Construct directional RIS phase profile
            a0 = self.UPA_array_response_vector(phi0, self.N_R, self.d_R) # size=(T/2, N_R)
            beta = np.exp(-1j*2*np.pi*np.cos(phi_az)*np.sin(phi_el)*self.d_R/self.lambda_)**(n_az-1)[None, :] # size=(T/2, N_R)
            gamma = np.exp(-1j*2*np.pi*np.sin(phi_az)*np.sin(phi_el)*self.d_R/self.lambda_)**(n_el-1)[None, :] # size=(T/2, N_R)
            a_dir = beta*gamma # size=(T/2, N_R)
            omega_tilde = np.conjugate(a0 * a_dir) # size=(T/2, N_R)

        else:
            random_phases = np.random.uniform(low=-np.pi, high=np.pi, size=(int(self.T/2), np.prod(self.N_R)))
            omega_tilde = np.exp(1j*random_phases)
        return omega_tilde

    def signal_separation_and_refinement(self, W_mat, H, f_tilde, eps):
        """
        Separate the signals originating from different paths.

        Outputs:
        --------
            y_tilde_N : ndarray, size=(T/2, N+1, K, E, N_U)
                Non-RIS signal.
            y_overline_D : ndarray, size=(T1, N+1, K, E, N_U-1)
                Directional received signal.
            y_overline_O : ndarray, size=(T2, N+1, K, E)
                Orthogonal received signal.
        """
        H_R = 1/2 * (H[1::2, :, :, :, :, :] - H[::2, :, :, :, :, :]) # size=(T/2, N+1, K, E, N_U, N_U)
        eps_R = 1/2 * (eps[1::2, :, :, :, :] - eps[::2, :, :, :, :])
        y_tilde_R = np.einsum("keji,tnkej->tnkei", np.conjugate(W_mat), np.einsum("tnkeij,tkej->tnkei", H_R, f_tilde)) \
                    + np.einsum("keji,tnkej->tnkei", np.conjugate(W_mat), eps_R) # size=(T/2, N+1, K, E, N_U)

        H_N = 1/2 * (H[1::2, :, :, :, :, :] + H[::2, :, :, :, :, :]) # size=(T/2, N+1, K, E, N_U, N_U)
        eps_N = 1/2 * (eps[1::2, :, :, :, :] + eps[::2, :, :, :, :])
        y_tilde_N = np.einsum("tnkeij,tkej->tnkei", H_N, f_tilde) + eps_N # size=(T/2, N+1, K, E, N_U)

        y_overline_D = y_tilde_R[:self.T1, :, :, :, 1:] # size=(T1, N+1, K, E, N_U-1)
        y_overline_O = y_tilde_R[self.T1:, :, :, :, 0] # size=(T2, N+1, K, E)
        return y_tilde_N, y_overline_D, y_overline_O

    def simulate_received_signal(self, H, f):
        """
        Simulate the received signal.

        Outputs:
        --------
            y : ndarray, size=(T, N+1, K, E, N_U)
                Received signal.
            eps : ndarray, size=(T, N+1, K, E, N_U)
                Additive white Gaussian noise.
        """
        # Simulate noise
        real_eps = np.random.normal(0, np.sqrt(self.p_noise_sc*2)/2, size=(self.T, self.N, self.K, self.E, np.prod(self.N_U)))
        imag_eps = np.random.normal(0, np.sqrt(self.p_noise_sc*2)/2, size=(self.T, self.N, self.K, self.E, np.prod(self.N_U)))
        eps = real_eps + 1j*imag_eps

        # Compute received signal
        y = np.einsum("tnkeij,tkej->tnkei", H, f) + eps
        return y, eps

    def precoder_time_sequence(self, f_tilde):
        """
        Construct the orthogonal precoder time sequence from the
        existing f_tilde precoders.
        """
        f = np.zeros((self.T, self.K, self.E, np.prod(self.N_U)), dtype=np.complex128)
        for t_tilde in range(0, int(self.T/2)):
            f[t_tilde*2, :, :, :] = f_tilde[t_tilde, :, :, :]
            f[t_tilde*2+1, :, :, :] = f_tilde[t_tilde, :, :, :]
        return f

    def construct_precoder(self, theta0):
        """
        Construct a sequence of precoders. The first T1 are towards the RIS
        while the remaining T2 are null towards the RIS (simulated randomly
        in the null space). The total length of the time sequence is T/2.
        """
        f_tilde = np.zeros((int(self.T/2), self.K, self.E, np.prod(self.N_U)), dtype=np.complex128)

        # Compute array response towards the RIS (redundant computations)
        aU_theta0 = self.UPA_array_response_vector(theta0, self.N_U, self.d_U) # size=(K, E, N_U)

        # # Construct precoder time sequence
        f_tilde[:self.T1, :, :, :] = np.conjugate(aU_theta0)[None, :, :, :] * np.ones(self.T1)[:, None, None, None] \
                            / np.linalg.norm(aU_theta0, axis=-1)[None, :, :, None]**2 # size=(T1, K, E, N_U)

        f_temp = np.exp(1j*2*np.pi*np.random.rand((np.prod(self.N_U))))/np.sqrt(np.prod(self.N_U))
        f_temp2 = f_temp[None, None, :] - np.einsum("kei,i->ke", np.conjugate(aU_theta0), f_temp)[:, :, None]*aU_theta0/(np.linalg.norm(aU_theta0, axis=-1)**2)[:, :, None]
        f_tilde[self.T1:, :, :, :] = np.conjugate(f_temp2/np.linalg.norm(f_temp2, axis=2)[:, :, None])[None, :, :, :]*np.ones(self.T//2-self.T1)[:, None, None, None]
        return f_tilde

    def construct_combiner(self, theta0):
        """
        Construct the combiner matrix. The first column of the combiner matrix
        is towards the RIS, while the remaining columns form an orthonormal
        basis for the space of vectors with a null towards the RIS.
        """
        W = np.zeros((self.K, self.E, np.prod(self.N_U), np.prod(self.N_U)), dtype=np.complex128)

        # Compute array response towards the RIS (redundant computations)
        aU_theta0 = self.UPA_array_response_vector(theta0, self.N_U, self.d_U) # size=(K, E, N_U)

        # DFT matrix
        DFT_idx = np.arange(np.prod(self.N_U))[None, :]*np.arange(np.prod(self.N_U))[:, None]
        DFTt = np.exp(1j*2*np.pi*DFT_idx/np.prod(self.N_U)) / np.sqrt(np.prod(self.N_U))

        # Construct combiner
        W = DFTt[None, None, :, :] * aU_theta0[:, :, :, None]

        # Check orthonormality
        WHW = np.einsum("keji,kejl->keil", np.conjugate(W), W)
        I = np.eye(np.prod(self.N_U), np.prod(self.N_U))
        assert np.allclose(WHW, I[None, None, :, :]), "The combiner must be an orthonormal matrix."
        return W

    def plot_configuration(self, p, p_U):
        """
        Visualize the UEs, SPs, and RIS in the observation window
        (projection onto xy-plane).
        """
        plt.plot(p_U[:, :, 0].flatten(), p_U[:, :, 1].flatten(), '^', color="tab:blue", markersize=5, label="UE")
        plt.plot(p[:, 0], p[:, 1], 'o', color="tab:orange", markersize=6, label="SP")
        plt.scatter(self.p_R[0], self.p_R[1], marker='|', color="tab:red", s=500, linewidths=3, label="RIS")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(self.window[0]-0.2, self.window[1])
        plt.ylim(self.window[2], self.window[3])
        plt.legend(loc="lower right")
        #plt.savefig(f"Results/{savename}.png", dpi=500, bbox_inches="tight")
        plt.show()

    def plot_configuration_3D(self, p, p_U):
        """
        Visualize the UEs, SPs, and RIS in the observation window
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(p_U[:, :, 0].flatten(), p_U[:, :, 1].flatten(), p_U[:, :, 2].flatten(), marker='^', color="tab:blue", s=7, label="UE")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker='o', color="tab:orange", s=7, label="SP")
        ax.scatter(self.p_R[0], self.p_R[1], self.p_R[2], marker='|', color="tab:red", s=500, linewidths=3, label="RIS")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(self.window[0]-0.2, self.window[1])
        ax.set_ylim(self.window[2], self.window[3])
        ax.set_zlim(self.window[4], self.window[5])
        ax.legend(loc="upper left")
        if self.save_verbose is True:
            plt.savefig("Plots/Configuration_plots/configuration.png", dpi=500, bbox_inches="tight")
        plt.show()

    def plot_response(self, y_tilde_N):
        """
        Plot the frequency and impulse responses for the different refined
        signals averaged across time for one antenna and one UE.
        """
        freq = (self.f_c + self.n*self.delta_f)*1e-09
        #plt.plot(freq, np.abs(np.sum(self.y[:, :, 0, 0, 0]*1e06, axis=0)), color="tab:red", label="$y$")
        plt.plot(freq, np.real(np.sum(y_tilde_N[:, :, 0, 0, 0]*1e06, axis=0)), color="tab:blue", label="$H^N$ Real part")
        plt.plot(freq, np.imag(np.sum(y_tilde_N[:, :, 0, 0, 0]*1e06, axis=0)), color="tab:red", label="$H^N$ Imaginary part")
        #plt.plot(freq, np.abs(np.sum(self.y_overline_D[:, :, 0, 0, 0]*1e06, axis=0)), color="tab:green", label="$y^D$")
        #plt.plot(freq, np.abs(np.sum(self.y_overline_O[:, :, 0, 0]*1e06, axis=0)), color="tab:orange", label="$y^O$")
        plt.ylabel("Frequency response [$\mu W$]")
        plt.xlabel("Frequency [GHz]")
        plt.legend()
        if self.save_verbose is True:
            plt.savefig("Plots/Signal_plots/frequency_response.png", dpi=500, bbox_inches="tight")
        plt.show()

        time = np.arange(self.N)/self.W*1e09
        #plt.plot(time[:120], np.abs(np.fft.ifft(np.sum(self.y[:, :, 0, 0, 0]*1e06, axis=0)))[:120], linestyle="dashed", color="tab:red", label="y")
        plt.plot(time[:120], np.abs(np.fft.ifft(np.sum(y_tilde_N[:, :, 0, 0, 0]*1e06, axis=0)))[:120], color="tab:blue", label="$h^N$")
        #plt.plot(time[:120], np.abs(np.fft.ifft(np.sum(self.y_overline_D[:, :, 0, 0, 0]*1e06, axis=0)))[:120], linestyle="dashed", color="tab:green", label="$y^D$")
        #plt.plot(time[:120], np.abs(np.fft.ifft(np.sum(self.y_overline_O[:, :, 0, 0]*1e06, axis=0)))[:120], color="tab:orange", label="$y^O$")
        plt.ylabel("Impulse response [$\mu W$]")
        plt.xlabel("Time [ns]")
        plt.legend()
        if self.save_verbose is True:
            plt.savefig("Plots/Signal_plots/impulse_response.png", dpi=500, bbox_inches="tight")
        plt.show()

    def plot_beampattern(self, theta0, W_mat, f_tilde):
        azimuth = np.linspace(-np.pi, np.pi, 360)
        elevation = np.array([theta0[1]])
        C1, C2 = np.meshgrid(azimuth, elevation)
        angle_grid = np.stack((C1.flatten(), C2.flatten()), axis = 1)
        a_angle = self.UPA_array_response_vector(angle_grid, self.N_U, self.d_U)

        amp_T1 = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[:self.T1, 0, 0, :]))**2, axis=1)
        self.beampattern_polar(azimuth, amp_T1, theta0[0], "Precoder towards RIS")

        amp_T2 = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[self.T1:, 0, 0, :]))**2, axis=1)
        self.beampattern_polar(azimuth, amp_T2, theta0[0], "Precoder with null towards RIS")

        amp_N = np.sum(np.abs(np.einsum("gi,gt->gti", a_angle, np.einsum("gi,ti->gt", a_angle, f_tilde[:, 0, 0, :])))**2, axis=(1, 2))
        self.beampattern_polar(azimuth, amp_N, theta0[0], "Non-RIS")

        amp_D = np.sum(np.abs(np.einsum("ij,gi->gj", np.conjugate(W_mat[0, 0, :, 1:]), a_angle))**2, axis=1)
        self.beampattern_polar(azimuth, amp_D, theta0[0], "Directional")

        amp_O = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[self.T1:, 0, 0, :]))**2, axis=1)
        self.beampattern_polar(azimuth, amp_O, theta0[0], "Orthogonal")

    def beampattern_polar(self, azimuth, amp, theta0, title):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='polar')
        ax1.plot(azimuth, amp/max(amp))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(azimuth*180/np.pi, np.array(amp)/max(amp))
        plt.axvline(x=theta0*180/np.pi, color="green")
        plt.title(title)
        if self.save_verbose is True:
            plt.savefig(f"Plots/Beampattern_plots/polar_{title}.png", dpi=500, bbox_inches="tight")
        plt.show()

    def plot_beampattern_map(self, theta0, W_mat, f_tilde):
        azimuth = np.linspace(-np.pi, np.pi, 360//4)
        elevation = np.linspace(0, np.pi, 180//4)
        C1, C2 = np.meshgrid(azimuth, elevation)
        angle_grid = np.stack((C1.flatten(), C2.flatten()), axis = 1)
        a_angle = self.UPA_array_response_vector(angle_grid, self.N_U, self.d_U)

        amp_T1 = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[:self.T1, 0, 0, :]))**2, axis=1)
        self.beampattern_map(azimuth, elevation, amp_T1/np.max(amp_T1), theta0, "Precoder towards RIS")

        amp_T2 = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[self.T1:, 0, 0, :]))**2, axis=1)
        self.beampattern_map(azimuth, elevation, amp_T2/np.max(amp_T2), theta0, "Precoder with null towards RIS")

        amp_N = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[:, 0, 0, :]))**2, axis=(1))
        self.beampattern_map(azimuth, elevation, amp_N/np.max(amp_N), theta0, "Non-RIS")

        amp_D = np.sum(np.abs(np.einsum("ij,gi->gj", np.conjugate(W_mat[0, 0, :, 1:]), a_angle))**2, axis=1)
        self.beampattern_map(azimuth, elevation, amp_D/np.max(amp_D), theta0, "Directional")

        amp_O = np.sum(np.abs(np.einsum("gi,ti->gt", a_angle, f_tilde[self.T1:, 0, 0, :]))**2, axis=1)
        self.beampattern_map(azimuth, elevation, amp_O/np.max(amp_O), theta0, "Orthogonal")

    def beampattern_map(self, azimuth, elevation, amp, theta0, label):
        spec = plt.pcolormesh(azimuth, elevation, np.reshape(amp, (180//4, 360//4)),
                              cmap="cool", shading="auto", norm=LogNorm())
        cb = plt.colorbar(spec)
        cb.set_label(label=label)
        plt.xlabel("Azimuth [rad]")
        plt.ylabel("Elevation [rad]")
        plt.plot(theta0[0], theta0[1], "or", markersize=5)
        if self.save_verbose is True:
            plt.savefig(f"Plots/Beampattern_plots/map_{label}.png", dpi=500, bbox_inches="tight")
        plt.show()


if __name__ == '__main__': # run demo
    np.random.seed(43)
    chn = channel(config_file="channel_config.toml", verbose=True, save_verbose=False)
    chn()
    print(chn)
    #profile.print_stats()

