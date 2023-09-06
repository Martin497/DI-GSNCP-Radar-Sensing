# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:03:16 2022

@author: Martin Voigt Vejling
E-mail: mvv@es.aau.dk

Track changes:
    Version 1.0: Implementation of homogenenous and inhomogeneous Poisson
                 point process simulation. (24/11/2022)
            1.1: Added functionality for 3D Poisson point processes. (12/01/2023)
            1.2: Added functionality to simulate Thomas processes.
                 Added functionality to simulate approximately a
                 hard core process. (27/07/2023)
            1.3 : Added functionality to simulate from a doubly inhomogeneous
                  Neyman-Scott model. (31/07/2023)
            1.4 : Added functionality to simulate from a doubly inhomogeneous
                  generalized shot noise Cox process model. (04/08/2023)
"""


import numpy as np
import matplotlib.pyplot as plt


class PoissonPointProcess(object):
    """
    Class for simulating data from a Poisson Point Process (PPP).

    Methods:
        __init__ : Initialize settings.
        homogeneous : Simulate homogeneous Poisson point process.
        inhomogeneous : Simulate inhomogeneous Poisson point process using
                        Lewis-Shedler thinning method.

    Attributes:
        window : Observation window boundaries.
        lambda_ : Intensity.
        points : Realization of Poisson point process.
        dim : Dimensionality of the window.
    """
    def __init__(self, window, lambda_, grid_dim=None):
        """
        Initialize the class attributes. Simulate the point process.
        If lambda_ is a function, simulate an inhomogeneous Poisson point
        process else simulate a homogeneous Poisson point process.
        """
        self.window = window
        self.lambda_ = lambda_
        self.grid_dim = grid_dim
        self.dim = len(self.window)//2
        assert self.dim == 2 or self.dim == 3, \
              "Only 2- or 3-dimensional processes are supported!"

        if hasattr(lambda_, "__call__"):
            assert hasattr(self, "grid_dim")
            self.points = self.inhomogeneous(lambda_)
        else:
            self.points = self.homogeneous(lambda_)

    def homogeneous(self, lambda_):
        """
        Simulate homogeneous Poisson point process. The method takes a
        float lambda_ as input and returns an ndarray of size=(n, dim) which is
        the realization of the point process.
        """
        area = (self.window[1]-self.window[0])*(self.window[3]-self.window[2])
        if self.dim == 3:
            area = area*(self.window[5]-self.window[4])

        n = np.random.poisson(area*lambda_)
        x = np.random.uniform(self.window[0], self.window[1], size=n)
        y = np.random.uniform(self.window[2], self.window[3], size=n)
        if self.dim == 2:
            points = np.vstack((x, y)).T
        elif self.dim == 3:
            z = np.random.uniform(self.window[4], self.window[5], size=n)
            points = np.vstack((x, y, z)).T
        return points

    def inhomogeneous(self, lambda_):
        """
        Simulate inhomogeneous Poisson point process using the Lewis-Shedler
        thinning method. The method takes a function lambda_ as input and
        returns an ndarray of size=(n, 2) which is the realization of the
        point process.
        """
        # Evaluate the intensity function in a grid to find the maximum M
        x_points = np.linspace(self.window[0], self.window[1], self.grid_dim[0])
        y_points = np.linspace(self.window[2], self.window[3], self.grid_dim[1])

        if self.dim == 2:
            C1, C2 = np.meshgrid(x_points, y_points, indexing="ij")
            grid_flat = np.stack((C1.flatten(), C2.flatten()), axis = 1)
            lambda_mat = np.reshape(lambda_(grid_flat), (self.grid_dim[0], self.grid_dim[1]))
        elif self.dim == 3:
            z_points = np.linspace(self.window[4], self.window[5], self.grid_dim[2])
            C1, C2, C3 = np.meshgrid(x_points, y_points, z_points, indexing="ij")
            grid_flat = np.stack((C1.flatten(), C2.flatten(), C3.flatten()), axis = 1)
            lambda_mat = np.reshape(lambda_(grid_flat), (self.grid_dim[0], self.grid_dim[1], self.grid_dim[2]))

        M = np.max(lambda_mat)

        # Simulate the homogeneous Poisson point process
        homogeneous_points = self.homogeneous(M)

        # Do the thinning based on the intensity function
        pi = lambda_(homogeneous_points)/M
        keep_bool = np.random.random(homogeneous_points.shape[0]) < pi
        points = homogeneous_points[keep_bool, :]
        return points


class CoxProcess(object):
    """
    Simulate Cox processes.
    """
    def __init__(self, window, Phi=None):
        """
        Initialize settings.
        """
        self.window = window
        self.dim = len(self.window)//2

    def simulate_driving_process(self, kappa, extend_window=False, type_="Poisson", **kwargs):
        """
        Used for shot noise Cox processes: Simulates the driving process as a
        homogeneous Poisson point process on an extended window.
        """
        if extend_window is False:
            window = np.copy(self.window)
        elif extend_window is True:
            window = np.copy(self.window)
            window[::2] -= kwargs["extend_dist"]
            window[::2] += kwargs["extend_dist"]

        if type_ == "Poisson":
            Phi = PoissonPointProcess(window, kappa).points
        elif type_ == "MaternTypeII":
            Phi = MaternTypeII(window).simulate_MaternTypeII(kappa, kwargs["HardCoreRadius"])
        return Phi

    def remove_points_outside_window(self, points, window):
        """
        Generic method to remove points in an array size=(N, d) which is outside
        the d-dimensional observation window.
        """
        if len(window)//2 == 3:
            x_remove = np.hstack((np.argwhere(points[:, 0]<window[0])[:, 0], np.argwhere(points[:, 0]>window[1])[:, 0]))
            y_remove = np.hstack((np.argwhere(points[:, 1]<window[2])[:, 0], np.argwhere(points[:, 1]>window[3])[:, 0]))
            z_remove = np.hstack((np.argwhere(points[:, 2]<window[4])[:, 0], np.argwhere(points[:, 2]>window[5])[:, 0]))
            remove = list(np.unique(np.hstack((x_remove, y_remove, z_remove))))
        elif len(window)//2 == 2:
            x_remove = np.hstack((np.argwhere(points[:, 0]<window[0])[:, 0], np.argwhere(points[:, 0]>window[1])[:, 0]))
            y_remove = np.hstack((np.argwhere(points[:, 1]<window[2])[:, 0], np.argwhere(points[:, 1]>window[3])[:, 0]))
            remove = list(np.unique(np.hstack((x_remove, y_remove))))
        keep = np.array([False if idx in remove else True for idx in range(points.shape[0])])
        points = points[keep]
        return points

    def simulate_Thomas(self, Phi, alpha, omega=1, window_restriction=False):
        """
        Simulate points from a Thomas process with driving process Phi with
        expected cluster size alpha and variance omega.
        """
        assert Phi.shape[1] == self.dim, "Input driving process must be of the proper dimensionality!"
        card_Phi = Phi.shape[0]
        cluster_sizes = np.random.poisson(alpha, size=card_Phi)
        daughter_points = list()
        for c, size in zip(Phi, cluster_sizes):
            daughter_points.append(np.random.multivariate_normal(c, np.eye(self.dim)*omega, size=size))
        daughter_points_arr = np.concatenate(daughter_points)
        if window_restriction is True:
            points = self.remove_points_outside_window(daughter_points_arr, self.window)
        else:
            points = daughter_points_arr
        return points

    def simulate_doubly_inhomogeneous_Neyman_Scott(self, Phi, alpha, omega, window_restriction=False):
        """
        Simulate points from a doubly inhomogeneous Neyman-Scott proces
        with driving process Phi and expected cluster size defined by the
        function alpha and covariance matrices defined by the function
        omega.
        """
        assert Phi.shape[1] == self.dim, "Input driving process must be of the proper dimensionality!"
        daughter_points = list()
        for c in Phi:
            daughter_points.append(np.random.multivariate_normal(c, omega(c)[0], size=np.random.poisson(alpha(c)[0])))
        if window_restriction is True:
            points = [self.remove_points_outside_window(daughter_points[l], self.window) for l in range(Phi.shape[0])]
        else:
            points = daughter_points
        return points

    def simulate_doubly_inhomogeneous_generalized_shot_noise_Cox_process(self, Tilde_Phi, alpha, omega, window_restriction=False):
        """
        Simulate points from a doubly inhomogeneous generalized shot noise
        Cox process with driving process \Tilde{Phi}=\sum_l \delta_{C_l}
        where C_l = (c_l, \sigma_l^E) consisting of cluster center positions
        and cluster extent standard deviation.
        The expected cluster size is given as alpha(c)*(1+(\sigma_l^E)^{2d})
        and the cluster spread is given as omega(c)+(\sigma_l^E)^2 I_d.
        """
        assert Tilde_Phi.shape[1]-1 == self.dim, "Input driving process must be of the proper dimensionality!"
        daughter_points = list()
        for C in Tilde_Phi:
            c, sigma_E = C[:self.dim], C[-1]
            daughter_points.append(np.random.multivariate_normal(c, omega(c)[0] + sigma_E**2*np.eye(self.dim),
                                                                 size=np.random.poisson(alpha(c)[0]*(1+sigma_E**(2*self.dim)))))
        if window_restriction is True:
            points = [self.remove_points_outside_window(daughter_points[l], self.window) for l in range(Phi.shape[0])]
        else:
            points = daughter_points
        return points


class MaternTypeII(object):
    """
    Simulate hard core process. This is done approximately by simulating
    a Matern Type II point process.

    Attributes:
    -----------
        window : list
            The boundary of the assumed rectangular domain. In 3D space the
            list should be specified as
                    [x_min, x_max, y_min, y_max, z_min, z_max]
        dim : int
            The dimension of the space.
    """
    def __init__(self, window):
        """
        Initialize settings.
        """
        self.window = window
        self.dim = len(self.window)//2

    def simulate_MaternTypeII(self, lambda_, R):
        """
        Simulate a Matern Type II point process with adjusted intensity
        lambda_ and interaction radius R.

        Inputs:
        -------
            lambda_ : float > 0
                The hard core adjusted intensity accounting for the
                approximate sampling method.
            R : float > 0
                The interaction radius, i.e., no two points can be
                within distance R.

        Output:
        -------
            points : ndarray, size=(M, dim)
                The simulated points.
        """
        assert lambda_*np.pi*R**2 < 1, "You be packing these points too tight, mon!"
        lambda_M = -np.log(1-lambda_*np.pi*R**2)/(np.pi*R**2)

        ext_window = [self.window[i] - R if i % 2 == 0 else self.window[i] + R for i in range(len(self.window))]
        ext_PPP_points = PoissonPointProcess(ext_window, lambda_M).points
        ext_PPP_age = np.random.rand(ext_PPP_points.shape[0])
        if self.dim == 1:
            bool_in_window = (ext_PPP_points[:, 0] > self.window[0]) & (ext_PPP_points[:, 0] < self.window[1])
        elif self.dim == 2:
            bool_in_window = (ext_PPP_points[:, 0] > self.window[0]) & (ext_PPP_points[:, 0] < self.window[1]) & \
                    (ext_PPP_points[:, 1] > self.window[2]) & (ext_PPP_points[:, 1] < self.window[3])
        elif self.dim == 3:
            bool_in_window = (ext_PPP_points[:, 0] > self.window[0]) & (ext_PPP_points[:, 0] < self.window[1]) & \
                    (ext_PPP_points[:, 1] > self.window[2]) & (ext_PPP_points[:, 1] < self.window[3]) & \
                    (ext_PPP_points[:, 2] > self.window[4]) & (ext_PPP_points[:, 2] < self.window[5])
        else:
            assert False, "Only 1, 2, and 3 dimensional data is supported!"
        index_window = np.arange(ext_PPP_points.shape[0])[bool_in_window]
        PPP_points = ext_PPP_points[bool_in_window]

        dist = PPP_points[:, None, :] - ext_PPP_points[None, :, :]
        dist_norm = np.linalg.norm(dist, axis=-1)
        bool_keep = np.zeros(PPP_points.shape[0], dtype=bool)
        for i in range(PPP_points.shape[0]):
            bool_in_disk  = (dist_norm[i, :] < R) & (dist_norm[i, :] > 0)
            if np.sum(bool_in_disk) == 0:
                bool_keep[i] = True
            else:
                bool_keep[i] = all(ext_PPP_age[index_window[i]] < ext_PPP_age[bool_in_disk])
        Mat_points = PPP_points[bool_keep]
        return Mat_points

def func(x):
    return 1/(np.linalg.norm(x, axis=-1)+4)*50


if __name__ == "__main__":
    plt.style.use("ggplot")
    use = 2

    if use == 0:
        window = [-5, 5, -5, 5]
        grid_dim = [10, 10]
        a = PoissonPointProcess(window, func, grid_dim)
        UE = a.points
    
        plt.rcParams['axes.grid'] = False
        x_points = np.linspace(window[0], window[1], grid_dim[0])
        y_points = np.linspace(window[2], window[3], grid_dim[1])
        C1, C2 = np.meshgrid(x_points, y_points)
        grid_flat = np.stack((C1.flatten(), C2.flatten()), axis = 1)
        lambda_mat = np.reshape(np.array([func(grid_flat[i, :]) for i in
                                range(grid_dim[0]*grid_dim[1])]),
                                (grid_dim[0], grid_dim[1]))
        spec = plt.pcolormesh(x_points, y_points,
                              lambda_mat.T, cmap="cool",
                              vmin=np.min(lambda_mat), vmax=np.max(lambda_mat), shading="auto")
        cb = plt.colorbar(spec)
        cb.set_label(label="Intensity")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(UE[:, 0], UE[:, 1], color="tab:orange", s=4)
        plt.show()
    elif use == 1:
        window = [-10, 10, -10, 10]
        Cox = CoxProcess(window)
        Psi, Phi = Cox.simulate_Thomas(2e-02, 7)

        plt.plot(Psi[:, 0], Psi[:, 1], "x", color="tab:green", markersize=4)
        plt.plot(Phi[:, 0], Phi[:, 1], "o", color="tab:red", markersize=6)
        plt.show()
    elif use == 2:
        intensity = np.zeros(1000)
        for j in range(1000):
            window = [-10, 10, -10, 10]
            Mat = MaternTypeII(window)
            Phi = Mat.simulate_MaternTypeII(0.05, 2)
            area = (window[1]-window[0])*(window[3]-window[2])
            intensity[j] = Phi.shape[0]/area
        intensity_trace = [np.mean(intensity[:k]) for k in range(1, 1000)]
        plt.plot(intensity_trace)
        plt.show()

        plt.plot(Phi[:, 0], Phi[:, 1], "o", color="tab:red", markersize=6)
        plt.show()