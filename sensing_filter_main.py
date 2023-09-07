# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:56:22 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Experiment script calling modules, loading sensing_filter.toml
           file, running sensing_filter class, and saving results.
           Has capability for multiprocessing Monte Carlo
           simulations. (25/08/2023)
"""


import os

import numpy as np
import matplotlib.pyplot as plt
import toml
import pickle

import line_profiler
profile = line_profiler.LineProfiler()

import multiprocessing as mp
from tqdm import tqdm

from sensing_filter import main_filter, simulate_driving_process


def fun_mp(alg, centers, i):
    """
    Run scenario and get sensing algorithm results.
    """
    print(f"Beginning simulation {i+1}")
    res_dict = alg(centers)
    return res_dict

def sim_mp(sims, alg, centers, N_workers=6):
    """
    Multiprocessing: Parallelisation over Monte Carlo runs.
    """
    # Pool with progress bar
    pool = mp.Pool(processes=N_workers,
                   initargs=(mp.RLock(),),
                   initializer=tqdm.set_lock)

    # run multiprocessing
    jobs = [pool.apply_async(fun_mp, args=(alg, centers[i], i))
            for i in range(sims)]
    pool.close()
    pool.join()

    # stack results
    pred_Map_list = list()
    for i in range(sims):
        pred_Map_list.append(jobs[i].get())
    return pred_Map_list

def get_driving_processes(lambda_Poi, hard_core_radius, extent_prior, sigma_extent_min,
                          sigma_extent_max, extent_IW_df, extent_IW_scale,
                          cor_extent_min, cor_extent_max,
                          use_fixed, use_saved, window, sims, savename):
    """
    Parameters
    ----------
    lambda_Poi : float
        The expected number of parent points.
    hard_core_radius : float
        The hard core radius.
    use_fixed : bool
        If true, use a fixed number of parent points (= lambda_Poi)
    use_saved : bool
        If true, use saved driving process.
    window : list, len=(8,)
        The rectangular boundaries of the point process domain.
    sims : int
        The number of Monte Carlo simulations.

    Returns
    -------
    driving_process : list of ndarrays of size=(L, d+e)
        The driving process consisting of L parent points with a
        spatial position and an object extent for each Monte Carlo simulation.
    """
    if use_saved is False:
        area = np.prod([window[2*i+1]-window[2*i] for i in range(3)])
        lambda_SP = lambda_Poi/area
        driving_process = list()
        for i in range(sims):
            Tilde_Phi = simulate_driving_process(lambda_SP, hard_core_radius, window, extent_prior, sigma_extent_min,
                                                 sigma_extent_max, extent_IW_df, extent_IW_scale, cor_extent_min, cor_extent_max)
            if use_fixed:
                L = int(lambda_SP+1)
                while Tilde_Phi.shape[0] < L:
                    Tilde_Phi = simulate_driving_process(lambda_SP, hard_core_radius, window, extent_prior, sigma_extent_min,
                                                         sigma_extent_max, extent_IW_df, extent_IW_scale, cor_extent_min, cor_extent_max)
            assert Tilde_Phi.shape[0] > 0, "The driving process must not be empty!"
            driving_process.append(Tilde_Phi)
        with open(f"Process_Data/driving_process/{use_fixed}_lambdaPoi{lambda_Poi}_{savename}.pickle", "wb") as file:
            pickle.dump(driving_process, file, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"Process_Data/driving_process/{use_fixed}_lambdaPoi{lambda_Poi}_{savename}.pickle", "rb") as file:
            driving_process = pickle.load(file)
    return driving_process


if __name__ == "__main__":
    plt.style.use("ggplot")

    toml_in = toml.load("sensing_filter.toml")
    toml_settings = toml_in["channel_settings"]
    toml_scenario = toml_in["scenario_settings"]
    toml_booleans = toml_in["boolean_specifications"]
    toml_experiment = toml_in["experiment_settings"]
    toml_oracle = toml_in["oracle_hparams"]
    toml_dbscan = toml_in["DBSCAN_hparams"]
    toml_MCMC = toml_in["MCMC_hparams"]

    seed = toml_experiment["seed"]
    verbose = toml_booleans["verbose"]
    np.random.seed(seed)

    savename = toml_experiment["savename"]
    # if toml_booleans["use_directional_RIS"] is True:
    #     folder = f"Results/{toml_experiment['sensing_alg']}/Directional_RIS2/{savename}"
    # else:
    #     folder = f"Results/{toml_experiment['sensing_alg']}/Random_RIS/{savename}"
    folder = f"Results/{savename}"

    # =============================================================================
    # Prepare input data structure
    # =============================================================================
    N_workers = toml_experiment["N_mp_workers"]

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

    extent_IW_df = toml_MCMC["extent_IW_df"]
    extent_IW_scale = toml_MCMC["extent_IW_scale"]*(extent_IW_df-3-1)*np.eye(3)
    toml_MCMC["extent_IW_scale"] = extent_IW_scale

    if toml_booleans["save_results"] is True:
        try:
            os.mkdir(folder)
        except OSError as error:
            print(error)

        with open(f"{folder}/{savename}.toml", "w") as toml_file:
            toml.dump(toml_in, toml_file)

    # =============================================================================
    # Run algorithm
    # =============================================================================
    driving_processes = get_driving_processes(toml_scenario["lambda_Poi"], toml_scenario["R"], toml_MCMC["extent_prior"],
                                              toml_MCMC["sigma_extent_min"], toml_MCMC["sigma_extent_max"],
                                              toml_MCMC["extent_IW_df"], toml_MCMC["extent_IW_scale"],
                                              toml_MCMC["cor_extent_min"], toml_MCMC["cor_extent_max"],
                                              toml_booleans["fixed_number_of_targets"], toml_booleans["use_saved_centers"],
                                              toml_scenario["window"], toml_experiment["sims"], savename)

    # =============================================================================
    # Main loop
    # =============================================================================
    for signal_types in toml_experiment["signal_types_list"]:
        for data_type in toml_experiment["data_types_list"]:

            if data_type == "One_User":
                resource_division = None
            elif data_type == "time_division":
                resource_division = "time_division"
            elif data_type == "frequency_division":
                resource_division = "frequency_division"

            if signal_types == "all":
                signal_types_in = ["nonRIS", "directional", "orthogonal"]
            elif signal_types == "nonRIS_directional":
                signal_types_in = ["nonRIS", "directional"]
            if signal_types == "nonRIS":
                signal_types_in = ["nonRIS"]
            if signal_types == "directional":
                signal_types_in = ["directional"]
            if signal_types == "RIS":
                signal_types_in = ["directional", "orthogonal"]
            if signal_types == "orthogonal":
                signal_types_in = ["orthogonal"]

            experiment_kwargs.update({"resource_division": resource_division,
                                      "signal_types": signal_types_in})

            # =============================================================================
            # Sensing
            # =============================================================================
            print(f"\n\nSignal types: {signal_types}\nData type: {data_type}")
            alg = main_filter(toml_experiment["sensing_alg"], toml_oracle, toml_dbscan, toml_MCMC, verbose, **experiment_kwargs)

            if toml_booleans["use_mp"] is True:
                res_dict = sim_mp(toml_experiment["sims"], alg, driving_processes, N_workers)
            else:
                # stack results
                res_dict = list()
                for i in range(toml_experiment["sims"]):
                    res_dict.append(fun_mp(alg, driving_processes[i], i))

            # =============================================================================
            # Save sensing results
            # =============================================================================
            if toml_booleans["save_results"] is True:
                with open(f"{folder}/{data_type}_{signal_types}_{savename}.pickle", "wb") as file:
                    pickle.dump(res_dict, file, pickle.HIGHEST_PROTOCOL)

