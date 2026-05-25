"""
A script demonstrating the entire end-to-end chain, propagating a continuous wave through a fibre during an earthquake.
"""

# Imports
from configparser import ConfigParser
import argparse
import os
import sys
import time
import logging
import csv
from requests import HTTPError, ConnectionError, Timeout

from quakefibre import FibreCNLSE, EarthquakeSubmarine, Transceiver, Device, Perturbation, Filter

import numpy as np
try:
    import cupy as cp
except:
    pass
import netCDF4 as nc
from obspy.clients.base import ClientHTTPException

# Configuration
logging.basicConfig(level = logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

if __name__ == '__main__':
    # Load command line flags
    logger.info("Reading parameters..")

    parser = argparse.ArgumentParser("Transmit a continuous-wave signal through an optical fibre during an earthquake. Save the input- and output signals in Jones space.")
    parser.add_argument("--configs", help = "Paths from the current working directory to all configuration files to load, detailing information on the fibre, earthquake, and signal to transmit. The configuration file(s together) must contain sections FIBRE, EARTHQUAKE, TRANSCEIVER and SIGNAL.", type = str, nargs = '+', required = True)
    parser.add_argument("--GPU", help = "Index of the CUDA-enabled GPU to use. Leave out to run on CPU.", type = int, required = False)
    parser.add_argument("--out", help = "Directory path from the current working directory where to save all results.", type = str, required = True)
    parser.add_argument("--make-out", help = "If --out doesn't exist, and the --make-out flag is passed, create a new directory at --out.", action = argparse.BooleanOptionalAction)
    parser.add_argument("--perturbation", help = "File path from the current working directory where to save/load earthquake strains obtained from Syngine. If not defined, don't save or load earthquake strains from disk.", type = str, required = False)
    parser.add_argument("--make-perturbation", help = "If the directory to --perturbation doesn't exist, and the --make-perturbation flag is passed, create a new directory to --perturbation.", type = bool, action = argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite-perturbation", help = "If --perturbation exists, delete it and rebuild it from scratch.", type = bool, action = argparse.BooleanOptionalAction)
    parser.add_argument("--alpha", help = "Extra parameter with which to scale the fibre strain.", type = float, nargs = '+', default = [1,])
    arguments = parser.parse_args()

    # Verify validity of passed flags
    for config in arguments.configs:
        assert config.endswith('.ini'), f"Config path must end with .ini, but was \"{config}\""
        assert os.path.isfile(config), f"Config \"{config}\" doesn't exist or is not a file"

    if 'GPU' in arguments:
        try:
            assert 'cupy' in sys.modules
            a = cp.array([1, 2, 3])
            a + a
        except:
            raise AssertionError("--GPU was specified, but cupy is not operational. Make sure you have a CUDA-enabled GPU, and installed the [cuda..x] or [cuda..x-local] version of quakefibre (see README.md)")

        cp.cuda.Device(arguments.GPU).use()

    if not os.path.exists(arguments.out) and arguments.make_out:
        logger.info(f"Output path \"{arguments.out}\" doesn't exist. Creating new directory..")
        os.makedirs(arguments.out)

    assert os.path.isdir(arguments.out), f"Output path \"{arguments.out}\" doesn't exist or is not a directory"

    if 'perturbation' in arguments:
        perturbation_directory, _ = os.path.split(arguments.perturbation)
        
        if len(perturbation_directory) == 0:
            perturbation_directory = '.'
        
        if not os.path.exists(perturbation_directory) and arguments.make_perturbation:
            logger.info(f"Perturbation directory \"{perturbation_directory}\" doesn't exist. Creating new directory..")
            os.makedirs(perturbation_directory)

        if os.path.exists(arguments.perturbation) and arguments.overwrite_perturbation:
            os.remove(arguments.perturbation)

        assert os.path.isdir(perturbation_directory), f"Perturbation directory \"{perturbation_directory}\" doesn't exist or is not a directory"

    else:
        if 'make_perturbation' in arguments:
            logger.warning("Flag --make-perturbation was passed, but doesn't do anything as --perturbation wasn't passed")
        if 'overwrite_perturbation' in arguments:
            logger.warning("Flag --overwrite-perturbation was passed, but doesn't do anything as --perturbation wasn't passed")

    # Load system parameters
    parameters = ConfigParser(inline_comment_prefixes = '#')
    parameters.read(arguments.configs)

    # Create system
    logger.info("Initialising simulation..")

    transceiver = Transceiver(parameters)
    
    fibre = FibreCNLSE(parameters)
    
    earthquake = EarthquakeSubmarine(parameters)
    earthquake_path = fibre.path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length'))
    
    filter_frequencies = []
    filter_responses = []
    with open(parameters.get('EARTHQUAKE', 'filter_csv_path'), newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            filter_frequencies.append(float(row[0]))
            filter_responses.append(float(row[1]))

    pressure_filter = Filter(filter_frequencies, filter_responses)

    # Transmit a continuous-wave signal
    logger.info("Transmitting signal..")
    signal = transceiver.transmit_continuous(
            symbol = [1, 0],
            symbol_count = parameters.getint('SIGNAL', 'symbol_count'),
            carrier_wavelength = parameters.getfloat('SIGNAL', 'carrier')
        )

    # Save transmitted signal
    signal_dataset = nc.Dataset(os.path.join(arguments.out, 'transmitted_signal.nc'), 'w')
    signal.save(signal_dataset)

    # At what times to start transmitting the signal, to 'catch' the earthquake at different moments
    transmission_start_times = np.arange(parameters.getfloat('SIGNAL', 'transmission_start'), parameters.getfloat('SIGNAL', 'transmission_stop'), parameters.getfloat('SIGNAL', 'transmission_step')) # Zhan et al. measured 20 samples / second; transmit a short signal every 1 / 20 seconds

    # Propagate signals
    logger.info(f"Starting signal propagation")

    # Propagate the signal
    propagated_signals = []
    for _ in arguments.alpha:
        propagated_signals.append(signal.copy())
        if 'GPU' in arguments: propagated_signals[-1].to_device(Device.CUDA)

    # Consider one piece of the fibre at a time, to limit memory usage
    steps_per_piece = parameters.getint('FIBRE', 'steps_per_piece')
    step_starts = np.arange(0, fibre.step_path.edge_count, steps_per_piece)

    start_time = time.time()
    for piece_index, step_start in enumerate(step_starts):
        logger.info(f"Evaluating fibre piece {piece_index + 1} of {len(step_starts)}")
        
        step_stop = step_start + steps_per_piece
        piece_step_path = fibre.step_path[step_start:step_stop + 1]
        piece_earthquake_path_start = max(0, np.min(np.where(earthquake_path.positions > fibre.step_path.positions[step_start])) - 1)
        piece_earthquake_path_stop  = min(len(earthquake_path), np.max(np.where(earthquake_path.positions < fibre.step_path.positions[step_stop])) + 1)
        
        # Obtain strain along this piece from Syngine or the saved perturbation
        perturbation = None
        if 'perturbation' in arguments:
            logger.info(f"Attempt to load fibre strains from file..")
            with nc.Dataset(arguments.perturbation, 'r') as perturbation_dataset:
                try:
                    perturbation = Perturbation.load(perturbation_dataset, step_start = piece_earthquake_path_start, step_stop = piece_earthquake_path_stop)
                    logger.info(f"..succeeded")
                    perturbation_loaded = True
                except:
                    logger.info(f"..failed")
                    perturbation_loaded = False
        
        # if len(syngine_earthquake_piece_step_path)
        if perturbation is None:
            logger.info(f"Requesting fibre strains from Syngine")
            timeout = 1
            max_timeout = 600
            while perturbation is None:
                try:
                    perturbation = earthquake(
                            path          = earthquake_path[piece_earthquake_path_start:piece_earthquake_path_stop],
                            duration      = parameters.getfloat('EARTHQUAKE', 'duration'),
                            batch_size    = parameters.getint('EARTHQUAKE', 'batch_size'),
                            worker_count  = parameters.getint('EARTHQUAKE', 'worker_count'),
                            request_delay = parameters.getfloat('EARTHQUAKE', 'request_delay')
                        )
                    perturbation = pressure_filter(perturbation)

                except (ClientHTTPException, HTTPError, ConnectionError, Timeout):
                    time.sleep(timeout)
                    timeout = max(2 * timeout, max_timeout) # Exponential backoff

        if 'perturbation' in arguments and not perturbation_loaded:
            logger.info(f"Saving fibre strains to file")
            with nc.Dataset(arguments.perturbation, 'a') as perturbation_dataset:
                perturbation.save(perturbation_dataset, piece_earthquake_path_start)

        # Propagate signal through fibre for all values of alpha
        for alpha_index, alpha in enumerate(arguments.alpha):
            logger.info(f"Propagating for alpha value {alpha_index + 1} of {len(arguments.alpha)} ({alpha})")
            perturbation_alpha = Perturbation(
                    start_time = perturbation.start_time,
                    strains = perturbation.strains * alpha,
                    twists = perturbation.twists,
                    sample_rate = perturbation.sample_rate,
                    domain = perturbation.domain
                )
            propagated_signals[alpha_index] = fibre(
                    signal = propagated_signals[alpha_index],
                    transmission_start_times = transmission_start_times,
                    perturbations = perturbation_alpha,
                    step_start = step_start,
                    step_stop = step_stop
                )
            del perturbation_alpha
        del perturbation

        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / (piece_index + 1) * (len(step_starts) - (piece_index + 1))
        logger.info(f"Evaluated {piece_index + 1} of {len(step_starts)} fibre pieces in {round(elapsed_time / 60, 1)} minutes ({round(elapsed_time / 3600, 1)} hours), an estimated {round(remaining_time / 60, 1)} minutes ({round(remaining_time / 3600, 1)} hours) remain")

    # Save received signal
    logger.info(f"Saving results")
    for propagated_signal, alpha in zip(propagated_signals, arguments.alpha):
        if 'GPU' in arguments: propagated_signal.to_device(Device.CPU)
        with nc.Dataset(os.path.join(arguments.out, f'propagated_signal_alpha={alpha}.nc'), 'w') as propagated_signal_dataset:
            propagated_signal.save(propagated_signal_dataset)