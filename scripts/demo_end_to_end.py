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

from quakefibre import FibreCNLSE, EarthquakeSubmarine, Transceiver, Device, Perturbation, Signal, Filter, Dimension

import numpy as np
try:
    import cupy as cp
except:
    pass
import netCDF4 as nc
from obspy.clients.base import ClientHTTPException

if __name__ == '__main__':
    # Load command line flags
    parser = argparse.ArgumentParser(
        prog = 'python scripts/demo_end_to_end.py',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = "Transmit a continuous-wave signal through an optical fibre during an earthquake. Save the input- and output signals in Jones space. Requires the testing version of quakefibre to be installed (see README.md).",
        argument_default = argparse.SUPPRESS
    )
    parser.add_argument("--configs", help = "Paths from the current working directory to all configuration files to load, detailing information on the fibre, earthquake, and signal to transmit. The configuration file(s together) must contain sections FIBRE, EARTHQUAKE, TRANSCEIVER and SIGNAL.", type = str, nargs = '+', required = True)
    parser.add_argument("--alpha", help = "Extra parameter with which to scale the fibre strain.", type = float, nargs = '+', default = [1,], required = False)
    parser.add_argument("--out", help = "Directory path from the current working directory where to save all results.", type = str, required = True)
    parser.add_argument("--make-out", help = "If --out doesn't exist, and the --make-out flag is passed, create a new directory at --out.", action = argparse.BooleanOptionalAction, default = False, required = False)
    parser.add_argument("--perturbation", help = "File path from the current working directory where to save/load earthquake strains obtained from Syngine. If not defined, don't save or load earthquake strains from disk.", type = str, required = False)
    parser.add_argument("--make-perturbation", help = "If the directory to --perturbation doesn't exist, and the --make-perturbation flag is passed, create a new directory to --perturbation if necessary.", type = bool, action = argparse.BooleanOptionalAction, default = False, required = False)
    parser.add_argument("--overwrite-perturbation", help = "If --perturbation exists, delete it and rebuild it from scratch.", type = bool, action = argparse.BooleanOptionalAction, default = False, required = False)
    parser.add_argument("--fibre", help = "File path from the current working directory where to save/load fibre realisations. If not defined, don't save or load fibre realisations from disk.", type = str, required = False)
    parser.add_argument("--make-fibre", help = "If the directory to --fibre doesn't exist, and the --make-fibre flag is passed, create a new directory to --fibre if necessary.", type = bool, action = argparse.BooleanOptionalAction, default = False, required = False)
    parser.add_argument("--overwrite-fibre", help = "If --fibre exists, delete it and rebuild it from scratch.", type = bool, action = argparse.BooleanOptionalAction, default = False, required = False)
    parser.add_argument("--GPU", help = "Index of the CUDA-enabled GPU to use. Leave out to run on CPU. Requires the CUDA version of quakefibre to be installed (see README.md).", type = int, choices = [] if 'cupy' not in sys.modules else list(range(cp.cuda.runtime.getDeviceCount())), required = False)
    parser.add_argument("--compression", help = "Compression algorithm to use when saving results, the fibre, and/or perturbations; see https://unidata.github.io/netcdf4-python/#efficient-compression-of-netcdf-variables.", type = str, default = 'zlib', required = False)
    parser.add_argument("--compression-level", help = "The level of compression from 0 (no compression, fast read/write, large file) to 9 (maximum compression, slow read/write, small file).", type = int, choices = list(range(10)), default = 4, required = False)
    parser.add_argument("--logging-level", help = "At which level to print logging messages.", type = str, choices = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default = "NOTSET", required = False)
    arguments = parser.parse_args()

    # Configure logging
    logging.basicConfig(level = getattr(logging, arguments.logging_level), format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()

    logger.info("Reading parameters")
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
        logger.info(f"Output path \"{arguments.out}\" doesn't exist, creating new directory")
        os.makedirs(arguments.out)

    assert os.path.isdir(arguments.out), f"Output path \"{arguments.out}\" doesn't exist or is not a directory"

    for flag in 'fibre', 'perturbation':
        if flag in arguments:
            flag_directory, _ = os.path.split(getattr(arguments, flag))
            
            if len(flag_directory) == 0:
                flag_directory = '.'
            
            if not os.path.exists(flag_directory) and getattr(arguments, f'make_{flag}'):
                logger.info(f"{flag.capitalize()} directory \"{flag_directory}\" doesn't exist, creating new directory")
                os.makedirs(flag_directory)

            if os.path.exists(getattr(arguments, flag)) and getattr(arguments, f'overwrite_{flag}'):
                os.remove(getattr(arguments, flag))

            assert os.path.isdir(flag_directory), f"{flag.capitalize()} directory \"{flag_directory}\" doesn't exist or is not a directory"

        else:
            if getattr(arguments, f'make_{flag}'):
                logger.warning(f"Flag --make-{flag} was passed, but doesn't do anything as --{flag} wasn't passed")
            if getattr(arguments, f'overwrite_{flag}'):
                logger.warning(f"Flag --overwrite-{flag} was passed, but doesn't do anything as --{flag} wasn't passed")

    # Load system parameters
    parameters = ConfigParser(inline_comment_prefixes = '#')
    parameters.read(arguments.configs)

    # Create system
    logger.info("Initialising simulation")

    transceiver = Transceiver(parameters)
    
    write_fibre = 'fibre' in arguments and (arguments.overwrite_fibre or not os.path.isfile(arguments.fibre))
    if 'fibre' in arguments and not write_fibre:
        with nc.Dataset(arguments.fibre, 'r') as fibre_dataset:
            fibre = FibreCNLSE.load(fibre_dataset)
    else:
        fibre = FibreCNLSE(parameters)
        if write_fibre:
            logger.info("Saving fibre")
            with nc.Dataset(arguments.fibre, 'w') as fibre_dataset:
                fibre.save(fibre_dataset, compression = arguments.compression, compression_level = arguments.compression_level)
    
    earthquake = EarthquakeSubmarine(parameters)
    earthquake_path = fibre.path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length'))
    
    pressure_filter = None
    if 'filter_csv_path' in parameters['EARTHQUAKE']:
        filter_frequencies = []
        filter_responses = []
        with open(parameters.get('EARTHQUAKE', 'filter_csv_path'), newline = '') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')
            for row in reader:
                filter_frequencies.append(float(row[0]))
                filter_responses.append(float(row[1]))

        pressure_filter = Filter(filter_frequencies, filter_responses)

    # Transmit a continuous-wave signal
    logger.info("Transmitting signal")
    signal = transceiver.transmit_continuous(
            symbol = [1, 0],
            symbol_count = parameters.getint('SIGNAL', 'symbol_count'),
            carrier_wavelength = parameters.getfloat('SIGNAL', 'carrier')
        )

    # Save transmitted signal
    with nc.Dataset(os.path.join(arguments.out, 'transmitted_signal.nc'), 'w') as signal_dataset:
        signal.save(signal_dataset, compression = arguments.compression, compression_level = arguments.compression_level)

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
        
        step_stop = min(step_start + steps_per_piece, fibre.step_path.edge_count)
        piece_step_path = fibre.step_path[step_start:step_stop]
        piece_earthquake_path_start = max(0, np.min(np.where(earthquake_path.centre_positions > fibre.step_path.centre_positions[step_start])) - 1)
        piece_earthquake_path_stop  = min(len(earthquake_path), np.max(np.where(earthquake_path.centre_positions < fibre.step_path.centre_positions[step_stop - 1])) + 2)
        piece_earthquake_path = earthquake_path[piece_earthquake_path_start:piece_earthquake_path_stop]
        
        # Obtain strain along this piece from Syngine or the saved perturbation
        water_depths = None
        normal_accelerations = None
        if 'perturbation' in arguments:
            logger.info(f"Attempt to load perturbations from file..")
            try:
                with nc.Dataset(arguments.perturbation, 'r') as perturbation_dataset:
                    # perturbation = Perturbation.load(perturbation_dataset, step_start = piece_earthquake_path_start, step_stop = piece_earthquake_path_stop)
                    water_depths = Signal.load(perturbation_dataset.groups['water_depths'], step_starts = {0: piece_earthquake_path_start}, step_stops = {0: piece_earthquake_path_stop})
                    normal_accelerations = Signal.load(perturbation_dataset.groups['normal_accelerations'], step_starts = {0: piece_earthquake_path_start}, step_stops = {0: piece_earthquake_path_stop})
                    logger.info(f"..succeeded")
                    perturbation_loaded = True
            except:
                logger.info(f"..failed")
                perturbation_loaded = False
        
        # if len(syngine_earthquake_piece_step_path)
        logger.info(f"Obtaining fibre strains")
        timeout = 1
        max_timeout = 600
        perturbation = None
        while perturbation is None:
            try:
                normal_accelerations, water_depths, fibre_strains = earthquake.request_fibre_strains(
                        path          = piece_earthquake_path,
                        duration      = parameters.getfloat('EARTHQUAKE', 'duration'),
                        batch_size    = parameters.getint('EARTHQUAKE', 'batch_size'),
                        worker_count  = parameters.getint('EARTHQUAKE', 'worker_count'),
                        request_delay = parameters.getfloat('EARTHQUAKE', 'request_delay'),
                        normal_accelerations = normal_accelerations,
                        water_depths  = water_depths,
                        return_normal_accelerations = True,
                        return_water_depths = True
                    )

                perturbation = Perturbation(
                        strains = fibre_strains.samples_time[:, :, 0],
                        sample_rate = fibre_strains.sample_rate
                    )

            except (ClientHTTPException, HTTPError, ConnectionError, Timeout):
                time.sleep(timeout)
                timeout = max(2 * timeout, max_timeout) # Exponential backoff

        if pressure_filter is not None:
            perturbation = pressure_filter(perturbation)

        if 'perturbation' in arguments and not perturbation_loaded:
            logger.info(f"Saving perturbations to file")
            with nc.Dataset(arguments.perturbation, 'a') as perturbation_dataset:
                for group in ('normal_accelerations', 'water_depths'):
                    if group not in perturbation_dataset.groups:
                        perturbation_dataset.createGroup(group)
                normal_accelerations.save(perturbation_dataset.groups['normal_accelerations'], step_starts = {0: piece_earthquake_path_start}, compression = arguments.compression, compression_level = arguments.compression_level)
                water_depths.save(perturbation_dataset.groups['water_depths'], step_starts = {0: piece_earthquake_path_start}, compression = arguments.compression, compression_level = arguments.compression_level)

        # Interpolate the perturbation from the sparse earthquake path to the dense fibre path
        perturbation = perturbation.interpolated(
                original_positions = earthquake_path.centre_positions[piece_earthquake_path_start] + piece_earthquake_path.centre_positions,
                new_positions = fibre.step_path.centre_positions[step_start] + piece_step_path.centre_positions
            )

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
            propagated_signal.save(propagated_signal_dataset, compression = arguments.compression, compression_level = arguments.compression_level)