# quakefibre

## Description
A toolbox to simulate submarine optical fibre propagation using the earthquake-perturbed coupled Schrödinger equation.
Earthquake seismograms are obtained through HTTP request to the [Syngine](https://ds.iris.edu/ds/products/syngine/) web service [\[1\]](#1).
Resulting fibre strain is modelled as a change in the birefringence, like in [\[2,](#2)[3\]](#3).
This repository is the implementation of [\[4\]](#4).


## Directory structure
- `src`: the toolbox source files.
- `scripts`: scripts that reproduce the results from [\[4\]](#4).
- `config`: model parameters to pass to the scripts.
- `tests`: unittests that were used to test correctness of the toolbox.


## Installation
1. Install [Python](https://www.python.org) or [Anaconda](https://www.anaconda.com).  
2. (Recommended) create a new virtual environment  
   - with [venv](https://docs.python.org/3/library/venv.html):  
     `python -m venv env`, `python3 -m venv env`, or `py -3 -m venv env` (depending on your system).  
     `call env/Scripts/activate` (Windows), or `source env/bin/activate` (Linux/MacOSX).
   - with [Anaconda](https://www.anaconda.com/download):  
     `conda create -n quakefibre`.  
     `conda activate quakefibre`.
6. Install quakefibre ([set up a personal SSH key pair](https://docs.gitlab.com/user/ssh/) first):  
      ```bat
      pip install "quakefibre @ git+ssh://git@github.com:RM-8vt13r/quake-fibre.git"
      ```
      On a system with a compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installation: `quakefibre[cuda..x]`, replacing `..` with `11`, `12` or `13`.  
      On a system without a compatible CUDA installation, but with a CUDA-enabled GPU: `quakefibre[cuda..x-local]`.  
      If you intend to run the scripts in `scripts` or unittests in `tests`: `quakefibre[testing]`.  
      Note that tags can be combined, e.g., `quakefibre[testing,cuda..x-local]`.  


## Testing and scripts
To reproduce the results from [\[4\]](#4):  
```bat
python scripts/demo_end_to_end.py --configs config/earthquake_oaxaca.ini fibre_curie.ini signal_continuous.ini transceiver_curie.ini --out results --make-out --alpha 1.5
```

To run unit tests:  
```bat
pytest tests
```

## Usage
Take a look at the `scripts` folder for a demonstration.  
Or, check out this simple example:
```python
from configparser import ConfigParser
from quakefibre import FibreCNLSE, Transceiver, EarthquakeSubmarine

# Load predefined parameters describing a fibre, earthquake, and transceiver
parameters = ConfigParser()
parameters.read([
        'config/earthquake_oaxaca.ini',
        'config/fibre_curie.ini',
        'config/signal_continuous.ini',
        'config/transceiver_curie.ini'
    ])

# Create a transceiver and transmit a continuous wave
transceiver = Transceiver(parameters)
transmitted_signal = transceiver.transmit_continuous(
        symbol = [1, 0],
        symbol_count = parameters.getint('SIGNAL', 'symbol_count'),
        carrier_wavelength = parameters.getfloat('SIGNAL', 'carrier')
    )

# Create an optical fibre model and earthquake
fibre = FibreCNLSE(parameters)
earthquake = EarthquakeSubmarine(parameters)

# Obtain fibre strain and propagate the signal
earthquake_perturbation = earthquake(fibre.step_path)
received_signal = fibre(transmitted_signal, perturbations = earthquake_perturbation)
```


## References
<a name="1">\[1\]</a>
L. Krischer, A. R. Hutko, M. van Driel, *et al.*,
"On-demand custom broad-band synthetic seismograms,"
*Seismol. Res. Lett.*,
vol. 88, no. 4, pp. 1127&ndash;1140,
Jul. 2017.
DOI: [10.1785/0220160210](https://doi.org/10.1785/0220160210)

<a name="2">\[2\]</a>
A. Mecozzi, M. Cantono, J. C. Castellanos, *et al.*,
"Polarization Sensing using Submarine Optical Cables,"
*Opt.*,
vol. 8, no. 6, pp. 788&ndash;795,
Jun. 2021.
DOI: [10.1364/OPTICA.424307](https://doi.org/10.1364/OPTICA.424307)

<a name="3">\[3\]</a>
H. Awad, F. Usmani, E. Virgillito, *et al.*,
"Environmental surveillance through machine learning-empowered utilization of optical networks,"
*Sens.*,
vol. 24, no. 10, pp. 3041,
May 2024.
DOI: [10.3390/s24103041](https://doi.org/10.3390/s24103041)

<a name="4">\[4\]</a>
R. M. Butler, J. Núñez-Kasaneva, *et al.*,
"End-to-End Modelling of Earthquake-Induced Polarisation Perturbations in Submarine Optical Fibres,"
*Eur. Conf. Opt. Commun.*,
In Review.


## Citation
If you use this work, please cite [\[4\]](#4).

## Authors and acknowledgment
Maintained by Rick M. Butler.  

## License
This project is published under the MIT license.
