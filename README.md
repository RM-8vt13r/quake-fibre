# tremor-waveplate-toolbox

## Description
A toolbox to simulate optical fibre strain from earthquake tremors using the waveplate model.
It models polarisation mode dispersion (PMD) as a constant differential group delay (DGD) along the fibre, and applies random rotations of the principal state of polarisation (PSP) after each correlation length.
Earthquake seismograms are obtained through https requests to the [Syngine](https://ds.iris.edu/ds/products/syngine/) web service [\[1\]](#1).
Resulting fibre strain is modelled as a change in DGD, like in [\[2,](#2)[3\]](#3).


## Demo
After [installing the toolbox](#testing-and-scripts), take a look at the notebooks in the `scripts' folder for demonstrations.


## Installation
To install and use this project:  
1. Install [Python](https://www.python.org) or [Anaconda](https://www.anaconda.com).  
2. Open a command line terminal in the directory of your project.  
3. (Recommended) create a new virtual environment:  
   - [venv](https://docs.python.org/3/library/venv.html):  
     `python -m venv env`,  
     `python3 -m venv env`, or  
     `py -3 -m venv env`  
     (depending on your OS and Python installation).  
   - [Anaconda](https://www.anaconda.com/download):  
     `conda create -n tremor-waveplate`.  
4. Activate the environment:  
   - venv:  
     `call env/Scripts/activate`
     on Windows, or  
     `source env/bin/activate`
     on Linux and MacOSX.  
   - Anaconda:  
     `conda activate tremor-waveplate`.  
5. Install this repository:  
   ```bat
   pip install git+https://gitlab.tue.nl/r.m.butler/tremor-waveplate-toolbox.git
   ```
   or, if you want to run unit tests from the 'tests' directory and/or demos from the 'scripts' directory (see [Testing and scripts](#testing_and_scripts)):  
   ```bat
   pip install "tremor-waveplate-toolbox[testing] @ git+https://gitlab.tue.nl/r.m.butler/tremor-waveplate-toolbox.git"
   ```
6. All done!


## Testing and scripts
To run scripts or unit tests from this project:  
1. Clone this repository:  
   `git clone git@gitlab.tue.nl:r.m.butler/tremor-waveplate-toolbox.git`  
2. Move into the newly created project folder.  
3. Follow steps 1-5 under [Installation](#installation).  
   Make sure to install the 'testing' version.  
4. To run scripts:  
   `jupyter notebook scripts`  
5. To run unit tests:  
   `pytest -s`  


## Usage
Take a look at the notebooks in the 'scripts' folder for specific demonstrations.  
After [installation](#installation), general usage looks like:
```python
from configparser import ConfigParser
from tremor_waveplate_toolbox import Fibre, Transmitter, Receiver, Earthquake

# Load predefined parameters describing a fibre and transceiver
parameters = ConfigParser()
parameters.read('config/parameters.ini')

# Create a transmitter, and generate a signal
transmitter = Transmitter(parameters)
transmitted_symbols, transmitted_signal = transmitter.transmit_random(shape = (
    parameters.getint('SIGNAL', 'batch_size'),
    int(parameters.getfloat('SIGNAL', 'symbol_count'))
))

# Create an optical fibre model, and transmit the signal
fibre = Fibre(parameters)
received_signal = fibre(transmitted_signal, verbose = True)

# Receive the signal
receiver = Receiver(parameters)
received_symbols = receiver(received_signal)


# Create an earthquake, and measure strain along the fibre link
earthquake = Earthquake(
    event = 'GCMT:C201002270634A',
    model = 'ak135f_5s'
)
time, _, _, _, _, strain = \
    earthquake(fibre, verbose = True)

# Effects of the earthquake on the fibre propagation will be included in the near future
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
A. Mecozzi, M. Cantono, J.C. Castellanos, *et al.*,
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


## Citation
If you use this work, please cite
```bibtex
@article{Krischer:Jul17:Syngine,
    author={Krischer, Lion and Hutko, Alexander R. and van Driel, Martin and St\"ahlar, Simon and Bahavar, Manochehr and Trabant, Chad and Nissen-Meyer, Tarje},
    title={On-Demand Custom Broadband Synthetic Seismograms},
    journal={Seismol. Res. Lett.},
    year={2017},
    month={07},
    volume={88},
    number={4},
    pages={1127--1140},
    publisher={Seismological Society of America},
    doi={10.1785/0220160210}
}

@article{Mecozzi:Jun21:polarization_sensing_submarine,
    author={Mecozzi, Antonio and Cantono, Mattia and Castellanos, Jorge C. and Kamalov, Valey and Muller, Rafael and Zhan, Zhongwen},
    title={Polarization Sensing using Submarine Optical Cables},
    journal={Opt.},
    year={2021},
    month={06},
    volume={8},
    number={6},
    pages={788--795},
    publisher={Optica Publishing Group},
    doi={10.1364/OPTICA.424307}
}

@article{Awad:May24:environmental_surveillance_networks,
    author={Awad, Hasan and Usmani, Fehmida and Virgillito, Emanuele and Bratovich, Rudi and Proietti, Roberto and Straullu, Stefano and Aquilino, Francesco and Pastorelli, Rosanna and Curri, Vittorio},
    title={Environmental Surveillance through Machine Learning-Empowered Utilization of Optical Networks},
    journal={Sens.},
    year={2024},
    month={05},
    volume={24},
    number={10},
    pages={3041},
    publisher={MDPI},
    doi={10.3390/s24103041}
}

@article{Butler::earthquake_PMD_modelling,
    author={Butler, Rick Maarten and Kasaneva, Jos\'e N\'u\~nez and H\"ager, Christian and Alvarado, Alex},
    title={Modelling Optical Fibre Birefringence under Seismic Strain},
    journal={Nature},
    year={},
    month={},
    volume={},
    number={},
    pages={},
    publisher={Springer},
    doi={}
}
```

## Authors and acknowledgment
Maintained by Rick M. Butler.  

## License
This project is published under the MIT license.
