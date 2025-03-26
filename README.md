# tremor-waveplate-toolbox

## Description
A toolbox to simulate optical fibre strain from earthquake tremors using the waveplate model.
It models polarisation mode dispersion (PMD) as a constant differential group delay (DGD) along the fibre, and applies random rotations of the principal state of polarisation (PSP) after each correlation length.
Earthquake seismograms are obtained through https requests to the [Syngine](https://ds.iris.edu/ds/products/syngine/) web service [\[1\]](#1).
Resulting fibre strain is modelled as a change in DGD, like in [\[2\]](#2).


## Demo

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
   - [Anaconda](https://www.anaconda.com/download): `conda create -n tremor-waveplate`.  
4. Activate the environment:  
   - venv:  
     `call env/Scripts/activate`
     on Windows, or  
     `source env/bin/activate`
     on Linux and MacOSX.  
   - Anaconda: `conda activate tremor-waveplate`.  
5. Install this repository:  
```bat
pip install https://gitlab.tue.nl/r.m.butler/tremor-waveplate-toolbox.git
```  
6. All done!

## Usage

## References
<a name="1">\[1\]</a>
L. Krischer, A. R. Hutko, M. van Driel, *et al.*,
"On-demand custom broad-band synthetic seismograms,"
*Seismol. Res. Lett.*,
vol. 88, no. 4, pp. 1127--1140,
Jul. 2017.
DOI: [10.1785/0220160210](https://doi.org/10.1785/0220160210)

<a name="2">\[2\]</a>
H. Awad, F. Usmani, E. Virgillito, *et al.*,
"Environmental surveillance through machine learning-empowered utilization of optical networks,"
*Sens.*,
vol. 24, no. 10, p. 3041,
May 2024.
DOI: [10.3390/s24103041](https://doi.org/10.3390/s24103041)

## Citation
If you use this work, please cite
```bibtex
@article{da_Silva:Jun24:OptiCommPy,
    author={da Silva, Edson Porto and Herbster, Adolfo Fernandes},
    title={{OptiCommPy}: Open-source Simulation of Fiber Optic Communications with {Python}},
    journal={J. Open Source Softw.},
    year={2024},
    month={06},
    volume={9},
    number={98},
    pages={6600},
    publisher={Open Journals},
    doi={10.21105/joss.06600}
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
```

## Authors and acknowledgment
Maintained by Rick M. Butler.  

## License
This project is published under the MIT license.
