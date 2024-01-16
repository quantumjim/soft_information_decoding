# Soft Info 

## Overview
Master thesis project repository. Test and development of new decoders using soft information.


## Installation

### Clone the Repository
To clone this repository, run the following command in your terminal:
```bash
git clone --recursive https://github.ibm.com/Maurice-Hanisch/Soft-Info.git
```

### Install C++ Dependencies
For macOS:
```bash
brew install armadillo
brew install mlpack
```


### Install Python Dependencies
After cloning the repository, navigate to the project directory and install the required Python packages using Pipenv: (WARNING THIS MAY TAKE A WHILE BECAUSE C++ PARTS ARE COMPILED)
```bash
pipenv install
```

### Rebuild C++ Parts
If you wish to change C++ parts, even if the soft_information_decoding package is in editable you need to rebuild it using the following command:
```bash
pip install .
```