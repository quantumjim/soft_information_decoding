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
brew install bazel
brew install armadillo
brew install mlpack
brew install llvm # for clang and parrallelism
```
Then add the Brew install path to your include path variable:
```bash
/opt/homebrew/include
# also add include for omp.h 
/opt/homebrew/opt/llvm/bin/../include/c++/v1
/opt/homebrew/Cellar/llvm/17.0.6/lib/clang/17/include
/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk/usr/include
```
NOTE: /opt/homebrew is the default path on Apple Silicon Macs, if you are using an Intel Mac, the path is /usr/local
CAUTION: if building with `'-DCMAKE_CXX_FLAGS=-fopenmp'` the kernel will crash when throwing errors!


### Install Python Dependencies
After cloning the repository, navigate to the project directory and install the required Python packages using Pipenv: (WARNING THIS MAY TAKE A WHILE BECAUSE C++ PARTS ARE COMPILED)
```bash
pipenv install
```

### Rebuild C++ Parts
If you wish to change C++ parts, even if the soft_information_decoding package is editable you need to rebuild it using the following command:
```bash
pip install .
```

### Install stim wheel
Install the stim wheel using pip. It is located in 'libs/PyMatching/bazel-bin/external/stim'.
```bash
pip install /path/to/stim-0.0.dev0-py3-none-any.whl
```