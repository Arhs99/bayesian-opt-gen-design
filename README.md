# bayesian-opt-gen-design
Generative design with bayesian optimization
## Description
Given a dataset of molecules with a specific property, we can employ bayesian optimization to generate molecules optimized for that property. An encoder/decoder NN can provide the ability to translate between molecules and their real vector representations. I used **cddd** by the Bayer group https://github.com/jrwnter/cddd but other alternatives can be plugged in.
## Installation
Use the ```conda.txt``` to build the conda environment. **cddd** can be installed by cloning the git repository from https://github.com/jrwnter/cddd and installing with **pip** e.g. ``` pip intall -e .```
## Usage
A notebook example is provided
## Aknowledgement
cddd publication: R. Winter, F. Montanari, F. Noe and D. Clevert, Chem. Sci, 2019, https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04175j#!divAbstract
