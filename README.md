# About
Simple multi agent system that runs a factory. Visualization of grid is provided.

# Installing
You can either use the anaconda environment or install the pre-requisites manually.

### Using Anaconda Environment
* Download [Anaconda - Python3](https://www.continuum.io/downloads)  
* Enable anaconda environment by adding it to the path
* `git clone git@github.com:nitred/simple-multi-agent-factory.git`
* `cd simple-multi-agent-factory`
* `conda env create --force -f environment.yaml`
* `source activate maas`

### Using Requirements
`pip install -r requirements.txt`


# Run Script
```
python factory_model.py
```


# Known Issues
* The robot agents do move when another agent is located at the departments agents (i.e. the Store Agent, Fitting/Manufacturing Agent etc).
* The robot agents pass directly through departments agents.
