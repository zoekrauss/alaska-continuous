This repository acts as a summary of the work produced for the 2022 Univ of Washington eScience Institute Winter Incubator Program.

## Machine-learning-based detection of offshore earthquakes

Project lead: Zoe Krauss, graduate student, School of Oceanography  
eScience Liaison: Scott Henderson

The project description can be found [here](https://escience.washington.edu/winter-2022-incubator-projects/).




The workflow we developed accomplishes the following:
- downloads continuous offshore seismic data
- preprocesses the seismic data using either filtering or [DeepDenoiser](https://github.com/wayneweiqiang/DeepDenoiser), a pretrained deep neural network denoiser
- extracts P- and S-wave picks from the continuous data using pretrained [EQTransformer](https://github.com/smousavi05/EQTransformer), a deep neural network phase picker
- evaluates the performance of the pre-trained network by comparison to a manually produced catalog    



Our workflow makes extensive use of the [Seisbench](https://github.com/seisbench/seisbench) package to apply these pretrained models.   
An example of running this workflow on offshore data from Alaska can be found in [EXAMPLE.ipynb](EXAMPLE.ipynb).


Resources to deploy this workflow on the cloud, using Azure, can be found in [this repository](https://github.com/Denolle-Lab/azure).


The next step in this project is to retrain the machine learning networks using offshore data. The repository where this work is beginning is [here](https://github.com/zoekrauss/ak-retraining) (but is very much still in the development phase).




