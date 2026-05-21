# seismic-data-quality
Investigate the quality of seismic data and apply corrections.

## Main Features
* Create seismic event catalogues
* Apply general preprocesssing techniques to seismic waveforms
* Find seismic wave amplitudes
* Correct for seismometer orientation
* Correct for seismometer decoupling
* Generate probabilistic power spectral density plots for a seismic waveform
## Repository Information
This repository contains the process.py file. It contains all the functions for the main features. 
For information and workflows on how to read seismic data and find sensor orientations, use the accompanying seismic-sensor-analysis repository (https://github.com/Ramirezs873/seismic-sensor-analysis) along side seismic-data-quality. 

## Instructions
Install this package by adding it your python working directory. 
* process.py requires a few other libraries to work properly. 
    * ObsPy (https://github.com/obspy/obspy)
    * Pandas (https://github.com/pandas-dev/pandas)
    * NumPy (https://github.com/numpy/numpy)

To get started:
```
import process.py
```
Current functions include:
* event_catalogue()
* find_channel()
* select_time()
* demean_detrend()
* apply_window()
* apply_filter()
* amplitudes()
* amplitude_correction()
* rotate_stream()
* ppsd()

## Tutorials
This repository features three tutorial Jupyter Notebooks:
* Sensor_Alignment_Tutorial.ipynb
  * Worflow for correcting for seismometer orientation   
* Sensor_Amplitude_Tutorial.ipynb
  * Worflow for correcting for seismometer amplitudes 
* PPSD_Tutorial
  * Workflow for generating probabilistic power spectral density plots.

To use these tutorials, a config file needs to be set up. See below.

### Config File

Create a (optional) `config.ymal` file as per that of below,

```
name: Insert_Name_Here
seismic_data_path: "...Example/Seismic/Path"
align_module: ".../align.py"
process_module: ".../process.py"
```
