from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from obspy import UTCDateTime as UTC
from obspy import Stream
from obspy import read
from scipy import signal


def preprocess(wave_dict,
               sensitivity = 2.994697576134245E8):
    """
    Applies a mean removal and detrending 
    to a wave dictionary with isolate NS and EW components
    then deconvolves the instrument sensitivity
    into velocity values (m/s).

    Parameters:
        wave_dict (dict):
            Dictionary containing seismic waveform data 
            in the form (EW_data, NS_data).
        sensitivity (float):
            Seismic instrument sensitivity.

    Returns:
        preprocessed_dict (dict):
            Corrected EW and NS components.
    """

    # Storage
    preprocessed_dict = {'EW': [], 'NS': []}


    for i in range(len(wave_dict)):
        # Define components
        NS = wave_dict[i]['Z9.CWA84.'][1].data
        EW = wave_dict[i]['Z9.CWA84.'][0].data
        # Mean removal
        NS_mr = NS - np.mean(NS)
        EW_mr = EW - np.mean(EW)
        # Detrend
        NS = signal.detrend(NS_mr)
        EW = signal.detrend(EW_mr)
        # Convert to m/s
        NS_v = NS / sensitivity
        EW_v = EW / sensitivity
        # Store Values
        preprocessed_dict['EW'].append(EW_v)
        preprocessed_dict['NS'].append(NS_v)

    
    return preprocessed_dict
        

    