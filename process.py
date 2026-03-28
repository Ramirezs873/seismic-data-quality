from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from obspy import UTCDateTime as UTC
from obspy import Stream
from obspy import read
from scipy import signal
from obspy.clients.fdsn import Client
from obspy import read_inventory
from collections import defaultdict
import obspy



def station_coords(client, 
                 network, 
                 station,   
                 user=None, 
                 password=None, 
                 config=None):

    """
    Gather seismic station coordinated from FDSN clients,
    or from local files if they already exist, 
    from one specified group of stations
    and stores it as a dictionary.

    Parameters:
    ----------
    client (str): 
        FDSN client URL for group 1. e.g 'IRIS', "AUSPASS".
    network (str): 
        Network code for group 1. e.g 'IU', 'AU'.
    station (str): 
        Station code for group 1. e.g 'CASY', 'CWA90'. 
    user (str):
        Username to access the client.
    password (str):
        Password to acces the client.

    Returns:
    coord_dict (dict):
        Dictionary containing station coordinate data.
    """
    
    # Check if file already exists
    # Config Support
    base_path = Path(config["seismic_data_path"]) if config else Path(".")
    base_path.mkdir(parents=True, exist_ok=True)

    title = f'{network}_{station}'
    
    filename = f"station_data_{title}.xml"
    file_path = base_path / filename


    if file_path.exists():
        print(f"Reading existing file: {file_path}")
        station_data = read_inventory(str(file_path))

    else:
        print("File not found. Downloading data")

        # Gather data from target stations
        g1 = Client(base_url=client, 
                    user=user, 
                    password=password) # FDSN client 

        station_data = g1.get_stations(network=network, 
                                        station=station) # Gather station data

        station_data.write(str(file_path), format="STATIONXML")
        print(f"Saved to: {file_path}")

    # Write to a dictionary
    coord_dict = defaultdict(list) 

    for net in station_data:
        for sta in net:
            key = f"{net.code}.{sta.code}"
            coord_dict[key] = {"latitude": sta.latitude, 
                                "longitude": sta.longitude}

    return coord_dict   

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
        

    