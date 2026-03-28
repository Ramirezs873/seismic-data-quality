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
import csv
from obspy.clients.fdsn.header import FDSNNoDataException



def event_catalogue(client, 
                 network, 
                 station, 
                 radius = 85,
                 min_mag = 6,
                 max_depth = 100,
                 title = 'CWA',  
                 user=None, 
                 password=None, 
                 config=None,
                 csv = True):

    """
    Gather seismic station coordinates and relevant event information
    from FDSN clients, or from local files if they already exist, 
    from a specified group of stations and stores it as a dictionary.
    The data can also be saved as a csv file.

    Parameters:
    ----------
    client (str): 
        FDSN client URL for group 1. e.g 'IRIS', "AUSPASS".
    network (str): 
        Network code for group 1. e.g 'IU', 'AU'.
    station (str): 
        Station code for group 1. e.g 'CASY', 'CWA90'. 
    min_mag (float):
        Minimum magnitude to search for events.
    max_depth (float):
        Maximum depth (km) to search for events.
    user (str):
        Username to access the client.
    password (str):
        Password to acces the client.
    config (yaml):
        (Optional) Configuration yaml file. 
    csv (bool):
        True/False. True to write station info and event catalogue to a csv file.

    Returns:
    data (list of dict):
        A dictionary containing station coordinate data and a dictionary containing event information.
        
    """
    
    # Check if file already exists
    # Config Support
    base_path = Path(config["seismic_data_path"]) if config else Path(".")
    base_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"station_data_{title}.xml"
    file_path = base_path / filename

    # Gather data from target stations
    g1 = Client(base_url=client, 
                user=user, 
                password=password) # FDSN client 

    if file_path.exists():
        print(f"Reading existing file: {file_path}")
        station_data = read_inventory(str(file_path))

    else:
        print("File not found. Downloading data")

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
                                "longitude": sta.longitude,
                                "t_start": sta.start_date,
                                "t_end": sta.end_date if sta.end_date else "active"}
    
    event_dict = defaultdict(list)
    stat_event_csv = []    
    
    for station, info in coord_dict.items():
        try:
            events = g1.get_events(
            latitude=info["latitude"],
            longitude=info["longitude"],
            maxradius= radius, # degrees
            starttime=info["t_start"],
            endtime=info["t_end"],
            minmagnitude=min_mag,
            maxdepth= max_depth # km
            )
        except FDSNNoDataException:
            print(f"No events found for {station} with min_mag={min_mag} and max_depth={max_depth}.")
            events = []

        event_dict[station] = events

        # loop through events
        for event in event_dict[station]:
            # get origin info (if exists)
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            
            stat_event_csv.append({
                "station": station,
                "station_latitude": info["latitude"],
                "station_longitude": info["longitude"],
                "station_t_start": info["t_start"],
                "station_t_end": info["t_end"],
                "event_t": origin.time,
                "event_latitude": origin.latitude,
                "event_longitude": origin.longitude,
                "event_depth_km": origin.depth / 1000 if origin.depth else None,
                "event_magnitude": magnitude.mag if magnitude else None
            })

    data = [coord_dict, event_dict]

    if csv == True:
        csv_filename = f"station_data_{title}.csv"
        csv_file_path = base_path / csv_filename

        if csv_file_path.exists():
            print(f"File already exists: {csv_file_path}")
            
        else:
            print("Writing CSV File")

            df = pd.DataFrame(stat_event_csv)
            df.to_csv(csv_file_path, index=False)

            print(f"Saved to: {csv_file_path}")

    return data

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
        

    