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

def preprocess(wave_list_dict,
               sensitivity = 2.994697576134245E8):
    """
    Applies a mean removal and detrending 
    to a list of wave dictionaries with isolate NS and EW components
    then deconvolves the instrument sensitivity
    into velocity values (m/s).

    Parameters:
        wave_dict (list):
            List of dictionaries containing seismic waveform data 
            in the form (EW_data, NS_data).
            Dictionary containing seismic waveform data 
            in the form (EW_data, NS_data).
        sensitivity (float):
            Seismic instrument sensitivity.

    Returns:
        preprocessed_dict (list):
            List of dictionaries containing preprocessed seismic waveform data 
            in the form (EW_data, NS_data).
    """

    # Storage
    preprocessed_dict = []

    for i, wave in enumerate(wave_list_dict):
        for station, (EW, NS, fs, t_start) in wave.items():
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
            preprocessed_dict.append({station: (EW_v, NS_v, fs, t_start)})
    
    return preprocessed_dict
        
def amplitudes(wave_list_dict):
    """
    Calculates the maximum, mean and median amplitudes 
    of the NS and EW seismic components from a list of wave dictionaries.
    Parameters:
        wave_list_dict (list):
            List of dictionaries containing seismic waveform data 
            in the form (EW_data, NS_data).
            Dictionary containing seismic waveform data 
            in the form (EW_data, NS_data).
    """

    #Storage
    amplitudes = []

    for i, wave in enumerate(wave_list_dict):
        for station, (EW, NS, fs, t_start) in wave.items():
            # Max amplitude
            NS_amp = np.max(np.abs(NS)) 
            EW_amp = np.max(np.abs(EW))
            # Mean amplitude
            NS_mean_amp = np.mean(np.abs(NS))
            EW_mean_amp = np.mean(np.abs(EW))
            # Median amplitude
            NS_median_amp = np.median(np.abs(NS))
            EW_median_amp = np.median(np.abs(EW))
            # Store Values
            amplitudes.append({"station": station,
                          "NS_max_amp": NS_amp,
                          "EW_max_amp": EW_amp,
                          "NS_mean_amp": NS_mean_amp,
                          "EW_mean_amp": EW_mean_amp,
                          "NS_median_amp": NS_median_amp,
                          "EW_median_amp": EW_median_amp})
            
            print(f"Amplitudes for {station}: {amplitudes}")
            
    return amplitudes

def find_channel(stream, options):
    """
    Find appropriate NS and EW Channels from a chosen stream. 
    Code from align.py.
    
    Parameters:
    stream (obspy.core.stream.Stream):
        An ObsPy stream object.
    options (list of str):
        A list of channel codes.
    """
    # Loop through streams and find the first associated channel code
    for ch in options:
        tr = stream.select(channel=ch)
        if len(tr) > 0:
            return tr[0] # Return first channel code
        
    # If none are found
    return None 

def signal_to_noise(wave_dict, 
                    filtered_dict, 
                    Z_channel, 
                    NS_channel, 
                    EW_channel):
    """
    Calculates the noise to signal ratio for the 
    Z, NS and EW components of seismic data and 
    stores the results as a dictionary.
    Parameters:
    wave_dict (dict):
        A wave dictionary containing seismic waveform data.
    filtered_dict (dict):
        A dictionary containing filtered seismic waveform data.
    Z_channel (list of str):
        List of channel codes for the Z component.
    NS_channel (list of str):
        List of channel codes for the NS component.
    EW_channel (list of str):
        List of channel codes for the EW component.
    """
    
    # Set up seismic waveform data
    # Setup Storage
    wave_data = {}
    # Loop through wave_dict, find the channels, and store the data in a new dictionary
    for station, stream in wave_dict.items():
            print(f"Processing {station}...")
            st = Stream(stream)
            st.sort(['channel'])
            Z = find_channel(st, Z_channel) # Try to find Z channel from function input
            NS = find_channel(st, NS_channel) # Try to find NS channel from function input
            EW = find_channel(st, EW_channel) # Try to find EW channel from function input
            # Warnings for missing channels.
            if Z is None:
                print(f"Warning: Missing Z channel for {station}. Skipping."
                        "This may cause issues if the Z channel is not missing in wave_dict.")
            if NS is None:
                print(f"Warning: Missing NS channel for {station}. Skipping."
                        "This may cause issues if the NS channel is not missing in wave_dict.")
            if EW is None:
                print(f"Warning: Missing EW channel for {station}. Skipping."
                        "This may cause issues if the EW channel is not missing in wave_dict.")
            wave_data[station] = {
                "Z": Z.data if Z is not None else None,
                "NS": NS.data if NS is not None else None,
                "EW": EW.data if EW is not None else None}

    # Set up filtered seismic waveform data
    # Setup Storage
    filt_data = {}
    # Loop through filtered_dict, find the channels, and store the data in a new dictionary
    for station, stream in filtered_dict.items():
            print(f"Processing {station}...")
            st = Stream(stream)
            st.sort(['channel'])
            Z = find_channel(st, Z_channel) # Try to find Z channel from function input
            NS = find_channel(st, NS_channel) # Try to find NS channel from function input
            EW = find_channel(st, EW_channel) # Try to find EW channel from function input
            # Warnings for missing channels.
            if Z is None:
                print(f"Warning: Missing Z channel for {station}. Skipping."
                        "This may cause issues if the Z channel is not missing in wave_dict.")
            if NS is None:
                print(f"Warning: Missing NS channel for {station}. Skipping."
                        "This may cause issues if the NS channel is not missing in wave_dict.")
            if EW is None:
                print(f"Warning: Missing EW channel for {station}. Skipping."
                        "This may cause issues if the EW channel is not missing in wave_dict.")
            filt_data[station] = {
                "Z": Z.data if Z is not None else None,
                "NS": NS.data if NS is not None else None,
                "EW": EW.data if EW is not None else None}
    
    # Calculate noise to signal ratio for each station and component, and store in a new dictionary
    # Setup storage
    ratio_dict = {}

    for station in filt_data:
        # Z component
        if wave_data[station]["Z"] is not None and filt_data[station]["Z"] is not None:
            noise_Z = wave_data[station]["Z"] - filt_data[station]["Z"]
            signal_P_Z = np.mean(filt_data[station]["Z"] ** 2)
            noise_P_Z = np.mean(noise_Z ** 2)
            ratio_Z = 10 * np.log10(signal_P_Z / noise_P_Z)
        else:
            ratio_Z = None
            print(f"No Data for Z component in {station}. Skipping.")
        # NS component
        if wave_data[station]["NS"] is not None and filt_data[station]["NS"] is not None:
            noise_NS = wave_data[station]["NS"] - filt_data[station]["NS"]
            signal_P_NS = np.mean(filt_data[station]["NS"] ** 2)
            noise_P_NS = np.mean(noise_NS ** 2)
            ratio_NS = 10 * np.log10(signal_P_NS / noise_P_NS)
        else:
            ratio_NS = None
            print(f"No Data for NS component in {station}. Skipping.")
        # EW component
        if wave_data[station]["EW"] is not None and filt_data[station]["EW"] is not None:
            noise_EW = wave_data[station]["EW"] - filt_data[station]["EW"]
            signal_P_EW = np.mean(filt_data[station]["EW"] ** 2)
            noise_P_EW = np.mean(noise_EW ** 2)
            ratio_EW = 10 * np.log10(signal_P_EW / noise_P_EW)
        else:
            ratio_EW = None
            print(f"No Data for EW component in {station}. Skipping.")
        # Store the ratios in a new dictionary
        ratio_dict[station] = {"Z": float(ratio_Z) if ratio_Z is not None else None,
                               "NS": float(ratio_NS) if ratio_NS is not None else None,
                               "EW": float(ratio_EW) if ratio_EW is not None else None}

    return ratio_dict

            




    