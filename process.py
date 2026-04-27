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
from scipy.signal import windows 
from scipy.signal import resample
from obspy.signal import PPSD
from obspy import Trace





def event_catalogue(client, 
                 network, 
                 station, 
                 event_t0 = None,
                 event_t1 = None,
                 radius = 85,
                 min_mag = 6,
                 max_depth = None,
                 title = 'CWA',  
                 user=None, 
                 password=None, 
                 config=None,
                 csv = True,
                 map = True,
                 projection = 'local'):

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
    event_t0 (UTC):
        (Optional) Start time for event search. If None, uses station start time.
    event_t1 (UTC):
        (Optional) End time for event search. If None, uses station end time.
    min_mag (float):
        Minimum magnitude to search for events.
    max_depth (float):
        (Optional) Maximum depth (km) to search for events.
    user (str):
        Username to access the client.
    password (str):
        Password to acces the client.
    config (yaml):
        (Optional) Configuration yaml file. 
    csv (bool):
        True/False. True to write station info and event catalogue to a csv file.
    map (bool):
        True/False. True to plot event locations for each station.
    projection (str):
        Projection for event location plots. See obspy.core.event.catalog.Catalog.plot for options.

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
                                        station=station,
                                        level="response") # Gather station data

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
            starttime=event_t0 if event_t0 is not None else info["t_start"],
            endtime=event_t1 if event_t1 is not None else info["t_end"],
            minmagnitude=min_mag,
            maxdepth=max_depth # km
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
                "station_t_start": event_t0 if event_t0 is not None else info["t_start"],
                "station_t_end": event_t1 if event_t1 is not None else info["t_end"],
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

    if map == True:
        for station in event_dict.keys():
            event_dict[station].plot(projection = projection, title = station)             
        
    return data, station_data

def find_channel(stream, options):
    """
    Find appropriate NS, EW, and Z Channels from a chosen stream.
    
    Parameters:
    stream (obspy.core.stream.Stream):
        An ObsPy stream object.
    options (list of str):
        A list of channel codes.
    """
    # Loop through streams and find the associated channel code
    traces = []
    for ch in options:
        traces.extend(stream.select(channel=ch))
        if len(traces) > 0:
            return traces # Return first channel code
        
    # If none are found
    return None 

def select_time(wave_dict, 
                t_start, 
                duration):
    
    """
    Select a specific time window from seismic waveform data stored in a dictionary without altering the original.
    Copied from align.py

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    t_start (UTCDateTime):
        Start time for the time window.
    duration (float):
        Duration of the time window in seconds.
    
    Returns:
    new_dict (dict):
        Dictionary containing selected seismic waveform data.
    """

    # Establish timespan
    t_end = t_start + duration

    # Trim to desired timespan and write to a direction
    new_dict = defaultdict(list)
    for station_name in wave_dict:
        st = Stream(wave_dict[station_name]).copy() # Copy to avoid overwriting data
        st.trim(starttime=t_start, endtime=t_end, pad=False)

        new_dict[station_name].extend(st.traces)

    return new_dict

def demean_detrend(wave_dict):
    
    """
    Demean and detrend seismic waveform data stored in a dictionary without altering the original.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    """
    # Demean and detrend the data and write to a dictionary
    new_dict = defaultdict(list)

    for station_name, traces in wave_dict.items():  
        st = Stream(traces).copy() # Copy to avoid overwriting data
        st.detrend("demean")
        st.detrend("linear")

        new_dict[station_name].extend(st.traces)
    
    return new_dict

def apply_window(wave_dict, 
                 type = 'hann',
                 max_percentage = None,
                 max_length = 1, 
                 side = 'both'):
    
    """
    Apply a window function from seismic waveform data stored in a dictionary without altering the original.
    Copied from align.py

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    type (str):
        Type of window function applied. 
        See the 'Supported Methods' for obspy.core.trace.Trace.taper function for a list of available window functions.
    max_percentage (float):
        Decimal percentage of window function applied at an end.
    max_length (float):
        Length in seconds of window function applied at an end.
    side (str):
        End(s) at which the window function is applied. 
        Available options are "left", "right", "both".
    """
    
    # Apply the window function and write to a dictionary
    new_dict = defaultdict(list)

    for station_name, traces in wave_dict.items():  
        st = Stream(traces).copy() # Copy to avoid overwriting data
        
        st.taper(type=type, max_percentage=max_percentage, max_length=max_length, side=side)

        new_dict[station_name].extend(st)
    
    return new_dict

def apply_filter(wave_dict, 
                 filter_type, 
                 freqmin=None, 
                 freqmax=None, 
                 freq=None, 
                 corners=4, 
                 zerophase=True):
    
    """
    Apply a filter to seismic waveform data stored in a dictionary without altering the original.
    Copied from align.py. 

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    filter_type (str):
        Type of filter to apply. 
        Options include 'bandpass', 'bandstop', 'lowpass', 'highpass'.
        'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently unsupported.
    freqmin (float):
        Minimum frequency for bandpass/bandstop filters.
    freqmax (float):
        Maximum frequency for bandpass/bandstop filters.
    freq (float):
        Cutoff frequency for lowpass/highpass filters.
    corners (int):
        Number of corners for the filter.
    zerophase (bool):
        True/False. If True, apply a zero-phase filter.

    Returns:
    filtered_dict (dict):
        Dictionary containing filtered seismic waveform data.
    """

    # Setup dictionary
    filtered_dict = {}

    # Loop through and apply filter to the traces
    for station_name, traces in wave_dict.items():
        st = Stream([tr.copy() for tr in traces]) # Copy to avoid overwriting data

        if filter_type in ('bandpass', 'bandstop'):
            st.filter(type=filter_type, 
                      freqmin=freqmin, 
                      freqmax=freqmax, 
                      corners=corners, 
                      zerophase=zerophase)
            
        elif filter_type in ('lowpass', 'highpass'): 
            st.filter(type=filter_type, 
                      freq=freq, 
                      corners=corners, 
                      zerophase=zerophase)
            
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}") #'lowpass_cheby_2', 'lowpass_fir', 'remez_fir' currently not setup

        filtered_dict[station_name] = st.traces # Write to a dictionary
        
    return filtered_dict

def amplitudes(wave_dict,
               NS_channel,
               EW_channel,
               Z_channel):
    """
    Calculates the maximum, mean and median amplitudes 
    of the NS and EW seismic components from a list of wave dictionaries.
    Parameters:
        wave_list_dict (list):
            List of dictionaries containing seismic waveform data 
            in the form (EW_data, NS_data).
            Dictionary containing seismic waveform data 
            in the form (EW_data, NS_data).
        NS_channel (list of str):
            Possible channel codes for North-South instrument component.
        EW_channel (list of str):
            Possible channel codes for East-West instrument component.
        NS_channel (list of str):
            Possible channel codes for North-South instrument component.
        Z_channel (list of str):
            Possible channel codes for vertical instrument component.
    """

    #Storage
    amplitudes = []

    for i, (station, stream) in enumerate(wave_dict.items(), start=2):
        print(f"Processing {station}...")
        st = Stream(stream)
        st.sort(['channel'])
        NS = find_channel(st, NS_channel) # Try to find NS channel from function input
        EW = find_channel(st, EW_channel) # Try to find EW channel from function input
        Z = find_channel(st, Z_channel)

        # Max amplitude
        NS_amp = np.max(np.abs(NS)) 
        EW_amp = np.max(np.abs(EW))
        Z_amp = np.max(np.abs(Z))   
        # Mean amplitude
        NS_mean_amp = np.mean(np.abs(NS))
        EW_mean_amp = np.mean(np.abs(EW))
        Z_mean_amp = np.mean(np.abs(Z))
        # Median amplitude
        NS_median_amp = np.median(np.abs(NS))
        EW_median_amp = np.median(np.abs(EW))
        Z_median_amp = np.median(np.abs(Z))
        # Store Values
        amplitudes.append({"station": station,
                        "NS_max_amp": float(NS_amp),
                        "EW_max_amp": float(EW_amp),
                        "Z_max_amp": float(Z_amp),
                        "NS_mean_amp": float(NS_mean_amp),
                        "EW_mean_amp": float(EW_mean_amp),
                        "Z_mean_amp": float(Z_mean_amp),
                        "NS_median_amp": float(NS_median_amp),
                        "EW_median_amp": float(EW_median_amp),
                        "Z_median_amp": float(Z_median_amp)})
        
        print(f"Amplitudes for {station}: {amplitudes}")
        
    return amplitudes

def amplitude_correction(wave_dict,
                         NS_channel,
                         EW_channel,
                         Z_channel,
                         NS_correction_factor,
                         EW_correction_factor,
                         Z_correction_factor):
    
    """
    Corrects the amplitude of a seismic waveform 
    from a list of correction factors for each channel.
    Parameters:
        wave_list_dict (list):
            List of dictionaries containing seismic waveform data 
            in the form (EW_data, NS_data).
            Dictionary containing seismic waveform data 
            in the form (EW_data, NS_data).
        NS_channel (list of str):
            Possible channel codes for North-South instrument component.
        EW_channel (list of str):
            Possible channel codes for East-West instrument component.
        NS_channel (list of str):
            Possible channel codes for North-South instrument component.
        Z_channel (list of str):
            Possible channel codes for vertical instrument component.
        NS_correction_factor (list of float):
            NS amplitude correction factor.
        EW_correction_factor (list of float):
            EW amplitude correction factor.
        Z_correction_factor (list of float):
            Z amplitude correction factor.
    """
    
    station_list = list(wave_dict.keys())
    amplitude_corrected_obspy = defaultdict(list) 

    for (station, stream, NS_correct, EW_correct, Z_correct) in zip(station_list, 
                                                                    wave_dict.values(), 
                                                                    NS_correction_factor, 
                                                                    EW_correction_factor, 
                                                                    Z_correction_factor):
        print(f"Processing {station}...")
        st = Stream(stream)
        st.sort(['channel'])
        NS = find_channel(st, NS_channel) 
        EW = find_channel(st, EW_channel) 
        Z = find_channel(st, Z_channel)

        t_start = min(tr.stats.starttime for tr in st)
        fs = st[0].stats.sampling_rate

        NS_corrected = NS[0].data * NS_correct if NS else None
        EW_corrected = EW[0].data * EW_correct if EW else None
        Z_corrected = Z[0].data * Z_correct if Z else None

        # Create new obspy stream with aligned data
        st = Stream()
        NS_name = NS[0].stats.channel if NS else None
        EW_name = EW[0].stats.channel if EW else None
        Z_name  = Z[0].stats.channel if Z else None
        components = {EW_name: EW_corrected, NS_name: NS_corrected, Z_name: Z_corrected}
        if NS_name is None or EW_name is None or Z_name is None:
            print(f"Skipping {station}. Missing channel")
            continue

        for channel, data in components.items():
            network_name, station_name, *_ = station.split('.')
            tr = Trace(data=data)
            tr.stats.network = network_name
            tr.stats.station = station_name
            tr.stats.channel = channel
            tr.stats.starttime = UTC(t_start)
            tr.stats.sampling_rate = fs
            st.append(tr)

        amplitude_corrected_obspy[station] = st
    
    return amplitude_corrected_obspy

def rotate_stream(wave_dict, 
                  NS_channel, 
                  EW_channel, 
                  Z_channel,
                  misalignment_angle):
    """
    Apply a complex transform to a seismic waveform to correct for rotation errors.

    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    EW_channel (list of str):
        Possible channel codes for East-West instrument component.
    NS_channel (list of str):
        Possible channel codes for North-South instrument component.
    Z_channel (list of str):
        Possible channel codes for vertical instrument component.
    misalignment_angle (list of float):
        The angle in degrees which the waveform is misaligned.
    """

    station_list = list(wave_dict.keys())
    aligned_wave_dict = {}
    aligned_obspy = defaultdict(list)

    for (station, stream, angle) in zip(station_list, wave_dict.values(), misalignment_angle):
        print(f"Processing {station}...")
        st = Stream(stream)
        st.sort(['channel'])
        NS = find_channel(st, NS_channel) 
        EW = find_channel(st, EW_channel) 
        Z = find_channel(st, Z_channel)

        t_start = min(tr.stats.starttime for tr in st)
        t_end   = max(tr.stats.endtime for tr in st)
        fs = st[0].stats.sampling_rate
        npts = int(round((t_end - t_start) * fs))

        # Make each channel a continuous data set by filling in the gaps with NaNs. 
        # This ensures that the cross correlation and rotation are applied to the entire signal, even if there are gaps in the data.
        E = np.full(npts, np.nan)
        N = np.full(npts, np.nan)
        Z_2 = np.full(npts, np.nan)

        for tr in EW:
            i0 = int(round((tr.stats.starttime - t_start) * fs))
            i1 = i0 + len(tr.data)
            if i1 > npts:
                i1 = npts
                data = tr.data[:(i1 - i0)]
            else:
                data = tr.data

            E[i0:i1] = data

        for tr in NS:
            i0 = int(round((tr.stats.starttime - t_start) * fs))
            i1 = i0 + len(tr.data)
            if i1 > npts:
                i1 = npts
                data = tr.data[:(i1 - i0)]
            else:
                data = tr.data

            N[i0:i1] = data

        for tr in Z:
            i0 = int(round((tr.stats.starttime - t_start) * fs))
            i1 = i0 + len(tr.data)
            if i1 > npts:
                i1 = npts
                data = tr.data[:(i1 - i0)]
            else:
                data = tr.data

            Z_2[i0:i1] = data

        n = min(len(N), len(E), len(Z_2)) 
        y = N[:n] 
        x = E[:n] 
        z = Z_2[:n]


        scale = np.nanmax(np.sqrt((x**2)+(y**2)))
        
        x = x / scale
        y = y / scale

        S_k = x + 1j*y
        S_k_aligned = S_k *np.exp(-1j * np.deg2rad(angle))

        x = np.real(S_k_aligned)
        y = np.imag(S_k_aligned)
        
        #times = NS.times("timestamp")[:n]        
        aligned_wave_dict[station] =  x, y, z, fs, t_start

        # Create new obspy stream with aligned data
        st = Stream()
        NS_name = NS[0].stats.channel if NS else None
        EW_name = EW[0].stats.channel if EW else None
        Z_name  = Z[0].stats.channel if Z else None
        components = {EW_name: x, NS_name: y, Z_name: z}
        if NS_name is None or EW_name is None or Z_name is None:
            print(f"Skipping {station}. Missing channel")
            continue

        for channel, data in components.items():
            network_name, station_name, *_ = station.split('.')
            tr = Trace(data=data)
            tr.stats.network = network_name
            tr.stats.station = station_name
            tr.stats.channel = channel
            tr.stats.starttime = UTC(t_start)
            tr.stats.sampling_rate = fs
            st.append(tr)

        aligned_obspy[station] = st

    return aligned_wave_dict, aligned_obspy

def preprocess(wave_dict, 
               window_type = "hann", 
               filter_order = 2, 
               filter_freq = 1, 
               filter_type = 'low', 
               sensitivity = 2.994697576134245E8,
               mean_removal = True,
                detrend = True,
               apply_window = True, 
               apply_filter = True):
    """
    ### Experimental Function. ###
    ### Does not work as well as normal ObsPy Process ###

    Applies prepocessing to a list of dictionary seismic waveform data.
    Mean removal and detrending is applied. There is also the option
    to apply a window function and a Butterworth filter.

    Parameters:
        wave_dict (list):
            List of dictionaries containing seismic waveform data.
        window_type (str):
            Type of window function to apply. 
            For options see scipy.signal.windows.
        filter_order (int):
            Order of the Butterworth filter.
        filter_freq (float):
            Cutoff frequency of the Butterworth filter in Hz. 
            Can be expressed as a single value for low/high pass, or a list of two values for bandpass.
        filter_type (str):
            Type of Butterworth filter to apply. 
            See scipy.signal.butter for options.
        sensitivity (float):
            Seismic instrument sensitivity.
        apply_window (bool):
            True/False. If True, applies a window function.
        apply_filter (bool):
            True/False. If True, applies a Butterworth filter.

    Returns:
        preprocessed_dict (list):
            List of dictionaries containing preprocessed seismic waveform data 
            in the form (EW_data, NS_data, sampling_rate, start_time).
    """

    preprocessed_dict = {}

    for station in wave_dict.keys():
        # Copy
        copyref = wave_dict[station].copy()
        # Frequency
        fs = copyref[0].stats.sampling_rate
        # Start time
        t_start = copyref[0].stats.starttime
        # Define Componetns
        EW = copyref[0].data
        NS = copyref[1].data
        Z = copyref[2].data
        if mean_removal == True:
            #Mean removal
            NS_mr = NS - np.mean(NS)
            EW_mr = EW - np.mean(EW)
            Z_mr = Z - np.mean(Z)
        else:
            NS_mr = NS
            EW_mr = EW
            Z_mr = Z
        if detrend == True:
            # Detrend
            NS = signal.detrend(NS_mr)
            EW = signal.detrend(EW_mr)
            Z = signal.detrend(Z_mr)
        else:
            NS = NS_mr
            EW = EW_mr
            Z = Z_mr
        # Convert to m/s
        NS_v = NS / sensitivity
        EW_v = EW / sensitivity
        Z_v = Z / sensitivity
        if apply_window == True:
            # Window function
            n = int(len(NS_v) * 0.05) # 5% taper
            taper = np.ones(len(NS_v))
            window = getattr(signal.windows, window_type)
            w = window(2 * n)
            taper[:n] = w[:n]
            taper[-n:] = w[-n:]
            NS_v = NS_v * taper
            EW_v = EW_v * taper
            Z_v = Z_v * taper
        if apply_filter == True:
            # Filter function
            sos = signal.butter(filter_order, filter_freq, fs=fs, btype=filter_type, analog=False, output='sos')
            NS_v = signal.sosfiltfilt(sos, NS_v)
            EW_v = signal.sosfiltfilt(sos, EW_v)
            Z_v = signal.sosfiltfilt(sos, Z_v)

        # Store Values
        preprocessed_dict[station] = (EW_v, NS_v, Z_v, fs, t_start)

    return preprocessed_dict
      

def ppsd(wave_dict, 
         metadata, 
         max_percentage=None,
         plot_channel = ['EW', 'NS', 'Z'],
         save_png = True,
         png_title = 'default_title',
         show_plot = True):
    """
    Create a Probabilistic Power Spectral Density (PPSD) plot 
    for seismic waveform data stored in a dictionary. 
    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data.
    metadata (Inventory):
        Obspy Inventory object containing station metadata for the seismic waveform data.
    max_percentage (float, optional):
        Maximum percentage of the distribution to display in the PPSD plot
    plot_channel (list of str):
        List of channel codes to plot. Options are 'EW', 'NS', and 'Z'.
    save_png (bool, optional):
        Whether to save the PPSD plot as a PNG file.
    png_title (str, optional):
        Title for the saved PNG file.
    show_plot (bool, optional):
        Whether to display the PPSD plot.
    """

    # No component selected
    if 'EW' not in plot_channel and 'NS' not in plot_channel and 'Z' not in plot_channel:
        print(f"No components selected. PPSD not calculated or plotted. Please select at least one component.")
    # Wrong input for plot_channel
    if any(ch not in ['EW', 'NS', 'Z'] for ch in plot_channel):
        print(f"Invalid plot_channel input. Please select from 'EW', 'NS', and 'Z'. No PPSD plotted for this station.")

    for station in wave_dict.keys():
        # Copy
        copy = wave_dict[station].copy()
        # Define Componetns
        EW = copy[0].data
        NS = copy[1].data
        Z = copy[2].data

        print(f"Processing {station}...")
        # Seismic Waveform Data
        # Warnings for missing channels.
        if Z is None:
            print(f"Warning: Missing Z channel for {station}. Skipping.")
        if NS is None:
            print(f"Warning: Missing NS channel for {station}. Skipping.")
        if EW is None:
            print(f"Warning: Missing EW channel for {station}. Skipping.")

        if png_title == 'default_title':
            title = f'ppsd_{station}.png'
        else:
            title = f'{png_title}.png'

        if 'EW' in plot_channel:
            # Plot EW componet
            print(f"Plotting PPSD for {station} EW component...")
            ppsd_EW = PPSD(stats = wave_dict[station][0].stats,metadata=metadata)
            ppsd_EW.add(wave_dict[station][0])
            ppsd_EW.plot(max_percentage=max_percentage, 
                         filename=title if save_png == True else None, 
                         show=True if show_plot == True else False)
        if 'NS' in plot_channel:
            # Plot NS componet
            print(f"Plotting PPSD for {station} NS component...")
            ppsd_NS = PPSD(stats = wave_dict[station][1].stats,metadata=metadata)
            ppsd_NS.add(wave_dict[station][1])
            ppsd_NS.plot(max_percentage=max_percentage, 
                         filename=title if save_png == True else None, 
                         show=True if show_plot == True else False)
        if 'Z' in plot_channel:
            # Plot Z componet
            print(f"Plotting PPSD for {station} Z component...")
            ppsd_Z = PPSD(stats = wave_dict[station][2].stats, metadata=metadata)
            ppsd_Z.add(wave_dict[station][2])
            ppsd_Z.plot(max_percentage=max_percentage, 
                       filename=title if save_png == True else None, 
                       show=True if show_plot == True else False)
