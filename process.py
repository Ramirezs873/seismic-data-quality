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
               window_type = "hann", 
               filter_order = 2, 
               filter_cutoff = 1, 
               filter_type = 'low', 
               sensitivity = 2.994697576134245E8,
               apply_window = True, 
               apply_filter = True):
    """
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
        filter_cutoff (float):
            Cutoff frequency of the Butterworth filter in Hz.
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
        NS = copyref[0].data
        EW = copyref[1].data
        #Mean removal
        NS_mr = NS - np.mean(NS)
        EW_mr = EW - np.mean(EW)
        # Detrend
        NS = signal.detrend(NS_mr)
        EW = signal.detrend(EW_mr)
        # Convert to m/s
        NS_v = NS / sensitivity
        EW_v = EW / sensitivity
        if apply_window == True:
            # Window function
            window = getattr(signal.windows, window_type)
            NS_v = NS_v * window(len(NS_v))
            EW_v = EW_v * window(len(EW_v))
        if apply_filter == True:
            # Filter function
            sos = signal.butter(filter_order, filter_cutoff, fs=fs, btype=filter_type, analog=False, output='sos')
            NS_v = signal.sosfiltfilt(sos, NS_v)
            EW_v = signal.sosfiltfilt(sos, EW_v)

        # Store Values
        preprocessed_dict[station] = (EW_v, NS_v, fs, t_start)

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

def cc_correction(ref_dict, 
                  target_dict, 
                  col_n=4,
                  underlying_plot='reference', 
                  png_title = 'default', 
                  save_png=True):
    """
    Create polar plots with cross-correlation correction to a reference station from seismic waveform data stored in a dictionary.
    Adapted for use with prepocess() from align.py.
    Parameters:
    ref_dict (dict):
        Dictionary containing seismic waveform data for a reference station.
    target_dict (dict):
        Dictionary containing seismic waveform data for target stations.
    col_n (int):
        Number of columns in the plot grid.
    underlying_plot (str):
        Choose what is plotted with the aligned target station signals.
        'reference' for reference station
        'original' for original target station signal
        Any other input skips the underlying plot.
    png_title (str):
        Title for the saved PNG file. If 'default', the file is named 'cross_correlation.png'.
    save_png (bool):
        True/False. If True, save each plot as a PNG file.
    """
    # Calculate number of rows needed for figure. Set number of columns in function call
    n_plots = len(target_dict) + 1
    row_n = int(np.ceil(n_plots / col_n)) 

    # Prepare Figure
    fig = plt.figure(figsize=(6*col_n, 6*row_n))

    # Setup reference station for first subplot
    ref_station = list(ref_dict.keys())[0]

    ref_NS = ref_dict[ref_station][1] 
    ref_EW = ref_dict[ref_station][0]

    if ref_NS is None or ref_EW is None:
        raise ValueError("Reference station missing required NS/EW channels")
        
    print(f"Processing reference station: {ref_station}...")

    # Ensure x and y data is of same length
    n_ref = min(len(ref_NS.data), len(ref_EW.data))
    y_ref = np.asarray(ref_NS)[:n_ref]
    x_ref = np.asarray(ref_EW)[:n_ref]

    # Apply peak normalization
    scale_ref = np.max(np.sqrt(x_ref**2 + y_ref**2))
    if scale_ref == 0:
        print(f"{ref_station}: zero amplitude signal.")
        
    ref_EW_norm = x_ref / scale_ref
    ref_NS_norm = y_ref / scale_ref

    # Gather polar coordinates 
    theta_ref = np.arctan2(ref_NS_norm,ref_EW_norm)
    r_ref = np.sqrt(ref_NS_norm**2 + ref_EW_norm**2)

    # Plot reference station
    ax = fig.add_subplot(row_n, col_n, 1, projection="polar")
    ax.plot(theta_ref,
            r_ref, 
            alpha=0.65, 
            color = 'red',
            label=f"Reference Station: {ref_station}")
    
    # Legend
    ax.legend(loc="upper right", 
              bbox_to_anchor=(1.3, 1.1), 
              fontsize=8,
              frameon=True)
    
    # Cardinal directions
    ax.set_rmax(1.2)
    cardinals = {
                "E": (0, 1.05 * 1.05),
                "N": (np.pi / 2, 1.05),
                "W": (np.pi, 1.05 * 1.05),
                "S": (3 * np.pi / 2, 1.05)}

    offset = 0.385 
    
    for label, (angle, radius) in cardinals.items():
        ax.text(angle,
                radius + offset,
                label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                clip_on=False)

    time = ref_dict[ref_station][3]
    timespan = len(ref_NS.data) / ref_dict[ref_station][2]

    ax.set_title(f"Horizontal Particle Motion Plot \n for {ref_station} at {time} for {timespan} seconds", y=1.15)    

    # Loop through stations, cross correlate, and plot
    for i, (station, stream) in enumerate(target_dict.items(), start=2):
            print(f"Processing {station}...")
            
            target_NS = target_dict[station][1] 
            target_EW = target_dict[station][0]

            if target_NS is None or target_EW is None:
                print(f"{station}: missing required channels, skipping.")
                continue
            
            # Match channel lengths
            n = min(len(ref_NS.data), len(ref_EW.data), len(target_NS.data), len(target_EW.data))
            y1 = np.asarray(ref_NS)[:n]
            x1 = np.asarray(ref_EW)[:n]
            y2 = np.asarray(target_NS)[:n]
            x2 = np.asarray(target_EW)[:n]

            # Apply peak normalization
            scale1 = np.max(np.sqrt((x1**2) + (y1**2)))
            x1 = x1 / scale1
            y1 = y1 / scale1

            scale2 = np.max(np.sqrt((x2**2) + (y2**2)))
            x2 = x2 / scale2
            y2 = y2 / scale2
        
            # Investigate cross correlation
            # Method from Misalignment Angle Correction of Borehole 
            # Seismic Sensors: The Case Study of
            # the Collalto Seismic Network
            # Diez Zaldívar
            # 2016
            # For full derivation, see the paper above. Only vital steps are conducted here.
            
            S_r = x1 +1j*y1 # Reference waveform
            S_k = x2 +1j*y2 # Target waveform

            # m = (G^H G)^-1 G^H d)
            # G = S_k, H is conjugate transpose matrix, d = S_r
            # => m_k = (S_k^H * S_k)^-1 *S_k^H * S_r
            # => m_k = sum(|S_k|^2)^-1 * sum(conj(S_k) *S_r)
            m_k = np.sum(np.conj(S_k)* S_r)/np.sum(np.abs(S_k)**2)
            phi = np.arctan2(np.imag(m_k), np.real(m_k)) # angle between the target and reference waveform

            # Rotate entire signal
            S_k_aligned = S_k * np.exp(1j *phi)
            x2_aligned =  np.real(S_k_aligned) # EW componet
            y2_aligned = np.imag(S_k_aligned) # NS component

            # Gather polar coordinates
            theta_aligned = np.arctan2(y2_aligned, x2_aligned)
            r_aligned = np.sqrt(x2_aligned**2 + y2_aligned**2)

            # Compute rotation angle
            angle_diff = np.rad2deg(phi)
            if angle_diff > 180:
                angle_diff = angle_diff - 360

            print(f"Rotation required for best correlation at {station}: {angle_diff:.2f}°")
            

            # Create polar plot
            ax = fig.add_subplot(row_n, col_n, i, projection="polar")
            if underlying_plot == 'reference':
                ax.plot(theta_ref,
                        r_ref, 
                        alpha=0.50, 
                        color = 'red',
                        label=f"Reference Station: {ref_station}")
            elif underlying_plot == 'original':
                theta_original = np.arctan2(y2,x2)
                r_original = np.sqrt(x2**2 + y2**2)
                ax.plot(theta_original, 
                        r_original, 
                        alpha=0.30, 
                        color = 'darkmagenta',
                        label="Uncorrected Target Station")
            ax.set_rlabel_position(5)

            ax.plot(theta_aligned, 
                    r_aligned, 
                    alpha=0.65, 
                    color = 'darkmagenta',
                    label="Corrected Target Station")
                
            # Legend
            ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            fontsize=8,
            frameon=True)

            # Add cardinal direction annotations
            ax.set_rmax(1.2)
            cardinals = {
                "E": (0, 1.05 * 1.05),
                "N": (np.pi / 2, 1.05),
                "W": (np.pi, 1.05 * 1.05),
                "S": (3 * np.pi / 2, 1.05)}

            offset = 0.385  # Offset for cardinal labels

            for label, (angle, radius) in cardinals.items():
                ax.text(
                    angle,
                    radius + offset,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False)
                
            # For title
            time = target_dict[station][3]
            timespan = len(target_NS.data) / target_dict[station][2]
            
            ax.set_title(f"Horizontal Particle Motion Plot \n for {station} at {time} for {timespan} seconds", y=1.15)    

    #Display
    plt.tight_layout()
    if save_png == True:
        if png_title == 'default':
            plt.savefig(f'cross_correlation.png', dpi=300)
        else:
            plt.savefig(f'{png_title}.png', dpi=300)

    plt.show()       


def tabulate_cc_correction(ref_dict, 
                           target_dict,
                           location='default_title'):
    
    """
    Tabulate correction angles for sensors from seismic waveform data stored in a dictionary.
    
    Parameters:
    wave_dict (dict):
        Dictionary containing seismic waveform data for the reference station.
    target_dict (dict):
        Dictionary containing seismic waveform data.  
    location (str):
        Title/location for the output table and CSV file.

    Returns:
    df (DataFrame):
        DataFrame containing peak angles for each station.
    """
    
    # Setup up table storage
    angle_results = []  
    
    # Setup reference station for first subplot
    ref_station = list(ref_dict.keys())[0]

    ref_NS = ref_dict[ref_station][1] 
    ref_EW = ref_dict[ref_station][0]

    if ref_NS is None or ref_EW is None:
        raise ValueError("Reference station missing required NS/EW channels")
    
    print(f"Processing reference station: {ref_station}...")

    # Loop through stations, cross correlate, and tabulate
    for i, (station, stream) in enumerate(target_dict.items(), start=2):
        print(f"Processing {station}...")
            
        target_NS = target_dict[station][1] 
        target_EW = target_dict[station][0]
        
        if target_NS is None or target_EW is None:
            print(f"{station}: missing required channels (NS options: {NS_channel}, EW options: {EW_channel}), skipping.")
            continue
            
        # Match channel lengths
        n = min(len(ref_NS.data), len(ref_EW.data), len(target_NS.data), len(target_EW.data))
        y1 = np.asarray(ref_NS)[:n]
        x1 = np.asarray(ref_EW)[:n]
        y2 = np.asarray(target_NS)[:n]
        x2 = np.asarray(target_EW)[:n]

        # Apply peak normalization
            
        scale1 = np.max(np.sqrt((x1**2) + (y1**2)))
        x1 = x1 / scale1
        y1 = y1 / scale1

        scale2 = np.max(np.sqrt((x2**2) + (y2**2)))
        x2 = x2 / scale2
        y2 = y2 / scale2
        
        # Investigate cross correlation
        # Method from Misalignment Angle Correction of Borehole 
        # Seismic Sensors: The Case Study of
        # the Collalto Seismic Network
        # Diez Zaldívar
        # 2016
        # For full derivation, see the paper above. Only vital steps are conducted here.
        
        S_r = x1 +1j*y1 # Reference waveform
        S_k = x2 +1j*y2 # Target waveform

        # m = (G^H G)^-1 G^H d)
        # G = S_k, H is conjugate transpose matrix, d = S_r
        # => m_k = (S_k^H * S_k)^-1 *S_k^H * S_r
        # => m_k = sum(|S_k|^2)^-1 * sum(conj(S_k) *S_r)
        m_k = np.sum(np.conj(S_k)* S_r)/np.sum(np.abs(S_k)**2)
        phi = np.arctan2(np.imag(m_k), np.real(m_k)) # angle between the target and reference waveform

        # Compute rotation angle
        angle_diff = np.rad2deg(phi)
        if angle_diff > 180:
            angle_diff = angle_diff - 360

        # Store results in table
        angle_results.append({"Station": station,"Angle Correction": f'{angle_diff:.2f}'})

    # Tabulate
    time = target_dict[station][3]
    df = pd.DataFrame(angle_results)
    df.to_csv(f'seismic_directions_{location}.csv', index=False)
    print(f"Alignments for {location} Earthquake @ {time} (UTC):")

    return df



    