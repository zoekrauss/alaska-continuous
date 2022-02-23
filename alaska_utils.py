import obspy
import obsplus
import pandas as pd
from obspy.clients.fdsn import Client
from datetime import timedelta
import datetime
import matplotlib.pyplot as plt
import numpy as np
import logging
import requests
import seisbench
import seisbench.models as sbm
import dask
import json
import postprocess


def station_list(dfS,t1,t2,elevation=False,network=False):
    """ 
    Function to make a station sublist from the master station list based on several choices
    
    INPUTS:
    dfS - pandas dataframe of station information for the entire AACSE catalog
    t1,t2 - datetime objects; only return stations operating between these two timestamps
    elevation - float; only return stations below this maximum elevation
    network - string; only return stations within this network
    
    OUTPUTS:
    dfS = subset of the original pandas station dataframe that meets specifications
    """
    if isinstance(dfS.iloc[0]['start_date'],str):
        dfS['start_date'] = pd.to_datetime(dfS['start_date'],infer_datetime_format=True,errors='coerce')
        dfS['end_date'] = pd.to_datetime(dfS['end_date'],infer_datetime_format=True,errors='coerce')

    
    dfS = dfS[(dfS['start_date'] < t1) & (dfS['end_date'] > t2)]
    
    
    if elevation:
        dfS = dfS[dfS['elevation(m)']<elevation]
        
    if network:
        dfS = dfS[dfS['network']==network]
    
    return dfS
    
    

def retrieve_waveforms(dfS,t1,t2,sampling_rate=100,separate=False):
    """
    Function to retrieve seismic time series using obspy get bulk waveforms function
    Also detrends, trims, and normalizes sampling rate for all traces
    
    
    INPUTS:
    dfS = pandas dataframe of station information (made using station_list)
    t1 = start time of desired time series, as datetime
    t2 = end time of desired time series, as datetime
    sampling_rate = sampling rate in Hz to normalize all downloaded time series to
    separate = bool, to separate into a list of streams for each station rather than returning one large stream
    
    OUTPUTS:
    st = obspy Stream object containing traces for each station in dfS for the desired time window
    """

    client = Client("iris")
    
    df = dfS.loc[:,['network','station']]
    df['location']='*'
    df['channels']=dfS.id.str[-2:]+'*'
    df['start']=obspy.UTCDateTime(t1)
    df['end']=obspy.UTCDateTime(t2)

    bulk_order = df.to_records(index=False).tolist()
    
    st = client.get_waveforms_bulk(bulk_order)
    
    # Detrend all traces:
    # try:
        # st = st.detrend("spline", order=2, dspline=5 * st[0].stats.sampling_rate)
    # except:
        # logging.error(f"Error: spline detrend failed at file {fname}")
        # st = st.detrend("demean")
    
    # Make sure all traces are the same length by padding with zeros:
    st = st.merge(fill_value=0)
    st = st.trim(obspy.UTCDateTime(t1),obspy.UTCDateTime(t2), pad=True, fill_value=0)
    
    # Double check that everything is the same sampling rate:
    for i in range(len(st)):
        if st[i].stats.sampling_rate != sampling_rate:
            st[i] = st[i].interpolate(sampling_rate, method="linear")
            
    if separate:
        # Separate the large stream into a list of streams, one for each instrument
        # This helps keep track of stations which have multiple channel sets for one station code,
        # which gets lost in seisbench
        sta=[];chn=[];stt=[];
        for tr in st:
            sta.append(tr.stats.station)
            chn.append(tr.stats.channel[0:2])
        st_meta = pd.DataFrame.from_dict({'sta':sta,'chn':chn})
        st_meta.drop_duplicates(inplace=True,ignore_index=True)
        stream2 = np.empty([len(st_meta)],dtype=object)
        for i in range(len(st_meta)):
            tmp = st.select(station=st_meta.iloc[i].sta,channel=st_meta.iloc[i].chn+'*')
            stream2[i] = tmp
        st = stream2

    
    return st

def retrieve_waveforms_raw(dfS,t1,t2,sampling_rate=100):
    """
    Function to retrieve seismic time series using obspy get bulk waveforms function
    
    
    INPUTS:
    dfS = pandas dataframe of station information (made using station_list)
    t1 = start time of desired time series, as datetime
    t2 = end time of desired time series, as datetime
    sampling_rate = sampling rate in Hz to normalize all downloaded time series to
    
    OUTPUTS:
    st = obspy Stream object containing traces for each station in dfS for the desired time window
    """

    client = Client("iris")
    
    df = dfS.loc[:,['network','station']]
    df['location']='*'
    df['channels']=dfS.id.str[-2:]+'*'
    df['start']=obspy.UTCDateTime(t1)
    df['end']=obspy.UTCDateTime(t2)

    bulk_order = df.to_records(index=False).tolist()
    
    st = client.get_waveforms_bulk(bulk_order)
            
    
    return st
    



def pick_quakeflow(st,dfS,remove_resp=False):
    """
    Runs a 30-s obspy stream through PhaseNet using QuakeFlow
    
    1. Converts stream to a set of numpy arrays, does various preprocessing and normalization
    2. Sends one set of 3 traces (Z,N,E) at a time to Quakeflow PhaseNet API
    3. Concats and saves Phasenet picks and returns them
    
    INPUTS:
    st = obspy Stream object containing traces that all have the same start and end time
    dfS = pandas dataframe of stations for which there are traces for
    
    OUTPUTS:
    phasenet_picks = array of pick information 
    """
    
    sampling_rate = 100
    n_channel = 3
    dtype = "float32"
    amplitude = True
    
    

    starttime = min([p.stats.starttime for p in st])
    endtime = min([p.stats.endtime for p in st])
    
    # Double check that everything is the same sampling rate:
    for i in range(len(st)):
        if st[i].stats.sampling_rate != sampling_rate:
            # logging.warning(
            #     f"Resampling {st[i].id} from {st[i].stats.sampling_rate} to {sampling_rate} Hz"
            # )
            st[i] = st[i].interpolate(sampling_rate, method="linear")
    
    # Set up an order to loop through the traces
    order = ['3', '2', '1', 'E', 'N', 'Z']
    order = {key: i for i, key in enumerate(order)}
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}
    
    nsta = len(dfS)
    nt = max(len(st[i].data) for i in range(len(st)))
    data = []
    station_id = []
    t0 = []
    
        # Loop through traces
        #for i in range(len(st)):
         #   trace_data = np.zeros([nt, n_channel], dtype=dtype)
          #  empty_station = True
           # sta_id = st[i].meta.network + '.' + st[i].meta.station + '..'+ st[i].meta.channel[0:2]

       # t1 = st[i].stats.starttime
       # t2 = st[i].stats.endtime
        
       # sta = dfS[dfS['id']==sta]
       # comp = sta.iloc[0].split(",")
        
       # if remove_resp:
       #     resp = sta.iloc[0]["response"].split(",")
            
            
        
    # Loop through stations and time bands
    
    for i in range(nsta):
        trace_data = np.zeros([nt, n_channel], dtype=dtype)
        empty_station = True
        # sta = station_locs.iloc[i]["station"]
        sta = dfS.id[i]
        comp = dfS.iloc[i]["component"].split(",")
        if remove_resp:
            resp = dfS.iloc[i]["response"].split(",")
            # resp = station_locs.iloc[i]["response"]

        for j, c in enumerate(sorted(comp, key=lambda x: order[x[-1]])):
            
            if remove_resp:
                resp_j = float(resp[j])
            
            
            if len(comp) != 3:  ## less than 3 component
                j = comp2idx[c]

            if len(st.select(id=sta + c)) == 0:
                # print(f"Empty trace: {sta+c} {starttime}")
                continue
            else:
                empty_station = False

            tmp = st.select(id=sta + c)[0].data.astype(dtype)
            trace_data[: len(tmp), j] = tmp[:nt]

            if dfS.iloc[i]["unit"] == "m/s**2":
                tmp = st.select(id=sta + c)[0]
                tmp = tmp.integrate()
                tmp = tmp.filter("highpass", freq=1.0)
                tmp = tmp.data.astype(dtype)
                trace_data[: len(tmp), j] = tmp[:nt]
            elif dfS.iloc[i]["unit"] == "m/s":
                tmp = st.select(id=sta + c)[0].data.astype(dtype)
                trace_data[: len(tmp), j] = tmp[:nt]
            else:
                print(
                    f"Error in {station_locs.iloc[i]['station']}\n{station_locs.iloc[i]['unit']} should be m/s**2 or m/s!"
                )
            
            if remove_resp:
                trace_data[:, j] /= resp_j
                
        if not empty_station:
            data.append(trace_data)
            station_id.append(sta)
            t0.append(starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])

    data = np.stack(data)
    
    # Make dictionary with data in numpy arrays, tied to station and time:
    meta = {"data": data, "t0": t0, "station_id": station_id, "fname": station_id}
    
    # Use that to call to QuakeFlow API
    
    PHASENET_API_URL = "http://phasenet.quakeflow.com"

    batch = 4
    phasenet_picks = []
    for j in range(0, len(meta["station_id"]), batch):
        req = {"id": meta['station_id'][j:j+batch],
            "timestamp": meta["t0"][j:j+batch],
            "vec": meta["data"][j:j+batch].tolist()}

        resp = requests.post(f'{PHASENET_API_URL}/predict', json=req)
        phasenet_picks.extend(resp.json())
    
    # print(phasenet_picks)
    picks = pd.DataFrame(phasenet_picks)
    
    return phasenet_picks

    

    
def calc_snr(trace,sampleind,phase):
    # Calculate SNR of arrival
    # INPUTS:
    # trace = obspy-formatted trace object
    # sampleind = index in the trace's data of desired arrival for which to calculate SNR
    # phase = type of arrival as a string, either 'P' or 'S'
    #
    # OUTPUT:
    # snr = float object of calculated SNR for the input index
    
    if phase == 'P':
        window = [5,5] # in seconds
    if phase == 'S':
        window = [5,5]
    try:
        data = trace.data
        sr = int(trace.stats.sampling_rate)
        snr_num = max(abs(data[sampleind:(sampleind+(window[0]*sr))]))
        snr_denom = np.sqrt(np.mean((data[(sampleind-(window[1]*sr)):sampleind])**2))
        snr = snr_num/snr_denom
    except:
        snr = float('NaN')
    return(snr)