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
import alaska_utils
import postprocess

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)


def ml_pick(dfS,t1,t2,waveform_length,waveform_overlap,filt_type,f1=False,f2=False):
    """
    Takes you through the whole workflow- downloads waveforms from IRIS, denoises/filters waveforms 
    as specified, applies EQTransformer to return phase picks, and saves all pick metadata including SNR values
    
    INPUTS
    dfS - pandas dataframe of metadata for all stations to pull data for
    t1 - desired start time, datetime format
    t2 - desired end time, datetime format
    waveform_length - window length of individual waveforms in s, float
    waveform_overlap - number of seconds with which to overlap windows in s, float
    filt_type - whether or not to filter the waveforms prior to applying EQTransformer
        0 = raw waveforms
        1 = bandpass filter, between f1 and f2 Hz
        2 = denoise using DeepDenoiser
    
    OUTPUTS
    pick_info - pandas dataframe with all metadata of successful phase picks
    gamma_picks - list of dictionaries, one for each pick, formatted for use with GAMMA
    """
    
    # Convert to pandas datetime
    dfS['start_date']=pd.to_datetime(dfS['start_date'],infer_datetime_format=True,errors='coerce')
    dfS['end_date']=pd.to_datetime(dfS['end_date'],infer_datetime_format=True,errors='coerce')

    # Download waveforms
    time_bins = pd.to_datetime(np.arange(t1,t2,pd.Timedelta(waveform_length-waveform_overlap,'seconds')))
    
    print('Downloading data from IRIS:')
    
    @dask.delayed
    def loop_times(dfS,t1,waveform_length):
        return alaska_utils.retrieve_waveforms(dfS,t1,t1+pd.Timedelta(waveform_length,'seconds'),separate=True)


    lazy_results = [loop_times(dfS,time,waveform_length) for time in time_bins]
    
    results = dask.compute(lazy_results)
    
    # Concat into big list of streams
    test = sum(results,[])
    stream = []
    for t in test:
        stream.extend(t)
    

    # Filter waveforms as specified, then apply EQTransformer
    if filt_type==0:
        print('Status: applying EQTransformer')
        annotation = apply_eqt(stream)
        pick_info = get_picks(stream,annotation,filt_type,denoise=False)
    if filt_type==1:
        print('Status: filtering data')
        filtered = filter_waveforms(stream,f1,f2)
        print('Status: applying EQTransformer')
        annotation = apply_eqt(filtered)
        pick_info = get_picks(stream,annotation,filt_type,denoise=filtered)
    if filt_type==2:
        print('Status: denoising data using DeepDenoiser')
        denoised = denoise_waveforms(stream)
        print('Status: applying EQTransformer')
        annotation = apply_eqt(denoised)
        pick_info = get_picks(stream,annotation,filt_type,denoise=denoised)
    
    # Convert to string to save as parquet:
    pick_info['timestamp']= [p.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for p in pick_info['timestamp']]
    # For the case of overlapping windows:
    pick_info = remove_duplicates(pick_info)
    pick_info['timestamp']= [p.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for p in pick_info['timestamp']]
    
    gamma_picks = convert_to_gamma(pick_info)  
    
    return pick_info,gamma_picks

def denoise_waveforms(stream):
    """
    Denoise waveforms using DeepDenoiser
    
    INPUTS
    stream - list of obspy streams, one for each station/channel combo, each 60 s long
    
    OUTPUTS
    denoise - same list of streams, but denoised
    """
    # Load model
    model = sbm.DeepDenoiser.from_pretrained("original")
    
    # Apply DeepDenoiser model
    denoise = np.empty([len(stream)],dtype=object)
    model.cuda()
    for i,st in enumerate(stream):
        den = model.annotate(st)
        denoise[i]=den
    
    return denoise

def filter_waveforms(stream,f1,f2):
    """
    Bandpass filter waveforms
    
    INPUTS
    stream - list of obspy streams, one for each station/channel combo, each 60 s long
    f1 - lower bandpass limit in Hz
    f2 - upper bandpass limit in Hz
    
    OUTPUTS
    stream - same list of streams, but filtered
    """
    for st in stream:
        st.filter('bandpass',freqmin=f1,freqmax=f2)
    
    return stream

def apply_eqt(stream):
    """
    Annotate streams with probability of P and S pick using EQT
    
    INPUTS
    stream - list of obspy streams, one for each station/channel combo, each 60 s long
    
    OUTPUTS
    annotation - list of obspy streams, each an annotated version of the input stream
    
    """

    # Load model
    model = sbm.EQTransformer.from_pretrained("original")
    # EDIT MODEL TO NOT CUT SAMPLES OFF 
    model.default_args["blinding"] = (0,0)

    model.cuda()
    annotation = np.empty([len(stream)],dtype=object)

    for i,st in enumerate(stream):
        at = model.annotate(st)
        annotation[i]=at; 

    return annotation

def get_picks(stream,annotation,filt_type,denoise=False):
    """
    Extract phase picks from streams of EQTransformer phase probabilities
    
    INPUTS
    stream - list of obspy streams, raw, one for each station/channel combo, each 60 s long
    annotation - list of obspy streams, each an EQT probability stream corresponding to the raw stream
    
    OUTPUTS
    pick_info - pandas dataframe with all metadata of successful phase picks
    
    """
    pick_meta=[];
    for i in range(int(len(annotation))):
        
        # For empty annotations:
        if not annotation[i]:
            continue
        
        annotation[i] = check_traces(annotation[i])
        preds = np.empty([1,annotation[i][0].stats.npts,1,3])
        preds[0,:,0,0] = annotation[i][0].data
        preds[0,:,0,1] = annotation[i][1].data
        preds[0,:,0,2] = annotation[i][2].data

        station_id = annotation[i][0].stats.network + '..' + annotation[i][0].stats.station + '.'
        final_id = stream[i][0].stats.network + '.' + stream[i][0].stats.station + '..' + stream[i][0].stats.channel[0:2]

        picks = postprocess.extract_picks(preds,station_ids = [station_id],fnames = [station_id],t0=[str(annotation[i][0].stats.starttime)])

        # now call to original data using the same i index to get amplitudes

        # Raw amplitudes
        stream[i] = check_traces(stream[i])
        raw = np.empty([1,stream[i][0].stats.npts,1,3])
        raw[0,:,0,0] = stream[i].select(channel="**Z")[0].data
        if stream[i].select(channel="**N"):
            raw[0,:,0,1] = stream[i].select(channel="**N")[0].data
            raw[0,:,0,2] = stream[i].select(channel="**E")[0].data
        else:
            raw[0,:,0,1] = stream[i].select(channel="**1")[0].data
            raw[0,:,0,2] = stream[i].select(channel="**2")[0].data
        raw_amps = postprocess.extract_amplitude(raw,picks)

        # Denoised amplitudes (if applicable)
        if filt_type!=0:
            denoise[i] = check_traces(denoise[i])
            dat = np.empty([1,denoise[i][0].stats.npts,1,3])
            dat[0,:,0,0] = denoise[i].select(channel="**Z")[0].data
            if stream[i].select(channel="**N"):
                dat[0,:,0,1] = denoise[i].select(channel="**N")[0].data
                dat[0,:,0,2] = denoise[i].select(channel="**E")[0].data
            else:
                dat[0,:,0,1] = denoise[i].select(channel="**1")[0].data
                dat[0,:,0,2] = denoise[i].select(channel="**2")[0].data
            den_amps = postprocess.extract_amplitude(dat,picks)

        # Then, if the pick isn't empty, calculate SNR of pick
        if picks[0].p_prob[0]:
            for j in range(len(picks[0].p_prob[0])):
                # Get timestamp of pick:
                ts = annotation[i][0].stats.starttime + (pd.Timedelta(1,'seconds')*annotation[i][0].stats.delta*picks[0].p_idx[0][j])
                # Get SNR of pick:
                z_raw_snr = calc_snr(stream[i].select(channel="**Z")[0],picks[0].p_idx[0][j],'P');
                if filt_type!=0:
                    z_den_snr = calc_snr(denoise[i].select(channel="**Z")[0],picks[0].p_idx[0][j],'P');
                if stream[i].select(channel="**N"):
                    n_raw_snr = calc_snr(stream[i].select(channel="**N")[0],picks[0].p_idx[0][j],'P');
                    e_raw_snr = calc_snr(stream[i].select(channel="**E")[0],picks[0].p_idx[0][j],'P');
                    if filt_type!=0:
                        n_den_snr = calc_snr(denoise[i].select(channel="**N")[0],picks[0].p_idx[0][j],'P');
                        e_den_snr = calc_snr(denoise[i].select(channel="**E")[0],picks[0].p_idx[0][j],'P');
                else:
                    n_raw_snr = calc_snr(stream[i].select(channel="**1")[0],picks[0].p_idx[0][j],'P');
                    e_raw_snr = calc_snr(stream[i].select(channel="**2")[0],picks[0].p_idx[0][j],'P');
                    if filt_type!=0:
                        n_den_snr = calc_snr(denoise[i].select(channel="**1")[0],picks[0].p_idx[0][j],'P');
                        e_den_snr = calc_snr(denoise[i].select(channel="**2")[0],picks[0].p_idx[0][j],'P');
                if filt_type==0:
                    z_den_snr = float('NaN'); n_den_snr = float('NaN'); e_den_snr = float('NaN');
                    den_amp=float('NaN')
                else:
                    den_amp = den_amps[0].p_amp[0][j]
                # Save all info in dictionary:
                p_dict = {'id':final_id,'network':stream[i][0].stats.network,'station':stream[i][0].stats.station,'channel':stream[i][0].stats.channel[0:2],'phase':'P',\
                          'timestamp':ts,'prob':picks[0].p_prob[0][j],'raw_amp':raw_amps[0].p_amp[0][j],'den_amp':den_amp,\
                          'z_raw_snr':z_raw_snr,'z_den_snr':z_den_snr,'n_raw_snr':n_raw_snr,'n_den_snr':n_den_snr,'e_raw_snr':e_raw_snr,'e_den_snr':e_den_snr}
                pick_meta.append(p_dict)
        if picks[0].s_prob[0]:
            for j in range(len(picks[0].s_prob[0])):
                # Get timestamp of pick:
                ts = annotation[i][0].stats.starttime + (pd.Timedelta(1,'seconds')*annotation[i][0].stats.delta*picks[0].s_idx[0][j])
                # Get SNR of pick:
                z_raw_snr = calc_snr(stream[i].select(channel="**Z")[0],picks[0].s_idx[0][j],'S');
                if filt_type!=0:
                    z_den_snr = calc_snr(denoise[i].select(channel="**Z")[0],picks[0].s_idx[0][j],'S');
                if stream[i].select(channel="**N"):
                    n_raw_snr = calc_snr(stream[i].select(channel="**N")[0],picks[0].s_idx[0][j],'S');
                    e_raw_snr = calc_snr(stream[i].select(channel="**E")[0],picks[0].s_idx[0][j],'S');
                    if filt_type!=0:
                        n_den_snr = calc_snr(denoise[i].select(channel="**N")[0],picks[0].s_idx[0][j],'S');
                        e_den_snr = calc_snr(denoise[i].select(channel="**E")[0],picks[0].s_idx[0][j],'S');
                else:
                    n_raw_snr = calc_snr(stream[i].select(channel="**1")[0],picks[0].s_idx[0][j],'S');
                    e_raw_snr = calc_snr(stream[i].select(channel="**2")[0],picks[0].s_idx[0][j],'S');
                    if filt_type!=0:
                        n_den_snr = calc_snr(denoise[i].select(channel="**1")[0],picks[0].s_idx[0][j],'S');
                        e_den_snr = calc_snr(denoise[i].select(channel="**2")[0],picks[0].s_idx[0][j],'S');
                if filt_type==0:
                    z_den_snr = float('NaN'); n_den_snr = float('NaN'); e_den_snr = float('NaN');
                    den_amp=float('NaN')
                else:
                    den_amp = den_amps[0].s_amp[0][j]
                # Save all info in dictionary:
                s_dict = {'id':final_id,'network':stream[i][0].stats.network,'station':stream[i][0].stats.station,'channel':stream[i][0].stats.channel[0:2],'phase':'S',\
                          'timestamp':ts,'prob':picks[0].s_prob[0][j],'raw_amp':raw_amps[0].s_amp[0][j],'den_amp':den_amp,\
                          'z_raw_snr':z_raw_snr,'z_den_snr':z_den_snr,'n_raw_snr':n_raw_snr,'n_den_snr':n_den_snr,'e_raw_snr':e_raw_snr,'e_den_snr':e_den_snr}
                pick_meta.append(s_dict)

    # Save all pick info as pandas dataframe
    pick_info = pd.DataFrame.from_dict(pick_meta)
    
    return(pick_info)

def convert_to_gamma(pick_df):
    """
    Converts pandas dataframe of pick information to a list of dictionaries, one for each pick, for use with GAMMA
    """
    gamma_picks = []
    for i in range(len(pick_df)):
        p = pick_df.iloc[i]
        gamma_dict = {'id':p['id'],'timestamp':str(p['timestamp']),'prob':p['prob'],'amp':p['raw_amp'],'type':p['phase']}
        gamma_picks.append(gamma_dict)
    return gamma_picks


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

def check_traces(stre):
    """
    Reads in a 3-component stream and checks to see if all traces are the same length
    If not, cuts all traces to minimum length
    """

    npts1 = stre[0].stats.npts
    npts2 = stre[1].stats.npts
    npts3 = stre[2].stats.npts

    if npts1==npts2 & npts2==npts3:
        return stre

    else:
        stre_edit = stre.copy()
        npts_cut = np.min([npts1,npts2,npts3])
        stre_edit[0].data = stre_edit[0].data[0:npts_cut]
        stre_edit[1].data = stre_edit[1].data[0:npts_cut]
        stre_edit[2].data = stre_edit[2].data[0:npts_cut]
        return stre_edit
    
def remove_duplicates(df):
    """
    For use when time windows have overlaps
    
    Removes duplicate picks that occur on the same station within several seconds of each other
    Keeps the pick with higher probability
    
    Input: pandas dataframe in the format output from the get_picks function
    
    Output: same dataframe but with duplicates dropped
    """
    df['timestamp']=pd.to_datetime(df['timestamp'],infer_datetime_format=True)
    df = df.sort_values(by=['timestamp']).reset_index()
    
    # Find duplicates of id and phase:
    df['dupes'] = df.duplicated(subset=['id','phase'])

    to_drop=[]
    # Loop through and get which repeated times should be dropped:
    for i,bol in enumerate(df['dupes']):
        if bol:
            row = df.iloc[i]
            diffs = row['timestamp']-df['timestamp']
            time_dupes = df[(np.abs(diffs)<pd.Timedelta(3,'second')) & (df['id']==row['id']) & (df['phase']==row['phase'])]

            # Check for time duplicates and drop the one with lower probability
            if len(time_dupes)>1:
                to_drop.append(time_dupes['prob'].idxmin())

    df = df.drop(index=to_drop)
    df.reset_index(inplace=True)
    
    return(df)




def merge_true(df,resid_max):
    """
    Compares pandas dataframe of ML picks to original AK catalog using a time residual threshold in seconds
    
    Returns a new dataframe with true picks merged
    """
    
    # Load master pick and event lists
    ground_truth = pd.read_parquet('https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_picks.parquet')
    events = pd.read_parquet('https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_events.parquet')
    ground_truth['og_timestamp']=pd.to_datetime(ground_truth['og_timestamp'],format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
    events['time']=pd.to_datetime(events['time'],format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
    

    # Let's only compare the part of the original catalog that we have EQT picks for:
    cat_start = min(df['timestamp'] - pd.Timedelta(30,'seconds'))
    cat_end = max(df['timestamp'] + pd.Timedelta(30,'seconds'))
    ground_truth = ground_truth[(ground_truth['og_timestamp'] > cat_start) & (ground_truth['og_timestamp'] < cat_end)]
    # Also only compare picks on stations that are in our station list:
    ground_truth = ground_truth[([t in df['id'].to_list() for t in ground_truth['sta_code']])]

    # Sort both dataframes by time:
    df.sort_values(by=['timestamp'],inplace=True)
    ground_truth.sort_values(by=['og_timestamp'],inplace=True)

    # Merge dataframes, only merging picks if they have matching station ID, phase type, and are within 0.1 s of each other
    comp = pd.merge_asof(left=df,right=ground_truth,left_on=['timestamp'],right_on=['og_timestamp'],left_by=['id','type'],right_by=['sta_code','og_phase'],tolerance = pd.Timedelta(resid_max,'seconds'),direction='nearest')

    # Add residual column: 
    comp['pick_resid'] = comp['og_timestamp'] - comp['timestamp']
    comp['pick_resid'] = comp['pick_resid'].dt.total_seconds()
    
    return comp






def plot_probscatter(df,comp):
    """
    Reads in dataframe of EQTransformer picks and its companion comparative dataframe from merge_true()
    
    Plots scatter plot of picks throughout time in comparison to their probability
    """
    
    
    
    # Load master pick and event lists
    ground_truth = pd.read_parquet('https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_picks.parquet')
    events = pd.read_parquet('https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_events.parquet')
    ground_truth['og_timestamp']=pd.to_datetime(ground_truth['og_timestamp'],format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
    events['time']=pd.to_datetime(events['time'],format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
    

    # Let's only compare the part of the original catalog that we have EQT picks for:
    cat_start = min(df['timestamp'] - pd.Timedelta(30,'seconds'))
    cat_end = max(df['timestamp'] + pd.Timedelta(30,'seconds'))
    ground_truth = ground_truth[(ground_truth['og_timestamp'] > cat_start) & (ground_truth['og_timestamp'] < cat_end)]
    events = events[(events['time'] > cat_start) & (events['time'] < cat_end)]
    events = events.sort_values(by=['time'])
    # Also only compare picks on stations that are in our station list:
    ground_truth = ground_truth[([t in df['id'].to_list() for t in ground_truth['sta_code']])]

    
    true_positives = comp[comp['og_timestamp'].notna()]


    # Filter by probability?
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['timestamp'],df['prob'])
    ax.scatter(ground_truth['og_timestamp'],np.full(np.shape(ground_truth['og_timestamp']),0.5))
    ax.scatter(true_positives['timestamp'],true_positives['prob'])
    ax.vlines(x=events['time'],ymin=0,ymax=1,colors ='r')
    ax.set_xlabel('time')
    ax.set_ylabel('pick probability')
    ax.legend(['EQT Picks','True Picks','Matched Picks','Earthquakes'])
    
    return fig,ax
