import datetime
import json
import logging
from datetime import timedelta

import alaska_utils
import dask
import eqt_utils
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import postprocess
import requests
import seisbench
import seisbench.models as sbm

# Progress bar for dask operations:
from dask.diagnostics import ProgressBar
from obspy.clients.fdsn import Client

pbar = dask.diagnostics.ProgressBar()
pbar.register()

import warnings

warnings.filterwarnings("ignore")

waveform_length = 60
waveform_overlap = 0
starttime = datetime.datetime(2018, 8, 6,18)
endtime = datetime.datetime(2018, 8, 7)

# Pre=saved station list in pandas dataframe format
dfS = pd.read_parquet(
    "https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_stations.parquet"
)
dfS = alaska_utils.station_list(dfS,starttime,endtime,elevation=False,network=False)

filt_type = 1
f1 = 5
f2 = 35

pick_info,gamma_picks = eqt_utils.ml_pick(
    dfS, starttime, endtime, waveform_length, waveform_overlap, filt_type, f1=f1, f2=f2
)

pick_info.to_parquet("picks_20180806_bp0535.parquet", version="2.6")
