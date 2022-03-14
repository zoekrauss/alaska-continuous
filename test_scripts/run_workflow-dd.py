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
starttime = datetime.datetime(2019, 5, 27)
endtime = datetime.datetime(2019, 5, 28)

# Pre=saved station list in pandas dataframe format
dfS = pd.read_parquet(
    "https://github.com/zoekrauss/alaska_catalog/raw/main/data_acquisition/alaska_stations.parquet"
)
dfS = alaska_utils.station_list(dfS,starttime,endtime,elevation=False,network=False)

filt_type = 2
f1 = None
f2 = None

pick_info,gamma_picks = eqt_utils.ml_pick(
    dfS, starttime, endtime, waveform_length, waveform_overlap, filt_type, f1=f1, f2=f2
)

pick_info.to_parquet("/home/adminuser/lynx/picks_20190527_dd.parquet", version="2.6")
