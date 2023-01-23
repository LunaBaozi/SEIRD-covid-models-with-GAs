import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from datetime import date, timedelta
memory = Memory(location='data_cache/', verbose=0)

# taken from https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

@memory.cache
def read_csv(url):
    return pd.read_csv(url, parse_dates=[0])

def get_data_from_day(cod_reg, yyyymmdd, col, url_base='https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-'):
    url_full = url_base+yyyymmdd+'.csv'
    df = read_csv(url_full)
    return df.loc[df["codice_regione"] == cod_reg, col]

def get_data_interval(start_date, end_date, cod_reg, col=["data", "nuovi_positivi", "totale_positivi", "totale_ospedalizzati", "deceduti", "dimessi_guariti", "totale_casi"]):
    daily_data = pd.DataFrame(columns=col)
    cumulative_data = pd.DataFrame()
    yyyy_s = int(start_date[0:4])
    mm_s = int(start_date[4:6])
    dd_s = int(start_date[6: :])
    yyyy_e = int(end_date[0:4])
    mm_e = int(end_date[4:6])
    dd_e = int(end_date[6: :])

    # print("START FETCHING DATA...")
    start_d = date(yyyy_s, mm_s, dd_s)
    end_d = date(yyyy_e, mm_e, dd_e) 
    for single_date in daterange(start_d, end_d):
        current_date = single_date.strftime("%Y%m%d")
        current_df = get_data_from_day(cod_reg, current_date, col)
        cumulative_df = current_df[["data", "totale_positivi", "deceduti", "dimessi_guariti", "totale_casi"]]
        
        cumulative_data = pd.concat([cumulative_data, cumulative_df])
        daily_df = current_df[["data", "nuovi_positivi", "deceduti", "dimessi_guariti"]]

    return cumulative_data

def augment_data(df):
    data = pd.DataFrame(columns=["data", "cum_pos", "cum_osp", "cum_dec"])

    cumulative = df.groupby("data")["nuovi_positivi"].sum()
    print(cumulative)

