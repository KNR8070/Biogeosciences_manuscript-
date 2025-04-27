#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:09:15 2024

@author: knreddy
"""

#%% Load Modules
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import math
#%% Bias calculation
def bias_var(obs,mod):
    # obs_s = pd.Series(obs)
    # mod_s = pd.Series(mod)
    # obs_mv_avg = (obs_s.rolling(5,center=True)).mean()
    # mod_mv_avg = (mod_s.rolling(5,center=True)).mean()
    # num = sum(abs(mod_mv_avg-obs_mv_avg))
    # den = sum(obs_mv_avg)
    num = sum(abs(mod-obs))
    den = sum(obs)
    bias_ = num/den
    return bias_
#%% RMSE calculation
def RMSE_var(obs,mod):
    MSE = np.square(np.subtract(obs,mod)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE
#%% load CLM data
wrk_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/CLM5_Data'

data_CTRL = xr.open_dataset(wrk_dir+'/CLM5_CTRL_1970_2014_CropData_22-Mar-2024.nc')

lat = data_CTRL['lat']
lon = data_CTRL['lon']
year = data_CTRL['year']

# CLM5 yield data
CTRL_GY_Ri = data_CTRL['GY_Rice']
CTRL_GY_SW = data_CTRL['GY_Wheat']

CTRL_GY_Ri_spatial_mean = (np.nanmean(np.nanmean(CTRL_GY_Ri,axis=1),axis=1))*(2/100) #units t/ha
CTRL_GY_SW_spatial_mean = (np.nanmean(np.nanmean(CTRL_GY_SW,axis=1),axis=1))*(2/100)

# CTRL_GY_Ri_spatial_mean = (np.nansum(np.nansum(CTRL_GY_Ri,axis=1),axis=1))/100#*(2/100) #units t/ha
# CTRL_GY_SW_spatial_mean = (np.nansum(np.nansum(CTRL_GY_SW,axis=1),axis=1))/100#*(2/100)
#%% FAO yield data
data_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Observations_CarbonFluxes/'
wheat_data = pd.read_csv(data_dir+'FAOSTAT_data_en_4-9-2024_Wheat.csv')
rice_data = pd.read_csv(data_dir+'FAOSTAT_data_en_4-9-2024_Rice.csv')

Units = wheat_data['Unit'];
W_data = wheat_data['Value'];
R_data = rice_data['Value'];

W_prod_data = np.array(W_data[Units=='t'],dtype=float)
W_area_data = np.array(W_data[Units=='ha'],dtype=float)

R_prod_data = np.array(R_data[Units=='t'],dtype=float)
R_area_data = np.array(R_data[Units=='ha'],dtype=float)

W_yield_data = W_prod_data/W_area_data #units t/ha
R_yield_data = R_prod_data/R_area_data

#%%
W_yield_data_s = pd.Series(W_yield_data)
R_yield_data_s = pd.Series(R_yield_data)

CTRL_GY_Ri_spatial_mean_s = pd.Series(CTRL_GY_Ri_spatial_mean)
CTRL_GY_SW_spatial_mean_s = pd.Series(CTRL_GY_SW_spatial_mean)

W_yield_data_5_avg = (W_yield_data_s.rolling(5,center=True)).mean()
R_yield_data_5_avg = (R_yield_data_s.rolling(5,center=True)).mean()

CTRL_GY_Ri_spatial_mean_5_avg = (CTRL_GY_Ri_spatial_mean_s.rolling(5,center=True)).mean()
CTRL_GY_SW_spatial_mean_5_avg = (CTRL_GY_SW_spatial_mean_s.rolling(5,center=True)).mean()
#%%
FAO_w_data = W_yield_data_5_avg
FAO_r_data = R_yield_data_5_avg

CLM_w_data = CTRL_GY_SW_spatial_mean_5_avg
CLM_r_data = CTRL_GY_Ri_spatial_mean_5_avg

fsize = 10

corr_pos_x = 2003
corr_pos_y = 1.5

bias_pos_x = 2003
bias_pos_y = 1

rmse_pos_x = 2003
rmse_pos_y = 0.5

fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,layout='constrained')

ax[0].plot(np.arange(1970,2015),FAO_r_data,label='FAO',c='k')
ax[0].plot(np.arange(1970,2015),CLM_r_data,label='CLM5',c='steelblue')

ax[1].plot(np.arange(1970,2015),FAO_w_data,c='k')
ax[1].plot(np.arange(1970,2015),CLM_w_data,c='steelblue')

###### calculating and writing statistics
bias_rice = bias_var(FAO_r_data, CLM_r_data)
bias_wheat = bias_var(FAO_w_data, CLM_w_data)

RMSE_rice = RMSE_var(FAO_r_data, CLM_r_data)
RMSE_wheat = RMSE_var(FAO_w_data, CLM_w_data)

corr_rice,p_val_rice = pearsonr(FAO_r_data[2:-2], CLM_r_data[2:-2])
corr_wheat,p_val_wheat = pearsonr(FAO_w_data[2:-2], CLM_w_data[2:-2])

rice_bias_text = '$MAB='+'{:.2f}$'.format(bias_rice)
rice_rmse_text = '$RMSE='+'{:.2f}$'.format(RMSE_rice)

wheat_bias_text = '$MAB='+'{:.2f}$'.format(bias_wheat)
wheat_rmse_text = '$RMSE='+'{:.2f}$'.format(RMSE_wheat)

rice_corr_text = f"Pearsons r = {corr_rice:.2f}*"
wheat_corr_text = f"Pearsons r = {corr_wheat:.2f}*"

ax[0].text(corr_pos_x,corr_pos_y,rice_corr_text,fontsize=fsize)
ax[0].text(bias_pos_x,bias_pos_y,rice_bias_text,fontsize=fsize)
ax[0].text(rmse_pos_x,rmse_pos_y,rice_rmse_text,fontsize=fsize)
ax[1].text(corr_pos_x,corr_pos_y,wheat_corr_text,fontsize=fsize)
ax[1].text(bias_pos_x,bias_pos_y,wheat_bias_text,fontsize=fsize)
ax[1].text(rmse_pos_x,rmse_pos_y,wheat_rmse_text,fontsize=fsize)
######

ax[0].legend()
ax[0].set_ylim([0,5])
ax[1].set_ylim([0,5])
ax[1].set_xlim([1970,2014])
ax[0].set_title('a. Rice')
ax[1].set_title('b. Wheat')

ax[0].set_ylabel('Yield (t/ha)')
ax[1].set_ylabel('Yield (t/ha)')


# fig.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Comparison_Yield_CLM5_FAO_18Oct.png',dpi=600)

