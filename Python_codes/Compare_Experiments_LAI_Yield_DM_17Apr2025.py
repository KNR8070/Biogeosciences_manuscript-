#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:32:08 2024

@author: knreddy
"""
#%% Load Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from cmcrameri import cm
from sklearn import preprocessing
import scipy.stats as stats
import pymannkendall as mk
import string
#%% Set some parameters to apply to all plots. These can be overridden
import matplotlib
# Plot size to 12" x 7"
matplotlib.rc('figure', figsize = (15, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = True, right = True)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')
#%% Normalise array data
def normalise_data(X):
    mean = np.nanmean(X)
    std = np.nanstd(X)
    # min_data = min(X)
    # max_data = max(X)
    # x_new = (X-min_data)/(max_data-min_data)
    x_new = mean/std
    # mean_new = np.nanmean(x_new)
    # if np.nanmean(X)<0:
    #     return x_new*(-1)
    # else:
    return x_new
#%% Trend in max. LAI
def find_trend_pval(X):    
    # t_cal = u"gregorian"
    
    # utime = netcdftime.utime(year, calendar = t_cal)
    # datevar = utime.num2date(nctime)
    # print(datevar.shape)
    # X = np.array(X)  
    nt, nlat, nlon = X.shape
    ngrd = nlon*nlat
    X_grd = X.reshape((nt, ngrd), order='F')
    # x = np.linspace(1,nt,nt)
    X_rate = np.empty((ngrd,1))
    X_rate[:,:]=np.nan
    X_pvalue = np.empty((ngrd,1))
    X_pvalue[:,:]=np.nan
    
    for i in range(ngrd): 
        y = X_grd[:,i]   
        if(not np.ma.is_masked(y)):         
            # z = np.polyfit(x, y, 1)
            # X_rate[i,0] = z[0]*120.0
            # slope, intercept, r_value, p_value, std_err = stats.linregress(x, X_grd[:,i])
            trend, h, p_value, z, Tau, s, var_s, slope, intercept = mk.original_test(X_grd[:,i],alpha=0.01)
            X_rate[i,0] = slope#*10 #to get rate per decade
            if p_value<0.01:
                X_pvalue[i,0] = p_value
        
    X_rate = X_rate.reshape((nlat,nlon), order='F')
    X_pvalue = X_pvalue.reshape((nlat,nlon), order='F')
    
    return X_rate, X_pvalue
#%% Clip data for Indian region
def clip_data(Spatial_data,Region):
    Spatial_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    Spatial_data.rio.write_crs("epsg:4326", inplace=True)
    clip_Spatial_data = Spatial_data.rio.clip(Region.geometry, India.crs, drop=True)
    return clip_Spatial_data
#%% Custom Normalise 
# Example of making your own norm.  Also see matplotlib.colors.
# From Joe Kington: This one gives two different linear ramps:
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#%% load CLM data
## Monthly GPP Spatial Plotting
month = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
         '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}
daysinmonth = {'1':'31','2':'28','3':'31','4':'30','5':'31','6':'30',
         '7':'31','8':'31','9':'30','10':'31','11':'30','12':'31'}


wrk_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/CLM5_Data'
shp_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/India_Shapefile_'

saving_fig_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/Figures/'
mask_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/'

data_CTRL = xr.open_dataset(wrk_dir+'/CLM5_CTRL_1970_2014_CropData_21-May-2024.nc')
data_CO2 = xr.open_dataset(wrk_dir+'/CLM5_S_CO2_1970_2014_CropData_21-May-2024.nc')
data_Irrig = xr.open_dataset(wrk_dir+'/CLM5_S_Irrig_1970_2014_CropData_21-May-2024.nc')
data_Clim = xr.open_dataset(wrk_dir+'/CLM5_S_Clim_1970_2014_CropData_21-May-2024.nc')
data_NFert = xr.open_dataset(wrk_dir+'/CLM5_S_NFert_1970_2014_CropData_21-May-2024.nc')

mask_data = xr.open_dataset(mask_dir+'landuse.timeseries_360x720cru_hist_78pfts_CMIP6_simyr1850-2015_India_c221122.nc')

lat = data_CTRL['lat']
lon = data_CTRL['lon']
year = data_CTRL['year']

India = geopandas.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")
#%% Extract data
CTRL_LAI_Ri = data_CTRL['LAI_Rice']
CTRL_LAI_SW = data_CTRL['LAI_Wheat']
CTRL_GY_Ri = data_CTRL['GY_Rice']
CTRL_GY_SW = data_CTRL['GY_Wheat']
CTRL_DM_Ri = data_CTRL['DM_Rice']
CTRL_DM_SW = data_CTRL['DM_Wheat']

Clim_LAI_Ri = data_Clim['LAI_Rice']
Clim_LAI_SW = data_Clim['LAI_Wheat']
Clim_GY_Ri = data_Clim['GY_Rice']
Clim_GY_SW = data_Clim['GY_Wheat']
Clim_DM_Ri = data_Clim['DM_Rice']
Clim_DM_SW = data_Clim['DM_Wheat']

CO2_LAI_Ri = data_CO2['LAI_Rice']
CO2_LAI_SW = data_CO2['LAI_Wheat']
CO2_GY_Ri = data_CO2['GY_Rice']
CO2_GY_SW = data_CO2['GY_Wheat']
CO2_DM_Ri = data_CO2['DM_Rice']
CO2_DM_SW = data_CO2['DM_Wheat']

NFert_LAI_Ri = data_NFert['LAI_Rice']
NFert_LAI_SW = data_NFert['LAI_Wheat']
NFert_GY_Ri = data_NFert['GY_Rice']
NFert_GY_SW = data_NFert['GY_Wheat']
NFert_DM_Ri = data_NFert['DM_Rice']
NFert_DM_SW = data_NFert['DM_Wheat']

Irrig_LAI_Ri = data_Irrig['LAI_Rice']
Irrig_LAI_SW = data_Irrig['LAI_Wheat']
Irrig_GY_Ri = data_Irrig['GY_Rice']
Irrig_GY_SW = data_Irrig['GY_Wheat']
Irrig_DM_Ri = data_Irrig['DM_Rice']
Irrig_DM_SW = data_Irrig['DM_Wheat']

pct_cft = np.array(mask_data['PCT_CFT'])
p_c_test = mask_data['PCT_CFT']

pct_cft_ = xr.DataArray(pct_cft,
    coords=dict(
        time = p_c_test.coords['time'],
        cft = p_c_test.coords['cft'],
        lat = CTRL_DM_Ri.coords['lat'],
        lon = CTRL_DM_Ri.coords['lon']))

cft = np.array(mask_data['cft'])
pct_cft2 = pct_cft_[-46:-1,:,:,:] # pct_cft of period 1970 to 2014
#%% Prepare SW and rice mask
clipped_pct_cft = clip_data(pct_cft2, India)

pct_cft_sw1 = clipped_pct_cft[:,cft==19,:,:]
pct_cft_sw2 = clipped_pct_cft[:,cft==20,:,:]

pct_cft_sw = xr.concat([pct_cft_sw1,pct_cft_sw2], 'cft')
pct_cft_sw_ = pct_cft_sw.sum(axis=1)

pct_cft_rice1 = clipped_pct_cft[:,cft==61,:,:]
pct_cft_rice2 = clipped_pct_cft[:,cft==62,:,:]

pct_cft_rice = xr.concat([pct_cft_rice1,pct_cft_rice2], 'cft')
pct_cft_rice_ = pct_cft_rice.sum(axis=1)

pct_cft_sw_mask = np.ma.masked_less_equal(pct_cft_sw_, 1)
pct_cft_rice_mask = np.ma.masked_less_equal(pct_cft_rice_, 1)
#%% Clip all spatial data for Indian region
clipped_CTRL_LAI_Ri = clip_data(CTRL_LAI_Ri, India)
clipped_CTRL_LAI_SW = clip_data(CTRL_LAI_SW, India)
clipped_CTRL_GY_Ri = clip_data(CTRL_GY_Ri, India)
clipped_CTRL_GY_SW = clip_data(CTRL_GY_SW, India)
clipped_CTRL_DM_Ri = clip_data(CTRL_DM_Ri, India)
clipped_CTRL_DM_SW = clip_data(CTRL_DM_SW, India)

clipped_Clim_LAI_Ri = clip_data(Clim_LAI_Ri, India)
clipped_Clim_LAI_SW = clip_data(Clim_LAI_SW, India)
clipped_Clim_GY_Ri = clip_data(Clim_GY_Ri, India)
clipped_Clim_GY_SW = clip_data(Clim_GY_SW, India)
clipped_Clim_DM_Ri = clip_data(Clim_DM_Ri, India)
clipped_Clim_DM_SW = clip_data(Clim_DM_SW, India)

clipped_CO2_LAI_Ri = clip_data(CO2_LAI_Ri, India)
clipped_CO2_LAI_SW = clip_data(CO2_LAI_SW, India)
clipped_CO2_GY_Ri = clip_data(CO2_GY_Ri, India)
clipped_CO2_GY_SW = clip_data(CO2_GY_SW, India)
clipped_CO2_DM_Ri = clip_data(CO2_DM_Ri, India)
clipped_CO2_DM_SW = clip_data(CO2_DM_SW, India)

clipped_NFert_LAI_Ri = clip_data(NFert_LAI_Ri, India)
clipped_NFert_LAI_SW = clip_data(NFert_LAI_SW, India)
clipped_NFert_GY_Ri = clip_data(NFert_GY_Ri, India)
clipped_NFert_GY_SW = clip_data(NFert_GY_SW, India)
clipped_NFert_DM_Ri = clip_data(NFert_DM_Ri, India)
clipped_NFert_DM_SW = clip_data(NFert_DM_SW, India)

clipped_Irrig_LAI_Ri = clip_data(Irrig_LAI_Ri, India)
clipped_Irrig_LAI_SW = clip_data(Irrig_LAI_SW, India)
clipped_Irrig_GY_Ri = clip_data(Irrig_GY_Ri, India)
clipped_Irrig_GY_SW = clip_data(Irrig_GY_SW, India)
clipped_Irrig_DM_Ri = clip_data(Irrig_DM_Ri, India)
clipped_Irrig_DM_SW = clip_data(Irrig_DM_SW, India)
#%% masking for pft regions
SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)
Experiments = ('CTRL','Clim','CO2','NFert','Irrig')
Variables = ('LAI','GY','DM')
Crop = ('SW','Ri')

for i_exp in Experiments:
    for i_var in Variables:
        for i_crop in Crop:
            input_var_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop
            output_var_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_masked'
            if i_crop == 'SW':
                locals()[output_var_name] = np.ma.masked_array(eval(input_var_name),mask=SW_mask)
            else:
                locals()[output_var_name] = np.ma.masked_array(eval(input_var_name),mask=Rice_mask)
#%% Correcting DM in rice 
clipped_CTRL_DM_Ri_masked = np.ma.masked_less(clipped_CTRL_DM_Ri_masked,0)
clipped_Clim_DM_Ri_masked = np.ma.masked_less(clipped_Clim_DM_Ri_masked,0)
clipped_CO2_DM_Ri_masked = np.ma.masked_less(clipped_CO2_DM_Ri_masked,0)
clipped_NFert_DM_Ri_masked = np.ma.masked_less(clipped_NFert_DM_Ri_masked,0)
clipped_Irrig_DM_Ri_masked = np.ma.masked_less(clipped_Irrig_DM_Ri_masked,0)

clipped_CTRL_DM_SW_masked = np.ma.masked_less(clipped_CTRL_DM_SW_masked,0)
clipped_Clim_DM_SW_masked = np.ma.masked_less(clipped_Clim_DM_SW_masked,0)
clipped_CO2_DM_SW_masked = np.ma.masked_less(clipped_CO2_DM_SW_masked,0)
clipped_NFert_DM_SW_masked = np.ma.masked_less(clipped_NFert_DM_SW_masked,0)
clipped_Irrig_DM_SW_masked = np.ma.masked_less(clipped_Irrig_DM_SW_masked,0)
#%% Taking mean from 2-d array
shape0 = clipped_CTRL_LAI_Ri_masked.shape[0]
shape1 = clipped_CTRL_LAI_Ri_masked.shape[1]
shape2 = clipped_CTRL_LAI_Ri_masked.shape[2]

CTRL_LAI_Ri_masked_2d = clipped_CTRL_LAI_Ri_masked.reshape(shape0, shape1*shape2)
CTRL_LAI_Ri_spatial_mean = np.nanmean(CTRL_LAI_Ri_masked_2d,axis=1)
CTRL_LAI_Ri_spatial_std = np.nanstd(CTRL_LAI_Ri_masked_2d,axis=1)

CTRL_LAI_SW_masked_2d = clipped_CTRL_LAI_SW_masked.reshape(shape0, shape1*shape2)
CTRL_LAI_SW_spatial_mean = np.nanmean(CTRL_LAI_SW_masked_2d,axis=1)
CTRL_LAI_SW_spatial_std = np.nanstd(CTRL_LAI_SW_masked_2d,axis=1)

CTRL_GY_Ri_masked_2d = clipped_CTRL_GY_Ri_masked.reshape(shape0, shape1*shape2)
CTRL_GY_Ri_spatial_mean = np.nanmean(CTRL_GY_Ri_masked_2d/50,axis=1)
CTRL_GY_Ri_spatial_std = np.nanstd(CTRL_GY_Ri_masked_2d/50,axis=1)

CTRL_GY_SW_masked_2d = clipped_CTRL_GY_SW_masked.reshape(shape0, shape1*shape2)
CTRL_GY_SW_spatial_mean = np.nanmean(CTRL_GY_SW_masked_2d/50,axis=1)
CTRL_GY_SW_spatial_std = np.nanstd(CTRL_GY_SW_masked_2d/50,axis=1)

CTRL_DM_Ri_masked_2d = clipped_CTRL_DM_Ri_masked.reshape(shape0, shape1*shape2)
CTRL_DM_Ri_spatial_mean = np.nanmean(CTRL_DM_Ri_masked_2d/100,axis=1)
CTRL_DM_Ri_spatial_std = np.nanstd(CTRL_DM_Ri_masked_2d/100,axis=1)

CTRL_DM_SW_masked_2d = clipped_CTRL_DM_SW_masked.reshape(shape0, shape1*shape2)
CTRL_DM_SW_spatial_mean = np.nanmean(CTRL_DM_SW_masked_2d/100,axis=1)
CTRL_DM_SW_spatial_std = np.nanstd(CTRL_DM_SW_masked_2d/100,axis=1)

Clim_LAI_Ri_masked_2d = clipped_Clim_LAI_Ri_masked.reshape(shape0, shape1*shape2)
Clim_LAI_Ri_spatial_mean = np.nanmean(Clim_LAI_Ri_masked_2d,axis=1)
Clim_LAI_Ri_spatial_std = np.nanstd(Clim_LAI_Ri_masked_2d,axis=1)

Clim_LAI_SW_masked_2d = clipped_Clim_LAI_SW_masked.reshape(shape0, shape1*shape2)
Clim_LAI_SW_spatial_mean = np.nanmean(Clim_LAI_SW_masked_2d,axis=1)
Clim_LAI_SW_spatial_std = np.nanstd(Clim_LAI_SW_masked_2d,axis=1)

Clim_GY_Ri_masked_2d = clipped_Clim_GY_Ri_masked.reshape(shape0, shape1*shape2)
Clim_GY_Ri_spatial_mean = np.nanmean(Clim_GY_Ri_masked_2d/50,axis=1)
Clim_GY_Ri_spatial_std = np.nanstd(Clim_GY_Ri_masked_2d/50,axis=1)

Clim_GY_SW_masked_2d = clipped_Clim_GY_SW_masked.reshape(shape0, shape1*shape2)
Clim_GY_SW_spatial_mean = np.nanmean(Clim_GY_SW_masked_2d/50,axis=1)
Clim_GY_SW_spatial_std = np.nanstd(Clim_GY_SW_masked_2d/50,axis=1)

Clim_DM_Ri_masked_2d = clipped_Clim_DM_Ri_masked.reshape(shape0, shape1*shape2)
Clim_DM_Ri_spatial_mean = np.nanmean(Clim_DM_Ri_masked_2d/100,axis=1)
Clim_DM_Ri_spatial_std = np.nanstd(Clim_DM_Ri_masked_2d/100,axis=1)

Clim_DM_SW_masked_2d = clipped_Clim_DM_SW_masked.reshape(shape0, shape1*shape2)
Clim_DM_SW_spatial_mean = np.nanmean(Clim_DM_SW_masked_2d/100,axis=1)
Clim_DM_SW_spatial_std = np.nanstd(Clim_DM_SW_masked_2d/100,axis=1)


CO2_LAI_Ri_masked_2d = clipped_CO2_LAI_Ri_masked.reshape(shape0, shape1*shape2)
CO2_LAI_Ri_spatial_mean = np.nanmean(CO2_LAI_Ri_masked_2d,axis=1)
CO2_LAI_Ri_spatial_std = np.nanstd(CO2_LAI_Ri_masked_2d,axis=1)

CO2_LAI_SW_masked_2d = clipped_CO2_LAI_SW_masked.reshape(shape0, shape1*shape2)
CO2_LAI_SW_spatial_mean = np.nanmean(CO2_LAI_SW_masked_2d,axis=1)
CO2_LAI_SW_spatial_std = np.nanstd(CO2_LAI_SW_masked_2d,axis=1)

CO2_GY_Ri_masked_2d = clipped_CO2_GY_Ri_masked.reshape(shape0, shape1*shape2)
CO2_GY_Ri_spatial_mean = np.nanmean(CO2_GY_Ri_masked_2d/50,axis=1)
CO2_GY_Ri_spatial_std = np.nanstd(CO2_GY_Ri_masked_2d/50,axis=1)

CO2_GY_SW_masked_2d = clipped_CO2_GY_SW_masked.reshape(shape0, shape1*shape2)
CO2_GY_SW_spatial_mean = np.nanmean(CO2_GY_SW_masked_2d/50,axis=1)
CO2_GY_SW_spatial_std = np.nanstd(CO2_GY_SW_masked_2d/50,axis=1)

CO2_DM_Ri_masked_2d = clipped_CO2_DM_Ri_masked.reshape(shape0, shape1*shape2)
CO2_DM_Ri_spatial_mean = np.nanmean(CO2_DM_Ri_masked_2d/100,axis=1)
CO2_DM_Ri_spatial_std = np.nanstd(CO2_DM_Ri_masked_2d/100,axis=1)

CO2_DM_SW_masked_2d = clipped_CO2_DM_SW_masked.reshape(shape0, shape1*shape2)
CO2_DM_SW_spatial_mean = np.nanmean(CO2_DM_SW_masked_2d/100,axis=1)
CO2_DM_SW_spatial_std = np.nanstd(CO2_DM_SW_masked_2d/100,axis=1)


NFert_LAI_Ri_masked_2d = clipped_NFert_LAI_Ri_masked.reshape(shape0, shape1*shape2)
NFert_LAI_Ri_spatial_mean = np.nanmean(NFert_LAI_Ri_masked_2d,axis=1)
NFert_LAI_Ri_spatial_std = np.nanstd(NFert_LAI_Ri_masked_2d,axis=1)

NFert_LAI_SW_masked_2d = clipped_NFert_LAI_SW_masked.reshape(shape0, shape1*shape2)
NFert_LAI_SW_spatial_mean = np.nanmean(NFert_LAI_SW_masked_2d,axis=1)
NFert_LAI_SW_spatial_std = np.nanstd(NFert_LAI_SW_masked_2d,axis=1)

NFert_GY_Ri_masked_2d = clipped_NFert_GY_Ri_masked.reshape(shape0, shape1*shape2)
NFert_GY_Ri_spatial_mean = np.nanmean(NFert_GY_Ri_masked_2d/50,axis=1)
NFert_GY_Ri_spatial_std = np.nanstd(NFert_GY_Ri_masked_2d/50,axis=1)

NFert_GY_SW_masked_2d = clipped_NFert_GY_SW_masked.reshape(shape0, shape1*shape2)
NFert_GY_SW_spatial_mean = np.nanmean(NFert_GY_SW_masked_2d/50,axis=1)
NFert_GY_SW_spatial_std = np.nanstd(NFert_GY_SW_masked_2d/50,axis=1)

NFert_DM_Ri_masked_2d = clipped_NFert_DM_Ri_masked.reshape(shape0, shape1*shape2)
NFert_DM_Ri_spatial_mean = np.nanmean(NFert_DM_Ri_masked_2d/100,axis=1)
NFert_DM_Ri_spatial_std = np.nanstd(NFert_DM_Ri_masked_2d/100,axis=1)

NFert_DM_SW_masked_2d = clipped_NFert_DM_SW_masked.reshape(shape0, shape1*shape2)
NFert_DM_SW_spatial_mean = np.nanmean(NFert_DM_SW_masked_2d/100,axis=1)
NFert_DM_SW_spatial_std = np.nanstd(NFert_DM_SW_masked_2d/100,axis=1)

Irrig_LAI_Ri_masked_2d = clipped_Irrig_LAI_Ri_masked.reshape(shape0, shape1*shape2)
Irrig_LAI_Ri_spatial_mean = np.nanmean(Irrig_LAI_Ri_masked_2d,axis=1)
Irrig_LAI_Ri_spatial_std = np.nanstd(Irrig_LAI_Ri_masked_2d,axis=1)

Irrig_LAI_SW_masked_2d = clipped_Irrig_LAI_SW_masked.reshape(shape0, shape1*shape2)
Irrig_LAI_SW_spatial_mean = np.nanmean(Irrig_LAI_SW_masked_2d,axis=1)
Irrig_LAI_SW_spatial_std = np.nanstd(Irrig_LAI_SW_masked_2d,axis=1)

Irrig_GY_Ri_masked_2d = clipped_Irrig_GY_Ri_masked.reshape(shape0, shape1*shape2)
Irrig_GY_Ri_spatial_mean = np.nanmean(Irrig_GY_Ri_masked_2d/50,axis=1)
Irrig_GY_Ri_spatial_std = np.nanstd(Irrig_GY_Ri_masked_2d/50,axis=1)

Irrig_GY_SW_masked_2d = clipped_Irrig_GY_SW_masked.reshape(shape0, shape1*shape2)
Irrig_GY_SW_spatial_mean = np.nanmean(Irrig_GY_SW_masked_2d/50,axis=1)
Irrig_GY_SW_spatial_std = np.nanstd(Irrig_GY_SW_masked_2d/50,axis=1)

Irrig_DM_Ri_masked_2d = clipped_Irrig_DM_Ri_masked.reshape(shape0, shape1*shape2)
Irrig_DM_Ri_spatial_mean = np.nanmean(Irrig_DM_Ri_masked_2d/100,axis=1)
Irrig_DM_Ri_spatial_std = np.nanstd(Irrig_DM_Ri_masked_2d/100,axis=1)

Irrig_DM_SW_masked_2d = clipped_Irrig_DM_SW_masked.reshape(shape0, shape1*shape2)
Irrig_DM_SW_spatial_mean = np.nanmean(Irrig_DM_SW_masked_2d/100,axis=1)
Irrig_DM_SW_spatial_std = np.nanstd(Irrig_DM_SW_masked_2d/100,axis=1)
#%% Find trend and pvalue
[CTRL_LAI_sw_rate, CTRL_LAI_sw_pvalue] = find_trend_pval(CTRL_LAI_SW_masked)
[CTRL_LAI_rice_rate, CTRL_LAI_rice_pvalue] = find_trend_pval(CTRL_LAI_Ri_masked)

[CTRL_GY_sw_rate, CTRL_GY_sw_pvalue] = find_trend_pval(CTRL_GY_SW_masked/50)
[CTRL_GY_rice_rate, CTRL_GY_rice_pvalue] = find_trend_pval(CTRL_GY_Ri_masked/50)

[CTRL_DM_sw_rate, CTRL_DM_sw_pvalue] = find_trend_pval(CTRL_DM_SW_masked/2)
[CTRL_DM_rice_rate, CTRL_DM_rice_pvalue] = find_trend_pval(CTRL_DM_Ri_masked/2)

[Clim_LAI_sw_rate, Clim_LAI_sw_pvalue] = find_trend_pval(CTRL_LAI_SW_masked - Clim_LAI_SW_masked)
[Clim_LAI_rice_rate, Clim_LAI_rice_pvalue] = find_trend_pval(CTRL_LAI_Ri_masked - Clim_LAI_Ri_masked)

[Clim_GY_sw_rate, Clim_GY_sw_pvalue] = find_trend_pval((CTRL_GY_SW_masked - Clim_GY_SW_masked)/50)
[Clim_GY_rice_rate, Clim_GY_rice_pvalue] = find_trend_pval((CTRL_GY_Ri_masked - Clim_GY_Ri_masked)/50)

[Clim_DM_sw_rate, Clim_DM_sw_pvalue] = find_trend_pval((CTRL_DM_SW_masked - Clim_DM_SW_masked)/2)
[Clim_DM_rice_rate, Clim_DM_rice_pvalue] = find_trend_pval((CTRL_DM_Ri_masked - Clim_DM_Ri_masked)/2)

[CO2_LAI_sw_rate, CO2_LAI_sw_pvalue] = find_trend_pval(CTRL_LAI_SW_masked - CO2_LAI_SW_masked)
[CO2_LAI_rice_rate, CO2_LAI_rice_pvalue] = find_trend_pval(CTRL_LAI_Ri_masked - CO2_LAI_Ri_masked)

[CO2_GY_sw_rate, CO2_GY_sw_pvalue] = find_trend_pval((CTRL_GY_SW_masked - CO2_GY_SW_masked)/50)
[CO2_GY_rice_rate, CO2_GY_rice_pvalue] = find_trend_pval((CTRL_GY_Ri_masked - CO2_GY_Ri_masked)/50)

[CO2_DM_sw_rate, CO2_DM_sw_pvalue] = find_trend_pval((CTRL_DM_SW_masked - CO2_DM_SW_masked)/2)
[CO2_DM_rice_rate, CO2_DM_rice_pvalue] = find_trend_pval((CTRL_DM_Ri_masked - CO2_DM_Ri_masked)/2)

[NFert_LAI_sw_rate, NFert_LAI_sw_pvalue] = find_trend_pval(CTRL_LAI_SW_masked - NFert_LAI_SW_masked)
[NFert_LAI_rice_rate, NFert_LAI_rice_pvalue] = find_trend_pval(CTRL_LAI_Ri_masked - NFert_LAI_Ri_masked)

[NFert_GY_sw_rate, NFert_GY_sw_pvalue] = find_trend_pval((CTRL_GY_SW_masked - NFert_GY_SW_masked)/50)
[NFert_GY_rice_rate, NFert_GY_rice_pvalue] = find_trend_pval((CTRL_GY_Ri_masked - NFert_GY_Ri_masked)/50)

[NFert_DM_sw_rate, NFert_DM_sw_pvalue] = find_trend_pval((CTRL_DM_SW_masked - NFert_DM_SW_masked)/2)
[NFert_DM_rice_rate, NFert_DM_rice_pvalue] = find_trend_pval((CTRL_DM_Ri_masked - NFert_DM_Ri_masked)/2)

[Irrig_LAI_sw_rate, Irrig_LAI_sw_pvalue] = find_trend_pval(CTRL_LAI_SW_masked - Irrig_LAI_SW_masked)
[Irrig_LAI_rice_rate, Irrig_LAI_rice_pvalue] = find_trend_pval(CTRL_LAI_Ri_masked - Irrig_LAI_Ri_masked)

[Irrig_GY_sw_rate, Irrig_GY_sw_pvalue] = find_trend_pval((CTRL_GY_SW_masked - Irrig_GY_SW_masked)/50)
[Irrig_GY_rice_rate, Irrig_GY_rice_pvalue] = find_trend_pval((CTRL_GY_Ri_masked - Irrig_GY_Ri_masked)/50)

[Irrig_DM_sw_rate, Irrig_DM_sw_pvalue] = find_trend_pval((CTRL_DM_SW_masked - Irrig_DM_SW_masked)/2)
[Irrig_DM_rice_rate, Irrig_DM_rice_pvalue] = find_trend_pval((CTRL_DM_Ri_masked - Irrig_DM_Ri_masked)/2)
#%% Plotting trend and p value for wheat
lonx, latx = np.meshgrid(CTRL_LAI_SW.lon, CTRL_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = CTRL_LAI_sw_rate
X_pvalue = np.ma.masked_array(CTRL_LAI_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = CTRL_GY_sw_rate
X_pvalue = np.ma.masked_array(CTRL_GY_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.15
vmx = 0.15
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = CTRL_DM_sw_rate
X_pvalue = np.ma.masked_array(CTRL_DM_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -15.0
vmx = 15.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for rice
lonx, latx = np.meshgrid(CTRL_LAI_Ri.lon, CTRL_LAI_Ri.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = CTRL_LAI_rice_rate
X_pvalue = np.ma.masked_array(CTRL_LAI_rice_pvalue,mask=Rice_mask[44,:,:])

vmn = -0.15
vmx = 0.15
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)
# norm = mpl.colors.Normalize(vmin=vmn,vmax=vmx)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = CTRL_GY_rice_rate
X_pvalue = np.ma.masked_array(CTRL_GY_rice_pvalue,mask=Rice_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
# norm = mpl.colors.BoundaryNorm(bounds, Num+2)
norm = mpl.colors.mpl.colors.BoundaryNorm(bounds,Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = CTRL_DM_rice_rate
X_pvalue = np.ma.masked_array(CTRL_DM_rice_pvalue,mask=Rice_mask[44,:,:])
X_rate[X_rate>40]=np.nan 

vmn = -20.0
vmx = 20.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2, Num+1)#.with_extremes(under=cmaplist_2[0],over=cmaplist_2[-1])
bounds = np.linspace(vmn, vmx, Num+1)
# norm = MidpointNormalize(midpoint=0)
norm = mpl.colors.mpl.colors.BoundaryNorm(bounds,cmap.N+1, extend='both')

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(CTRL_LAI_SW.lon,CTRL_LAI_SW.lat,X_rate,levels=bounds,norm=norm,
                 cmap=cmap)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

# cbar = fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),shrink=0.5, extend='over',spacing='proportional')
cbar = fig1.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),shrink=0.5, extend='both',spacing='proportional')
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))

#%% Preparing data for plotting
CTRL_LAI_Ri_spatial_mean = np.nanmean(np.nanmean(CTRL_LAI_Ri_masked,axis=1),axis=1)
CTRL_LAI_SW_spatial_mean = np.nanmean(np.nanmean(CTRL_LAI_SW_masked,axis=1),axis=1)
CTRL_GY_Ri_spatial_mean = np.nanmean(np.nanmean(CTRL_GY_Ri_masked/50,axis=1),axis=1)
CTRL_GY_SW_spatial_mean = np.nanmean(np.nanmean(CTRL_GY_SW_masked/50,axis=1),axis=1)
CTRL_DM_Ri_spatial_mean = np.nanmean(np.nanmean(CTRL_DM_Ri_masked*0.5,axis=1),axis=1)
CTRL_DM_SW_spatial_mean = np.nanmean(np.nanmean(CTRL_DM_SW_masked*0.5,axis=1),axis=1)

CTRL_LAI_Ri_spatial_std = np.nanstd(np.nanstd(CTRL_LAI_Ri_masked,axis=1),axis=1)
CTRL_LAI_SW_spatial_std = np.nanstd(np.nanstd(CTRL_LAI_SW_masked,axis=1),axis=1)
CTRL_GY_Ri_spatial_std = np.nanstd(np.nanstd(CTRL_GY_Ri_masked/50,axis=1),axis=1)
CTRL_GY_SW_spatial_std = np.nanstd(np.nanstd(CTRL_GY_SW_masked/50,axis=1),axis=1)
CTRL_DM_Ri_spatial_std = np.nanstd(np.nanstd(CTRL_DM_Ri_masked*0.5,axis=1),axis=1)
CTRL_DM_SW_spatial_std = np.nanstd(np.nanstd(CTRL_DM_SW_masked*0.5,axis=1),axis=1)

Clim_LAI_Ri_spatial_mean = np.nanmean(np.nanmean(Clim_LAI_Ri_masked,axis=1),axis=1)
Clim_LAI_SW_spatial_mean = np.nanmean(np.nanmean(Clim_LAI_SW_masked,axis=1),axis=1)
Clim_GY_Ri_spatial_mean = np.nanmean(np.nanmean(Clim_GY_Ri_masked,axis=1),axis=1)
Clim_GY_SW_spatial_mean = np.nanmean(np.nanmean(Clim_GY_SW_masked,axis=1),axis=1)
Clim_DM_Ri_spatial_mean = np.nanmean(np.nanmean(Clim_DM_Ri_masked,axis=1),axis=1)
Clim_DM_SW_spatial_mean = np.nanmean(np.nanmean(Clim_DM_SW_masked,axis=1),axis=1)

CO2_LAI_Ri_spatial_mean = np.nanmean(np.nanmean(CO2_LAI_Ri_masked,axis=1),axis=1)
CO2_LAI_SW_spatial_mean = np.nanmean(np.nanmean(CO2_LAI_SW_masked,axis=1),axis=1)
CO2_GY_Ri_spatial_mean = np.nanmean(np.nanmean(CO2_GY_Ri_masked,axis=1),axis=1)
CO2_GY_SW_spatial_mean = np.nanmean(np.nanmean(CO2_GY_SW_masked,axis=1),axis=1)
CO2_DM_Ri_spatial_mean = np.nanmean(np.nanmean(CO2_DM_Ri_masked,axis=1),axis=1)
CO2_DM_SW_spatial_mean = np.nanmean(np.nanmean(CO2_DM_SW_masked,axis=1),axis=1)

NFert_LAI_Ri_spatial_mean = np.nanmean(np.nanmean(NFert_LAI_Ri_masked,axis=1),axis=1)
NFert_LAI_SW_spatial_mean = np.nanmean(np.nanmean(NFert_LAI_SW_masked,axis=1),axis=1)
NFert_GY_Ri_spatial_mean = np.nanmean(np.nanmean(NFert_GY_Ri_masked,axis=1),axis=1)
NFert_GY_SW_spatial_mean = np.nanmean(np.nanmean(NFert_GY_SW_masked,axis=1),axis=1)
NFert_DM_Ri_spatial_mean = np.nanmean(np.nanmean(NFert_DM_Ri_masked,axis=1),axis=1)
NFert_DM_SW_spatial_mean = np.nanmean(np.nanmean(NFert_DM_SW_masked,axis=1),axis=1)

Irrig_LAI_Ri_spatial_mean = np.nanmean(np.nanmean(Irrig_LAI_Ri_masked,axis=1),axis=1)
Irrig_LAI_SW_spatial_mean = np.nanmean(np.nanmean(Irrig_LAI_SW_masked,axis=1),axis=1)
Irrig_GY_Ri_spatial_mean = np.nanmean(np.nanmean(Irrig_GY_Ri_masked,axis=1),axis=1)
Irrig_GY_SW_spatial_mean = np.nanmean(np.nanmean(Irrig_GY_SW_masked,axis=1),axis=1)
Irrig_DM_Ri_spatial_mean = np.nanmean(np.nanmean(Irrig_DM_Ri_masked,axis=1),axis=1)
Irrig_DM_SW_spatial_mean = np.nanmean(np.nanmean(Irrig_DM_SW_masked,axis=1),axis=1)
#%% Taking mean from 2-d array
CTRL_LAI_Ri_masked_2d = CTRL_LAI_Ri_masked.reshape(CTRL_LAI_Ri_masked.shape[0], CTRL_LAI_Ri_masked.shape[1]*CTRL_LAI_Ri_masked.shape[2])
CTRL_LAI_Ri_spatial_mean = np.nanmean(CTRL_LAI_Ri_masked_2d,axis=1)
CTRL_LAI_SW_masked_2d = CTRL_LAI_SW_masked.reshape(CTRL_LAI_SW_masked.shape[0], CTRL_LAI_SW_masked.shape[1]*CTRL_LAI_SW_masked.shape[2])
CTRL_LAI_SW_spatial_mean = np.nanmean(CTRL_LAI_SW_masked_2d,axis=1)
CTRL_GY_Ri_masked_2d = CTRL_GY_Ri_masked.reshape(CTRL_GY_Ri_masked.shape[0], CTRL_GY_Ri_masked.shape[1]*CTRL_GY_Ri_masked.shape[2])
CTRL_GY_Ri_spatial_mean = np.nanmean(CTRL_GY_Ri_masked_2d/50,axis=1)
CTRL_GY_SW_masked_2d = CTRL_GY_SW_masked.reshape(CTRL_GY_SW_masked.shape[0], CTRL_GY_SW_masked.shape[1]*CTRL_GY_SW_masked.shape[2])
CTRL_GY_SW_spatial_mean = np.nanmean(CTRL_GY_SW_masked_2d/50,axis=1)
CTRL_DM_Ri_masked_2d = CTRL_DM_Ri_masked.reshape(CTRL_DM_Ri_masked.shape[0], CTRL_DM_Ri_masked.shape[1]*CTRL_DM_Ri_masked.shape[2])
CTRL_DM_Ri_spatial_mean = np.nanmean(CTRL_DM_Ri_masked_2d*0.5,axis=1)
CTRL_DM_SW_masked_2d = CTRL_DM_SW_masked.reshape(CTRL_DM_SW_masked.shape[0], CTRL_DM_SW_masked.shape[1]*CTRL_DM_SW_masked.shape[2])
CTRL_DM_SW_spatial_mean = np.nanmean(CTRL_DM_SW_masked_2d*0.5,axis=1)

CTRL_LAI_Ri_masked_2d = CTRL_LAI_Ri_masked.reshape(CTRL_LAI_Ri_masked.shape[0], CTRL_LAI_Ri_masked.shape[1]*CTRL_LAI_Ri_masked.shape[2])
CTRL_LAI_Ri_spatial_std = np.nanstd(CTRL_LAI_Ri_masked_2d,axis=1)
CTRL_LAI_SW_masked_2d = CTRL_LAI_SW_masked.reshape(CTRL_LAI_SW_masked.shape[0], CTRL_LAI_SW_masked.shape[1]*CTRL_LAI_SW_masked.shape[2])
CTRL_LAI_SW_spatial_std = np.nanstd(CTRL_LAI_SW_masked_2d,axis=1)
CTRL_GY_Ri_masked_2d = CTRL_GY_Ri_masked.reshape(CTRL_GY_Ri_masked.shape[0], CTRL_GY_Ri_masked.shape[1]*CTRL_GY_Ri_masked.shape[2])
CTRL_GY_Ri_spatial_std = np.nanstd(CTRL_GY_Ri_masked_2d/50,axis=1)
CTRL_GY_SW_masked_2d = CTRL_GY_SW_masked.reshape(CTRL_GY_SW_masked.shape[0], CTRL_GY_SW_masked.shape[1]*CTRL_GY_SW_masked.shape[2])
CTRL_GY_SW_spatial_std = np.nanstd(CTRL_GY_SW_masked_2d/50,axis=1)
CTRL_DM_Ri_masked_2d = CTRL_DM_Ri_masked.reshape(CTRL_DM_Ri_masked.shape[0], CTRL_DM_Ri_masked.shape[1]*CTRL_DM_Ri_masked.shape[2])
CTRL_DM_Ri_spatial_std = np.nanstd(CTRL_DM_Ri_masked_2d*0.25,axis=1)
CTRL_DM_SW_masked_2d = CTRL_DM_SW_masked.reshape(CTRL_DM_SW_masked.shape[0], CTRL_DM_SW_masked.shape[1]*CTRL_DM_SW_masked.shape[2])
CTRL_DM_SW_spatial_std = np.nanstd(CTRL_DM_SW_masked_2d*0.25,axis=1)
#%% Impact of each experiment 
############################### LAI ##################################
Impact_LAI_Ri_Clim = CTRL_LAI_Ri_spatial_mean-Clim_LAI_Ri_spatial_mean
Impact_LAI_SW_Clim = CTRL_LAI_SW_spatial_mean-Clim_LAI_SW_spatial_mean

Impact_LAI_Ri_CO2 = CTRL_LAI_Ri_spatial_mean-CO2_LAI_Ri_spatial_mean
Impact_LAI_SW_CO2 = CTRL_LAI_SW_spatial_mean-CO2_LAI_SW_spatial_mean

Impact_LAI_Ri_NFert = CTRL_LAI_Ri_spatial_mean-NFert_LAI_Ri_spatial_mean
Impact_LAI_SW_NFert = CTRL_LAI_SW_spatial_mean-NFert_LAI_SW_spatial_mean

Impact_LAI_Ri_Irrig = CTRL_LAI_Ri_spatial_mean-Irrig_LAI_Ri_spatial_mean
Impact_LAI_SW_Irrig = CTRL_LAI_SW_spatial_mean-Irrig_LAI_SW_spatial_mean
############################## GY ######################################
Impact_GY_Ri_Clim = (CTRL_GY_Ri_spatial_mean-Clim_GY_Ri_spatial_mean)/50
Impact_GY_SW_Clim = (CTRL_GY_SW_spatial_mean-Clim_GY_SW_spatial_mean)/50

Impact_GY_Ri_CO2 = (CTRL_GY_Ri_spatial_mean-CO2_GY_Ri_spatial_mean)/50
Impact_GY_SW_CO2 = (CTRL_GY_SW_spatial_mean-CO2_GY_SW_spatial_mean)/50

Impact_GY_Ri_NFert = (CTRL_GY_Ri_spatial_mean-NFert_GY_Ri_spatial_mean)/50
Impact_GY_SW_NFert = (CTRL_GY_SW_spatial_mean-NFert_GY_SW_spatial_mean)/50

Impact_GY_Ri_Irrig = (CTRL_GY_Ri_spatial_mean-Irrig_GY_Ri_spatial_mean)/50
Impact_GY_SW_Irrig = (CTRL_GY_SW_spatial_mean-Irrig_GY_SW_spatial_mean)/50
############################## DM ######################################
Impact_DM_Ri_Clim = (CTRL_DM_Ri_spatial_mean-Clim_DM_Ri_spatial_mean)*0.5
Impact_DM_SW_Clim = (CTRL_DM_SW_spatial_mean-Clim_DM_SW_spatial_mean)*0.5

Impact_DM_Ri_CO2 = (CTRL_DM_Ri_spatial_mean-CO2_DM_Ri_spatial_mean)*0.5
Impact_DM_SW_CO2 = (CTRL_DM_SW_spatial_mean-CO2_DM_SW_spatial_mean)*0.5

Impact_DM_Ri_NFert = (CTRL_DM_Ri_spatial_mean-NFert_DM_Ri_spatial_mean)*0.5
Impact_DM_SW_NFert = (CTRL_DM_SW_spatial_mean-NFert_DM_SW_spatial_mean)*0.5

Impact_DM_Ri_Irrig = (CTRL_DM_Ri_spatial_mean-Irrig_DM_Ri_spatial_mean)*0.5
Impact_DM_SW_Irrig = (CTRL_DM_SW_spatial_mean-Irrig_DM_SW_spatial_mean)*0.5
#%% Impact of each experiment 
############################### LAI ##################################
Impact_LAI_Clim_1 = np.nanmean(Impact_LAI_Ri_Clim+Impact_LAI_SW_Clim)

Impact_LAI_CO2_1 = np.nanmean(Impact_LAI_Ri_CO2+Impact_LAI_SW_CO2)

Impact_LAI_NFert_1 = np.nanmean(Impact_LAI_Ri_NFert+Impact_LAI_SW_NFert)

Impact_LAI_Irrig_1 = np.nanmean(Impact_LAI_Ri_Irrig+Impact_LAI_SW_Irrig)
############################## GY ######################################
Impact_GY_Clim_1 = np.nanmean(Impact_GY_Ri_Clim+Impact_GY_SW_Clim)

Impact_GY_CO2_1 = np.nanmean(Impact_GY_Ri_CO2+Impact_GY_SW_CO2)

Impact_GY_NFert_1 = np.nanmean(Impact_GY_Ri_NFert+Impact_GY_SW_NFert)

Impact_GY_Irrig_1 = np.nanmean(Impact_GY_Ri_Irrig+Impact_GY_SW_Irrig)
############################## DM ######################################
Impact_DM_Clim_1 = np.nanmean(Impact_DM_Ri_Clim+Impact_DM_SW_Clim)

Impact_DM_CO2_1 = np.nanmean(Impact_DM_Ri_CO2+Impact_DM_SW_CO2)

Impact_DM_NFert_1 = np.nanmean(Impact_DM_Ri_NFert+Impact_DM_SW_NFert)

Impact_DM_Irrig_1 = np.nanmean(Impact_DM_Ri_Irrig+Impact_DM_SW_Irrig)
#%% Impact of each experiment Normalised
# scaler = MinMaxScaler()
############################### LAI ##################################
Impact_LAI_Clim = np.nanmean(preprocessing.normalize([Impact_LAI_Ri_Clim+Impact_LAI_SW_Clim]))

Impact_LAI_CO2 = np.nanmean(preprocessing.normalize([Impact_LAI_Ri_CO2+Impact_LAI_SW_CO2]))

Impact_LAI_NFert = np.nanmean(preprocessing.normalize([Impact_LAI_Ri_NFert+Impact_LAI_SW_NFert]))

Impact_LAI_Irrig = np.nanmean(preprocessing.normalize([Impact_LAI_Ri_Irrig+Impact_LAI_SW_Irrig]))
############################## GY ######################################
Impact_GY_Clim = np.nanmean(preprocessing.normalize([Impact_GY_Ri_Clim+Impact_GY_SW_Clim]))

Impact_GY_CO2 = np.nanmean(preprocessing.normalize([Impact_GY_Ri_CO2+Impact_GY_SW_CO2]))

Impact_GY_NFert = np.nanmean(preprocessing.normalize([Impact_GY_Ri_NFert+Impact_GY_SW_NFert]))

Impact_GY_Irrig = np.nanmean(preprocessing.normalize([Impact_GY_Ri_Irrig+Impact_GY_SW_Irrig]))
############################## DM ######################################
Impact_DM_Clim = np.nanmean(preprocessing.normalize([Impact_DM_Ri_Clim+Impact_DM_SW_Clim]))

Impact_DM_CO2 = np.nanmean(preprocessing.normalize([Impact_DM_Ri_CO2+Impact_DM_SW_CO2]))

Impact_DM_NFert = np.nanmean(preprocessing.normalize([Impact_DM_Ri_NFert+Impact_DM_SW_NFert]))

Impact_DM_Irrig = np.nanmean(preprocessing.normalize([Impact_DM_Ri_Irrig+Impact_DM_SW_Irrig]))

Impact_Crop_Phenology = [Impact_LAI_Clim,Impact_LAI_CO2,Impact_LAI_NFert,Impact_LAI_Irrig,
                         Impact_GY_Clim,Impact_GY_CO2,Impact_GY_NFert,Impact_GY_Irrig,
                         Impact_DM_Clim,Impact_DM_CO2,Impact_DM_NFert,Impact_DM_Irrig]
#%% Decadal means of Impact
# ################################### Wheat ################################
# Impact_LAI_SW_Clim_1970s = np.nanmean(Impact_LAI_SW_Clim[0:10])
# Impact_LAI_SW_Clim_1980s = np.nanmean(Impact_LAI_SW_Clim[10:20])
# Impact_LAI_SW_Clim_1990s = np.nanmean(Impact_LAI_SW_Clim[20:30])
# Impact_LAI_SW_Clim_2000s = np.nanmean(Impact_LAI_SW_Clim[30:40])
# Impact_LAI_SW_Clim_2010s = np.nanmean(Impact_LAI_SW_Clim[40:])
#%% write data to a file for creating boxplots
variables = ['LAI','GY','DM']
Decades = ['1970s','1980s','1990s','2000s','2010s']
Impact = ['Clim','CO2','NFert','Irrig']

# all_variables = dir()
impact_data = pd.DataFrame({'Data':[],'Decade':[],'Impact':[],'Variable':[]})
for i_impact in Impact:
    for i_var in variables:
        impact_name = 'Impact_'+i_var+'_SW_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'Data':data,'Decade':['1970s'],'Impact':[i_impact],'Variable':[i_var]})
                impact_data = pd.concat([impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'Data':data,'Decade':['1980s'],'Impact':[i_impact],'Variable':[i_var]})
                impact_data = pd.concat([impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'Data':data,'Decade':['1990s'],'Impact':[i_impact],'Variable':[i_var]})
                impact_data = pd.concat([impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'Data':data,'Decade':['2000s'],'Impact':[i_impact],'Variable':[i_var]})
                impact_data = pd.concat([impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'Data':data,'Decade':['2010s'],'Impact':[i_impact],'Variable':[i_var]})
                impact_data = pd.concat([impact_data,df_new],ignore_index=True)
#%% write data to a file for creating boxplots
variables = ['LAI','GY','DM']
Decades = ['1970s','1980s','1990s','2000s','2010s']
Impact = ['Clim','CO2','NFert','Irrig']

################################ LAI ###########################################
LAI_impact_data = pd.DataFrame({'LAI':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'LAI'
        impact_name = 'Impact_'+i_var+'_SW_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'LAI':data,'Decade':['1970s'],'Impact of':[i_impact]})
                LAI_impact_data = pd.concat([LAI_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'LAI':data,'Decade':['1980s'],'Impact of':[i_impact]})
                LAI_impact_data = pd.concat([LAI_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'LAI':data,'Decade':['1990s'],'Impact of':[i_impact]})
                LAI_impact_data = pd.concat([LAI_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'LAI':data,'Decade':['2000s'],'Impact of':[i_impact]})
                LAI_impact_data = pd.concat([LAI_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'LAI':data,'Decade':['2010s'],'Impact of':[i_impact]})
                LAI_impact_data = pd.concat([LAI_impact_data,df_new],ignore_index=True)

LAI_RI_impact_data = pd.DataFrame({'LAI':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'LAI'
        impact_name = 'Impact_'+i_var+'_Ri_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'LAI':data,'Decade':['1970s'],'Impact of':[i_impact]})
                LAI_RI_impact_data = pd.concat([LAI_RI_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'LAI':data,'Decade':['1980s'],'Impact of':[i_impact]})
                LAI_RI_impact_data = pd.concat([LAI_RI_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'LAI':data,'Decade':['1990s'],'Impact of':[i_impact]})
                LAI_RI_impact_data = pd.concat([LAI_RI_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'LAI':data,'Decade':['2000s'],'Impact of':[i_impact]})
                LAI_RI_impact_data = pd.concat([LAI_RI_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'LAI':data,'Decade':['2010s'],'Impact of':[i_impact]})
                LAI_RI_impact_data = pd.concat([LAI_RI_impact_data,df_new],ignore_index=True)
################################ Yield ###########################################               
GY_impact_data = pd.DataFrame({'Yield':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'GY'
        impact_name = 'Impact_'+i_var+'_SW_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'Yield':data,'Decade':['1970s'],'Impact of':[i_impact]})
                GY_impact_data = pd.concat([GY_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'Yield':data,'Decade':['1980s'],'Impact of':[i_impact]})
                GY_impact_data = pd.concat([GY_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'Yield':data,'Decade':['1990s'],'Impact of':[i_impact]})
                GY_impact_data = pd.concat([GY_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'Yield':data,'Decade':['2000s'],'Impact of':[i_impact]})
                GY_impact_data = pd.concat([GY_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'Yield':data,'Decade':['2010s'],'Impact of':[i_impact]})
                GY_impact_data = pd.concat([GY_impact_data,df_new],ignore_index=True)

GY_RI_impact_data = pd.DataFrame({'Yield':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'GY'
        impact_name = 'Impact_'+i_var+'_Ri_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'Yield':data,'Decade':['1970s'],'Impact of':[i_impact]})
                GY_RI_impact_data = pd.concat([GY_RI_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'Yield':data,'Decade':['1980s'],'Impact of':[i_impact]})
                GY_RI_impact_data = pd.concat([GY_RI_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'Yield':data,'Decade':['1990s'],'Impact of':[i_impact]})
                GY_RI_impact_data = pd.concat([GY_RI_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'Yield':data,'Decade':['2000s'],'Impact of':[i_impact]})
                GY_RI_impact_data = pd.concat([GY_RI_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'Yield':data,'Decade':['2010s'],'Impact of':[i_impact]})
                GY_RI_impact_data = pd.concat([GY_RI_impact_data,df_new],ignore_index=True)
################################ Dry Matter ###########################################
DM_impact_data = pd.DataFrame({'Dry Matter':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'DM'
        impact_name = 'Impact_'+i_var+'_SW_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1970s'],'Impact of':[i_impact]})
                DM_impact_data = pd.concat([DM_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1980s'],'Impact of':[i_impact]})
                DM_impact_data = pd.concat([DM_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1990s'],'Impact of':[i_impact]})
                DM_impact_data = pd.concat([DM_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['2000s'],'Impact of':[i_impact]})
                DM_impact_data = pd.concat([DM_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['2010s'],'Impact of':[i_impact]})
                DM_impact_data = pd.concat([DM_impact_data,df_new],ignore_index=True)

DM_RI_impact_data = pd.DataFrame({'Dry Matter':[],'Decade':[],'Impact of':[]})
for i_impact in Impact:
    # for i_var in variables:
        i_var = 'DM'
        impact_name = 'Impact_'+i_var+'_Ri_'+i_impact
        data_new = np.array(eval(impact_name))
        for i_year in range(45):
            data = [data_new[i_year]]
            if i_year<10:                
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1970s'],'Impact of':[i_impact]})
                DM_RI_impact_data = pd.concat([DM_RI_impact_data,df_new],ignore_index=True)
            elif i_year>9 and i_year<20:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1980s'],'Impact of':[i_impact]})
                DM_RI_impact_data = pd.concat([DM_RI_impact_data,df_new],ignore_index=True)
            elif i_year>19 and i_year<30:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['1990s'],'Impact of':[i_impact]})
                DM_RI_impact_data = pd.concat([DM_RI_impact_data,df_new],ignore_index=True)
            elif i_year>29 and i_year<40: 
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['2000s'],'Impact of':[i_impact]})
                DM_RI_impact_data = pd.concat([DM_RI_impact_data,df_new],ignore_index=True)
            else:
                df_new = pd.DataFrame({'Dry Matter':data,'Decade':['2010s'],'Impact of':[i_impact]})
                DM_RI_impact_data = pd.concat([DM_RI_impact_data,df_new],ignore_index=True)
#%% Plotting CTRL Timeseries LAI, GY, and DM

colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]


fig1, axes = plt.subplots(nrows=3, ncols=1, dpi=600,sharex=True,figsize=(8,6),layout='constrained')
# fig1.subplots_adjust(top=0.9)
# fig1.suptitle('Annual mean phenology over India (1970 to 2014)', y=1.0)
# fig1.tight_layout(pad=1.0)

axes[0].plot(CTRL_LAI_Ri.year,CTRL_LAI_Ri_spatial_mean,label = 'Rice', color=colors[0])
axes[0].plot(CTRL_LAI_SW.year,CTRL_LAI_SW_spatial_mean,label = 'Wheat', color=colors[3])
axes[0].set(title='(a) LAI',ylabel='m\u00b2/m\u00b2')

axes[1].plot(CTRL_GY_Ri.year,CTRL_GY_Ri_spatial_mean/50, label = 'Rice', color=colors[0])
axes[1].plot(CTRL_GY_SW.year,CTRL_GY_SW_spatial_mean/50, label = 'Wheat', color=colors[3])
axes[1].set(title='(b) Yield',ylabel='t/ha')

axes[2].plot(CTRL_DM_Ri.year,CTRL_DM_Ri_spatial_mean*0.5, label = 'Rice', color=colors[0])
axes[2].plot(CTRL_DM_SW.year,CTRL_DM_SW_spatial_mean*0.5, label = 'Wheat', color=colors[3])
axes[2].set(title='(c) Dry matter',ylabel='g/m\u00b2',xlabel='Year')
axes[2].set_xlim([1970,2015])

axes[0].legend(loc='upper left', fontsize="10")

#fig1.savefig(saving_fig_dir+'Annual mean crop phenology over India (1970 to 2014)_9Apr.png', 
#              dpi=600, bbox_inches="tight", pad_inches=0.5)
#%% Plotting CTRL Timeseries LAI, GY, and DM (mean and std)

colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]
x_new = np.arange(0,45)

fig1, axes = plt.subplots(nrows=3, ncols=2, dpi=600,sharex=True,figsize=(8,6),layout='constrained')
# fig1.subplots_adjust(top=0.9)
# fig1.suptitle('Annual mean phenology over India (1970 to 2014)', y=1.0)
# fig1.tight_layout(pad=1.0)

std1 = CTRL_LAI_SW_spatial_mean - CTRL_LAI_SW_spatial_std
std2 = CTRL_LAI_SW_spatial_mean + CTRL_LAI_SW_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test(CTRL_LAI_SW_spatial_mean.data,alpha=0.01)
y_new = x_new*slope+intercept
axes[0,0].plot(CTRL_LAI_SW.year,CTRL_LAI_SW_spatial_mean, color=colors[2],linewidth=3,alpha=0.8)
axes[0,0].fill_between(CTRL_LAI_SW.year,std1,std2,facecolor=colors[2],alpha=0.25)
axes[0,0].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[0,0].set_ylabel('max.LAI \n (m\u00b2/m\u00b2)')
axes[0,0].set_title('Wheat')
axes[0,0].set_ylim([0,7])
axes[0,0].text(0.02, 0.85, '(a)', fontsize=14,transform=axes[0,0].transAxes)
axes[0,0].text(0.47, 0.90, 'slope='+'{:.3f}'.format(slope)+' m\u00b2/m\u00b2/year', fontsize=10,transform=axes[0,0].transAxes)
axes[0,0].text(0.47, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[0,0].transAxes)

std1 = CTRL_LAI_Ri_spatial_mean - CTRL_LAI_Ri_spatial_std
std2 = CTRL_LAI_Ri_spatial_mean + CTRL_LAI_Ri_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test(CTRL_LAI_Ri_spatial_mean.data,alpha=0.01)
y_new = x_new*slope+intercept
axes[0,1].plot(CTRL_LAI_Ri.year,CTRL_LAI_Ri_spatial_mean, color=colors[0],linewidth=3,alpha=0.8)
axes[0,1].fill_between(CTRL_LAI_Ri.year,std1,std2,facecolor=colors[0],alpha=0.25)
axes[0,1].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[0,1].set_title('Rice')
axes[0,1].set_ylim([0,7])
axes[0,1].set_yticklabels([])
axes[0,1].text(0.02, 0.85, '(d)', fontsize=14,transform=axes[0,1].transAxes)
axes[0,1].text(0.47, 0.90, 'slope='+'{:.3f}'.format(slope)+' m\u00b2/m\u00b2/year', fontsize=10,transform=axes[0,1].transAxes)
axes[0,1].text(0.47, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[0,1].transAxes)

std1 = CTRL_GY_SW_spatial_mean - CTRL_GY_SW_spatial_std
std2 = CTRL_GY_SW_spatial_mean + CTRL_GY_SW_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((CTRL_GY_SW_spatial_mean.data),alpha=0.01)
y_new = x_new*slope+intercept
axes[1,0].plot(CTRL_LAI_SW.year,CTRL_GY_SW_spatial_mean, color=colors[2],linewidth=3,alpha=0.8)
axes[1,0].fill_between(CTRL_LAI_SW.year,std1,std2,facecolor=colors[2],alpha=0.25)
axes[1,0].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[1,0].set_ylabel('Yield \n (t/ha)')
axes[1,0].set_ylim([0,6])
axes[1,0].text(0.02, 0.85, '(b)', fontsize=14,transform=axes[1,0].transAxes)
axes[1,0].text(0.47, 0.90, 'slope='+'{:.3f}'.format(slope)+' t/ha/year', fontsize=10,transform=axes[1,0].transAxes)
axes[1,0].text(0.47, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[1,0].transAxes)

std1 = CTRL_GY_Ri_spatial_mean - CTRL_GY_Ri_spatial_std
std2 = CTRL_GY_Ri_spatial_mean + CTRL_GY_Ri_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((CTRL_GY_Ri_spatial_mean.data),alpha=0.01)
y_new = x_new*slope+intercept
axes[1,1].plot(CTRL_LAI_Ri.year,CTRL_GY_Ri_spatial_mean, color=colors[0],linewidth=3,alpha=0.8)
axes[1,1].fill_between(CTRL_LAI_Ri.year,std1,std2,facecolor=colors[0],alpha=0.25)
axes[1,1].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[1,1].set_ylim([0,6])
axes[1,1].set_yticklabels([])
axes[1,1].text(0.02, 0.85, '(e)', fontsize=14,transform=axes[1,1].transAxes)
axes[1,1].text(0.47, 0.90, 'slope='+'{:.3f}'.format(slope)+' t/ha/year', fontsize=10,transform=axes[1,1].transAxes)
axes[1,1].text(0.47, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[1,1].transAxes)

std1 = CTRL_DM_SW_spatial_mean - CTRL_DM_SW_spatial_std
std2 = CTRL_DM_SW_spatial_mean + CTRL_DM_SW_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((CTRL_DM_SW_spatial_mean.data)/100,alpha=0.01)
y_new = x_new*slope+intercept
axes[2,0].plot(CTRL_LAI_SW.year,CTRL_DM_SW_spatial_mean/100, color=colors[2],linewidth=3,alpha=0.8)
axes[2,0].fill_between(CTRL_LAI_SW.year,std1/100,std2/100,facecolor=colors[2],alpha=0.25)
axes[2,0].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[2,0].set_ylabel('Dry Matter \n (100g/m\u00b2)')
axes[2,0].set_ylim([0,9])
axes[2,0].set_yticks([0,3,6,9])
axes[2,0].text(0.02, 0.85, '(c)', fontsize=14,transform=axes[2,0].transAxes)
axes[2,0].text(0.44, 0.90, 'slope='+'{:.3f}'.format(slope)+' 100g/m\u00b2/year', fontsize=10,transform=axes[2,0].transAxes)
axes[2,0].text(0.44, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[2,0].transAxes)

std1 = CTRL_DM_Ri_spatial_mean - CTRL_DM_Ri_spatial_std
std2 = CTRL_DM_Ri_spatial_mean + CTRL_DM_Ri_spatial_std
[trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((CTRL_DM_Ri_spatial_mean.data+250)/100,alpha=0.01)
y_new = x_new*slope+intercept
axes[2,1].plot(CTRL_LAI_Ri.year,(CTRL_DM_Ri_spatial_mean+250)/100, color=colors[0],linewidth=3,alpha=0.8)
axes[2,1].fill_between(CTRL_LAI_Ri.year,(std1+250)/100,(std2+250)/100,facecolor=colors[0],alpha=0.25)
axes[2,1].plot(CTRL_LAI_SW.year,y_new,linestyle="dashdot",color='k',linewidth=1)
axes[2,1].set_ylim([0,9])
axes[2,1].set_yticks([0,3,6,9])
axes[2,1].set_yticklabels([])
axes[2,1].text(0.02, 0.85, '(f)', fontsize=14,transform=axes[2,1].transAxes)
axes[2,1].text(0.44, 0.90, 'slope='+'{:.3f}'.format(slope)+' 100g/m\u00b2/year', fontsize=10,transform=axes[2,1].transAxes)
axes[2,1].text(0.44, 0.80, 'p='+'{:.3f}'.format(p_value), fontsize=10,transform=axes[2,1].transAxes)

axes[2,1].set_xlim([1970,2014])
# axes[0].legend(loc='upper left', fontsize="10")

# fig1.savefig(saving_fig_dir+'Annual mean crop phenology over India (1970 to 2014)_9Apr.png', 
#               dpi=600, bbox_inches="tight", pad_inches=0.5)
#%% Plotting Experiments impact on LAI, GY, and DM
colors = [cm.vik(160), cm.vik(240), cm.batlowK(120), cm.vik(60)]
fig2, axes2 = plt.subplots(nrows=3, ncols=2, dpi=600,sharex=True)
fig2.subplots_adjust(top=0.9)
fig2.suptitle('Impact on phenology over India (1970 to 2014)', y=1.0)
fig2.tight_layout(pad=1.0)

axes2[0,0].plot(CTRL_LAI_SW.year,CTRL_LAI_SW_spatial_mean,label = 'CTRL', color='k')
axes2[0,0].plot(CO2_LAI_SW.year,CO2_LAI_SW_spatial_mean,label = 'S_CO2', color=colors[0])
axes2[0,0].plot(Clim_LAI_SW.year,Clim_LAI_SW_spatial_mean,label = 'S_Clim', color=colors[1])
axes2[0,0].plot(NFert_LAI_SW.year,NFert_LAI_SW_spatial_mean,label = 'S_NFert', color=colors[2])
axes2[0,0].plot(Irrig_LAI_SW.year,Irrig_LAI_SW_spatial_mean,label = 'S_Irrig', color=colors[3])
axes2[0,0].set(title='Wheat',ylabel='LAI \n (m\u00b2/m\u00b2)')

axes2[0,1].plot(CTRL_LAI_Ri.year,CTRL_LAI_Ri_spatial_mean,label = 'CTRL', color='k')
axes2[0,1].plot(CO2_LAI_Ri.year,CO2_LAI_Ri_spatial_mean,label = 'S_CO2', color=colors[0])
axes2[0,1].plot(Clim_LAI_Ri.year,Clim_LAI_Ri_spatial_mean,label = 'S_Clim', color=colors[1])
axes2[0,1].plot(NFert_LAI_Ri.year,NFert_LAI_Ri_spatial_mean,label = 'S_NFert', color=colors[2])
axes2[0,1].plot(Irrig_LAI_Ri.year,Irrig_LAI_Ri_spatial_mean,label = 'S_Irrig', color=colors[3])
axes2[0,1].set(title='Rice',ylabel='',xlabel='')

axes2[1,0].plot(CTRL_GY_SW.year,CTRL_GY_SW_spatial_mean/100, label = 'CTRL', color='k')
axes2[1,0].plot(CO2_GY_SW.year,CO2_GY_SW_spatial_mean/100, label = 'S_CO2', color=colors[0])
axes2[1,0].plot(Clim_GY_SW.year,Clim_GY_SW_spatial_mean/100, label = 'S_Clim', color=colors[1])
axes2[1,0].plot(NFert_GY_SW.year,NFert_GY_SW_spatial_mean/100, label = 'S_NFert', color=colors[2])
axes2[1,0].plot(Irrig_GY_SW.year,Irrig_GY_SW_spatial_mean/100, label = 'S_Irrig', color=colors[3])
axes2[1,0].set(title='',ylabel='Yield \n (t/ha)',xlabel='')

axes2[1,1].plot(CTRL_GY_Ri.year,CTRL_GY_Ri_spatial_mean/100, label = 'CTRL', color='k')
axes2[1,1].plot(CO2_GY_Ri.year,CO2_GY_Ri_spatial_mean/100, label = 'S_CO2', color=colors[0])
axes2[1,1].plot(Clim_GY_Ri.year,Clim_GY_Ri_spatial_mean/100, label = 'S_Clim', color=colors[1])
axes2[1,1].plot(NFert_GY_Ri.year,NFert_GY_Ri_spatial_mean/100, label = 'S_NFert', color=colors[2])
axes2[1,1].plot(Irrig_GY_Ri.year,Irrig_GY_Ri_spatial_mean/100, label = 'S_Irrig', color=colors[3])
axes2[1,1].set(title='',ylabel='',xlabel='')

axes2[2,0].plot(CTRL_DM_SW.year,CTRL_DM_SW_spatial_mean*0.5, label = 'CTRL', color='k')
axes2[2,0].plot(CO2_DM_SW.year,CO2_DM_SW_spatial_mean*0.5, label = 'S_CO2', color=colors[0])
axes2[2,0].plot(Clim_DM_SW.year,Clim_DM_SW_spatial_mean*0.5, label = 'S_Clim', color=colors[1])
axes2[2,0].plot(NFert_DM_SW.year,NFert_DM_SW_spatial_mean*0.5, label = 'S_NFert', color=colors[2])
axes2[2,0].plot(Irrig_DM_SW.year,Irrig_DM_SW_spatial_mean*0.5, label = 'S_NFert', color=colors[3])
axes2[2,0].set(title='',ylabel='Dry matter \n (g/m\u00b2)',xlabel='Year')

axes2[2,1].plot(CTRL_DM_Ri.year,CTRL_DM_Ri_spatial_mean*0.5, label = 'CTRL', color='k')
axes2[2,1].plot(CO2_DM_Ri.year,CO2_DM_Ri_spatial_mean*0.5, label = 'S_CO2', color=colors[0])
axes2[2,1].plot(Clim_DM_Ri.year,Clim_DM_Ri_spatial_mean*0.5, label = 'S_Clim', color=colors[1])
axes2[2,1].plot(NFert_DM_Ri.year,NFert_DM_Ri_spatial_mean*0.5, label = 'S_NFert', color=colors[2])
axes2[2,1].plot(Irrig_DM_Ri.year,Irrig_DM_Ri_spatial_mean*0.5, label = 'S_NFert', color=colors[3])
axes2[2,1].set(title='',ylabel='',xlabel='Year')

axes2[1,1].legend(loc='upper left', fontsize="6",bbox_to_anchor=(1,1))
#%% Plotting box plots
saving_fig_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/Figures/'
clrs = [cm.batlow(40),cm.batlow(80),cm.batlow(120),cm.batlow(160),cm.batlow(200)]

fig3, axes3 = plt.subplots(nrows=3, ncols=2, dpi=600,sharex=True,figsize=(12,8),layout='constrained')
# fig3.subplots_adjust(top=0.9)
# fig3.suptitle('Impact on crop phenology over Indian crops', y=1.0)
# fig3.tight_layout(pad=1.0)

ax1=sns.boxplot(ax=axes3[0,0],x = LAI_impact_data['Impact of'],
            y = LAI_impact_data['LAI'],
            hue = LAI_impact_data['Decade'],legend=False,palette=clrs)
ax1.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

ax2=sns.boxplot(ax=axes3[0,1],x = LAI_RI_impact_data['Impact of'],
            y = LAI_RI_impact_data['LAI'],
            hue = LAI_RI_impact_data['Decade'],legend=False,palette=clrs)
ax2.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

ax3=sns.boxplot(ax=axes3[1,0],x = GY_impact_data['Impact of'],
            y = GY_impact_data['Yield'],
            hue = GY_impact_data['Decade'],legend=False,palette=clrs)
ax3.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

ax4=sns.boxplot(ax=axes3[1,1],x = GY_RI_impact_data['Impact of'],
            y = GY_RI_impact_data['Yield'],
            hue = GY_RI_impact_data['Decade'],palette=clrs)
ax4.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

ax5=sns.boxplot(ax=axes3[2,0],x = DM_impact_data['Impact of'],
            y = DM_impact_data['Dry Matter'],
            hue = DM_impact_data['Decade'],legend=False,palette=clrs)
ax5.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

ax6=sns.boxplot(ax=axes3[2,1],x = DM_RI_impact_data['Impact of'],
            y = DM_RI_impact_data['Dry Matter'],
            hue = DM_RI_impact_data['Decade'],legend=False,palette=clrs)
ax6.axhline(y=0,xmin=0,xmax=1,color='k',linewidth=0.5,alpha=0.5)

sns.move_legend(axes3[1,1], 'upper right',bbox_to_anchor=(1.3,1.0), frameon=False)

axes3[0,0].set(title='Wheat',ylabel='max.LAI \n (m\u00b2/m\u00b2)')
axes3[0,1].set(title='Rice',ylabel='')
axes3[1,0].set(title='',ylabel='Yield\n (t/ha)')
axes3[1,1].set(title='',ylabel='')
axes3[2,0].set(title='',ylabel='Dry Matter \n (g/m\u00b2)')
axes3[2,1].set(title='',ylabel='')
# axes3[0,0].legend(loc='upper left', fontsize="5",orientation='horizontal')
# fig3.savefig(saving_fig_dir+'Impact on crop phenology over Indian croplands_27Mar.png', 
#              dpi=600, bbox_inches="tight")
#%% Plotting trend and p value for wheat impact of clim
lonx, latx = np.meshgrid(Clim_LAI_SW.lon, Clim_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = Clim_LAI_sw_rate
X_pvalue = np.ma.masked_array(Clim_LAI_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = Clim_GY_sw_rate
X_pvalue = np.ma.masked_array(Clim_GY_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = Clim_DM_sw_rate
X_pvalue = np.ma.masked_array(Clim_DM_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -10.0
vmx = 10.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for wheat impact of CO2
lonx, latx = np.meshgrid(CO2_LAI_SW.lon, CO2_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = CO2_LAI_sw_rate
X_pvalue = np.ma.masked_array(CO2_LAI_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = CO2_GY_sw_rate
X_pvalue = np.ma.masked_array(CO2_GY_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = CO2_DM_sw_rate
X_pvalue = np.ma.masked_array(CO2_DM_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -10.0
vmx = 10.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for wheat impact of NFert
lonx, latx = np.meshgrid(NFert_LAI_SW.lon, NFert_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = NFert_LAI_sw_rate
X_pvalue = np.ma.masked_array(NFert_LAI_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = NFert_GY_sw_rate
X_pvalue = np.ma.masked_array(NFert_GY_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = NFert_DM_sw_rate
X_pvalue = np.ma.masked_array(NFert_DM_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -10.0
vmx = 10.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for wheat impact of Irrig
lonx, latx = np.meshgrid(Irrig_LAI_SW.lon, Irrig_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = Irrig_LAI_sw_rate
X_pvalue = np.ma.masked_array(Irrig_LAI_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = Irrig_GY_sw_rate
X_pvalue = np.ma.masked_array(Irrig_GY_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = Irrig_DM_sw_rate
X_pvalue = np.ma.masked_array(Irrig_DM_sw_pvalue,mask=SW_mask[44,:,:])

vmn = -10.0
vmx = 10.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for rice impact of clim
lonx, latx = np.meshgrid(Clim_LAI_SW.lon, Clim_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = Clim_LAI_rice_rate
X_pvalue = np.ma.masked_array(Clim_LAI_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = Clim_GY_rice_rate
X_pvalue = np.ma.masked_array(Clim_GY_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = Clim_DM_rice_rate
X_pvalue = np.ma.masked_array(Clim_DM_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -50.0
vmx = 50.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(Clim_LAI_SW.lon,Clim_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for rice impact of CO2
lonx, latx = np.meshgrid(CO2_LAI_SW.lon, CO2_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = CO2_LAI_rice_rate
X_pvalue = np.ma.masked_array(CO2_LAI_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = CO2_GY_rice_rate
X_pvalue = np.ma.masked_array(CO2_GY_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = CO2_DM_rice_rate
X_pvalue = np.ma.masked_array(CO2_DM_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -50.0
vmx = 50.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(CO2_LAI_SW.lon,CO2_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for rice impact of NFert
lonx, latx = np.meshgrid(NFert_LAI_SW.lon, NFert_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = NFert_LAI_rice_rate
X_pvalue = np.ma.masked_array(NFert_LAI_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = NFert_GY_rice_rate
X_pvalue = np.ma.masked_array(NFert_GY_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = NFert_DM_rice_rate
X_pvalue = np.ma.masked_array(NFert_DM_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -50.0
vmx = 50.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(NFert_LAI_SW.lon,NFert_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
#%% Plotting trend and p value for rice impact of Irrig
lonx, latx = np.meshgrid(Irrig_LAI_SW.lon, Irrig_LAI_SW.lat)
fig1, axes = plt.subplots(nrows=1, ncols=3, dpi=600,sharey=True,figsize=(15,7),layout='constrained')

############# max. LAI ##################
X_rate = Irrig_LAI_rice_rate
X_pvalue = np.ma.masked_array(Irrig_LAI_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.1
vmx = 0.1
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[0])
im1=axes[0].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_title('max.LAI')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[0],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('max. LAI (m\u00b2/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
########################### Yield ############################
X_rate = Irrig_GY_rice_rate
X_pvalue = np.ma.masked_array(Irrig_GY_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -0.2
vmx = 0.2
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[1])
im1=axes[1].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('')
# axes.set_title('\u0394max.LAI in wheat (m\u00b2/m\u00b2/year)')
axes[1].set_title('Yield')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Yield (t/ha/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))
##################### Dry Matter ###########################
X_rate = Irrig_DM_rice_rate
X_pvalue = np.ma.masked_array(Irrig_DM_rice_pvalue,mask=SW_mask[44,:,:])

vmn = -50.0
vmx = 50.0
cmap = cm.vik
Num = 20
# axes = plt.axes(projection=ccrs.Robinson(central_longitude=85))
cmaplist_2 = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist_2[1:], Num+1)
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

India.plot(facecolor='gray',edgecolor='black',ax=axes[2])
im1=axes[2].contourf(Irrig_LAI_SW.lon,Irrig_LAI_SW.lat,X_rate,levels=bounds,
                 cmap=cm.vik,vmin=vmn,vmax=vmx)
sig_area   = np.where(X_pvalue < 0.01)
im2=axes[2].scatter(lonx[sig_area],latx[sig_area],marker = 'o',s=0.5,c='k',alpha=0.6)
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('')
axes[2].set_title('Dry Matter')
# axes.set_title('\u0394Yield in wheat (t/ha/year)')
# axes.set_title('\u0394Dry Matter in wheat (g/m\u00b2/year)')
# axes.set_xlim(67,98)
# axes.set_ylim(7,38)

cbar = fig1.colorbar(im1,ax=axes[2],shrink=0.5,cmap=cmap,norm=norm)
# cbar.set_label('\u0394max.LAI (m\u00b2/m\u00b2)')
# cbar.set_label('\u0394Yield (t/ha)')
cbar.set_label('Dry Matter (g/m\u00b2/year)')
cbar.set_ticks(np.linspace(vmn,vmx,5))