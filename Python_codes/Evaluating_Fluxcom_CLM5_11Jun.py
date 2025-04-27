#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:15:34 2024

@author: knreddy
"""

#%% Loading required Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import matplotlib as mpl
from cmcrameri import cm
import pymannkendall as mk
import scipy.stats as stats
import warnings
import string
import seaborn as sns
warnings.filterwarnings("ignore")
#%% Adjust longitude 
def adjust_long(Var,lat):
    Var_ = np.empty(np.shape(Var))
    Var_[:,:,:] = Var[:,-1:,:]
    # Var_[:,int(len(lat)/2):] = Var[:,0:int(len(lat)/2)]
    return Var_
#%% Clip data for Indian region
def clip_data(Spatial_data,Region):
    Spatial_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    Spatial_data.rio.write_crs("epsg:4326", inplace=True)
    clip_Spatial_data = Spatial_data.rio.clip(Region.geometry, India.crs, drop=True)
    return clip_Spatial_data
#%% Pearson's r and bias
def find_r_and_bias(data1,data2):
    shape_data = shape(data1)
    fluxcom_data = np.reshape(data1, np.prod(shape_data))
    clm_data = np.reshape(data2, np.prod(shape_data))
    
    f_nans = np.isnan(fluxcom_data)
    c_nans = np.isnan(clm_data)
    
    fluxcom_data_ = fluxcom_data[((~f_nans) & (~c_nans))]
    clm_data_ = clm_data[((~f_nans) & (~c_nans))]
    
    r,p = stats.pearsonr(fluxcom_data_,clm_data_)
    
    bias = np.nansum(abs(clm_data_) - abs(fluxcom_data_))/np.nansum(abs(fluxcom_data_))
    return r,p,bias
#%%FLUXCOM Data
Foldername = '/Users/knreddy/Documents/GMD_Paper/FluxCom_Monthly_data/RS_METEO_monthly/'
test_SH_name = Foldername+'H.RS_METEO.EBC-ALL.MLM-ALL.METEO-GSWP3.720_360.monthly.1950.nc'
test_data = xr.open_dataset(test_SH_name)
test_data_SH = test_data.H
test_SH_india = test_data_SH.sel(lat=slice(40,0),lon=slice(60,100))


lat = test_data.lat
lon = test_data.lon

start_year = 1970
end_year = 2014

N_years = len(range(start_year,end_year))+1
shape_data = shape(test_SH_india)

SH_Fluxcom_1970_2014 = np.empty([N_years,shape_data[0],shape_data[1],shape_data[2]])
LH_Fluxcom_1970_2014 = np.empty([N_years,shape_data[0],shape_data[1],shape_data[2]])

for i_count,i_year in enumerate(range(start_year,end_year+1)):
    file_name = Foldername+'H.RS_METEO.EBC-ALL.MLM-ALL.METEO-GSWP3.720_360.monthly.'+str(i_year)+'.nc'
    temp_data = xr.open_dataset(file_name)
    temp_data_SH = temp_data.H
    temp_SH_india = temp_data_SH.sel(lat=slice(40,0),lon=slice(60,100))
    temp_SH_india=temp_SH_india.reindex(lat=list(reversed(temp_SH_india['lat'])))
    SH_Fluxcom_1970_2014[i_count,:,:,:] = temp_SH_india
    
    file_name2 = Foldername+'LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-GSWP3.720_360.monthly.'+str(i_year)+'.nc'
    temp_data2 = xr.open_dataset(file_name2)
    temp_data_LH = temp_data2.LE
    temp_LH_india = temp_data_LH.sel(lat=slice(40,0),lon=slice(60,100))
    temp_LH_india=temp_LH_india.reindex(lat=list(reversed(temp_LH_india['lat'])))
    LH_Fluxcom_1970_2014[i_count,:,:,:] = temp_LH_india


SH_Fluxcom_1970_2014_ = SH_Fluxcom_1970_2014/0.0864 # converting units from MJ/m2-d to W/m2
LH_Fluxcom_1970_2014_ = LH_Fluxcom_1970_2014/0.0864
#%% Loading carbon fluxes data
# CLM5 CTRL DATA
wrk_dir = '/Users/knreddy/Documents/CLM_DataAnalysis/Data_Extract/CLM5_Carbon_Exp/CLM5_Exp_Data/'
shp_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/India_Shapefile_'

data_CTRL = xr.open_dataset(wrk_dir+'/CLM5_CTRL_1970_2014_Monthly_Total_08-Apr-2024.nc')

India = gpd.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")

mask_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/'
mask_data = xr.open_dataset(mask_dir+'landuse.timeseries_360x720cru_hist_78pfts_CMIP6_simyr1850-2015_India_c221122.nc')

lat = data_CTRL['lat']
lon = data_CTRL['lon']
year = data_CTRL['year']
#%% Creating dataarray
Fluxcom_SH = xr.DataArray(SH_Fluxcom_1970_2014_,
    coords=dict(
        year = data_CTRL.coords['year'],
        month = data_CTRL.coords['month'],
        lat = data_CTRL.coords['lat'],
        lon = data_CTRL.coords['lon']),
    attrs=dict(
        description="Sensible heat FLUXCOM observations",
        units="W/m^2"))

Fluxcom_LH = xr.DataArray(LH_Fluxcom_1970_2014_,
    coords=dict(
        year = data_CTRL.coords['year'],
        month = data_CTRL.coords['month'],
        lat = data_CTRL.coords['lat'],
        lon = data_CTRL.coords['lon']),
    attrs=dict(
        description="Latent heat FLUXCOM observations",
        units="W/m^2"))
#%% Extract data (Units are gC/m2/year)
exp = ['CTRL']
var = ['LH','SH']
crop = ['SW','Rice']

for i_var in var:
    output_name = exp[0]+'_'+i_var
    dataset_name = 'data_'+exp[0]          
    data_var_name = i_var
    locals()[output_name] = eval(dataset_name)[data_var_name]
            
#%% 
pct_cft = np.array(mask_data['PCT_CFT'])
p_c_test = mask_data['PCT_CFT']
area = np.array(mask_data['AREA'])

pct_cft_ = xr.DataArray(pct_cft,
    coords=dict(
        time = p_c_test.coords['time'],
        cft = p_c_test.coords['cft'],
        lat = CTRL_LH.coords['lat'],
        lon = CTRL_LH.coords['lon']))

area_ = xr.DataArray(area,
    coords=dict(
        lat = CTRL_LH.coords['lat'],
        lon = CTRL_LH.coords['lon']),
    attrs=dict(
        description="area of grid cell",
        units="km^2"))

cft = np.array(mask_data['cft'])
pct_cft2 = pct_cft_[-46:-1,:,:,:] # pct_cft of period 1970 to 2014

clipped_pct_cft = clip_data(pct_cft2, India)
clipped_area = clip_data(area_,India)

pct_cft_sw1 = clipped_pct_cft[:,cft==19,:,:]
pct_cft_sw2 = clipped_pct_cft[:,cft==20,:,:]

pct_cft_sw = xr.concat([pct_cft_sw1,pct_cft_sw2], 'cft')
pct_cft_sw_ = pct_cft_sw.sum(axis=1)

pct_cft_rice1 = clipped_pct_cft[:,cft==61,:,:]
pct_cft_rice2 = clipped_pct_cft[:,cft==62,:,:]

pct_cft_rice = xr.concat([pct_cft_rice1,pct_cft_rice2], 'cft')
pct_cft_rice_ = pct_cft_rice.sum(axis=1)

pct_cft_sw_mask = np.ma.masked_less_equal(pct_cft_sw_, 10)
pct_cft_rice_mask = np.ma.masked_less_equal(pct_cft_rice_, 10)

area_SW = (pct_cft_sw_mask*10**4)*clipped_area # units are m2
area_Rice = (pct_cft_rice_mask*10**4)*clipped_area
#%% Clip all spatial data for Indian region
exp = ['CTRL','Fluxcom']
for i_exp in exp:
    for i_var in var:
        input_name = i_exp+'_'+i_var
        output_name = 'clipped_'+input_name
        locals()[output_name] = clip_data(eval(input_name),India)
#%% Masking for wheat growing and rice growing regions over India
SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_var = 'clipped_'+i_exp+'_'+i_var
            output_var = i_exp+'_'+i_var+'_'+i_crop
            if i_crop == 'SW':        
                temp_var = i_exp+'_'+i_var+'_'+i_crop+'_temp'         
                locals()[temp_var] = xr.concat([eval(input_var)[:-1,-2:,:,:],eval(input_var)[1:,0:4,:,:]],dim="month")
                locals()[output_var] = np.ma.masked_array(np.nanmean(eval(temp_var),axis=1),mask=eval(i_crop+'_mask'))
            else:
                locals()[output_var] = np.ma.masked_array(np.nanmean(eval(input_var)[:,6:10,:,:],axis=1),mask=eval(i_crop+'_mask'))

#%% taking spatial means and std
d_s = eval(output_var).shape

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
           input_3_name = i_exp+'_'+i_var+'_'+i_crop
           output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean'
           output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std'
           
           locals()[output_3_name1] = np.nanmean(eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
           locals()[output_3_name2] = np.nanstd(eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
#%% taking spatial means and std for CTRL and impact of exps 
d_s = eval(output_var).shape

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            if i_exp == 'CTRL':
               input_3_name = i_exp+'_'+i_var+'_'+i_crop
               output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean'
               output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std'
               
               locals()[output_3_name1] = np.nanmean(eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_3_name2] = np.nanstd(eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1) # units in kgC/m2/yr
               
            else:
                input_CTRL_name = 'CTRL_'+i_var+'_'+i_crop
                input_3_name = i_exp+'_'+i_var+'_'+i_crop
                output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean2'
                output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std2'
                
                impact_data = (eval(input_CTRL_name).reshape(d_s[0],d_s[1]*d_s[2])) - (eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))
                locals()[output_3_name1] = np.nanmean(impact_data,axis=1)
                locals()[output_3_name2] = np.nanstd(impact_data,axis=1)
#%% plotting trends and impact of drivers
colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]

fsize=14
plotid_x = 0.02
plotid_y = 0.85

slope_pos_x = 0.45
slope_pos_y = 0.85

p_pos_x = 0.52
p_pos_y = 0.8

LH_lims = [-15,120]
SH_lims = [-15,120]

LH_ticks = [0,25,50,75,100]
SH_ticks = [0,25,50,75,100]

LH_ticklabels = [0,25,50,75,100]
SH_ticklabels = [0,25,50,75,100]

x_new = np.arange(0,45)
year = np.arange(1970,2015)
fig1, axes = plt.subplots(nrows=2, ncols=2, dpi=600,sharex=True,figsize=(8,5),layout='constrained')

exp = ['CTRL', 'Fluxcom']
var = ['LH','SH']
crop = ['SW','Rice']
i_crop = crop[1]
# var in rows and exp in cols
for i_col,i_exp in enumerate(exp):
    for i_row,i_var in enumerate(var):

        if i_col ==0:
            color_line = 'k'
        else:
            # color_line = colors[i_col-1]
            color_line = 'k'
        # if i_exp == 'CTRL':
        mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean'
        std_data = i_exp+'_'+i_var+'_'+i_crop+'_std'
        # else:
        #     mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean2'
        #     std_data = i_exp+'_'+i_var+'_'+i_crop+'_std2'
        std1 = eval(mean_data) - eval(std_data)
        std2 = eval(mean_data) + eval(std_data)
        [trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((eval(mean_data)),alpha=0.05)
        # [slope2,intercept2] = mk.sens_slope((eval(mean_data)))
        y_new = x_new*slope+intercept

        axes[i_row,i_col].plot(year,(eval(mean_data)), color=color_line,linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1,std2,facecolor=color_line,alpha=0.25)
        if i_exp == 'CTRL':
            axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        else:
            if slope >0:
                axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            else:
                axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        axes[i_row,i_col].set_ylim(eval(i_var+'_lims'))
        axes[i_row,i_col].set_yticks(eval(i_var+'_ticks'))
        ## title
        # if i_col == 0:
        axes[0,i_col].set_title(i_exp,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        # else:
        #     title_text = r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
        #     axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[i_row,i_col].set_ylabel(i_var+' \n (W/m\u00b2)',fontsize=fsize)
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels'), fontsize=fsize)
        else:
            axes[i_row,i_col].set_ylabel('')
            axes[i_row,i_col].set_yticklabels('')
        ## x label
        if i_row == 2:
            axes[i_row,i_col].set_xlabel('Year',fontsize=fsize)

            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            axes[i_row,i_col].set_xlabel('')

        ## ticklabels
        axes[i_row,i_col].set_xticks([1975,1995,2010])
        axes[i_row,i_col].set_xticklabels([1975,1995,2010], fontsize=fsize)
        panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize+1,transform=axes[i_row,i_col].transAxes)
        # axes[i_row,i_col].text(plotid_x,plotid_y-0.1, trend,fontsize=fsize-2,transform=axes[i_row,i_col].transAxes)
        if p_value < 0.05:
            slope_text = '$slope='+'{:.3f}^*$'.format(slope)
        else:
            slope_text = '$slope='+'{:.3f}$'.format(slope)
        if i_col == 0:
            if slope < 0:
                if p_value <0.05:
                    slope_pos_x_ = slope_pos_x - 0.085
                else:
                    slope_pos_x_ = slope_pos_x - 0.05
            elif slope > 0:
                slope_pos_x_ = slope_pos_x
            bbbox_facecolor = (0.75, 0.75, 0.75)
        else:
            if slope < 0:
                if p_value <0.05:
                    slope_pos_x_ = slope_pos_x - 0.085
                    bbbox_facecolor = (1., 0.75, 0.75)
                else:
                    slope_pos_x_ = slope_pos_x - 0.05
                    bbbox_facecolor = (1., 0.75, 0.75)
            elif slope > 0:
                slope_pos_x_ = slope_pos_x
                bbbox_facecolor = (0.75, 0.8, 0.75)
            else:
                bbbox_facecolor = (0.75, 0.75, 0.75)
        axes[i_row,i_col].text(slope_pos_x_,slope_pos_y,slope_text, 
                               fontsize=fsize-1,transform=axes[i_row,i_col].transAxes,
                               bbox=dict(boxstyle="square",
                                         ec=bbbox_facecolor,
                                         fc=bbbox_facecolor,
                                         ))
        axes[i_row,i_col].axhline(y=0, color='grey', linestyle='--',zorder=0)
        # axes[i_row,i_col].text(p_pos_x,p_pos_y, 'p='+'{:.3f}'.format(p_value), fontsize=fsize-2,transform=axes[i_row,i_col].transAxes)

# fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/'+i_crop+'_Trends_energyfluxes_Sens_Slope_5Jun.png',dpi=600)
#%% Create dummy dataset
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            output_dataset = i_exp+'_'+i_var+'_'+i_crop+'_masked'
            locals()[output_dataset] = xr.DataArray(np.empty(shape(clipped_CTRL_LH)),
                                             coords=dict(
                                                 year = clipped_CTRL_LH.coords['year'],
                                                 month = clipped_CTRL_LH.coords['month'],
                                                 lat = clipped_CTRL_LH.coords['lat'],
                                                 lon = clipped_CTRL_LH.coords['lon']))
#%% Mean of 1970-80 and 2005-1014
SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_var = 'clipped_'+i_exp+'_'+i_var
            output_var1 = i_exp+'_'+i_var+'_'+i_crop+'_masked'
            output_var2 = i_exp+'_'+i_var+'_'+i_crop+'_1970_79'
            output_var3 = i_exp+'_'+i_var+'_'+i_crop+'_2005_14'
            output_var4 = i_exp+'_'+i_var+'_'+i_crop+'_masked2'
            for i_month,month in enumerate(CTRL_LH.month):
                if i_crop == 'SW':        
                    locals()[output_var1][:,i_month,:,:] = np.ma.masked_array(eval(input_var).data[:,i_month,:,:],mask=eval(i_crop+'_mask'))
                else:
                    locals()[output_var1][:,i_month,:,:] = np.ma.masked_array(eval(input_var).data[:,i_month,:,:],mask=eval(i_crop+'_mask'))
            locals()[output_var2] = np.nanmean(eval(output_var1).data[:10,:,:,:],axis=0)
            locals()[output_var3] = np.nanmean(eval(output_var1).data[35:,:,:,:],axis=0)
            locals()[output_var4] = np.nanmean(eval(output_var1).data,axis=0)
#%% finding mean and std over first deacde and last decade of simulation
d_s = eval(output_var2).shape

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
               input_1_name = i_exp+'_'+i_var+'_'+i_crop+'_1970_79'
               input_2_name = i_exp+'_'+i_var+'_'+i_crop+'_2005_14'
               input_name = i_exp+'_'+i_var+'_'+i_crop+'_masked2'
               output_1_name1 = input_1_name+'_mean'
               output_1_name2 = input_1_name+'_std'
               output_2_name1 = input_2_name+'_mean'
               output_2_name2 = input_2_name+'_std'
               output_name1 = input_name+'_mean'
               output_name2 = input_name+'_std'
               
               locals()[output_1_name1] = np.nanmean(eval(input_1_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_1_name2] = np.nanstd(eval(input_1_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_2_name1] = np.nanmean(eval(input_2_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_2_name2] = np.nanstd(eval(input_2_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_name1] = np.nanmean(eval(input_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
               locals()[output_name2] = np.nanstd(eval(input_name).reshape(d_s[0],d_s[1]*d_s[2]),axis=1)
#%% plotting trends and impact of drivers
colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]

fsize=14
plotid_x = 0.02
plotid_y = 0.85

data_pos_x = 0.535
data_pos_y = 0.27

slope_pos_x = 0.45
slope_pos_y = 0.85

p_pos_x = 0.52
p_pos_y = 0.8

LH_lims = [-15,120]
SH_lims = [-15,120]

LH_ticks = [0,25,50,75,100]
SH_ticks = [0,25,50,75,100]

LH_ticklabels = [0,25,50,75,100]
SH_ticklabels = [0,25,50,75,100]

x_new = np.arange(0,12)
year = np.arange(0,12)
fig1, axes = plt.subplots(nrows=2, ncols=2, dpi=600,sharex=True,figsize=(8,5),layout='constrained')

exp = ['CTRL', 'Fluxcom']
var = ['LH','SH']
crop = ['SW','Rice']
i_crop = crop[1]
# var in rows and exp in cols
for i_col,i_exp in enumerate(exp):
    for i_row,i_var in enumerate(var):

        if i_col ==0:
            color_line = 'k'
        else:
            # color_line = colors[i_col-1]
            color_line = 'k'
        # if i_exp == 'CTRL':
        mean_data1 = i_exp+'_'+i_var+'_'+i_crop+'_1970_79_mean'
        std_data1 = i_exp+'_'+i_var+'_'+i_crop+'_1970_79_std'
        
        mean_data2 = i_exp+'_'+i_var+'_'+i_crop+'_2005_14_mean'
        std_data2 = i_exp+'_'+i_var+'_'+i_crop+'_2005_14_std'
        # else:
        #     mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean2'
        #     std_data = i_exp+'_'+i_var+'_'+i_crop+'_std2'
        std1_1 = eval(mean_data1) - eval(std_data1)
        std2_1 = eval(mean_data1) + eval(std_data1)
        std1_2 = eval(mean_data2) - eval(std_data2)
        std2_2 = eval(mean_data2) + eval(std_data2)
        [trend, h, p_value, z, Tau, s, var_s, slope1, intercept1] = mk.original_test((eval(mean_data1)),alpha=0.05)
        [trend, h, p_value, z, Tau, s, var_s, slope2, intercept2] = mk.original_test((eval(mean_data2)),alpha=0.05)
        # [slope2,intercept2] = mk.sens_slope((eval(mean_data)))
        # y_new1 = x_new*slope1+intercept1

        axes[i_row,i_col].plot(year,(eval(mean_data1)), color='b',linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1_1,std2_1,facecolor='b',alpha=0.25)
        axes[i_row,i_col].plot(year,(eval(mean_data2)), color='r',linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1_2,std2_2,facecolor='r',alpha=0.25)
        # if i_exp == 'CTRL':
        #     axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        # else:
        #     if slope >0:
        #         axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        #     else:
        #         axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        axes[i_row,i_col].set_ylim(eval(i_var+'_lims'))
        axes[i_row,i_col].set_yticks(eval(i_var+'_ticks'))
        ## title
        # if i_col == 0:
        axes[0,i_col].set_title(i_exp,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        # else:
        #     title_text = r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
        #     axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[i_row,i_col].set_ylabel(i_var+' \n (W/m\u00b2)',fontsize=fsize)
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels'), fontsize=fsize)
        else:
            axes[i_row,i_col].set_ylabel('')
            axes[i_row,i_col].set_yticklabels('')
        ## x label
        if i_row == 2:
            axes[i_row,i_col].set_xlabel('Year',fontsize=fsize)

            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            axes[i_row,i_col].set_xlabel('')

        ## ticklabels
        axes[i_row,i_col].set_xticks([0,2,4,6,8,10,0])
        axes[i_row,i_col].set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov','Jan'], fontsize=fsize)
        panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize+1,transform=axes[i_row,i_col].transAxes)
        # axes[i_row,i_col].text(plotid_x,plotid_y-0.1, trend,fontsize=fsize-2,transform=axes[i_row,i_col].transAxes)
#%% plotting trends and impact of drivers
colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]

fsize=14
plotid_x = 0.02
plotid_y = 0.85

LH_lims = [-15,120]
SH_lims = [-15,120]

LH_ticks = [0,25,50,75,100]
SH_ticks = [0,25,50,75,100]

LH_ticklabels = [0,25,50,75,100]
SH_ticklabels = [0,25,50,75,100]

x_new = np.arange(0,12)
year = np.arange(0,12)
fig1, axes = plt.subplots(nrows=2, ncols=2, dpi=600,sharex=True,figsize=(8,5),layout='constrained')

exp = ['CTRL', 'Fluxcom']
var = ['LH','SH']
crop = ['Rice','SW']
# var in rows and exp in cols
for i_col,i_crop in enumerate(crop):
    for i_row,i_var in enumerate(var):
        # if i_exp == 'CTRL':
        mean_data1 = 'CTRL_'+i_var+'_'+i_crop+'_masked2_mean'
        std_data1 = 'CTRL_'+i_var+'_'+i_crop+'_masked2_std'
        
        mean_data2 = 'Fluxcom_'+i_var+'_'+i_crop+'_masked2_mean'
        std_data2 = 'Fluxcom_'+i_var+'_'+i_crop+'_masked2_std'
        # else:
        #     mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean2'
        #     std_data = i_exp+'_'+i_var+'_'+i_crop+'_std2'
        std1_1 = eval(mean_data1) - eval(std_data1)
        std2_1 = eval(mean_data1) + eval(std_data1)
        std1_2 = eval(mean_data2) - eval(std_data2)
        std2_2 = eval(mean_data2) + eval(std_data2)
        # [trend, h, p_value, z, Tau, s, var_s, slope1, intercept1] = mk.original_test((eval(mean_data1)),alpha=0.05)
        # [trend, h, p_value, z, Tau, s, var_s, slope2, intercept2] = mk.original_test((eval(mean_data2)),alpha=0.05)
        # [slope2,intercept2] = mk.sens_slope((eval(mean_data)))
        # y_new1 = x_new*slope1+intercept1

        axes[i_row,i_col].plot(year,(eval(mean_data1)), color='b',linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1_1,std2_1,facecolor='b',alpha=0.25)
        axes[i_row,i_col].plot(year,(eval(mean_data2)), color='r',linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1_2,std2_2,facecolor='r',alpha=0.25)
        # if i_exp == 'CTRL':
        #     axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        # else:
        #     if slope >0:
        #         axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        #     else:
        #         axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        axes[i_row,i_col].set_ylim(eval(i_var+'_lims'))
        axes[i_row,i_col].set_yticks(eval(i_var+'_ticks'))
        ## title
        # if i_col == 0:
        axes[0,0].set_title('Rice',fontsize=fsize+2)
        axes[0,1].set_title('Wheat',fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        # else:
        #     title_text = r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
        #     axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[i_row,i_col].set_ylabel(i_var+' \n (W/m\u00b2)',fontsize=fsize)
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels'), fontsize=fsize)
        else:
            axes[i_row,i_col].set_ylabel('')
            axes[i_row,i_col].set_yticklabels('')
        ## x label
        if i_row == 2:
            axes[i_row,i_col].set_xlabel('Year',fontsize=fsize)

            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            axes[i_row,i_col].set_xlabel('')

        ## ticklabels
        axes[i_row,i_col].set_xlim([0,11])
        axes[i_row,i_col].set_xticks([0,2,4,6,8,10,0])
        axes[i_row,i_col].set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov','Jan'], fontsize=fsize)
        panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize+1,transform=axes[i_row,i_col].transAxes)      
#%% scatter plot all data for fluxes

##%% DON'T RUN ########################
a = np.array(CTRL_LH_Rice_masked).reshape(np.prod(CTRL_LH_Rice_masked.shape))
b = np.array(Fluxcom_LH_Rice_masked).reshape(np.prod(CTRL_LH_Rice_masked.shape))

a_ = a[((~np.isnan(a)) & (~np.isnan(b)))]
b_ = b[((~np.isnan(a)) & (~np.isnan(b)))]

ax = sns.regplot(x=a[((~np.isnan(a)) & (~np.isnan(b)))], y=b[((~np.isnan(a)) & (~np.isnan(b)))], 
                 fit_reg=True,ci=95, dropna=True,robust=True,
                 line_kws=dict(color="r"),
                 scatter_kws=dict(color='k',s=0.01, alpha=0.5))
r, p = stats.pearsonr(a[((~np.isnan(a)) & (~np.isnan(b)))],b[((~np.isnan(a)) & (~np.isnan(b)))])
#%% Data for plotting monthly spatial means
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_name = input_name = i_exp+'_'+i_var+'_'+i_crop+'_masked'
            output_name = input_name+'_monthly'
            locals()[output_name] = np.nanmean(eval(input_name),axis=0)
#%% Plotting monthly data of wheat
################## Spatial Plotting ###########################################
###############################################################################
Data = CTRL_SH_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

data_pos_x = 0.5
data_pos_y = 0.8

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=14

fig3, axes3 = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, dpi=600, figsize=(10,11),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 100
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -50
vmx2 = 50

cmap2 = cm.vik
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+5,5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

wheat_month_dim = [11, 0, 1, 2]
wheat_month_names = ['Dec','Jan','Feb','Mar']
for i_row in range(4):
    for i_col in range(4):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=12)
        panel_no = i_row
        ax.text(panel_pos_x,panel_pos_y,string.ascii_lowercase[panel_no]+'.'+str(i_col+1),
                    fontsize=fsize,transform=ax.transAxes)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_col==0:
            plot_data1 = Fluxcom_LH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data1,axis=0)
            im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.set_ylabel(wheat_month_names[i_row]+' \n\n Latitude', fontsize=fsize)
            if i_row==0:
                ax.set(title='FLUXCOM',xlabel='')
                ax.text(varname_pos_x,varname_pos_y,'Latent Heat',
                        fontsize=fsize-2,transform=ax.transAxes)
            else:
                ax.set(title='',xlabel='')
        elif i_col ==1:
            plot_data1 = Fluxcom_LH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data2 = CTRL_LH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
            
            im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
            r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
            if p<0.01:
                data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
            else:
                data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
            ax.text(data_pos_x,data_pos_y,data_text,
                        fontsize=fsize-3,transform=ax.transAxes)
            # a = plot_data2 - plot_data1
            # ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
            # sig_area = np.where(ttest_val.pvalue<0.05)
            # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
            if i_row==0:
                ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')
            else:
                ax.set(title='',xlabel='', ylabel='')
        elif i_col ==2:
            plot_data1 = Fluxcom_SH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data1,axis=0)
            im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            if i_row==0:
                ax.set(title='FLUXCOM',xlabel='', ylabel='')
                ax.text(varname_pos_x,varname_pos_y,'Sensible Heat',
                        fontsize=fsize-2,transform=ax.transAxes)
            else:
                ax.set(title='',xlabel='', ylabel='')
        else:
            plot_data1 = Fluxcom_SH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data2 = CTRL_SH_SW_masked[:,wheat_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
            im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
            r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
            if p<0.01:
                data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
            else:
                data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
            ax.text(data_pos_x,data_pos_y,data_text,
                        fontsize=fsize-3,transform=ax.transAxes)
            # a = plot_data2 - plot_data1
            # ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
            # sig_area = np.where(ttest_val.pvalue<0.05)
            # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
            if i_row==0:
                ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')
            else:
                ax.set(title='',xlabel='', ylabel='')
                
        if i_row==3:
            ax.set_xlabel('Longitude',fontsize=fsize)

            if ((i_col==0) | (i_col==2)):
                cbar = fig3.colorbar(im1,ax=axes3[3,i_col],shrink=0.8,cmap=cmap,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn,vmx,5))
            else:
                cbar = fig3.colorbar(im2,ax=axes3[3,i_col],shrink=0.8,cmap=cmap2,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn2,vmx2,5))
                
            if (i_col==0):
                cbar.set_label('LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==1):
                cbar.set_label('\u0394LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==2):
                cbar.set_label('SH (W/m\u00b2)',fontsize=fsize)
            else:   
                cbar.set_label('\u0394SH (W/m\u00b2)',fontsize=fsize)
                
# cbar.remove()
#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Wheat_growingseason_monthly_EnergyFluxes_r_and_bias.png', 
#              dpi=600, bbox_inches="tight")

#%% Plotting monthly data of rice
################## Spatial Plotting ###########################################
###############################################################################
Data = CTRL_SH_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

data_pos_x = 0.5
data_pos_y = 0.8

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=14

fig3, axes3 = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, dpi=600, figsize=(10,11),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 150
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -70
vmx2 = 70

cmap2 = cm.vik
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+5,5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

rice_month_dim = [6, 7, 8, 9]
rice_month_names = ['Jul','Aug','Sep','Oct']
for i_row in range(4):
    for i_col in range(4):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=12)
        panel_no = i_row
        ax.text(panel_pos_x,panel_pos_y,string.ascii_lowercase[panel_no]+'.'+str(i_col+1),
                    fontsize=fsize,transform=ax.transAxes)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_col==0:
            plot_data1 = Fluxcom_LH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data1,axis=0)
            im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            ax.set_ylabel(rice_month_names[i_row]+' \n\n Latitude', fontsize=fsize)
            if i_row==0:
                ax.set(title='FLUXCOM',xlabel='')
                ax.text(varname_pos_x,varname_pos_y,'Latent Heat',
                        fontsize=fsize-2,transform=ax.transAxes)
            else:
                ax.set(title='',xlabel='')
        elif i_col ==1:
            plot_data1 = Fluxcom_LH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data2 = CTRL_LH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
            
            im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
            r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
            if p<0.01:
                data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
            else:
                data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
            ax.text(data_pos_x,data_pos_y,data_text,
                        fontsize=fsize-3,transform=ax.transAxes)
            # a = plot_data2 - plot_data1
            # ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
            # sig_area = np.where(ttest_val.pvalue<0.05)
            # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
            if i_row==0:
                ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')
            else:
                ax.set(title='',xlabel='', ylabel='')
        elif i_col ==2:
            plot_data1 = Fluxcom_SH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data1,axis=0)
            im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
            if i_row==0:
                ax.set(title='FLUXCOM',xlabel='', ylabel='')
                ax.text(varname_pos_x,varname_pos_y,'Sensible Heat',
                        fontsize=fsize-2,transform=ax.transAxes)
            else:
                ax.set(title='',xlabel='', ylabel='')
        else:
            plot_data1 = Fluxcom_SH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data2 = CTRL_SH_Rice_masked[:,rice_month_dim[i_row],:,:]
            plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
            im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
            r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
            if p<0.01:
                data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
            else:
                data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
            ax.text(data_pos_x,data_pos_y,data_text,
                        fontsize=fsize-3,transform=ax.transAxes)
            # a = plot_data2 - plot_data1
            # ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
            # sig_area = np.where(ttest_val.pvalue<0.05)
            # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
            if i_row==0:
                ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')
            else:
                ax.set(title='',xlabel='', ylabel='')
                
        if i_row==3:
            ax.set_xlabel('Longitude',fontsize=fsize)

            if ((i_col==0) | (i_col==2)):
                cbar = fig3.colorbar(im1,ax=axes3[3,i_col],shrink=0.8,cmap=cmap,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn,vmx,3))
            else:
                cbar = fig3.colorbar(im2,ax=axes3[3,i_col],shrink=0.8,cmap=cmap2,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn2,vmx2,5))
                
            if (i_col==0):
                cbar.set_label('LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==1):
                cbar.set_label('\u0394LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==2):
                cbar.set_label('SH (W/m\u00b2)',fontsize=fsize)
            else:   
                cbar.set_label('\u0394SH (W/m\u00b2)',fontsize=fsize)
                
# cbar.remove()
# fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Rice_growingseason_monthly_EnergyFluxes_r_and_bias.png', 
#               dpi=600, bbox_inches="tight")
#%% Plotting differnce between 1970-79 and 2005-14
Data = CTRL_SH_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

data_pos_x = 0.5
data_pos_y = 0.8

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=12

fig3, axes3 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, dpi=600, figsize=(8,5),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 100
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -50
vmx2 = 50

cmap2 = cm.vik
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+5,5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

wheat_month_dim = [11, 0, 1, 2]
wheat_month_names = ['Dec','Jan','Feb','Mar']
for i_row in range(2):
    for i_col in range(4):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=fsize)
        panel_no = i_col+4*(i_row)
        ax.text(panel_pos_x,panel_pos_y,'('+string.ascii_lowercase[panel_no]+')',
                    fontsize=fsize,transform=ax.transAxes)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row==0:            
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_ylabel('Mean of 1970-79 \n Latitude', fontsize=fsize-1)
                ax.set_title('FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.text(varname_pos_x,varname_pos_y,'Latent Heat',
                        fontsize=fsize-3,transform=ax.transAxes)
            elif i_col ==1:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_LH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                # sig_area = np.where(ttest_val.pvalue<0.01)
                # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.6)
                ax.set_title('CLM-FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.set_ylabel('')
            elif i_col ==2:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_title('FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.text(varname_pos_x,varname_pos_y,'Sensible Heat',
                        fontsize=fsize-3,transform=ax.transAxes)
            else:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_SH_SW_masked[:10,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                # sig_area = np.where(ttest_val.pvalue<0.01)
                # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.6)
                ax.set_title('CLM-FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.set_ylabel('')
        if i_row==1:
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_ylabel('Mean of 2005-14 \n Latitude', fontsize=fsize-1)
                ax.set(title='',xlabel='')
            elif i_col ==1:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_LH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                # sig_area = np.where(ttest_val.pvalue<0.01)
                # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.6)
                ax.set(title='',xlabel='', ylabel='')
            elif i_col ==2:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='', ylabel='')
            else:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_SH_SW_masked[-10:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                # sig_area = np.where(ttest_val.pvalue<0.01)
                # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.6)
                ax.set(title='',xlabel='', ylabel='')               
        if i_row==1:
            ax.set_xlabel('Longitude',fontsize=fsize)

            if ((i_col==0) | (i_col==2)):
                cbar = fig3.colorbar(im1,ax=axes3[1,i_col],shrink=0.75,cmap=cmap,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn,vmx,5))
            else:
                cbar = fig3.colorbar(im2,ax=axes3[1,i_col],shrink=0.75,cmap=cmap2,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn2,vmx2,5))
                
            if (i_col==0):
                cbar.set_label('LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==1):
                cbar.set_label('\u0394LH (W/m\u00b2)',fontsize=fsize)
            elif (i_col==2):
                cbar.set_label('SH (W/m\u00b2)',fontsize=fsize)
            else:   
                cbar.set_label('\u0394SH (W/m\u00b2)',fontsize=fsize)
#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Wheat_EnergyFluxes_1970s_vs_2010s_r_and_bias.png', 
#              dpi=600, bbox_inches="tight")
#%% Latest plotting for rice
Data = CTRL_SH_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

data_pos_x = 0.5
data_pos_y = 0.8

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=12

fig3, axes3 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, 
                           dpi=600, figsize=(5,6),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 150
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -70
vmx2 = 70

cmap2 = cm.vik
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+5,5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

rice_month_dim = [6, 7, 8, 9]
rice_month_names = ['Jul','Aug','Sep','Oct']
for i_row in range(2):
    for i_col in range(2):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=fsize)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        if i_row==0:
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_LH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_ylabel('Latent Heat', fontsize=fsize-1)
                if i_row==0:
                    ax.set_title('(i)\nFLUXCOM',fontsize=fsize-2)
                    ax.set_xlabel('')
                else:
                    ax.set(title='',xlabel='')
            elif i_col ==1:
                plot_data1 = np.nanmean(Fluxcom_LH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_LH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                axes3[0,0].text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=axes3[0,0].transAxes)
                ax.set_title('(ii)\nCLM-FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.set_ylabel('')
        elif i_row==1:
            if i_col ==0:
                plot_data1 = np.nanmean(Fluxcom_SH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)

                ax.set_xlabel('')
                ax.set_ylabel('Sensible Heat', fontsize=fsize-1)

            else:
                plot_data1 = np.nanmean(Fluxcom_SH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_SH_Rice_masked[:10,rice_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                axes3[1,0].text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=axes3[1,0].transAxes)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
        if i_row==1:

            if (i_col==0):
                cbar = fig3.colorbar(im1,ax=axes3[1,i_col],shrink=0.75,cmap=cmap,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn,vmx,3))
                cbar.set_ticklabels(np.linspace(vmn,vmx,3), fontsize=fsize-2)
            else:
                cbar = fig3.colorbar(im2,ax=axes3[1,i_col],shrink=0.75,cmap=cmap2,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn2,vmx2,5))
                cbar.set_ticklabels(np.linspace(vmn2,vmx2,5), fontsize=fsize-3)
                
            if (i_col==0):
                cbar.set_label('(W/m\u00b2)',fontsize=fsize)
            elif (i_col==1):
                cbar.set_label('\u0394 (W/m\u00b2)',fontsize=fsize)


# Add rotated bold 'A.' manually
axes3[0, 0].text(-0.3, 0.5, '(a)', transform=axes3[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[1, 0].text(-0.3, 0.5, '(b)', transform=axes3[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')


#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Rice_growingseason_monthly_EnergyFluxes_r_and_bias_18Apr.png', 
#              dpi=600, bbox_inches="tight")

#%% Latest plotting for wheat
Data = CTRL_SH_SW_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

data_pos_x = 0.5
data_pos_y = 0.8

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=12

fig3, axes3 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, 
                           dpi=600, figsize=(5,6),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 150
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -70
vmx2 = 70

cmap2 = cm.vik
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+5,5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

rice_month_dim = [6, 7, 8, 9]
rice_month_names = ['Jul','Aug','Sep','Oct']

sw_month_dim = [11, 1, 2, 3]
sw_month_names = ['Dec','Jan','Feb','Mar']
for i_row in range(2):
    for i_col in range(2):
        p_row = i_row
        ax = axes3[p_row,i_col]
        ax.tick_params('both', labelsize=fsize)
        #panel_no = i_col+4*(i_row)
        #ax.text(panel_pos_x,panel_pos_y,'('+string.ascii_lowercase[panel_no]+')',
        #            fontsize=fsize-2,transform=ax.transAxes)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        if i_row==0:
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_ylabel('Latent Heat', fontsize=fsize-1)
                if i_row==0:
                    ax.set_title('(i)\nFLUXCOM',fontsize=fsize-2)
                    ax.set_xlabel('')

                else:
                    ax.set(title='',xlabel='')
            elif i_col ==1:
                plot_data1 = np.nanmean(Fluxcom_LH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_LH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                axes3[0,0].text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=axes3[0,0].transAxes)
                ax.set_title('(ii)\nCLM-FLUXCOM',fontsize=fsize-2)
                ax.set_xlabel('')
                ax.set_ylabel('')
        elif i_row==1:
            if i_col ==0:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set_xlabel('')
                ax.set_ylabel('Sensible Heat', fontsize=fsize-1)
            else:
                plot_data1 = np.nanmean(Fluxcom_SH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_SH_SW_masked[:,sw_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(np.array(plot_data1), np.array(plot_data2))
                if p<0.01:
                    data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                axes3[1,0].text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3.5,transform=axes3[1,0].transAxes)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
        if i_row==1:

            if (i_col==0):
                cbar = fig3.colorbar(im1,ax=axes3[1,i_col],shrink=0.75,cmap=cmap,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn,vmx,3))
                cbar.set_ticklabels(np.linspace(vmn,vmx,3), fontsize=fsize-2)
            else:
                cbar = fig3.colorbar(im2,ax=axes3[1,i_col],shrink=0.75,cmap=cmap2,orientation='horizontal')
                
                cbar.set_ticks(np.linspace(vmn2,vmx2,5))
                cbar.set_ticklabels(np.linspace(vmn2,vmx2,5), fontsize=fsize-3)
                
            if (i_col==0):
                cbar.set_label('(W/m\u00b2)',fontsize=fsize)
            elif (i_col==1):
                cbar.set_label('\u0394 (W/m\u00b2)',fontsize=fsize)

# Add rotated bold 'A.' manually
axes3[0, 0].text(-0.3, 0.5, '(a)', transform=axes3[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[1, 0].text(-0.3, 0.5, '(b)', transform=axes3[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')


#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Rice_growingseason_monthly_EnergyFluxes_r_and_bias_18Apr.png', 
#              dpi=600, bbox_inches="tight")

#%%
Data = CTRL_SH_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.45
varname_pos_y = 0.85

fsize=14

fig3, axes3 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, dpi=600, figsize=(10,4),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 120
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -50
vmx2 = 50

cmap2 = cm.vik
cmaplist2 = [cmap2(i) for i in np.arange(0,cmap2.N,int(256/Num+1))]
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist2, Num+1)

# define the bins and normalize
bounds2 = np.linspace(vmn2, vmx2, Num+1)
norm2 = mpl.colors.BoundaryNorm(bounds2, Num+2)

wheat_month_dim = [11, 0, 1, 2]
wheat_month_names = ['Dec','Jan','Feb','Mar']
for i_col in range(4):
    ax = axes3[i_col]
    ax.tick_params('both', labelsize=15)
    India.plot(facecolor='gray',edgecolor='black',ax=ax)
    if i_col==0:
        plot_data = np.nanmean(Fluxcom_LH_Rice_masked_monthly[6:10,:,:],axis=0)
        im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
        ax.set_ylabel('Latitude', fontsize=fsize)
        ax.set(title='FLUXCOM',xlabel='')
        ax.text(varname_pos_x,varname_pos_y,'Latent Heat',
                    fontsize=fsize-2,transform=ax.transAxes)
    elif i_col ==1:
        plot_data = np.nanmean(CTRL_LH_Rice_masked_monthly[6:10,:,:],axis=0) - np.nanmean(Fluxcom_LH_Rice_masked_monthly[6:10,:,:],axis=0)       
        im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
        a = CTRL_LH_Rice_masked_monthly[6:10,:,:] - Fluxcom_LH_Rice_masked_monthly[6:10,:,:]
        ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
        sig_area = np.where(ttest_val.pvalue<0.05)
        im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
        ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')

    elif i_col ==2:
        plot_data = np.nanmean(Fluxcom_SH_Rice_masked_monthly[6:10,:,:],axis=0)
        im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
        ax.set(title='FLUXCOM',xlabel='', ylabel='')
        ax.text(varname_pos_x,varname_pos_y,'Sensible Heat',
                    fontsize=fsize-2,transform=ax.transAxes)
    else:
        plot_data = np.nanmean(CTRL_SH_Rice_masked_monthly[6:10,:,:],axis=0) - np.nanmean(Fluxcom_SH_Rice_masked_monthly[6:10,:,:],axis=0)
        im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
        a = CTRL_SH_Rice_masked_monthly[6:10,:,:] - Fluxcom_SH_Rice_masked_monthly[6:10,:,:]
        ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
        sig_area = np.where(ttest_val.pvalue<0.05)
        im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
        ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')

    ax.set_xlabel('Longitude',fontsize=fsize)

    if ((i_col==0) | (i_col==2)):
        cbar = fig3.colorbar(im1,ax=axes3[i_col],shrink=0.8,cmap=cmap,orientation='horizontal')
        
        cbar.set_ticks(np.linspace(vmn,vmx,3))
    else:
        cbar = fig3.colorbar(im2,ax=axes3[i_col],shrink=0.8,cmap=cmap2,orientation='horizontal')
        
        cbar.set_ticks(np.linspace(vmn2,vmx2,5))
        
    if ((i_col==0) | (i_col==1)):
        cbar.set_label('LH (W/m\u00b2)',fontsize=fsize)
    else:   
        cbar.set_label('SH (W/m\u00b2)',fontsize=fsize)
#%%
fig3, axes3 = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, dpi=600, figsize=(10,4),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = 0
vmx = 120
Num = 10

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+1))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+1)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+1)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)

vmn2 = -50
vmx2 = 50

cmap2 = cm.vik
cmaplist2 = [cmap2(i) for i in np.arange(0,cmap2.N,int(256/Num+1))]
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist2, Num+1)

# define the bins and normalize
bounds2 = np.linspace(vmn2, vmx2, Num+1)
norm2 = mpl.colors.BoundaryNorm(bounds2, Num+2)

wheat_month_dim = [11, 0, 1, 2]
wheat_month_names = ['Dec','Jan','Feb','Mar']
for i_col in range(4):
    ax = axes3[i_col]
    ax.tick_params('both', labelsize=15)
    India.plot(facecolor='gray',edgecolor='black',ax=ax)
    if i_col==0:
        plot_data1 = np.concatenate((Fluxcom_LH_SW_masked_monthly[-1:,:,:],Fluxcom_LH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data = np.nanmean(plot_data1,axis=0)
        im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
        ax.set_ylabel('Latitude', fontsize=fsize)
        ax.set(title='FLUXCOM',xlabel='')
        ax.text(varname_pos_x,varname_pos_y,'Latent Heat',
                    fontsize=fsize-2,transform=ax.transAxes)
    elif i_col ==1:
        plot_data1 = np.concatenate((Fluxcom_LH_SW_masked_monthly[-1:,:,:],Fluxcom_LH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data2 = np.concatenate((CTRL_LH_SW_masked_monthly[-1:,:,:],CTRL_LH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)       
        im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
        a = plot_data2 - plot_data1
        ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
        sig_area = np.where(ttest_val.pvalue<0.05)
        im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
        ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')

    elif i_col ==2:
        plot_data1 = np.concatenate((Fluxcom_SH_SW_masked_monthly[-1:,:,:],Fluxcom_SH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data = np.nanmean(plot_data1,axis=0)
        im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
        ax.set(title='FLUXCOM',xlabel='', ylabel='')
        ax.text(varname_pos_x,varname_pos_y,'Sensible Heat',
                    fontsize=fsize-2,transform=ax.transAxes)
    else:
        plot_data1 = np.concatenate((Fluxcom_SH_SW_masked_monthly[-1:,:,:],Fluxcom_SH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data2 = np.concatenate((CTRL_SH_SW_masked_monthly[-1:,:,:],CTRL_SH_SW_masked_monthly[0:3,:,:]),axis=0)
        plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
        im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
        a = plot_data2 - plot_data1
        ttest_val = stats.ttest_1samp(a, popmean=0, nan_policy='omit')
        sig_area = np.where(ttest_val.pvalue<0.05)
        im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=1,c='k',alpha=0.6)
        ax.set(title='CLM-FLUXCOM',xlabel='', ylabel='')

    ax.set_xlabel('Longitude',fontsize=fsize)

    if ((i_col==0) | (i_col==2)):
        cbar = fig3.colorbar(im1,ax=axes3[i_col],shrink=0.8,cmap=cmap,orientation='horizontal')
        
        cbar.set_ticks(np.linspace(vmn,vmx,3))
    else:
        cbar = fig3.colorbar(im2,ax=axes3[i_col],shrink=0.8,cmap=cmap2,orientation='horizontal')
        
        cbar.set_ticks(np.linspace(vmn2,vmx2,5))
        
    if ((i_col==0) | (i_col==1)):
        cbar.set_label('LH (W/m\u00b2)',fontsize=fsize)
    else:   
        cbar.set_label('SH (W/m\u00b2)',fontsize=fsize)

