#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:40:30 2025

@author: knreddy
"""

#%% Loading required Modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import math
import netCDF4 as nc
import warnings
warnings.filterwarnings("ignore")
#%% closest function
def closest(data,K):
    list_min = abs(data - K);
    indx = min(list_min) == list_min;
    x_closest = list(indx);
    return x_closest
#%% Bias calculation
def bias_var(obs,mod):
    num = sum(abs(mod-obs))
    den = sum(obs)
    bias_ = num/den
    return bias_
#%% RMSE calculation
def RMSE_var(obs,mod):
    MSE = np.square(np.subtract(obs,mod)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE
#%% Load CLM data
CLM_data_dir = '/Users/knreddy/Documents/CLM_DataAnalysis/Data_Extract/CLM5_Carbon_Exp/CLM5_Exp_Data'
filename = CLM_data_dir+'/CLM5_CTRL_1970_2014_Daily_PFT_21-Mar-2024.nc'
filename_mon = CLM_data_dir+'/CLM5_CTRL_1970_2014_Monthly_PFT_18-Mar-2024.nc'

data = nc.Dataset(filename)
data_mon = nc.Dataset(filename_mon)

CLM_year = data['year']
CLM_doy = data['day_of_year']
CLM_lat = data['lat']
CLM_lon = data['lon']

# CLM_LAI = data['LAI']
# CLM_DM = data['TOTPFTC']
# CLM_GY = data_mon['GY']

CLM_Rice_LAI = data['LAI'][:,:,2:,:,:]
CLM_Rice_DryMatter = data['TOTPFTC'][:,:,2:,:,:]
CLM_Rice_Yield = data_mon['GY'][:,:,2:,:,:]

CLM_Wheat_LAI = data['LAI'][:,:,:2,:,:]
CLM_Wheat_DryMatter = data['TOTPFTC'][:,:,:2,:,:]
CLM_Wheat_Yield = data_mon['GY'][:,:,:2,:,:]
#%% Observations data
obs_dir = '/Users/knreddy/Documents/Carbon_Fluxes_CLM5/Analysis/Observation_Data/'
crops = ['Rice', 'Wheat']
variables = ['LAI', 'DryMatter','Yield','GSL']
site_var = ['Lat','Lon','doy','year']

for crop in crops:
    for var in variables:
        if var == 'GSL':
            input_file = crop+'_'+var+'_'+'Observations_7Jan.xlsx'
        else:
            input_file = crop+'_'+var+'_'+'Observations.xlsx'
        output_name1 = 'Obs_'+crop+'_'+var+'_data'
        locals()[output_name1] = pd.read_excel(obs_dir+input_file)
#%% extracting data for sites from CLM5
for crop in crops:
    for var in variables[0:3]:
        input_name = 'Obs_'+crop+'_'+var+'_data'
        CLM_inputname = 'CLM_'+crop+'_'+var
        output_main = 'CLM_'+crop+'_'+var+'_data'
        locals()[output_main] = np.empty(len(eval(input_name)))
        for i_count in range(len(eval(output_main))):
            year_mask = CLM_year == eval(input_name)['Year'][i_count]
            lat_mask = closest(CLM_lat,eval(input_name)['Lat'][i_count])
            lon_mask = closest(CLM_lon,eval(input_name)['Lon'][i_count])
            if ((var == 'LAI') | (var == 'DryMatter')):
                doy_mask = CLM_doy == eval(input_name)['DOY'][i_count]
                clm_dummy_data = eval(CLM_inputname)[year_mask,doy_mask,:,lat_mask,lon_mask]
                if var == 'LAI':
                    clm_dummy_data2 = np.nanmax(clm_dummy_data)
                else:
                    clm_dummy_data2 = np.nansum(abs(clm_dummy_data))/100 # units converted from g/m2 to t/ha
                locals()[output_main][i_count] = clm_dummy_data2
            else:
                clm_dummy_data = eval(CLM_inputname)[year_mask,:,:,lat_mask,lon_mask]
                locals()[output_main][i_count] = (np.nansum(clm_dummy_data))/100 # units in g/m2 to t/ha
#%% Plotting data
fig1, axes = plt.subplots(nrows=3, ncols=2, dpi=600,figsize=(8,12),layout='constrained')     

fsize = 14

plotid_x = 0.02
plotid_y = 0.9

corr_pos_x = 0.45
corr_pos_y = 0.9

bias_pos_x = 0.45
bias_pos_y = 0.83

rmse_pos_x = 0.45
rmse_pos_y = 0.76

for i_col,crop in enumerate(crops):
    for i_row,var in enumerate(variables[:3]):
        obs_data = eval('Obs_'+crop+'_'+var+'_data')['Data']
        clm_data = eval('CLM_'+crop+'_'+var+'_data')
        if var=='DryMatter':
            if crop == 'Rice':
                obs_data = (obs_data[((clm_data!=0)&(clm_data<20)&(clm_data>2))])/100
                clm_data = clm_data[((clm_data!=0)&(clm_data<20)&(clm_data>2))]
            else:
                obs_data = (obs_data[~np.isnan(clm_data)])/100
                clm_data = clm_data[~np.isnan(clm_data)]
        else:
            obs_data = obs_data
            clm_data = clm_data
        bias = bias_var(obs_data,clm_data)
        rmse = RMSE_var(obs_data,clm_data)
        corr_,p_val = pearsonr(obs_data,clm_data)
        # m,b = np.polyfit(clm_data,obs_data,1)
        # axes[i_row,i_col].scatter(obs_data,clm_data,c='k')
        # axes[i_row,i_col].plot(obs_data,obs_data*m+b,'k',alpha=0.6,linewidth=3)
        sns.regplot(x=obs_data, y=clm_data, fit_reg=True,ci=95, 
                    dropna=True,line_kws=dict(color="k"),scatter_kws=dict(color='k',s=25),
                    robust=True,ax=axes[i_row,i_col])
        #panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        #axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize+1,transform=axes[i_row,i_col].transAxes)
        if p_val < 0.05:
            corr_text = f"Pearsons r = {corr_:.2f}*"
        else:
            corr_text = f"Pearsons r = {corr_:.2f}"
        bias_text = '$MAB='+'{:.2f}$'.format(bias)
        rmse_text = '$RMSE='+'{:.2f}$'.format(rmse)
        axes[i_row,i_col].text(corr_pos_x,corr_pos_y,corr_text, 
                               fontsize=fsize-1,transform=axes[i_row,i_col].transAxes)
        axes[i_row,i_col].text(bias_pos_x,bias_pos_y,bias_text, 
                               fontsize=fsize-1,transform=axes[i_row,i_col].transAxes)
        axes[i_row,i_col].text(rmse_pos_x,rmse_pos_y,rmse_text, 
                               fontsize=fsize-1,transform=axes[i_row,i_col].transAxes)
        
# axes[0,0].set_xlim([-0.5,10]);axes[0,1].set_xlim([-0.5,10]);
# axes[1,0].set_xlim([-0.5,20]);axes[1,1].set_xlim([-0.5,20]);
# axes[2,0].set_xlim([-0.5,7]) ;axes[2,1].set_xlim([-0.5,7]);

# axes[0,0].set_ylim([-0.5,10]);axes[0,1].set_ylim([-0.5,10]);
# axes[1,0].set_ylim([-0.5,20]);axes[1,1].set_ylim([-0.5,20]);
# axes[2,0].set_ylim([-0.5,7]) ;axes[2,1].set_ylim([-0.5,7]);
        
axes[0,0].set_xticks(np.linspace(0,9,4));axes[0,1].set_xticks(np.linspace(0,9,4));
axes[1,0].set_xticks(np.linspace(0,18,4));axes[1,1].set_xticks(np.linspace(0,18,4));
axes[2,0].set_xticks(np.linspace(0,6,4)) ;axes[2,1].set_xticks(np.linspace(0,6,4));

axes[0,0].set_yticks(np.linspace(0,9,4));axes[0,1].set_yticks(np.linspace(0,9,4));
axes[1,0].set_yticks(np.linspace(0,18,4));axes[1,1].set_yticks(np.linspace(0,18,4));
axes[2,0].set_yticks(np.linspace(0,6,4)) ;axes[2,1].set_yticks(np.linspace(0,6,4));

axes[0,0].set_xticklabels(np.linspace(0,9,4));axes[0,1].set_xticklabels(np.linspace(0,9,4));
axes[1,0].set_xticklabels(np.linspace(0,18,4));axes[1,1].set_xticklabels(np.linspace(0,18,4));
axes[2,0].set_xticklabels(np.linspace(0,6,4)) ;axes[2,1].set_xticklabels(np.linspace(0,6,4));

axes[0,0].set_yticklabels(np.linspace(0,9,4));axes[0,1].set_yticklabels([]);
axes[1,0].set_yticklabels(np.linspace(0,18,4));axes[1,1].set_yticklabels([]);
axes[2,0].set_yticklabels(np.linspace(0,6,4)) ;axes[2,1].set_yticklabels([]);

axes[0,0].set_xlabel('') ;axes[0,1].set_xlabel('',);
axes[1,0].set_xlabel('') ;axes[1,1].set_xlabel('',);
axes[2,0].set_xlabel('Observations') ;axes[2,1].set_xlabel('Observations',);

# Add rotated bold 'A.' manually
axes[0, 0].text(-0.30, 0.5, '(a)', transform=axes[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[1, 0].text(-0.30, 0.5, '(b)', transform=axes[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[2, 0].text(-0.30, 0.5, '(c)', transform=axes[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')

axes[0,0].set_ylabel('$CLM5_{LAI} (m^2/m^2)$',fontsize=fsize-1);axes[0,1].set_ylabel('');
axes[1,0].set_ylabel('$CLM5_{DryMatter} (t/ha)$',fontsize=fsize-1);axes[1,1].set_ylabel('');
axes[2,0].set_ylabel('$CLM5_{Yield} (t/ha)$',fontsize=fsize-1) ;axes[2,1].set_ylabel('');

# axes[0,0].set_xlabel('');axes[0,1].set_xlabel('$CLM5_{LAI} (m^2/m^2)$',fontsize=fsize-1);
# axes[1,0].set_xlabel('');axes[1,1].set_xlabel('$CLM5_{DryMatter} (t/ha)$',fontsize=fsize-1);
# axes[2,0].set_xlabel('') ;axes[2,1].set_xlabel('$CLM5_{Yield} (t/ha)$',fontsize=fsize-1);

# axes[0,0].set_ylabel('$Obs_{LAI} (m^2/m^2)$',fontsize=fsize-1);axes[0,1].set_ylabel('$Obs_{LAI} (m^2/m^2)$',fontsize=fsize-1);
# axes[1,0].set_ylabel('$Obs_{DryMatter} (t/ha)$',fontsize=fsize-1);axes[1,1].set_ylabel('$Obs_{DryMatter} (t/ha)$',fontsize=fsize-1);
# axes[2,0].set_ylabel('$Obs_{Yield} (t/ha)$',fontsize=fsize-1) ;axes[2,1].set_ylabel('$Obs_{Yield} (t/ha)$',fontsize=fsize-1);

# axes[0,0].set_title('');axes[1,0].set_title('');
axes[0,0].set_title('(i)\nRice',fontsize=fsize);#,fontweight='bold');
axes[0,1].set_title('(ii)\nWheat',fontsize=fsize);#,fontweight='bold');
# axes[2,0].set_title('') ;axes[1,2].set_title('');

# fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Comparison_Crop_Phenology_CLM5_Observations_18Oct.png',
            #dpi=600, bbox_inches="tight")