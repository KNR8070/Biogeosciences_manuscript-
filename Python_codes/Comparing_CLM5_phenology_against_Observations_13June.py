#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:34:27 2024

@author: knreddy
"""
#%% Loading required Modules
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import scipy as sp
import math
import netCDF4 as nc
import scipy.stats as stats
import warnings
import string
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pymannkendall as mk
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
        # out_var_name = 'Obs_'+crop+'_'+var
        # locals()[]
        
#%% LAI selection for CTRL simulations
for i_crop in ['Rice', 'Wheat']:
    unique_file_names = eval('Obs_'+i_crop+'_LAI_data').Event.unique()
    
    globals()['Obs_'+i_crop+'_LAI_new'] = {'Event':[],'DOY': [], 'LAI': []}
    
    for i_filename in unique_file_names:
        LAI_data = eval('Obs_'+i_crop+'_LAI_data')[eval('Obs_'+i_crop+'_LAI_data').Event==i_filename]    
        # Step 1: Segment DOY and LAI into experiments
    # Assume a new experiment starts when the difference between consecutive DOYs decreases
        experiments = []
        current_experiment = {'doy': [], 'lai': []}
        
        for i in range(len(LAI_data.DOY)):
            if i > 0 and LAI_data.DOY.iloc[i] < LAI_data.DOY.iloc[i-1]:  # New experiment starts
                experiments.append(current_experiment)  # Save current experiment
                current_experiment = {'doy': [], 'lai': []}  # Start a new one
            # Add data to the current experiment
            current_experiment['doy'].append(LAI_data.DOY.iloc[i])
            current_experiment['lai'].append(LAI_data.Data.iloc[i])
        experiments.append(current_experiment)
        
        lowest_doy_experiment = min(experiments, key=lambda exp: min(exp['doy']))
        
        globals()['Obs_'+i_crop+'_LAI_new']['Event'].append(i_filename)
        globals()['Obs_'+i_crop+'_LAI_new']['DOY'].append(lowest_doy_experiment['doy'])
        globals()['Obs_'+i_crop+'_LAI_new']['LAI'].append(lowest_doy_experiment['lai'])
    
    globals()['Obs_'+i_crop+'_LAI_df'] = pd.DataFrame({'Event': [],
                                   'DOY': [],
                                   'LAI': [],
                                    })
    for i_num,i_event in enumerate(eval('Obs_'+i_crop+'_LAI_new')['Event']):
        dummy_df = pd.DataFrame({'Event': [i_event]*len(eval('Obs_'+i_crop+'_LAI_new')['DOY'][i_num]),
                                 'DOY': eval('Obs_'+i_crop+'_LAI_new')['DOY'][i_num],
                                 'LAI': eval('Obs_'+i_crop+'_LAI_new')['LAI'][i_num],
                                    })
        globals()['Obs_'+i_crop+'_LAI_df'] = pd.concat([eval('Obs_'+i_crop+'_LAI_df'), dummy_df])
    
#%% Plotting LAI data from observations
wheat_lai_data = (Obs_Wheat_LAI_data.sort_values('DOY')).reset_index()
wheat_lai_data['DOY'][wheat_lai_data['DOY']<150] = wheat_lai_data['DOY']+365

# wheat_lai_data = (Obs_Wheat_LAI_df.sort_values('DOY')).reset_index()
# wheat_lai_data['DOY'][wheat_lai_data['DOY']<150] = wheat_lai_data['DOY']+365

#%%
fig,ax = plt.subplots(ncols=2)
# Obs_Rice_LAI_data.plot(kind='scatter',x='DOY', y='Data', ax=ax[0])
# wheat_lai_data.plot(kind='scatter',x='DOY',y='Data',ax=ax[1])
sns.regplot(data=Obs_Rice_LAI_df, x='DOY',y='LAI',ax=ax[0],order=2,line_kws=dict(color="k"))
sns.regplot(data=wheat_lai_data, x='DOY',y='Data',ax=ax[1],order=2,line_kws=dict(color="k"))
ax[1].set_xlim([274,365+274])
ax[0].set_xlim([0,366])

ax[0].set_xticks([0, 91, 182, 274, 366])
ax[0].set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Jan'])

ax[1].set_xticks([274, 365+1, 365+91, 365+182, 365+274])
ax[1].set_xticklabels(['Oct', 'Jan', 'Apr', 'Jul', 'Oct'])
#%%
Obs_Rice_LAI_data['Crop'] = 'Rice'
wheat_lai_data['Crop'] = 'Wheat'

LAI_data = pd.concat([Obs_Rice_LAI_data, wheat_lai_data], ignore_index=True)
#%%
fig,ax = plt.subplots(figsize=(12,6))
# Obs_Rice_LAI_data.plot(kind='scatter',x='DOY', y='Data', ax=ax[0])
# wheat_lai_data.plot(kind='scatter',x='DOY',y='Data',ax=ax[1])
sns.regplot(data=Obs_Rice_LAI_df, x='DOY',y='LAI',
            ax=ax,order=2,line_kws=dict(color="k"),color='red')
sns.regplot(data=wheat_lai_data, x='DOY',y='Data',
            ax=ax,order=2,line_kws=dict(color="k"),color='blue')
ax.set_xlim([0,365+182])
ax.set_xlabel('Month', fontsize='x-large')
# ax[0].set_xlim([0,366])
ax.set_ylim([-0.1, 12])
ax.set_yticks(np.arange(0,13,2))
ax.set_yticklabels(np.arange(0,13,2), fontsize='x-large')
ax.set_ylabel('LAI $(m^{2}/m^{2})$', fontsize='x-large')
ax.set_xticks([0, 60, 121, 182, 244, 305, 366, 365+60, 365+121,365+182])
ax.set_xticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan', 'Mar', 'May', 'Jul'], fontsize='x-large')

legend_elements1= {Line2D([0], [0], marker='', color='blue', label='Wheat',alpha=0.6),
                   Line2D([0], [0], marker='', color='red', label='Rice', alpha=0.6),}
legend1 = ax.legend(handles=legend_elements1,loc = 'upper right',
            title='Crop', fontsize='x-large')
ax.add_artist(legend1)

plt.vlines(x=[152,274,305,365+91],ymin = 0, ymax = 12,colors='grey',ls='--')
plt.text((152+274)/2, 10, "Kharif Season", color='black', fontsize='x-large', ha='center')
plt.text((305+365+91)/2, 10, "Rabi Season", color='black', fontsize='x-large', ha='center')

plt.annotate(
    "", 
    xy=(152, 9.5), 
    xytext=(274, 9.5), 
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)

plt.annotate(
    "", 
    xy=(305, 9.5), 
    xytext=(365+91, 9.5), 
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)
#fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Observation_LAI_9Jan.png',
#            dpi=600, bbox_inches="tight")
#%% Plotting CLM5 default LAI and Irrigation against LAI observations and irrigation observations
fig,ax = plt.subplots(figsize=(12,5))
# Obs_Rice_LAI_data.plot(kind='scatter',x='DOY', y='Data', ax=ax[0])
# wheat_lai_data.plot(kind='scatter',x='DOY',y='Data',ax=ax[1])
sns.regplot(data=Obs_Rice_LAI_df, x='DOY',y='LAI',
            ax=ax,order=2,fit_reg=False,color='red')
sns.regplot(data=Obs_Wheat_LAI_data, x='DOY',y='Data',
            ax=ax,order=2,fit_reg=False,color='blue')

ax2 = ax.twinx()

### Irrigation data from "Extracting_Irrigtaion_Obs_CLM5_9Jan.py"
ax2.plot(np.arange(len(QI_w_Def)),QI_w_Def,color='blue',label='CLM5_Def (Wheat)')
ax2.plot(np.arange(len(QI_r_Def)),QI_r_Def,color='red',label='CLM5_Def (Rice)')
ax2.plot(np.arange(len(QI_w_Obs)),QI_w_Obs,color='k',label='Beimenas et al., 2016 (Wheat)')
ax2.plot(np.arange(len(QI_r_Obs)),QI_r_Obs,'k--',label='Beimenas et al., 2016 (Rice)')

ax2.set_ylim([0, 3])
ax2.set_ylabel('Irrigation (BCM/day)')
ax2.legend(title = 'Irrigation in',loc='upper left', fontsize='small', ncols=2)

ax.set_xlim([0,365])
ax.set_xlabel('Month')
# ax[0].set_xlim([0,366])
ax.set_ylim([-0.1, 14])
ax.set_ylabel('LAI $(m^{2}/m^{2})$')
ax.xaxis.set_minor_locator(MultipleLocator(30))
ax.set_xticks([0, 60, 121, 182, 244, 305, 366])
ax.set_xticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan'])
# ax.set_xticks([0, 60, 121, 182, 244, 305, 366, 365+60, 365+121,365+182])
# ax.set_xticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan', 'Mar', 'May', 'Jul'])

legend_elements1= {Line2D([0], [0], marker='o', markerfacecolor='blue',color='w', label='Wheat',alpha=0.6),
                   Line2D([0], [0], marker='o', markerfacecolor='red',color='w', label='Rice', alpha=0.6),}
legend1 = ax.legend(handles=legend_elements1,loc = 'upper right',
            title='Crop LAI', fontsize="small")
ax.add_artist(legend1)

ax.vlines(x=[152,274,305,91],ymin = 0, ymax = 14,colors='grey',ls='--')
ax.text((152+274)/2, 10, "Kharif Season", color='black', fontsize=12, ha='center')
ax.text((305+365)/2, 10, "Rabi", color='black', fontsize=12, ha='center')
ax.text((0+91)/2, 10, "Rabi", color='black', fontsize=12, ha='center')

ax.annotate(
    "", 
    xy=(152, 9.5), 
    xytext=(274, 9.5),
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)

ax.annotate(
    "", 
    xy=(305, 9.5), 
    xytext=(365, 9.5),
    arrowprops=dict(arrowstyle="->", color='grey', lw=1.5)
)

ax.annotate(
    "", 
    xy=(0, 9.5), 
    xytext=(91, 9.5),
    arrowprops=dict(arrowstyle="<-", color='grey', lw=1.5)
)
fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Observation_LAI_Irrigation_Against_CLM_Def_9Jan.png',
            dpi=600, bbox_inches="tight")

#%%
fig,ax = plt.subplots(figsize=(8,4))
# Obs_Rice_LAI_data.plot(kind='scatter',x='DOY', y='Data', ax=ax[0])
# wheat_lai_data.plot(kind='scatter',x='DOY',y='Data',ax=ax[1])
sns.regplot(data=Obs_Rice_LAI_data, x='DOY',y='Data',
            ax=ax,order=2,line_kws=dict(color="k"),color='red')
sns.regplot(data=wheat_lai_data, x='DOY',y='Data',
            ax=ax,order=2,line_kws=dict(color="k"),color='blue')
ax.set_xlim([0,365+182])
ax.set_xlabel('Month')
# ax[0].set_xlim([0,366])
ax.set_ylim([-0.1, 9])
ax.set_ylabel('LAI $(m^{2}/m^{2})$')
ax.set_xticks([0, 60, 121, 182, 244, 305, 366, 365+60, 365+121,365+182])
ax.set_xticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan', 'Mar', 'May', 'Jul'])

legend_elements1= {Line2D([0], [0], marker='', color='blue', label='Wheat',alpha=0.6),
                   Line2D([0], [0], marker='', color='red', label='Rice', alpha=0.6),}
legend1 = ax.legend(handles=legend_elements1,loc = 'upper right',
            title='Crop', fontsize="small")
ax.add_artist(legend1)

plt.vlines(x=[152,274,305,365+91],ymin = 0, ymax = 9,colors='grey',ls='--')
plt.text((152+274)/2, 8, "Kharif Season", color='black', fontsize=12, ha='center')
plt.text((305+365+91)/2, 8, "Rabi Season", color='black', fontsize=12, ha='center')

plt.annotate(
    "", 
    xy=(152, 7.5), 
    xytext=(274, 7.5), 
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)

plt.annotate(
    "", 
    xy=(305, 7.5), 
    xytext=(365+91, 7.5), 
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)
# fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Observation_LAI_6Jan.png',
#             dpi=600, bbox_inches="tight")
#%%
Obs_Rice_Yield_data['Decade'] = (Obs_Rice_Yield_data['Year'] // 10) * 10
Obs_Wheat_Yield_data['Decade'] = (Obs_Wheat_Yield_data['Year'] // 10) * 10

Rice_yield_data = (Obs_Rice_Yield_data.sort_values('Decade')).reset_index()
Wheat_yield_data = (Obs_Wheat_Yield_data.sort_values('Decade')).reset_index()

Rice_yield_data['Crop'] = 'Rice'
Wheat_yield_data['Crop'] = 'Wheat'

yield_data = pd.concat([Rice_yield_data,Wheat_yield_data], ignore_index=True)

yield_data = yield_data[yield_data['Decade'] >= 1970]

yield_data['Decade'] = pd.Categorical(yield_data['Decade'])

# Map Year to Decade Indices
decades = sorted(yield_data['Decade'].unique())
decade_to_index = {decade: idx for idx, decade in enumerate(decades)}

# Add a mapped index column for alignment
yield_data['Decade_Index'] = yield_data['Decade'].map(decade_to_index)
# gsl_data['Year_Index'] = gsl_data['Year'].apply(lambda y: decade_to_index[(y // 10) * 10])


yield_data['Year_Index'] = (yield_data['Year'] - 
                          yield_data['Year'].min())/ (yield_data['Year'].max() - 
                                                    yield_data['Year'].min()) * (5) # multiplying with 5 since max decades are 5
#%%
fig,ax = plt.subplots(figsize=(8,4))

sns.boxplot(x='Decade',y='Data',data=yield_data, 
            ax=ax,hue='Crop',palette={"Rice": "red", "Wheat": "blue"})#color='red')
# sns.boxplot(x='Decade',y='Data',data=Wheat_yield_data,ax=ax,color='blue')

ax.set_xlim([0.5, 5.5])
ax.set_ylabel('Yield (t/ha)')
fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Observation_Yield_6Jan.png',
            dpi=600, bbox_inches="tight")
#%%
Obs_Rice_GSL_data['Decade'] = (Obs_Rice_GSL_data['Year'] // 10) * 10
Obs_Wheat_GSL_data['Decade'] = (Obs_Wheat_GSL_data['Year'] // 10) * 10

Rice_GSL_data = (Obs_Rice_GSL_data.sort_values('Decade')).reset_index()
Wheat_GSL_data = (Obs_Wheat_GSL_data.sort_values('Decade')).reset_index()

Rice_GSL_data['Crop'] = 'Rice'
Wheat_GSL_data['Crop'] = 'Wheat'

Rice_GSL_data['Sow_DOY'] = Rice_GSL_data['Sowing_date'].dt.dayofyear
Wheat_GSL_data['Sow_DOY'] = Wheat_GSL_data['Sowing_date'].dt.dayofyear

Rice_GSL_data['Harv_DOY'] = Rice_GSL_data['Harvest_date'].dt.dayofyear
Wheat_GSL_data['Harv_DOY'] = Wheat_GSL_data['Harvest_date'].dt.dayofyear
# Wheat_GSL_data.drop(Wheat_GSL_data['Sow_DOY']<250)

gsl_data = pd.concat([Rice_GSL_data,Wheat_GSL_data], ignore_index=True)

gsl_data = gsl_data[gsl_data['Decade']<=2019]

gsl_data['Decade'] = pd.Categorical(gsl_data['Decade'])

# Map Year to Decade Indices
decades = sorted(gsl_data['Decade'].unique())
decade_to_index = {decade: idx for idx, decade in enumerate(decades)}

# Add a mapped index column for alignment
gsl_data['Decade_Index'] = gsl_data['Decade'].map(decade_to_index)
# gsl_data['Year_Index'] = gsl_data['Year'].apply(lambda y: decade_to_index[(y // 10) * 10])


gsl_data['Year_Index'] = (gsl_data['Year'] - 
                          gsl_data['Year'].min())/ (gsl_data['Year'].max() - 
                                                    gsl_data['Year'].min()) * (5) # multiplying with 5 since max decades are 5
#%%
fig,ax = plt.subplots(figsize=(8,4))

sns.boxplot(x='Decade',y='GSL',data=gsl_data, 
            ax=ax,hue='Crop',palette={"Rice": "red", "Wheat": "blue"})#color='red')
# sns.boxplot(x='Decade',y='Data',data=Wheat_yield_data,ax=ax,color='blue')
sns.lmplot(x='Year',y='GSL',data=gsl_data
          ,hue='Crop',palette={"Rice": "red", "Wheat": "blue"})

# ax.set_xlim([0.5, 5.5])
ax.set_ylabel('GSL (days)')
# fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Observation_Yield_6Jan.png',
#             dpi=600, bbox_inches="tight")
#%%
fig,ax = plt.subplots(figsize=(8,4))

g = sns.lmplot(x='Year', y='GSL', data=gsl_data, hue='Crop',
               col='Crop', height=3, aspect=1, palette={"Rice": "red", "Wheat": "blue"},
               )
ax.set_ylabel('Growing Season \n Length (days)')
def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['Year'], data['GSL'])
    ax = plt.gca()
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
    
    
g.map_dataframe(annotate)
plt.show()
#%% 
fig,ax = plt.subplots(figsize=(8,4))

sns.regplot(data=Wheat_GSL_data,x='Year',y='Sow_DOY') 

ax.set_ylim([274,365])
#%% 
fig,ax = plt.subplots(nrows=2,figsize=(8,8))

sns.regplot(ax = ax[0],data=gsl_data[gsl_data['Crop'] == 'Rice'], x='Year', y='Sow_DOY'
            ,order=1,line_kws=dict(color="red"),color='red')
sns.regplot(ax = ax[0],data=gsl_data[gsl_data['Crop'] == 'Wheat'], x='Year', y='Sow_DOY'
            ,order=1,line_kws=dict(color="blue"),color='blue')

sns.regplot(ax = ax[1],data=gsl_data[gsl_data['Crop'] == 'Rice'], x='Year', y='Harv_DOY'
            ,order=1,line_kws=dict(color="red"),color='red')
sns.regplot(ax = ax[1],data=gsl_data[gsl_data['Crop'] == 'Wheat'], x='Year', y='Harv_DOY'
            ,order=1,line_kws=dict(color="blue"),color='blue')

ax[0].set_yticks([0, 60, 121, 182, 244, 305, 366])
ax[0].set_yticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan'])

ax[0].set_ylabel('Sowing Day')

ax[1].set_yticks([0, 60, 121, 182, 244, 305, 366])
ax[1].set_yticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan'])

ax[1].set_ylabel('Harvest Day')
ax[0].text(0.01,0.9,'(a)',transform=ax[0].transAxes)
ax[1].text(0.01,0.9,'(b)',transform=ax[1].transAxes)

legend_elements1= {Line2D([0], [0], marker='', color='red', label='Rice', alpha=0.6),
                   Line2D([0], [0], marker='', color='blue', label='Wheat',alpha=0.6),}

legend1 = ax[0].legend(handles=legend_elements1,loc = 'lower left',
            title='Crop', fontsize="small")

fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Analysis_Observation_Sowing_Harvest_Dates_7Jan.png',
            dpi=600, bbox_inches="tight")
#%% Plotting lai and yield data
fig,ax = plt.subplots(nrows = 3, figsize=(8,12))

###### LAI
sns.regplot(data=Obs_Rice_LAI_df, x='DOY',y='LAI',
            ax=ax[0],order=2,line_kws=dict(color="k"),color='red')
sns.regplot(data=wheat_lai_data, x='DOY',y='Data',
            ax=ax[0],order=2,line_kws=dict(color="k"),color='blue')
ax[0].set_xlim([0,365+182])
ax[0].set_xlabel('Month')
# ax[0].set_xlim([0,366])
ax[0].set_ylim([-0.1, 9])
ax[0].set_ylabel('LAI $(m^{2}/m^{2})$')
ax[0].set_xticks([0, 60, 121, 182, 244, 305, 366, 365+60, 365+121,365+182])
ax[0].set_xticklabels(['Jan','Mar', 'May', 'Jul', 'Sep', 'Nov','Jan', 'Mar', 'May', 'Jul'])

legend_elements1= {Line2D([0], [0], marker='', color='red', label='Rice', alpha=0.6),
                   Line2D([0], [0], marker='', color='blue', label='Wheat',alpha=0.6),}

legend1 = ax[0].legend(handles=legend_elements1,loc = 'upper right',
            title='Crop', fontsize="small")

ax[0].add_artist(legend1)

ax[0].vlines(x=[152,274,305,365+91],ymin = 0, ymax = 9,colors='grey',ls='--')
ax[0].text((152+274)/2, 8, "Kharif Season",color='black', fontsize=12, ha='center')
ax[0].text((305+365+91)/2, 8, "Rabi Season", color='black', fontsize=12, ha='center')

ax[0].annotate(
    "", 
    xy=(152, 7.5), 
    xytext=(274, 7.5),
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)

ax[0].annotate(
    "", 
    xy=(305, 7.5), 
    xytext=(365+91, 7.5),
    arrowprops=dict(arrowstyle="<->", color='grey', lw=1.5)
)
######### Yield
sns.boxplot(x='Decade_Index',y='Data',data=yield_data,  fill=None, legend=False,
            ax=ax[1],hue='Crop',palette={"Rice": "red", "Wheat": "blue"},label=None)#color='red')
# sns.boxplot(x='Decade',y='Data',data=Wheat_yield_data,ax=ax,color='blue')
sns.regplot(data=yield_data[yield_data['Crop'] == 'Rice'], x='Year_Index',y='Data',
            ax=ax[1],order=1,line_kws=dict(color="Red"),color='red')
sns.regplot(data=yield_data[yield_data['Crop'] == 'Wheat'], x='Year_Index',y='Data',
            ax=ax[1],order=1,line_kws=dict(color="blue"),color='blue')

ax[1].set_xticks(np.arange(len(yield_data['Decade'].cat.categories)))
ax[1].set_xticklabels(yield_data['Decade'].cat.categories)

# ax[1].set_xlim([0.5, 5.5])
ax[1].set_ylabel('Yield (t/ha)')
ax[1].set_xlabel('Year')
# ax[1].legend(title='Crop',loc='upper right')

ax[0].text(0.01,0.9,'(a)',transform=ax[0].transAxes)
ax[1].text(0.01,0.9,'(b)',transform=ax[1].transAxes)
ax[2].text(0.01,0.9,'(c)',transform=ax[2].transAxes)
############ GSL
sns.boxplot(x='Decade_Index',y='GSL',data=gsl_data, fill=None,
            ax=ax[2],hue='Crop',palette={"Rice": "red", "Wheat": "blue"})#color='red')
# sns.boxplot(x='Decade',y='Data',data=Wheat_yield_data,ax=ax,color='blue')
sns.regplot(data=gsl_data[gsl_data['Crop'] == 'Rice'], x='Year_Index',y='GSL',
            ax=ax[2],order=1,line_kws=dict(color="Red"),color='red')
sns.regplot(data=gsl_data[gsl_data['Crop'] == 'Wheat'], x='Year_Index',y='GSL',
            ax=ax[2],order=1,line_kws=dict(color="blue"),color='blue')

ax[2].set_xticks(np.arange(len(gsl_data['Decade'].cat.categories)))
ax[2].set_xticklabels(gsl_data['Decade'].cat.categories)

# ax.set_xlim([0.5, 5.5])
ax[2].set_ylabel('Growing season \n length (days)')
ax[2].set_xlabel('Year')

ax[2].legend(title='Crop',loc='upper right',)

####### annotate trend (p_value) and slope 


fig.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/Analysis_Observation_9Jan.png',
            dpi=600, bbox_inches="tight")
#%% Trend analysis
[gsl_rice_trend, gsl_rice_h, gsl_rice_p, gsl_rice_z, 
 gsl_rice_Tau, gsl_rice_s, gsl_rice_var_s, 
 gsl_rice_slope, gsl_rice_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Rice']['GSL'], 
                                                        alpha=0.1)

[gsl_wheat_trend, gsl_wheat_h, gsl_wheat_p, gsl_wheat_z, 
 gsl_wheat_Tau, gsl_wheat_s, gsl_wheat_var_s, 
 gsl_wheat_slope, gsl_wheat_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Wheat']['GSL'], 
                                                          alpha=0.1)

[sow_rice_trend, sow_rice_h, sow_rice_p, sow_rice_z, 
 sow_rice_Tau, sow_rice_s, sow_rice_var_s, 
 sow_rice_slope, sow_rice_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Rice']['Sow_DOY'], 
                                                        alpha=0.1)

[sow_wheat_trend, sow_wheat_h, sow_wheat_p, sow_wheat_z, 
 sow_wheat_Tau, sow_wheat_s, sow_wheat_var_s, 
 sow_wheat_slope, sow_wheat_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Wheat']['Sow_DOY'], 
                                                          alpha=0.1)

[harv_rice_trend, harv_rice_h, harv_rice_p, harv_rice_z, 
 harv_rice_Tau, harv_rice_s, harv_rice_var_s, 
 harv_rice_slope, harv_rice_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Rice']['Harv_DOY'], 
                                                          alpha=0.1)

[harv_wheat_trend, harv_wheat_h, harv_wheat_p, harv_wheat_z, 
 harv_wheat_Tau, harv_wheat_s, harv_wheat_var_s, 
 harv_wheat_slope, harv_wheat_intercept] = mk.original_test(gsl_data[gsl_data['Crop'] == 'Wheat']['Harv_DOY'], 
                                                            alpha=0.1)

[yield_rice_trend, yield_rice_h, yield_rice_p, yield_rice_z, 
 yield_rice_Tau, yield_rice_s, yield_rice_var_s, 
 yield_rice_slope, yield_rice_intercept] = mk.original_test(yield_data[yield_data['Crop'] == 'Rice']['Data'], 
                                                            alpha=0.1)

[yield_wheat_trend, yield_wheat_h, yield_wheat_p, yield_wheat_z, 
 yield_wheat_Tau, yield_wheat_s, yield_wheat_var_s, 
 yield_wheat_slope, yield_wheat_intercept] = mk.original_test(yield_data[yield_data['Crop'] == 'Wheat']['Data'], 
                                                              alpha=0.1)
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
#%%

p=sns.regplot(x=obs_data, y=clm_data, fit_reg=True,ci=95, 
            dropna=True,line_kws=dict(color="k"),scatter_kws=dict(color='k',s=25),
            robust=True,ax=axes[i_row,i_col])
slope, intercept, r, p, sterr = stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                       y=p.get_lines()[0].get_ydata())



