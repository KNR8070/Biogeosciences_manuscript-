#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:23:20 2024

@author: knreddy
"""
#%% Load Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas
import numpy as np
import matplotlib as mpl
from cmcrameri import cm
import pymannkendall as mk
import string
import roman
#%% Clip data for Indian region
def clip_data(Spatial_data,Region):
    Spatial_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    Spatial_data.rio.write_crs("epsg:4326", inplace=True)
    clip_Spatial_data = Spatial_data.rio.clip(Region.geometry, India.crs, drop=True)
    return clip_Spatial_data
#%% Trend in max. LH
def find_trend_pval(X):     
    nt, nlat, nlon = X.shape
    ngrd = nlon*nlat
    X_grd = X.reshape((nt, ngrd), order='F')
    X_rate = np.empty((ngrd,1))
    X_rate[:,:]=np.nan
    X_pvalue = np.empty((ngrd,1))
    X_pvalue[:,:]=np.nan
    
    for i in range(ngrd): 
        y = X_grd[:,i]   
        if(not np.ma.is_masked(y)):         
            trend, h, p_value, z, Tau, s, var_s, slope, intercept = mk.original_test(X_grd[:,i],alpha=0.05)
            X_rate[i,0] = slope
            if p_value<0.01:
                X_pvalue[i,0] = p_value
        
    X_rate = X_rate.reshape((nlat,nlon), order='F')
    X_pvalue = X_pvalue.reshape((nlat,nlon), order='F')
    
    return X_rate, X_pvalue
#%% Moving averages
def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')
#%% cummulative Moving average
def moving_average2(data_set, periods):
    i = 1
    # Initialize an empty list to store cumulative moving
    # averages
    moving_averages = []
     
    # Store cumulative sums of array in cum_sum array
    cum_sum = np.cumsum(data_set);
     
    # Loop through the array elements
    while i <= len(data_set):
     
        # Calculate the cumulative average by dividing
        # cumulative sum by number of elements till 
        # that position
        window_average = round(cum_sum[i-1] / i, 2)
         
        # Store the cumulative average of
        # current window in moving average list
        moving_averages.append(window_average)
         
        # Shift window to right by one position
        i += 1
    return moving_averages
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

data_CTRL = xr.open_dataset(wrk_dir+'/CLM5_CTRL_1970_2014_CropFluxData_11-Apr-2024.nc')
data_CO2 = xr.open_dataset(wrk_dir+'/CLM5_S_CO2_1970_2014_CropFluxData_11-Apr-2024.nc')
data_Irrig = xr.open_dataset(wrk_dir+'/CLM5_S_Irrig_1970_2014_CropFluxData_11-Apr-2024.nc')
data_Clim = xr.open_dataset(wrk_dir+'/CLM5_S_Clim_1970_2014_CropFluxData_11-Apr-2024.nc')
data_NFert = xr.open_dataset(wrk_dir+'/CLM5_S_NFert_1970_2014_CropFluxData_11-Apr-2024.nc')

mask_data = xr.open_dataset(mask_dir+'landuse.timeseries_360x720cru_hist_78pfts_CMIP6_simyr1850-2015_India_c221122.nc')

lat = data_CTRL['lat']
lon = data_CTRL['lon']
year = data_CTRL['year']

India = geopandas.read_file(shp_dir+'/india_administrative_outline_boundary.shp', crs="epsg:4326")

#%% Extract data (Units are gC/m2/year)
exp = ['CTRL','Clim','CO2','NFert','Irrig']
var = ['GPP','AR','NPP']
crop = ['SW','Rice']

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            output_name = i_exp+'_'+i_var+'_'+i_crop
            dataset_name = 'data_'+i_exp          
            if i_crop == 'SW':
                data_var_name = i_var+'_Wheat'
                locals()[output_name] = eval(dataset_name)[data_var_name]
            else:
                data_var_name = i_var+'_Rice'
                locals()[output_name] = eval(dataset_name)[data_var_name]

pct_crop = np.array(mask_data['PCT_CROP'])
pct_cft = np.array(mask_data['PCT_CFT'])
p_c_test = mask_data['PCT_CFT']
area = np.array(mask_data['AREA'])

pct_cft_ = xr.DataArray(pct_cft,
    coords=dict(
        time = p_c_test.coords['time'],
        cft = p_c_test.coords['cft'],
        lat = CTRL_GPP_Rice.coords['lat'],
        lon = CTRL_GPP_Rice.coords['lon']))

pct_crop_ = xr.DataArray(pct_crop,
    coords=dict(
        time = p_c_test.coords['time'],
        lat = CTRL_GPP_Rice.coords['lat'],
        lon = CTRL_GPP_Rice.coords['lon']))

area_ = xr.DataArray(area,
    coords=dict(
        lat = CTRL_GPP_Rice.coords['lat'],
        lon = CTRL_GPP_Rice.coords['lon']),
    attrs=dict(
        description="area of grid cell",
        units="km^2"))

cft = np.array(mask_data['cft'])
pct_cft2 = pct_cft_[-46:-1,:,:,:] # pct_cft of period 1970 to 2014
pct_crop2 = pct_crop_[-46:-1,:,:]

pct_crop2_ = clip_data(pct_crop2, India)
#%% Prepare SW and rice mask
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

pct_cft_sw_mask = np.ma.masked_less_equal(pct_cft_sw_, 1)
pct_cft_rice_mask = np.ma.masked_less_equal(pct_cft_rice_, 1)

# pct_cft_sw_mask = np.ma.masked_greater(pct_cft_sw_, 100)
# pct_cft_rice_mask = np.ma.masked_greater(pct_cft_rice_, 100)

area_SW = (pct_cft_sw_mask*10**4)*clipped_area # units are m2
area_Rice = (pct_cft_rice_mask*10**4)*clipped_area
#%% Trends in crop harvested area for wheat and rice
area_sw_1970_2014 = (pct_cft2[:,((cft==19) | (cft==20)),:,:].mean(dim='cft',skipna=True))*(pct_crop2)*area_
# area_sw_1970_2014 = pct_cft2[:,cft==19,:,:]*(pct_crop2)*area_
area_sw_1970_2014_latmean = area_sw_1970_2014.mean(dim='lat',skipna=True)
area_sw_1970_2014_mean = (area_sw_1970_2014_latmean.mean(dim='lon',skipna=True))*100 # units in ha

area_rice_1970_2014 = (pct_cft2[:,((cft==61) | (cft==62)),:,:].mean(dim='cft',skipna=True))*(pct_crop2)*area_
# area_rice_1970_2014 = pct_cft2[:,cft==61,:,:]*(pct_crop2)*area_
area_rice_1970_2014_latmean = area_rice_1970_2014.mean(dim='lat',skipna=True)
area_rice_1970_2014_mean = (area_rice_1970_2014_latmean.mean(dim='lon',skipna=True))*100 # units in ha
#%% trends in area
area_sw_1970_2014_ = clip_data(area_sw_1970_2014, India)
area_rice_1970_2014_ = clip_data(area_rice_1970_2014, India)

SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)

area_sw_1970_2014_masked = np.ma.masked_array(area_sw_1970_2014_,mask=SW_mask)
area_rice_1970_2014_masked = np.ma.masked_array(area_rice_1970_2014_,mask=Rice_mask)

area_sw_trend = find_trend_pval(area_sw_1970_2014_masked/(10**4))
area_rice_trend = find_trend_pval(area_rice_1970_2014_masked/(10**4))
#%% plotting trend in area
lonx, latx = np.meshgrid(area_sw_1970_2014_.lon, area_sw_1970_2014_.lat)
fig1, axes = plt.subplots(nrows=1, ncols=2, dpi=600,sharex=True,sharey=True,figsize=(8,6),layout='constrained')

fsize=14

vmn = -2
vmx = 2
Num = 18

plotid_x = 0.02
plotid_y = 0.9

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

bounds = np.arange(vmn,vmx+0.2,0.2)
norm = mpl.colors.BoundaryNorm(bounds, Num+2)
####################### Wheat #################################
India.plot(facecolor='gray',edgecolor='black',ax=axes[0])

X_rate = area_sw_trend[0]
X_pvalue = area_sw_trend[1]
im1=axes[0].contourf(area_sw_1970_2014_.lon,area_sw_1970_2014_.lat,X_rate,
                               levels=bounds, cmap=cmap2,vmin=vmn,vmax=vmx)    
sig_area   = np.where(X_pvalue < 0.05)
im2=axes[0].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.4)

axes[0].set_title('Wheat', fontsize=fsize)
axes[0].set_xlabel('Longitude', fontsize=fsize)
axes[0].set_ylabel('Latitude', fontsize=fsize)
axes[0].set_xticks([70, 80, 90])
axes[0].set_xticklabels([70,80,90], fontsize=fsize)
axes[0].set_yticks([10,20,30])
axes[0].set_yticklabels([10,20,30], fontsize=fsize)

panel_no = '('+list(string.ascii_lowercase)[0]+')'
axes[0].text(plotid_x,plotid_y, panel_no,fontsize=fsize,transform=axes[0].transAxes)
####################### Rice ####################################
India.plot(facecolor='gray',edgecolor='black',ax=axes[1])

X_rate = area_rice_trend[0]
X_pvalue = area_rice_trend[1]
im1=axes[1].contourf(area_sw_1970_2014_.lon,area_sw_1970_2014_.lat,X_rate,
                               levels=bounds, cmap=cmap2,vmin=vmn,vmax=vmx)    
sig_area   = np.where(X_pvalue < 0.05)
im2=axes[1].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.5,c='k',alpha=0.4)
# im2=axes[1].add_patch(Rectangle(lonx[sig_area],latx[sig_area],hatch='/',fill=False))

axes[1].set_title('Rice',fontsize=fsize)
axes[1].set_xlabel('Longitude',fontsize=fsize)

axes[1].set_xticks([70, 80, 90])
axes[1].set_xticklabels([70,80,90], fontsize=fsize)
axes[1].set_yticks([10,20,30])
axes[1].set_yticklabels([10,20,30], fontsize=fsize)
panel_no = '('+list(string.ascii_lowercase)[1]+')'
axes[1].text(plotid_x,plotid_y, panel_no,fontsize=fsize,transform=axes[1].transAxes)

cbar1 = fig1.colorbar(im1,ax=axes[1],shrink=0.5,cmap=cmap2,norm=norm)
cbar1.set_label('Mha/year',fontsize=fsize+2)
# cbar1.set_lims(GPP_vmn,GPP_vmx)
cbar1.set_ticks(np.linspace(vmn,vmx,5))
cbar1.set_ticklabels(np.linspace(vmn,vmx,5),fontsize=fsize-1)

svfig_dir = '/Users/knreddy/Documents/PhD_Thesis/Figures/'

#fig1.savefig(svfig_dir+'Increase_in_Crop_Area_24July.png', 
#               dpi=600, bbox_inches="tight")
#%% Clip all spatial data for Indian region
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_name = i_exp+'_'+i_var+'_'+i_crop
            output_name = 'clipped_'+input_name
            locals()[output_name] = clip_data(eval(input_name),India)
#%% masking for pft regions
SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_var_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop
            output_var_name = input_var_name+'_masked'
            if i_crop == 'SW':
                locals()[output_var_name] = np.ma.masked_array(eval(input_var_name),mask=SW_mask)
            else:
                locals()[output_var_name] = np.ma.masked_array(eval(input_var_name),mask=Rice_mask)
#%% taking spatial means and std
d_s = eval(output_name).shape

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
           input_3_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_masked'
           output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean'
           output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std'
           
           locals()[output_3_name1] = np.nanmean((eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))/1000,axis=1)
           locals()[output_3_name2] = np.nanstd((eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))/1000,axis=1) # units in kgC/m2/yr
#%% taking spatial means and std for CTRL and impact of exps 
d_s = eval(output_name).shape

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            if i_exp == 'CTRL':
               input_3_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_masked'
               output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean'
               output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std'
               
               locals()[output_3_name1] = np.nanmean((eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))/1000,axis=1)
               locals()[output_3_name2] = np.nanstd((eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))/1000,axis=1) # units in kgC/m2/yr
               
            else:
                input_CTRL_name = 'clipped_CTRL_'+i_var+'_'+i_crop+'_masked'
                input_3_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_masked'
                output_3_name1 = i_exp+'_'+i_var+'_'+i_crop+'_mean'
                output_3_name2 = i_exp+'_'+i_var+'_'+i_crop+'_std'
                
                impact_data = (eval(input_CTRL_name).reshape(d_s[0],d_s[1]*d_s[2])) - (eval(input_3_name).reshape(d_s[0],d_s[1]*d_s[2]))
                locals()[output_3_name1] = np.nanmean((impact_data)/1000,axis=1)
                locals()[output_3_name2] = np.nanstd((impact_data)/1000,axis=1)
#%% Run cells from Compare_Experiments_GPP_AR_NPP_11_Apr.py
colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]

fsize=14
plotid_x = 0.02
plotid_y = 0.85

slope_pos_x = 0.45
slope_pos_y = 0.85

p_pos_x = 0.52
p_pos_y = 0.8

GPP_lims = [-0.3,3.5]
AR_lims = [-0.3,1.75]
NPP_lims = [-0.3,1.75]

GPP_ticks = [0,1.0,2.0,3.0]
AR_ticks = [0,0.5,1.0,1.5]
NPP_ticks = [0,0.5,1.0,1.5]

GPP_ticklabels = [0,1.0,2.0,3.0]
AR_ticklabels = [0,0.5,1.0,1.5]
NPP_ticklabels = [0,0.5,1.0,1.5]

GPP_lims2 = [-0.3,2.0]
AR_lims2 = [-0.3,1.0]
NPP_lims2 = [-0.3,1.0]

GPP_ticks2 = [0,1.0,2.0]
AR_ticks2 = [0,0.5,1.0]
NPP_ticks2 = [0,0.5,1.0]

GPP_ticklabels2 = [0,1.0,2.0]
AR_ticklabels2 = [0,0.5,1.0]
NPP_ticklabels2 = [0,0.5,1.0]

x_new = np.arange(0,45)
year = np.arange(1970,2015)
fig1, axes = plt.subplots(nrows=3, ncols=5, dpi=600,sharex=True,figsize=(15,7),layout='constrained')

exp = ['CTRL', 'Clim', 'CO2', 'NFert', 'Irrig']
var = ['GPP','AR','NPP']
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
        mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean'
        std_data = i_exp+'_'+i_var+'_'+i_crop+'_std'
        std1 = eval(mean_data) - eval(std_data)
        std2 = eval(mean_data) + eval(std_data)
        [trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((eval(mean_data)),alpha=0.05)
        # [slope2,intercept2] = mk.sens_slope((eval(mean_data)))
        y_new = x_new*slope+intercept
        # plot_data = 
        axes[i_row,i_col].plot(year,(eval(mean_data)), color=color_line,linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(year,std1,std2,facecolor=color_line,alpha=0.25)
        if i_exp == 'CTRL':
            axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            axes[i_row,i_col].set_ylim(eval(i_var+'_lims'))
            axes[i_row,i_col].set_yticks(eval(i_var+'_ticks'))
        else:
            if slope >0:
                axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            else:
                axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            axes[i_row,i_col].set_ylim(eval(i_var+'_lims2'))
            axes[i_row,i_col].set_yticks(eval(i_var+'_ticks2'))
        ## title
        if i_col == 0:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[i_row,i_col].set_ylabel(i_var+' \n (kgC/m\u00b2/year)',fontsize=fsize)
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels'), fontsize=fsize)
        elif i_col==1:
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels2'), fontsize=fsize)
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

# Add rotated bold 'A.' manually
axes[0, 0].text(-0.45, 0.5, '(a)', transform=axes[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[1, 0].text(-0.45, 0.5, '(b)', transform=axes[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[2, 0].text(-0.45, 0.5, '(c)', transform=axes[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')


fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/'+i_crop+'_Trends_carbonfluxes_Sens_Slope_16Apr.png',
             dpi=600,bbox_inches="tight")
#%% Plotting moving average mean data
colors = [cm.vik(180), cm.vik(220), cm.batlowK(80), cm.vik(20)]

fsize=14
plotid_x = 0.02
plotid_y = 0.85

slope_pos_x = 0.45
slope_pos_y = 0.85

p_pos_x = 0.52
p_pos_y = 0.8

GPP_lims = [-0.3,3.5]
AR_lims = [-0.3,1.75]
NPP_lims = [-0.3,1.75]

GPP_ticks = [0,1.0,2.0,3.0]
AR_ticks = [0,0.5,1.0,1.5]
NPP_ticks = [0,0.5,1.0,1.5]

GPP_ticklabels = [0,1.0,2.0,3.0]
AR_ticklabels = [0,0.5,1.0,1.5]
NPP_ticklabels = [0,0.5,1.0,1.5]

GPP_lims2 = [-0.3,2.0]
AR_lims2 = [-0.3,1.0]
NPP_lims2 = [-0.3,1.0]

GPP_ticks2 = [0,1.0,2.0]
AR_ticks2 = [0,0.5,1.0]
NPP_ticks2 = [0,0.5,1.0]

GPP_ticklabels2 = [0,1.0,2.0]
AR_ticklabels2 = [0,0.5,1.0]
NPP_ticklabels2 = [0,0.5,1.0]

x_new = np.arange(0,45)
year = np.arange(1970,2015)
fig1, axes = plt.subplots(nrows=3, ncols=5, dpi=600,sharex=True,figsize=(16,7),layout='constrained')

exp = ['CTRL', 'Clim', 'CO2', 'NFert', 'Irrig']
var = ['GPP','AR','NPP']
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
        mean_data = i_exp+'_'+i_var+'_'+i_crop+'_mean'
        std_data = i_exp+'_'+i_var+'_'+i_crop+'_std'
        std1 = eval(mean_data) - eval(std_data)
        std2 = eval(mean_data) + eval(std_data)
        [trend, h, p_value, z, Tau, s, var_s, slope, intercept] = mk.original_test((eval(mean_data)),alpha=0.05)
        # [slope2,intercept2] = mk.sens_slope((eval(mean_data)))
        y_new = x_new*slope+intercept
        # plot_mean_data = np.array([np.nanmean((eval(mean_data)[:10])),np.nanmean((eval(mean_data)[10:20])),
        #              np.nanmean((eval(mean_data)[20:30])),np.nanmean((eval(mean_data)[30:40])),
        #              np.nanmean((eval(mean_data)[40:]))])
        # plot_std_data = np.array([np.nanmean((eval(std_data)[:10])),np.nanmean((eval(std_data)[10:20])),
        #              np.nanmean((eval(std_data)[20:30])),np.nanmean((eval(std_data)[30:40])),
        #              np.nanmean((eval(std_data)[40:]))])
        plot_mean_data = moving_average(eval(mean_data), 3)
        plot_std_data = moving_average(eval(std_data), 3)
        plot_std1_data = plot_mean_data - plot_std_data
        plot_std2_data = plot_mean_data + plot_std_data
        years_data = moving_average(year,3)
        # decades = [1970,1980,1990,2000,2010]
        
        [trend2, h2, p_value2, z2, Tau2, s2, var_s2, slope2, intercept2] = mk.original_test(plot_mean_data,alpha=0.05)

        x_new = np.arange(0,len(years_data))
        y_new = x_new*slope2 + intercept2
        
        axes[i_row,i_col].plot(years_data,plot_mean_data, color=color_line,linewidth=3,alpha=0.6)
        axes[i_row,i_col].fill_between(years_data,plot_std1_data,plot_std2_data,facecolor=color_line,alpha=0.25)
        axes[i_row,i_col].plot(years_data,y_new,linestyle="dashdot",color='k',linewidth=1.5)
        if i_exp == 'CTRL':
            # axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            axes[i_row,i_col].set_ylim(eval(i_var+'_lims'))
            axes[i_row,i_col].set_yticks(eval(i_var+'_ticks'))
        else:
            # if slope >0:
            #     # axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            # else:
                # axes[i_row,i_col].plot(year,y_new,linestyle="dashdot",color='k',linewidth=1.5)
            axes[i_row,i_col].set_ylim(eval(i_var+'_lims2'))
            axes[i_row,i_col].set_yticks(eval(i_var+'_ticks2'))
        ## title
        if i_col == 0:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[i_row,i_col].set_ylabel(i_var+' \n (kgC/m\u00b2/year)',fontsize=fsize)
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels'), fontsize=fsize)
        elif i_col==1:
            axes[i_row,i_col].set_yticklabels(eval(i_var+'_ticklabels2'), fontsize=fsize)
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
        axes[i_row,i_col].set_xticks([1975,1985,1995,2005,2015])
        axes[i_row,i_col].set_xticklabels([1975,1985,1995,2005,2015], fontsize=fsize-2)#, rotation=15)
        #panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        #axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize+1,transform=axes[i_row,i_col].transAxes)
        # axes[i_row,i_col].text(plotid_x,plotid_y-0.1, trend,fontsize=fsize-2,transform=axes[i_row,i_col].transAxes)
        if p_value < 0.05:
            slope_text = '$slope='+'{:.3f}^*$'.format(slope)
        else:
            slope_text = '$slope='+'{:.3f}$'.format(slope)
        if i_col == 0:
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

# Add rotated bold 'A.' manually
axes[0, 0].text(-0.45, 0.5, '(a)', transform=axes[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[1, 0].text(-0.45, 0.5, '(b)', transform=axes[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[2, 0].text(-0.45, 0.5, '(c)', transform=axes[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')


fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/'+i_crop+'_Trends_carbonfluxes_Sens_Slope_16Apr.png',
             dpi=600,bbox_inches="tight")
#%%considering the impact with area
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            area = eval('area_'+i_crop)
            pct_cft_data = eval('pct_cft_'+i_crop.lower()+'_')
            input_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_masked'
            output_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_area'
            locals()[output_name] = eval(input_name)*(area*10**-9)*(pct_cft_data/100)*(pct_crop2_/100)
#%% Find trend and pvalue at regional scale
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_area'
            out1 = i_exp+'_'+i_var+'_'+i_crop+'_rate'
            locals()[out1] = find_trend_pval(eval(input_name))
#%% Find trend and pvalue at regional scale impact of drivers
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:            
            out1 = i_exp+'_'+i_var+'_'+i_crop+'_rate'
            if i_exp == 'CTRL':
                input_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_area'
                locals()[out1] = find_trend_pval(eval(input_name))
            else:
                CTRL_input_name = 'clipped_CTRL_'+i_var+'_'+i_crop+'_area'
                input_name = 'clipped_'+i_exp+'_'+i_var+'_'+i_crop+'_area'
                locals()[out1] = find_trend_pval(eval(CTRL_input_name) - eval(input_name))
#%% Plotting trend and p value
lonx, latx = np.meshgrid(clipped_CTRL_GPP_SW.lon, clipped_CTRL_GPP_SW.lat)
fig1, axes = plt.subplots(nrows=3, ncols=5, dpi=600,sharex=True,sharey=True,figsize=(16,10),layout='constrained')

fsize=14

Num = 10
plotid_x = 0.02
plotid_y = 0.9

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)

# vmn = -0.5
# vmx = 0.5
# bounds = np.linspace(vmn, vmx, Num+1)
# norm = mpl.colors.BoundaryNorm(bounds, Num+2)

# GPP_vmn = -100
# GPP_vmx = 100
GPP_vmn = -15
GPP_vmx = 15
GPP_bounds = np.arange(GPP_vmn,GPP_vmx+0.5,3)
GPP_norm = mpl.colors.BoundaryNorm(GPP_bounds, Num+2)

# AR_vmn = -40
# AR_vmx = 40
AR_vmn = -5
AR_vmx = 5
AR_bounds = np.arange(AR_vmn, AR_vmx+0.5, 1)
AR_norm = mpl.colors.BoundaryNorm(AR_bounds, Num+2)

NPP_vmn = -10
NPP_vmx = 10
NPP_bounds = np.arange(NPP_vmn, NPP_vmx, 2)
NPP_norm = mpl.colors.BoundaryNorm(NPP_bounds, Num+2)

exp = ['CTRL', 'Clim', 'CO2', 'NFert', 'Irrig']
var = ['GPP','AR','NPP']
crop = ['SW','Rice']
i_crop = crop[0]
# var in rows and exp in cols
for i_row,i_var in enumerate(var):
    for i_col,i_exp in enumerate(exp):
        if i_var == 'GPP':
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            if i_exp == 'CTRL':
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im1=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                               levels=GPP_bounds, cmap=cmap2,vmin=GPP_vmn,
                                               vmax=GPP_vmx)    
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
            else:
                input_name_CTRL = 'CTRL_'+i_var+'_'+i_crop+'_rate'
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate_CTRL = eval(input_name_CTRL)[0]
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im1=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate_CTRL-X_rate,
                                               levels=GPP_bounds, cmap=cmap2,vmin=GPP_vmn,
                                               vmax=GPP_vmx)   
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
        elif i_var == 'AR':
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            if i_exp == 'CTRL':
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im3=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                               levels=AR_bounds, cmap=cmap2,vmin=AR_vmn,vmax=AR_vmx)    
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
            else:
                input_name_CTRL = 'CTRL_'+i_var+'_'+i_crop+'_rate'
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate_CTRL = eval(input_name_CTRL)[0]
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im3=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate_CTRL-X_rate,
                                               levels=AR_bounds, cmap=cmap2,vmin=AR_vmn,vmax=AR_vmx)   
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
        else:
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            if i_exp == 'CTRL':
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im4=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                               levels=NPP_bounds, cmap=cmap2,vmin=NPP_vmn,
                                               vmax=NPP_vmx)    
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
            else:
                input_name_CTRL = 'CTRL_'+i_var+'_'+i_crop+'_rate'
                input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
                X_rate_CTRL = eval(input_name_CTRL)[0]
                X_rate = eval(input_name)[0]
                X_pvalue = eval(input_name)[1]
                im4=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate_CTRL-X_rate,
                                               levels=NPP_bounds, cmap=cmap2,vmin=NPP_vmn,
                                               vmax=NPP_vmx)   
                sig_area   = np.where(X_pvalue < 0.01)
                im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
            
        ## title
        if i_col == 0:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
        ## y label   
        if i_col == 0:
            axes[0,i_col].set_ylabel('GPP \n\n Latitude',fontsize=fsize+1)
            axes[1,i_col].set_ylabel('AR \n\n Latitude',fontsize=fsize+1)
            axes[2,i_col].set_ylabel('NPP \n\n Latitude',fontsize=fsize+1)
        else:
            axes[i_row,i_col].set_ylabel('')
        ## x label
        if i_row == 2:
            axes[i_row,i_col].set_xlabel('Longitude',fontsize=fsize)
        else:
            axes[i_row,i_col].set_xlabel('')
        ## ticklabels
        axes[i_row,i_col].set_xticks([70, 80, 90])
        axes[i_row,i_col].set_xticklabels([70,80,90], fontsize=fsize)
        axes[i_row,i_col].set_yticks([10,20,30])
        axes[i_row,i_col].set_yticklabels([10,20,30], fontsize=fsize)
        panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize-3,transform=axes[i_row,i_col].transAxes)

# Add rotated bold 'A.' manually
axes[0, 0].text(-0.45, 0.5, '(a)', transform=axes[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[1, 0].text(-0.45, 0.5, '(b)', transform=axes[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[2, 0].text(-0.45, 0.5, '(c)', transform=axes[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')

## colorbar at last column
cbar1 = fig1.colorbar(im1,ax=axes[0,4],shrink=0.8,cmap=cmap2,norm=GPP_norm)
cbar1.set_label('GgC/year',fontsize=fsize+2)
# cbar1.set_lims(GPP_vmn,GPP_vmx)
cbar1.set_ticks(np.linspace(GPP_vmn,GPP_vmx,5))

cbar2 = fig1.colorbar(im3,ax=axes[1,4],shrink=0.8,cmap=cmap2,norm=AR_norm)
cbar2.set_label('GgC/year',fontsize=fsize+2)
# cbar2.set_lims(AR_vmn,AR_vmx)
cbar2.set_ticks(np.linspace(AR_vmn,AR_vmx,5))

cbar3 = fig1.colorbar(im4,ax=axes[2,4],shrink=0.8,cmap=cmap2,norm=NPP_norm)
cbar3.set_label('GgC/year',fontsize=fsize+2)
# cbar3.set_lims(NPP_vmn,NPP_vmx)
cbar3.set_ticks(np.linspace(NPP_vmn,NPP_vmx,5))

fig1.savefig('/Users/knreddy/Documents/PhD_Thesis/Figures/'+i_crop+
             '_SpatialTrends_Carbonfluxes_AllExperiments_16Apr.png',
             dpi=600,bbox_inches="tight")
#%% Plotting trend and p value with impact of drivers
lonx, latx = np.meshgrid(clipped_CTRL_GPP_SW.lon, clipped_CTRL_GPP_SW.lat)
fig1, axes = plt.subplots(nrows=3, ncols=5, dpi=600,sharex=True,sharey=True,figsize=(16,10),layout='constrained')

fsize=16


plotid_x = 0.02
plotid_y = 0.9

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num+1)



exp = ['CTRL', 'Clim', 'CO2', 'NFert', 'Irrig']
var = ['GPP','AR','NPP']
crop = ['SW','Rice']
i_crop = crop[1]
if i_crop == 'Rice':
    GPP_vmn = -25
    GPP_vmx = 25
    GPP_Num = 18
    GPP_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=GPP_Num+1)
    GPP_bounds = np.arange(GPP_vmn,GPP_vmx+2.5,2.5)
    GPP_norm = mpl.colors.BoundaryNorm(GPP_bounds, len(GPP_bounds)+1)
    
    AR_vmn = -10
    AR_vmx = 10
    AR_Num = 18
    AR_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=AR_Num+1)
    AR_bounds = np.arange(AR_vmn,AR_vmx+1,1)
    AR_norm = mpl.colors.BoundaryNorm(AR_bounds, len(AR_bounds)+1)
    
    NPP_vmn = -15
    NPP_vmx = 15
    NPP_Num = 18
    NPP_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=NPP_Num+1)
    NPP_bounds = np.arange(NPP_vmn,NPP_vmx+1.5,1.5)
    NPP_norm = mpl.colors.BoundaryNorm(NPP_bounds, len(NPP_bounds)+1)
else:
    GPP_vmn = -6
    GPP_vmx = 6
    GPP_Num = 18
    GPP_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=GPP_Num+1)
    GPP_bounds = np.arange(GPP_vmn,GPP_vmx+0.6,0.6)
    GPP_norm = mpl.colors.BoundaryNorm(GPP_bounds, len(GPP_bounds)+1)
    
    AR_vmn = -2
    AR_vmx = 2
    AR_Num = 18
    AR_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=AR_Num+1)
    AR_bounds = np.arange(AR_vmn,AR_vmx+0.2,0.2)
    AR_norm = mpl.colors.BoundaryNorm(AR_bounds, len(AR_bounds)+1)
    
    NPP_vmn = -4
    NPP_vmx = 4
    NPP_Num = 18
    NPP_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                        colors =[cm.vik(40), 
                                                              (1, 1., 1), 
                                                              cm.vik(220)], 
                                                        N=NPP_Num+1)
    NPP_bounds = np.arange(NPP_vmn,NPP_vmx+0.4,0.4)
    NPP_norm = mpl.colors.BoundaryNorm(NPP_bounds, len(NPP_bounds)+1)


# var in rows and exp in cols
for i_row,i_var in enumerate(var):
    for i_col,i_exp in enumerate(exp):
        axes[i_row,i_col].set_xticks([])
        axes[i_row,i_col].set_yticks([])
        axes[i_row,i_col].set_xticklabels([])
        axes[i_row,i_col].set_yticklabels([])
        axes[i_row,i_col].spines["top"].set_visible(False)
        axes[i_row,i_col].spines["right"].set_visible(False)
        axes[i_row,i_col].spines["bottom"].set_visible(False)
        axes[i_row,i_col].spines["left"].set_visible(False)
        if i_var == 'GPP':
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
            X_rate = eval(input_name)[0]
            X_pvalue = eval(input_name)[1]
            im1=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                           levels=GPP_bounds, cmap=GPP_cmap2,vmin=GPP_vmn,
                                           vmax=GPP_vmx,extend='both')    
            sig_area   = np.where(X_pvalue < 0.01)
            im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
        elif i_var == 'AR':
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
            X_rate = eval(input_name)[0]
            X_pvalue = eval(input_name)[1]
            im3=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                           levels=AR_bounds, cmap=AR_cmap2,vmin=AR_vmn,vmax=AR_vmx,extend='both')    
            sig_area   = np.where(X_pvalue < 0.01)
            im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
        else:
            India.plot(facecolor='gray',edgecolor='black',ax=axes[i_row,i_col])
            input_name = i_exp+'_'+i_var+'_'+i_crop+'_rate'
            X_rate = eval(input_name)[0]
            X_pvalue = eval(input_name)[1]
            im4=axes[i_row,i_col].contourf(clipped_CTRL_GPP_SW.lon,clipped_CTRL_GPP_SW.lat,X_rate,
                                           levels=NPP_bounds, cmap=NPP_cmap2,vmin=NPP_vmn,
                                           vmax=NPP_vmx,extend='both')    
            sig_area   = np.where(X_pvalue < 0.01)
            im2=axes[i_row,i_col].scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.1,c='k',alpha=0.6)
            
        ## title
        if i_col == 0:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
            # axes[i_row,1].set_title(i_exp,fontsize=fsize)
        else:
            title_text = '('+roman.toRoman(i_col+1).lower()+')\n'r'$\mathrm{CTRL - S}_\mathrm{'+i_exp+'}$'
            axes[0,i_col].set_title(title_text,fontsize=fsize+2)
            
        ## y label   
        if i_col == 0:
            axes[0,i_col].set_ylabel('GPP',fontsize=fsize+1)
            axes[1,i_col].set_ylabel('AR',fontsize=fsize+1)
            axes[2,i_col].set_ylabel('NPP',fontsize=fsize+1)
        else:
            axes[i_row,i_col].set_ylabel('')
        ## x label
        # if i_row == 2:
        #     axes[i_row,i_col].set_xlabel('Longitude',fontsize=fsize)
        # else:
        #     axes[i_row,i_col].set_xlabel('')
        # ## ticklabels
        # axes[i_row,i_col].set_xticks([70, 80, 90])
        # axes[i_row,i_col].set_xticklabels([70,80,90], fontsize=fsize)
        # axes[i_row,i_col].set_yticks([10,20,30])
        # axes[i_row,i_col].set_yticklabels([10,20,30], fontsize=fsize)
        #panel_no = '('+list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)+')'
        #axes[i_row,i_col].text(plotid_x,plotid_y, panel_no,fontsize=fsize-3,transform=axes[i_row,i_col].transAxes)
## colorbar at last column
cbar1 = fig1.colorbar(im1,ax=axes[0,4],shrink=0.8,cmap=GPP_cmap2,norm=GPP_norm)
cbar1.set_label('GgC/year',fontsize=fsize+2)
# cbar1.set_lims(GPP_vmn,GPP_vmx)
cbar1.set_ticks(np.linspace(GPP_vmn,GPP_vmx,5))
cbar1.set_ticklabels(np.linspace(GPP_vmn,GPP_vmx,5),fontsize=fsize)

cbar2 = fig1.colorbar(im3,ax=axes[1,4],shrink=0.8,cmap=AR_cmap2,norm=AR_norm)
cbar2.set_label('GgC/year',fontsize=fsize+2)
# cbar2.set_lims(AR_vmn,AR_vmx)
cbar2.set_ticks(np.linspace(AR_vmn,AR_vmx,5))
cbar2.set_ticklabels(np.linspace(AR_vmn,AR_vmx,5),fontsize=fsize)

cbar3 = fig1.colorbar(im4,ax=axes[2,4],shrink=0.8,cmap=NPP_cmap2,norm=NPP_norm)
cbar3.set_label('GgC/year',fontsize=fsize+2)
# cbar3.set_lims(NPP_vmn,NPP_vmx)
cbar3.set_ticks(np.linspace(NPP_vmn,NPP_vmx,5))
cbar3.set_ticklabels(np.linspace(NPP_vmn,NPP_vmx,5),fontsize=fsize)

# Add rotated bold 'A.' manually
axes[0, 0].text(-0.25, 0.5, '(a)', transform=axes[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[1, 0].text(-0.25, 0.5, '(b)', transform=axes[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes[2, 0].text(-0.25, 0.5, '(c)', transform=axes[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')

# fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/'+i_crop+'_SpatialTrends_Carbonfluxes_AllExperiments_6July.png',dpi=600)
fig1.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/'+i_crop+
             '_SpatialTrends_Carbonfluxes_AllExperiments_16Apr.png',dpi=600,bbox_inches="tight")




