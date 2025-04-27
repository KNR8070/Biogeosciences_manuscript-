#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:28:01 2025

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
    
#%% Clip all spatial data for Indian region
exp = ['CTRL','Fluxcom']
for i_exp in exp:
    for i_var in var:
        input_name = i_exp+'_'+i_var
        output_name = 'clipped_'+input_name
        locals()[output_name] = clip_data(eval(input_name),India)
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
#              dpi=600, bbox_inches="tigh