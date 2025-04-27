#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:23:18 2024

@author: knreddy
"""

#%% Loading required Modules
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import matplotlib as mpl
from cmcrameri import cm
import scipy.stats as stats
import warnings
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
Foldername = '/Users/knreddy/Documents/GMD_Paper/FluxCom_Monthly_data/CarbonFluxes_monthly/'
test_GPP_name = Foldername+'GPP.RS.FP-ALL.MLM-ALL.METEO-NONE.720_360.monthly.2001.nc'
test_data = xr.open_dataset(test_GPP_name)
test_data_GPP = test_data.GPP
test_GPP_india = test_data_GPP.sel(latitude=slice(40,0),longitude=slice(60,100))


lat = test_data.latitude
lon = test_data.longitude

start_year = 2001
end_year = 2014

N_years = len(range(start_year,end_year))+1
shape_data = np.shape(test_GPP_india)

GPP_Fluxcom_2001_2014 = np.empty([N_years,shape_data[0],shape_data[1],shape_data[2]])
NEE_Fluxcom_2001_2014 = np.empty([N_years,shape_data[0],shape_data[1],shape_data[2]])
TER_Fluxcom_2001_2014 = np.empty([N_years,shape_data[0],shape_data[1],shape_data[2]])

for i_count,i_year in enumerate(range(start_year,end_year+1)):
    file_name = Foldername+'GPP.RS.FP-ALL.MLM-ALL.METEO-NONE.720_360.monthly.'+str(i_year)+'.nc'
    temp_data = xr.open_dataset(file_name)
    temp_data_GPP = temp_data.GPP
    temp_GPP_india = temp_data_GPP.sel(latitude=slice(40,0),longitude=slice(60,100))
    temp_GPP_india=temp_GPP_india.reindex(latitude=list(reversed(temp_GPP_india['latitude'])))
    GPP_Fluxcom_2001_2014[i_count,:,:,:] = temp_GPP_india
    
    file_name2 = Foldername+'NEE.RS.FP-NONE.MLM-ALL.METEO-NONE.720_360.monthly.'+str(i_year)+'.nc'
    temp_data2 = xr.open_dataset(file_name2)
    temp_data_NEE = temp_data2.NEE
    temp_NEE_india = temp_data_NEE.sel(latitude=slice(40,0),longitude=slice(60,100))
    temp_NEE_india=temp_NEE_india.reindex(latitude=list(reversed(temp_NEE_india['latitude'])))
    NEE_Fluxcom_2001_2014[i_count,:,:,:] = temp_NEE_india
    
    file_name3 = Foldername+'TER.RS.FP-ALL.MLM-ALL.METEO-NONE.720_360.monthly.'+str(i_year)+'.nc'
    temp_data3 = xr.open_dataset(file_name3)
    temp_data_TER = temp_data3.TER
    temp_TER_india = temp_data_TER.sel(latitude=slice(40,0),longitude=slice(60,100))
    temp_TER_india=temp_TER_india.reindex(latitude=list(reversed(temp_TER_india['latitude'])))
    TER_Fluxcom_2001_2014[i_count,:,:,:] = temp_TER_india

# units of fluxes are gC/m2/day
#%% CLM5 CTRL DATA
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
Fluxcom_GPP = xr.DataArray(GPP_Fluxcom_2001_2014,
    coords=dict(
        year = data_CTRL.coords['year'][-14:],
        month = data_CTRL.coords['month'],
        lat = data_CTRL.coords['lat'],
        lon = data_CTRL.coords['lon']),
    attrs=dict(
        description="gross primary productivity",
        units="gC m-2 day-1"))

Fluxcom_NEP = xr.DataArray((NEE_Fluxcom_2001_2014)*-1,
    coords=dict(
        year = data_CTRL.coords['year'][-14:],
        month = data_CTRL.coords['month'],
        lat = data_CTRL.coords['lat'],
        lon = data_CTRL.coords['lon']),
    attrs=dict(
        description="net ecosystem productivity",
        units="gC m-2 day-1"))

Fluxcom_TER = xr.DataArray(TER_Fluxcom_2001_2014,
    coords=dict(
        year = data_CTRL.coords['year'][-14:],
        month = data_CTRL.coords['month'],
        lat = data_CTRL.coords['lat'],
        lon = data_CTRL.coords['lon']),
    attrs=dict(
        description="terrestrial ecosystem respiration",
        units="gC m-2 day-1"))
#%% Extract data (Units are gC/m2/year)
exp = ['CTRL']
var = ['GPP','NEP','TER']
crop = ['SW','Rice']

for i_var in var:
    output_name = exp[0]+'_'+i_var
    dataset_name = 'data_'+exp[0]          
    data_var_name = i_var
    locals()[output_name] = eval(dataset_name)[data_var_name][-14:,:,:,:]
#%% cft per land unit masking
pct_cft = np.array(mask_data['PCT_CFT'])
p_c_test = mask_data['PCT_CFT']
area = np.array(mask_data['AREA'])

pct_cft_ = xr.DataArray(pct_cft,
    coords=dict(
        time = p_c_test.coords['time'],
        cft = p_c_test.coords['cft'],
        lat = CTRL_GPP.coords['lat'],
        lon = CTRL_GPP.coords['lon']))

area_ = xr.DataArray(area,
    coords=dict(
        lat = CTRL_GPP.coords['lat'],
        lon = CTRL_GPP.coords['lon']),
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

#%% Create dummy dataset
for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            output_dataset = i_exp+'_'+i_var+'_'+i_crop+'_masked'
            locals()[output_dataset] = xr.DataArray(np.empty(shape(clipped_CTRL_GPP)),
                                             coords=dict(
                                                 year = clipped_CTRL_GPP.coords['year'],
                                                 month = clipped_CTRL_GPP.coords['month'],
                                                 lat = clipped_CTRL_GPP.coords['lat'],
                                                 lon = clipped_CTRL_GPP.coords['lon']))
#%% Mean of 1970-80 and 2005-1014
SW_mask = np.ma.getmask(pct_cft_sw_mask)
Rice_mask = np.ma.getmask(pct_cft_rice_mask)

for i_exp in exp:
    for i_var in var:
        for i_crop in crop:
            input_var = 'clipped_'+i_exp+'_'+i_var
            output_var1 = i_exp+'_'+i_var+'_'+i_crop+'_masked'
            output_var4 = i_exp+'_'+i_var+'_'+i_crop+'_masked2'
            for i_month,month in enumerate(CTRL_GPP.month):
                if i_crop == 'SW':        
                    locals()[output_var1][:,i_month,:,:] = np.ma.masked_array(eval(input_var).data[:,i_month,:,:],mask=eval(i_crop+'_mask')[-14:,:,:])
                else:
                    locals()[output_var1][:,i_month,:,:] = np.ma.masked_array(eval(input_var).data[:,i_month,:,:],mask=eval(i_crop+'_mask')[-14:,:,:])
            locals()[output_var4] = np.nanmean(eval(output_var1).data,axis=0)
#%% Plotting GPP, NEP, and TER
Data = CTRL_GPP_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.65
varname_pos_y = 0.85

data_pos_x = 0.535
data_pos_y = 0.27

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=12

fig3, axes3 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, dpi=600, figsize=(6,7),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = -0.5
vmx = 6
Num = 11

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+3))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+3)

# define the bins and normalize
bounds = np.linspace(vmn, vmx, Num+3)
norm = mpl.colors.BoundaryNorm(bounds, Num+4)

vmn2 = -4
vmx2 = 4
Num2 = 8

cmap2 = cm.vik
# cmaplist2 = [cmap2(i) for i in np.arange(0,cmap2.N,int(256/Num2+1))]
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num2+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+0.5,0.5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

wheat_month_dim = [11, 0, 1, 2]
wheat_month_names = ['Dec','Jan','Feb','Mar']
for i_row in range(3):
    for i_col in range(3):
        p_row = i_row
        ax = axes3[p_row,i_col]
        # panel_no = list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)
        # ax.text(panel_pos_x,panel_pos_y,panel_no,
        #             fontsize=fsize,transform=ax.transAxes)
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row==0:         
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_GPP_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                # ax.set_ylabel('a. GPP \n\n Latitude', fontsize=fsize-1)
                if i_row==0:
                    ax.set(title='(i)\nFLUXCOM',ylabel='GPP')
                else:
                    ax.set(title='',xlabel='')
                # ax.text(varname_pos_x,varname_pos_y,'GPP',
                #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==2:
                plot_data1 = np.nanmean(Fluxcom_GPP_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_GPP_SW_masked[:,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                if p<0.01:
                    data_text = 'r = %0.2f*\nbias = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nbias = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val_less = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='less')
                # ttest_val_great = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='greater')
                # sig_area_l = np.where(ttest_val_less.pvalue<0.001)
                # sig_area_g = np.where(ttest_val_great.pvalue<0.001)
                # im3=ax.scatter(lonx[sig_area_l],latx[sig_area_l],marker='o',s=0.25,c='k',alpha=0.6)
                # im3=ax.scatter(lonx[sig_area_g],latx[sig_area_g],marker='o',s=0.25,c='k',alpha=0.6)
                if i_row==0:
                    ax.set(title='(iii)\nCLM-FLUXCOM',xlabel='', ylabel='')
                else:
                    ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'GPP',
                #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==1:
                plot_data1 = np.nanmean(CTRL_GPP_SW_masked[:1,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data1,axis=0)
                im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                
                if i_row==0:
                    ax.set(title='(ii)\nCLM',xlabel='', ylabel='')
                else:
                    ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'GPP',
                #             fontsize=fsize,transform=ax.transAxes)
        if i_row==2:
            # p_row = i_row
            # ax = axes3[p_row,i_col]
            # ax.tick_params('both', labelsize=15)
            # India.plot(facecolor='gray',edgecolor='black',ax=ax)
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_NEP_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                # ax.set_ylabel('c. NEP \n\n Latitude', fontsize=fsize-1)
                ax.set(title='',ylabel='NEP')
                # ax.text(varname_pos_x,varname_pos_y,'NEP',
                #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==2:
                plot_data1 = np.nanmean(Fluxcom_NEP_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data2 = np.nanmean(CTRL_NEP_SW_masked[:,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                if p<0.01:
                    data_text = 'r = %0.2f*\nbias = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nbias = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val_less = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='less')
                # ttest_val_great = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='greater')
                # sig_area_l = np.where(ttest_val_less.pvalue<0.001)
                # sig_area_g = np.where(ttest_val_great.pvalue<0.001)
                # im3=ax.scatter(lonx[sig_area_l],latx[sig_area_l],marker='o',s=0.25,c='k',alpha=0.6)
                # im3=ax.scatter(lonx[sig_area_g],latx[sig_area_g],marker='o',s=0.25,c='k',alpha=0.6)
                ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'NEP',
                #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==1:
                plot_data1 = np.nanmean(CTRL_NEP_SW_masked[:,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data1,axis=0)
                im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'NEP',
                #             fontsize=fsize,transform=ax.transAxes)
        if i_row==1:
            # p_row = i_row
            # ax = axes3[p_row,i_col]
            # ax.tick_params('both', labelsize=15)
            # India.plot(facecolor='gray',edgecolor='black',ax=ax)
            if i_col==0:
                plot_data1 = np.nanmean(Fluxcom_TER_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data1[plot_data1>10] = np.nan
                plot_data = np.nanmean(plot_data1,axis=0)
                im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                # ax.set_ylabel('b. TER \n\n Latitude', fontsize=fsize-1)
                ax.set(title='',ylabel='TER')
                # ax.text(varname_pos_x,varname_pos_y,'TER',
                #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==2:
                plot_data1 = np.nanmean(Fluxcom_TER_SW_masked[:,wheat_month_dim,:,:],axis=1)
                plot_data1[plot_data1>10] = np.nan
                plot_data2 = np.nanmean(CTRL_TER_SW_masked[:,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                if p<0.01:
                    data_text = 'r = %0.2f*\nbias = %0.2f' % (r, bias)
                else:
                    data_text = 'r = %0.2f\nbias = %0.2f' % (r, bias)
                ax.text(data_pos_x,data_pos_y,data_text,
                            fontsize=fsize-3,transform=ax.transAxes)
                # a = plot_data2 - plot_data1
                # ttest_val_less = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='less')
                # ttest_val_great = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit',alternative='greater')
                # sig_area_l = np.where(ttest_val_less.pvalue<0.001)
                # sig_area_g = np.where(ttest_val_great.pvalue<0.001)
                # im3=ax.scatter(lonx[sig_area_l],latx[sig_area_l],marker='o',s=0.25,c='k',alpha=0.6)
                # im3=ax.scatter(lonx[sig_area_g],latx[sig_area_g],marker='o',s=0.25,c='k',alpha=0.6)
                ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'TER',
                #             fontsize=fsize,transform=ax.transAxes)
                
            elif i_col ==1:
                plot_data1 = np.nanmean(CTRL_TER_SW_masked[:,wheat_month_dim,:,:],axis=1)*86400
                plot_data = np.nanmean(plot_data1,axis=0)
                im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                ax.set(title='',xlabel='', ylabel='')
                # ax.text(varname_pos_x,varname_pos_y,'TER',
                #             fontsize=fsize,transform=ax.transAxes)

# Add rotated bold 'A.' manually
axes3[0, 0].text(-0.30, 0.5, '(a)', transform=axes3[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[1, 0].text(-0.30, 0.5, '(b)', transform=axes3[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[2, 0].text(-0.30, 0.5, '(c)', transform=axes3[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')


cbar1 = fig3.colorbar(im4,ax=axes3[2,:2],shrink=0.4,cmap=cmap,orientation='horizontal')
cbar1.set_label('gC/m\u00b2/day',fontsize=fsize-3)
cbar1.set_ticks(np.linspace(0,vmx,7)) 

cbar2 = fig3.colorbar(im2,ax=axes3[2,2],shrink=0.8,cmap=cmap2,orientation='horizontal')
cbar2.set_label('\u0394 gC/m\u00b2/day',fontsize=fsize-3)
cbar2.set_ticks(np.linspace(vmn2,vmx2,5))

#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Comparison_carbonFluxes_Wheat_Fluxcom_spatial_r_and_bias_16Apr.png',
                #dpi=600, bbox_inches="tight")
#%% Plotting GPP, NEP, and TER for rice
Data = CTRL_GPP_Rice_masked
lonx, latx = np.meshgrid(Data.lon, Data.lat)

varname_pos_x = 0.65
varname_pos_y = 0.85

data_pos_x = 0.535
data_pos_y = 0.27

panel_pos_x = 0.025
panel_pos_y = 0.85

fsize=12

fig3, axes3 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, dpi=600, figsize=(6,7),layout='constrained')
# fig3.set_size_inches(8,12)

vmn = -0.5
vmx = 8
Num = 11

cmap = cm.batlowW_r  # define the colormap
cmaplist = [cmap(i) for i in np.arange(0,cmap.N,int(256/Num+3))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, Num+3)

# define the bins and normalize
bounds = np.arange(vmn,vmx+0.5,0.5)
norm = mpl.colors.BoundaryNorm(bounds, len(bounds)+1)

vmn2 = -5
vmx2 = 5
Num2 = 10

cmap2 = cm.vik
# cmaplist2 = [cmap2(i) for i in np.arange(0,cmap2.N,int(256/Num2+1))]
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
                                                    colors =[cm.vik(40), 
                                                          (1, 1., 1), 
                                                          cm.vik(220)], 
                                                    N=Num2+1)

# define the bins and normalize
bounds2 = np.arange(vmn2,vmx2+0.5,0.5)
norm2 = mpl.colors.BoundaryNorm(bounds2, len(bounds2)+1)

rice_month_dim = [6, 7, 8, 9]
rice_month_names = ['Jul','Aug','Sep','Oct']
for i_row in range(3):
    for i_col in range(3):
        p_row = i_row
        ax = axes3[p_row,i_col]
        # panel_no = list(string.ascii_lowercase)[i_row]+'.'+str(i_col+1)
        # ax.text(panel_pos_x,panel_pos_y,panel_no,
        #             fontsize=fsize,transform=ax.transAxes)
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        India.plot(facecolor='gray',edgecolor='black',ax=ax)
        if i_row==0:           
            if i_col==0:
                    plot_data1 = np.nanmean(Fluxcom_GPP_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    # ax.set_ylabel('a. GPP \n\n Latitude', fontsize=fsize-1)
                    # ax.set_ylabel('a. GPP', fontsize=fsize-1)
                    if i_row==0:
                        ax.set(title='(i)\nFLUXCOM',ylabel='GPP')
                    else:
                        ax.set(title='',xlabel='')                   
                    # ax.text(varname_pos_x,varname_pos_y,'GPP',
                    #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==2:
                    plot_data1 = np.nanmean(Fluxcom_GPP_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data2 = np.nanmean(CTRL_GPP_Rice_masked[:,wheat_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                    
                    im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                    r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                    if p<0.01:
                        data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                    else:
                        data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                    ax.text(data_pos_x,data_pos_y,data_text,
                                fontsize=fsize-4,transform=ax.transAxes)
                    # a = plot_data2 - plot_data1
                    # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                    # sig_area = np.where(ttest_val.pvalue<0.001)
                    # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.25,c='k',alpha=0.6)
                    if i_row==0:
                        ax.set(title='(iii)\nCLM-FLUXCOM',xlabel='', ylabel='')
                    else:
                        ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'GPP',
                    #             fontsize=fsize,transform=ax.transAxes)
            elif i_col ==1:
                    plot_data1 = np.nanmean(CTRL_GPP_Rice_masked[:1,rice_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    
                    if i_row==0:
                        ax.set(title='(ii)\nCLM',xlabel='', ylabel='')
                    else:
                        ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'GPP',
                    #             fontsize=fsize,transform=ax.transAxes)
        if i_row==2:
                if i_col==0:
                    plot_data1 = np.nanmean(Fluxcom_NEP_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    # ax.set_ylabel('c. NEP \n\n Latitude', fontsize=fsize-1)
                    ax.set(title='',ylabel='NEP')
                    # ax.text(varname_pos_x,varname_pos_y,'NEP',
                    #             fontsize=fsize,transform=ax.transAxes)
                elif i_col ==2:
                    plot_data1 = np.nanmean(Fluxcom_NEP_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data2 = np.nanmean(CTRL_NEP_Rice_masked[:,wheat_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                    
                    im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                    r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                    if p<0.01:
                        data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                    else:
                        data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                    ax.text(data_pos_x,data_pos_y,data_text,
                                fontsize=fsize-4,transform=ax.transAxes)
                    # a = plot_data2 - plot_data1
                    # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                    # sig_area = np.where(ttest_val.pvalue<0.001)
                    # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.25,c='k',alpha=0.6)
                    ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'NEP',
                    #             fontsize=fsize,transform=ax.transAxes)
                elif i_col ==1:
                    plot_data1 = np.nanmean(CTRL_NEP_Rice_masked[:,rice_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'NEP',
                    #             fontsize=fsize,transform=ax.transAxes)
        if i_row==1:
                if i_col==0:
                    plot_data1 = np.nanmean(Fluxcom_TER_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data1[plot_data1>10] = np.nan
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im1 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    # ax.set_ylabel('b. TER', fontsize=fsize-1)
                    ax.set(title='',ylabel='TER')
                    # ax.text(varname_pos_x,varname_pos_y,'TER',
                    #             fontsize=fsize,transform=ax.transAxes)
                elif i_col ==2:
                    plot_data1 = np.nanmean(Fluxcom_TER_Rice_masked[:,rice_month_dim,:,:],axis=1)
                    plot_data1[plot_data1>10] = np.nan
                    plot_data2 = np.nanmean(CTRL_TER_Rice_masked[:,rice_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data2,axis=0) - np.nanmean(plot_data1,axis=0)
                    
                    im2 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds2,cmap=cmap2,vmin=vmn2,vmax=vmx2)
                    r,p,bias = find_r_and_bias(plot_data1, plot_data2)
                    if p<0.01:
                        data_text = 'r = %0.2f*\nMAB = %0.2f' % (r, bias)
                    else:
                        data_text = 'r = %0.2f\nMAB = %0.2f' % (r, bias)
                    ax.text(data_pos_x,data_pos_y,data_text,
                                fontsize=fsize-4,transform=ax.transAxes)
                    # a = plot_data2 - plot_data1
                    # ttest_val = stats.ttest_1samp(a, popmean=np.nanmean(a), nan_policy='omit')
                    # sig_area = np.where(ttest_val.pvalue<0.001)
                    # im3=ax.scatter(lonx[sig_area],latx[sig_area],marker='o',s=0.25,c='k',alpha=0.6)
                    ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'TER',
                    #             fontsize=fsize,transform=ax.transAxes)
                elif i_col ==1:
                    plot_data1 = np.nanmean(CTRL_TER_Rice_masked[:,rice_month_dim,:,:],axis=1)*86400
                    plot_data = np.nanmean(plot_data1,axis=0)
                    im4 = ax.contourf(Data.lon,Data.lat,plot_data,levels=bounds,cmap=cmap,vmin=vmn,vmax=vmx)
                    ax.set(title='',xlabel='', ylabel='')
                    # ax.text(varname_pos_x,varname_pos_y,'TER',
                    #             fontsize=fsize,transform=ax.transAxes)

# Add rotated bold 'A.' manually
axes3[0, 0].text(-0.30, 0.5, '(a)', transform=axes3[0, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[1, 0].text(-0.30, 0.5, '(b)', transform=axes3[1, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')
axes3[2, 0].text(-0.30, 0.5, '(c)', transform=axes3[2, 0].transAxes,
                fontsize=fsize,# rotation=90,
                va='center', ha='center')

cbar1 = fig3.colorbar(im4,ax=axes3[2,:2],shrink=0.5,cmap=cmap,norm=norm,orientation='horizontal')
cbar1.set_label('gC/m\u00b2/day',fontsize=fsize-3)
cbar1.set_ticks(np.linspace(0,vmx,9)) 

cbar2 = fig3.colorbar(im2,ax=axes3[2,2],shrink=0.8,cmap=cmap2,norm=norm2,orientation='horizontal')
cbar2.set_label('\u0394 gC/m\u00b2/day',fontsize=fsize-3)
cbar2.set_ticks(np.linspace(vmn2,vmx2,6))
# cbar2.set_ticks([-5,-3,-1,0,1,3,5])

#fig3.savefig('/Users/knreddy/Documents/Biogeosciences_Paper/Figures/Comparison_carbonFluxes_Rice_Fluxcom_spatial_r_and_bias_16Apr.png',
             #dpi=600, bbox_inches="tight")



