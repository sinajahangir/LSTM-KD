# -*- coding: utf-8 -*-
"""
# This code is used to assess the perfromance of KD on 421 catchments
# In this version, two different scenarios are compared: Daymet vs ERA5 and 
 DI+Daymet vs Daymet 

#scenario_random: Random seed for selecting the 50 catchments used for
training the regional model

# At this stage, only NSE and KGE are compared
# Results are compared for two different groups: same (50), and different (371)
-------------------------------------------------------------------------
Dependencies: 
- matplotlib
- numpy
- pandas

First version: March 2025
@author: SinaJahangir (Ph.D.)
contact:mohammadsina.jahangir@gmail.com
"""
#%%
#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#%% plotting options
#change based on your preferance
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams['font.family'] = 'Calibri'  # Set font to Calibri
plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
plt.rcParams['axes.labelsize'] = 16  # Large axis labels
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 14  # Set x-tick label size
plt.rcParams['ytick.labelsize'] = 14  # Set y-tick label size
plt.rcParams['xtick.major.width'] = 2  # Set x-tick major width to 2
plt.rcParams['ytick.major.width'] = 2  # Set y-tick major width to 2
plt.rcParams['figure.titlesize'] = 18  # Set title font size
plt.rcParams['figure.titleweight'] = 'bold'  # Set title font weight
#%%
def nash_sutcliffe_error(Q_obs,Q_sim):
    """
    Written by: SJ
    Q_obs: observed discharge; 1D vector
    Q_sim: simulated discharge; 1D vector
    This function calculates the NSE between observed and simulated discharges
    returns: NSE; float
    """
    if len(Q_sim)!=len(Q_obs):
        print('Length of simulated and observed discharges do not match')
        return
    else:
        num=np.sum(np.square(Q_sim-Q_obs))
        den=np.sum(np.square(Q_obs-np.mean(Q_obs)))
        NSE=1-(num/den)
        return NSE

def CC(Pr,Y):
    from scipy import stats
    Pr=np.reshape(Pr,(-1,1))
    Y=np.reshape(Y,(-1,1))
    return stats.pearsonr(Pr.flatten(),Y.flatten())[0]
#modified KGE
def KGE(prediction,observation):

    nas = np.logical_or(np.isnan(prediction), np.isnan(observation))
    pred=np.copy(np.reshape(prediction,(-1,1)))
    obs=np.copy(np.reshape(observation,(-1,1)))
    r=CC(pred[~nas],obs[~nas])
    beta=np.nanmean(pred)/np.nanmean(obs)
    gamma=(np.nanstd(pred)/np.nanstd(obs))/beta
    kge=1-((r-1)**2+(beta-1)**2+(gamma-1)**2)**0.5
    return kge
#%%
#change directory
os.chdir(r'D:\Paper\Code\KD')
#%%
#Read perfromance of DI
seed=213
df_reg_371=pd.read_csv('NSE_KGE_in_list_seed_%d.csv'%(seed))[['nse','kge']]
df_reg_50=pd.read_csv('NSE_KGE_not_in_list_seed_%d.csv'%(seed))[['nse','kge']]
df_kd_371=pd.read_csv('NSE_KGE_in_list_seed_%d_100_DI.csv'%(seed))[['nse','kge']]
df_kd_50=pd.read_csv('NSE_KGE_not_in_list_seed_%d_100_DI.csv'%(seed))[['nse','kge']]
#%%
# Function to style the boxplot
# Function to style the boxplot
def style_boxplot(box_parts, ax, labels, colors,hatch_patterns):
    for box, color, hatch in zip(box_parts['boxes'], colors, hatch_patterns):
        box.set_facecolor(color)
        box.set_edgecolor('k')
        box.set_linewidth(1.75)
        box.set_alpha(0.75)
        box.set_hatch(hatch)  # Add hatching
    
    for whisker in box_parts['whiskers']:
        whisker.set_color('k')
        whisker.set_linewidth(1.75)
    
    for cap in box_parts['caps']:
        cap.set_color('k')
        cap.set_linewidth(1.5)
    
    for median in box_parts['medians']:
        median.set_color('k')
        median.set_linewidth(1.75)
        # Display median value as text above the box
        x_median = median.get_xdata().mean()  # Get the x-position of the median
        y_median = median.get_ydata()[0]  # Y-position (median value)
        median_value = median.get_ydata()[0]
        # Add text with median value, slightly above the box
        ax.text(x_median, y_median + 0.01, f'{median_value:.3f}', 
                ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
    
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    
    ax.tick_params(direction='inout', length=6, width=2, colors='k')
    ax.grid(axis='y', color='black', alpha=0.5)
    ax.set_xticks([1.125, 2.125])  # Center the labels between the pairs
    ax.set_xticklabels(labels, rotation=30)
#%%
# Create a figure and two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4),dpi=400,sharey=True)



# Second boxplot: data3 vs data7 and data4 vs data8
boxplot1 = ax1.boxplot([df_kd_371.iloc[:,1],df_kd_50.iloc[:,1], df_reg_371.iloc[:,1],df_reg_50.iloc[:,1]], patch_artist=True, 
                       positions=[1, 1.4, 2, 2.4], widths=0.2,  # Adjust positions and widths
                       showfliers=False)
ax1.set_title('KGE')
# Define colors for the second boxplot: gray, blue, gray, blue
colors1 = ['lightskyblue', 'lightskyblue', 'orange', 'orange']
hatch_patterns1 = [None, '///', None, '///']  # Hatching for blue boxes
style_boxplot(boxplot1, ax1, ['KD', 'Vanilla'], colors1,hatch_patterns1)



boxplot2 = ax2.boxplot([df_kd_371.iloc[:,0],df_kd_50.iloc[:,0], df_reg_371.iloc[:,0],df_reg_50.iloc[:,0]], patch_artist=True, 
                       positions=[1, 1.4, 2, 2.4], widths=0.2,  # Adjust positions and widths
                       showfliers=False)


ax2.set_title('NSE')
# Define colors for the first boxplot: gray, blue, gray, blue
colors2 = ['lightskyblue', 'lightskyblue', 'orange', 'orange']
hatch_patterns2 = [None, '///', None, '///']  # Hatching for blue boxes
style_boxplot(boxplot2, ax2, ['KD', 'Vanilla'], colors2,hatch_patterns2)


# Adjust layout
plt.tight_layout()
plt.savefig('Compare_Box_Daymet_KD_%d_v1.png'%(seed))
#%%
#Read perfromance of ERA5
seed=113
df_reg_371=pd.read_csv('NSE_KGE_in_list_era_seed_%d.csv'%(seed))[['nse','kge']]
df_reg_50=pd.read_csv('NSE_KGE_not_in_list_era_seed_%d.csv'%(seed))[['nse','kge']]
df_kd_371=pd.read_csv('NSE_KGE_in_list_era_seed_%d_kd.csv'%(seed))[['nse','kge']]
df_kd_50=pd.read_csv('NSE_KGE_not_in_list_era_seed_%d_kd.csv'%(seed))[['nse','kge']]
#%%
# Create a figure and two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4),dpi=400,sharey=True)



# Second boxplot: data3 vs data7 and data4 vs data8
boxplot1 = ax1.boxplot([df_kd_371.iloc[:,1],df_kd_50.iloc[:,1], df_reg_371.iloc[:,1],df_reg_50.iloc[:,1]], patch_artist=True, 
                       positions=[1, 1.4, 2, 2.4], widths=0.2,  # Adjust positions and widths
                       showfliers=False)
ax1.set_title('KGE')
# Define colors for the second boxplot: gray, blue, gray, blue
colors1 = ['lightskyblue', 'lightskyblue', 'orange', 'orange']
hatch_patterns1 = [None, '///', None, '///']  # Hatching for blue boxes
style_boxplot(boxplot1, ax1, ['KD', 'Vanilla'], colors1,hatch_patterns1)



boxplot2 = ax2.boxplot([df_kd_371.iloc[:,0],df_kd_50.iloc[:,0], df_reg_371.iloc[:,0],df_reg_50.iloc[:,0]], patch_artist=True, 
                       positions=[1, 1.4, 2, 2.4], widths=0.2,  # Adjust positions and widths
                       showfliers=False)


ax2.set_title('NSE')
# Define colors for the first boxplot: gray, blue, gray, blue
colors2 = ['lightskyblue', 'lightskyblue', 'orange', 'orange']
hatch_patterns2 = [None, '///', None, '///']  # Hatching for blue boxes
style_boxplot(boxplot2, ax2, ['KD', 'Vanilla'], colors2,hatch_patterns2)


# Adjust layout
plt.tight_layout()
plt.savefig('Compare_Box_ERA_KD_%d_v1.png'%(seed))
