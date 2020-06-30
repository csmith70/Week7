# Christopher Smith
# HW9: Multivariate Linear Regression of Global Surface Temperature
# 10/11/19

#import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import glob 
import LHD
import os.path as pth
import h5py
import statsmodels.api as sm

#Read in data
file_d = LHD.load_space_data('modern_climate_time_series.dat', 22)
#print(file_d)

#Mask values where data is filled with -999
masked_data = np.ma.masked_where(file_d==-999, file_d)
#Delete rows with masked data
clean_data = np.ma.compress_rows(masked_data)
'''
#Future Data
yearf = masked_data[:,0]
tsif = masked_data[:,2]
ensoS = masked_data[:,3]
ensof = masked_data[:,4]
volcS = masked_data[:,5]
volcf = masked_data[:,6]
CO2f = masked_data[:,8]
print(tsif)
print(yearf)
'''


#Extract variables
year = clean_data[:,0]
#print(year)
#Exisiting Data
obs_T = clean_data[:,1]
tsi = clean_data[:,2]
enso = clean_data[:,4]
volc = clean_data[:,6]
CO2 = clean_data[:,8]
#print(CO2)

#Defines the number of observations and regressors
nobs = len(year) #Number of observations
nrgs = 5 #Number of regressors + 1 (1 is for the y-intercept const.)

#Create a matrix of 1s with the dimensions defined above
regressors = np.ones((nobs, nrgs), dtype=np.float)

#Populate the matrix columns with regressors (leaving column 0 as 1s)
regressors[:,1] = tsi
regressors[:,2] = enso
regressors[:,3] = volc
regressors[:,4] = CO2


#Create an ordinary least squares model
model = sm.OLS(obs_T, regressors)

#Extract the constants 
constants = model.fit().params
print(constants)
print('____Summary_____')
print(model.fit().summary())

#Calculate the model fit result
model_T = constants[0] + constants[1]*tsi + constants[2]*enso + constants[3]*volc + constants[4]*CO2

#_______Plot Setup________

#Axes position setup parameters
left = 0.1
base = 0.06
base_top = 0.689
del_base = 0.135
width = 0.8
height_upper = 0.25
height_lower = 0.12

#Axis parameters
xmin = 1880
xmax = 2013

#Figure positions 
fig = plt.figure(figsize=[9,12])
ax1 = fig.add_axes([left, base_top, width, height_upper])
ax2 = fig.add_axes([left, base+del_base*3.73, width, height_lower])
ax3 = fig.add_axes([left, base+del_base*2.80, width, height_lower])
ax4 = fig.add_axes([left, base+del_base*1.85, width, height_lower])
ax5 = fig.add_axes([left, base+del_base*0.92, width, height_lower])
ax6 = fig.add_axes([left, base, width, height_lower])

#Remove x tick labels
ax1.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax3.xaxis.set_ticklabels([])
ax4.xaxis.set_ticklabels([])
ax5.xaxis.set_ticklabels([])

#Set axis ranges
ax1.set_xlim(xmin, xmax)
ax2.set_xlim(xmin, xmax)
ax3.set_xlim(xmin, xmax)
ax4.set_xlim(xmin, xmax)
ax5.set_xlim(xmin, xmax)
ax6.set_xlim(xmin, xmax)
ax1.set_ylim(-1.0, 1.0)
ax2.set_ylim(-0.6, 0.6)
ax3.set_ylim(-0.05, 0.10)
ax4.set_ylim(-0.3, 0.3)
ax5.set_ylim(-0.50, 0.0)
#ax6.set_ylim(0.0, 1.0)

#Plotting 
ax1.plot(year, obs_T, c = 'g', label='CRU Temp Anomaly Data')
ax1.legend()
ax1.set_ylabel('Anomaly \n (\xb0 C)')
ax1.plot(year, model_T, c='black', label = 'Modeled Temp Anomaly')
ax1.legend()
ax2.plot(year, (obs_T - model_T), c='y', label = 'Measured-Model')
ax2.legend()
ax2.set_ylabel('Anomaly \n (\xb0 C)')
ax3.plot(year, constants[1]*tsi, c='black', label = 'Solar Irradiance Impact')
ax3.legend()
ax3.set_ylabel('Anomaly \n (\xb0 C)')
ax4.plot(year, constants[2]*enso, color='r', label = 'ENSO Impact')
ax4.legend()
ax4.set_ylabel('Anomaly \n (\xb0 C)')
ax5.plot(year, constants[3]*volc, color='purple', label = 'Volcanic Aerosols Impact')
ax5.legend()
ax5.set_ylabel('Anomaly \n (\xb0 C)')
ax6.plot(year, constants[4]*CO2, color='orange', label= 'CO2/Anthropogenic Impact') 
ax6.legend()
ax6.set_ylabel('Anomaly \n (\xb0 C)')
plt.suptitle('Temperature Anomaly and Impacts \n 1890 - 2012')
plt.xlabel('Year')
plt.show()
#print(CO2)

#***********PART TWO*************#

#Extract all data, including the future years' data. This is included in the masked data.
yearf = masked_data[:,0]
tsif = masked_data[:,2]
ensoS = masked_data[:,3] #Super enso data
ensof = masked_data[:,4] #flat enso data
volcS = masked_data[:,5] #super volcano data
volcf = masked_data[:,6] #flat volcano data
CO2f = masked_data[:,8]
#print(tsif)
#print(yearf)

#Calculate the model fit result for the flat enso and volcano data using the constants from Part 1
model_Tf = constants[0] + constants[1]*tsif + constants[2]*ensof + constants[3]*volcf + constants[4]*CO2f
#Calculate the model fit result for the super enso and super volcano data using the constants from Part 1
model_TSup = constants[0] + constants[1]*tsif + constants[2]*ensoS + constants[3]*volcS + constants[4]*CO2f

#Practice plotting
'''
#fig2 = plt.figure(figsize=[9,12])
#plt.plot(yearf, constants[2]*ensof)
plt.plot(yearf, model_TSup)
plt.plot(yearf, model_Tf)
#ax1.plot(yearf, constants[1]*tsif)
plt.show()
'''

#Plot On a New Figure
#Figure positions 
fig2 = plt.figure(figsize=[9,12])
left = 0.1
base = 0.07
base_top = 0.69
del_base = 0.135
width = 0.8
height_upperf = .85
height_lower = 0.15
ax1 = fig2.add_axes([left, base, width, height_upperf])
#If future plots needed on axes
#ax2 = fig2.add_axes([left, base, width, height_lower])
#ax3 = fig2.add_axes([left, base+del_base*2.8, width, height_lower])
#ax4 = fig2.add_axes([left, base+del_base*1.8, width, height_lower])
#ax5 = fig2.add_axes([left, base+del_base*0.8, width, height_lower])
#ax6 = fig2.add_axes([left, base, width, height_lower])

#Remove x tick labels if needed
#ax1.xaxis.set_ticklabels([])
#ax2.xaxis.set_ticklabels([])
#ax3.xaxis.set_ticklabels([])
#ax4.xaxis.set_ticklabels([])
#ax5.xaxis.set_ticklabels([])

#plot from years 1980-2030
nxmin = 1980
nxmax = 2030
#Set axis ranges
ax1.set_xlim(nxmin, nxmax)
#ax2.set_xlim(nxmin, nxmax)
#ax3.set_xlim(nxmin, nxmax)
#ax4.set_xlim(nxmin, nxmax)
#ax5.set_xlim(nxmin, nxmax)
#ax6.set_xlim(nxmin, nxmax)
ax1.set_ylim(-0.5, 1.5)
#ax2.set_ylim(-0.6, 0.6)
#ax3.set_ylim(-0.05, 0.10)
#ax4.set_ylim(-0.3, 0.3)
#ax5.set_ylim(-0.50, 0.0)
#ax6.set_ylim(0.0, 1.0)

#Plotting data
ax1.plot(yearf, model_Tf, c='blue', label = 'Flat')
ax1.legend()
ax1.plot(yearf, model_TSup, c = 'r', ls = '--', label='Eruption + Super El Nino') 
ax1.legend()
plt.suptitle('Global Surface Temperature Projections (1980-2030)')
ax1.set_ylabel('Anomaly \n (\xb0 C)')
plt.xlabel('Year')

plt.show()
#Save figures
fig.savefig('existingdata.png')
fig2.savefig('modeleddata.png')
#Future plots
'''
#ax2.plot(year, (obs_T - model_T), c='y', label = 'Measured-Model')
ax2.legend()
ax3.plot(yearf, constants[1]*tsif, c='black', label = 'Solar Irradiance Impact')
ax3.legend()
ax4.plot(yearf, constants[2]*ensof, color='r', label = 'ENSO Impact')
ax4.legend()
ax5.plot(yearf, constants[3]*volcf, color='purple', label = 'Volcanic Aerosols Impact')
ax5.legend()
ax6.plot(yearf, constants[4]*CO2f, color='orange', label= 'CO2 Impact') 
ax6.legend()
#plt.show()
'''



