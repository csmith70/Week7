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
file_d = LHD.load_space_data('global_temperature_record.dat', 7)
#print(file_d)

#Mask values where data is filled with -999
masked_data = np.ma.masked_where(file_d==-999, file_d)
#Delete rows with masked data
clean_data = np.ma.compress_rows(masked_data)
#Extract variables
year = clean_data[:,0]
#print(year)
#Exisiting Data
obs_T = clean_data[:,1]
#T=np.float(obs_T)
print(obs_T)
#compute 5-year running mean

rmean = []  #empty values added to beginning of running mean

x=2
while x< 136:

	#calculate running mean
	total = (obs_T[x]+obs_T[x-1]+obs_T[x-2]+obs_T[x+1]\
		 +clean_data[x+2])/5
	
	#round the mean to 2 decimal places
	mean = float(round(total[1],2))

	#add mean to the end of existing array
	rmean.append(mean)	
	x+=1
print(rmean)
