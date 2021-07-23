#this works good
from math import pi
import numpy as np
import pandas as pd
from lmfit.models import LorentzianModel, GaussianModel
import matplotlib.pyplot as plt
from scipy import signal


# file input and assign
inputFilename ='DPPC_22C_scan162_163_qxy_gid'

file = r'/Users/dayeen/Downloads/'+inputFilename+'.txt'
df0=pd.read_csv(file ,delimiter='\t',names=['q','intensity','error'])
X=df0['q'].values
Y=df0['intensity'].values
yerr=df0['error'].values

Y_fix=signal.detrend(Y)-min(signal.detrend(Y))

x = X
y = Y_fix


#fit declaration
gauss_mod = GaussianModel(prefix='gauss_')
lorentz1 = LorentzianModel(prefix='l1_')
lorentz2 = LorentzianModel(prefix='l2_')


#for  peaks fitting


def index_of(arrval, value):
    """return index of array *at or below* value """
    if value < min(arrval):
        return 0
    return max(np.where(arrval <= value)[0])
    
def get_peaks():
    #rough peak positions
    ix1 = index_of(x, 1.31)
    ix2 = index_of(x, 1.43)
    ix3 = index_of(x, 1.56)

    pars1 = gauss_mod.guess(y[:ix1], x=x[:ix1])
    pars2 = lorentz1.guess(y[ix1:ix2], x=x[ix1:ix2])
    pars3 = lorentz2.guess(y[ix2:ix3], x=x[ix2:ix3])

    pars_peaks = pars1 + pars2 + pars3
    mod_peaks = lorentz1 + lorentz2 + gauss_mod

    out_peaks = mod_peaks.fit(y, pars_peaks, x=x)

    #print output
    print(out_peaks.fit_report(min_correl=0.5))

    #plot with fit
    plt.scatter(x, y, color='blue')
    # plt.plot(x, out_peaks.init_fit, 'k--', label='initial fit')
    plt.plot(x, out_peaks.best_fit, 'r-', label='best fit')
    plt.legend(loc='best')
    plt.show()
    #assign values to variables for calculation
    peak1 = out_peaks.params['l1_center'].value
    peak2 = out_peaks.params['l2_center'].value
    fwhm1 = out_peaks.params['l1_fwhm'].value
    fwhm2 = out_peaks.params['l2_fwhm'].value
    return peak1, peak2, fwhm1, fwhm2


result = get_peaks()

out_peak1 = result[0]
out_peak2 = result[1]
out_fwhm1 = result[2]
out_fwhm2 = result[3]

correlation_length1 = (0.9*2*np.pi)/np.sqrt(out_fwhm1**2-0.0014**2)
correlation_length2 = (0.9*2*np.pi)/np.sqrt(out_fwhm2**2-0.0014**2)
d_spacing_a = 2*np.pi/out_peak1
d_spacing_b = 2*np.pi/out_peak2
Area_unit_cell_a = (np.sqrt(3)/2)*d_spacing_a**2

print('peak1 = ',out_peak1,
'\npeak2 = ', out_peak2,
'\ncorrelation1 = ',correlation_length1, 
'\ncorrelation2 = ',correlation_length2,
'\nd-spacing_a = ',d_spacing_a,
'\nd-spacing_a = ',d_spacing_b,
'\nArea_unitcell_a = ',Area_unit_cell_a
)