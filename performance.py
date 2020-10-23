import numpy as np
import warnings
from sklearn.metrics import auc
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def performance(bkg_events, sig_events, bkg_weights = 'ones', sig_weights = 'ones'):
    # bkg_events is a 1D array of anomaly scores for the background dataset
    # sig_events is a 1D array of anomaly scores for the signal dataset
    # bkg_weights is an optional 1D array of weights for the background dataset defaults to ones
    # sig_weights is an optional 1D array of weights for the signal dataset defaults to ones
    # Returns: Area under the ROC curve, and S/sqrt(S+B) for three background efficiencies: 10^-2, 10^-3, 10^-4

    #if sig/bkg weights unused, initialise to ones. This returns a warning if weights are defined, hence the warning filter.
    warnings.simplefilter(action='ignore',category=FutureWarning)
    if bkg_weights == 'ones':
        bkg_weights = np.ones(len(bkg_events))
    if sig_weights == 'ones':
        sig_weights = np.ones(len(sig_events))
    warnings.simplefilter(action='default',category=FutureWarning)

    ### AUC CALCULATION ###
    #loop over N slices between bkg_max and bkg_min and calculate bkg/sig efficiency to the right of each cut.
    N = 1000 #number of points to build the ROC curve for. Increase this if more accuracy is necessary. 1000 seems to be appropriate for AUC.
    MIN = min(bkg_events)
    MAX = max(bkg_events)
    bins = np.linspace(MIN,MAX, N+1)

    #make histograms to perform cuts
    bkg_hist, bins = np.histogram(np.clip(bkg_events, MIN, MAX), bins = bins, weights = bkg_weights)
    sig_hist, bins = np.histogram(np.clip(sig_events, MIN, MAX), bins = bins, weights = sig_weights)

    #number of sig/bkg events
    nsig = float(sum(sig_hist))
    nbkg = float(sum(bkg_hist))

    #sig/bkg efficiency
    sig_eff = np.full(N, float(-1))
    bkg_eff = np.full(N, float(-1))

    #construct ROC curve and get AUC
    for i in range(N):        
        #number of sig/bkg events greater than a cut at x
        sig_gtx = float(sum(sig_hist[i:]))
        bkg_gtx = float(sum(bkg_hist[i:]))                                        
        
        #calculate efficiencies
        sig_eff[i] = float(sig_gtx)/float(nsig)
        bkg_eff[i] = float(bkg_gtx)/float(nbkg)

    #calculate area under the curve
    AUC = auc(bkg_eff, sig_eff)

    ### EPSILON CALCULATION ###
    #Calculate very low background efficiencies. Need high value of N
    N = 100000 #number of points to build the ROC curve for. Increase this if more accuracy is necessary. 100000 seems to be appropriate for epsilon calculations
    MIN = min(bkg_events)
    MAX = max(bkg_events)
    bins = np.linspace(MIN,MAX, N+1)

    #make histograms to perform cuts
    bkg_hist, bins = np.histogram(np.clip(bkg_events, MIN, MAX), bins = bins, weights = bkg_weights)
    sig_hist, bins = np.histogram(np.clip(sig_events, MIN, MAX), bins = bins, weights = sig_weights)

    #background efficiencies
    efficiency1 = 10.0**-2
    efficiency2 = 10.0**-3
    efficiency3 = 10.0**-4
    #epsilon values
    epsilon1 = 0.0
    epsilon2 = 0.0
    epsilon3 = 0.0
    #flags to tell when done
    done1 = False
    done2 = False
    done3 = False

    for i in range(N)[::-1]:
        #number of sig/bkg events greater than a cut at x
        sig_gtx = float(sum(sig_hist[i:]))
        bkg_gtx = float(sum(bkg_hist[i:]))                                        
        
        #calculate background efficiency
        bkg_eff = float(bkg_gtx)/float(nbkg)

        #calculate signal efficiencies as close to the desired background efficiency as possible. 
        #If this is not close enough to the desired efficiency, increase N
        if bkg_eff >= efficiency1 and done1 == False:
            epsilon1 = sig_gtx/np.sqrt(sig_gtx+bkg_gtx)
            done1 = True
        if bkg_eff >= efficiency2 and done2 == False:
            epsilon2 = sig_gtx/np.sqrt(sig_gtx+bkg_gtx)
            done2 = True
        if bkg_eff >= efficiency3 and done3 == False:
            epsilon3 = sig_gtx/np.sqrt(sig_gtx+bkg_gtx)
            done3 = True

        if done1 and done2 and done3:
            break
            
    return AUC, epsilon1, epsilon2, epsilon3
