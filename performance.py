import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

def performance(bkg_events, sig_events):
    # bkg_events is a 1D array of anomaly scores for the background dataset
    # sig_events is a 1D array of anomaly scores for the signal dataset
    # Returns: Area under the ROC curve, and signal efficiencies for three background efficiencies: 10^-2, 10^-3, 10^-4

    #Create background and signal labels
    bkg_labels = np.zeros(len(bkg_events))
    sig_labels = np.ones(len(sig_events))
    
    #stitch all results together
    events = np.append(bkg_events, sig_events)
    labels = np.append(bkg_labels, sig_labels)

    #Build ROC curve using sklearns roc_curve function
    FPR, TPR, thresholds = roc_curve(labels, events)

    #Calculate area under the ROC curve
    AUC = auc(FPR, TPR)

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

    #iterate through bkg efficiencies and get as close as possible to the desired efficiencies.
    for i in range(len(FPR)):
        bkg_eff = FPR[i]
        if bkg_eff >= efficiency1 and done1 == False:
            epsilon1 = TPR[i]
            done1 = True
        if bkg_eff >= efficiency2 and done2 == False:
            epsilon2 = TPR[i]
            done2 = True
        if bkg_eff >= efficiency3 and done3 == False:
            epsilon3 = TPR[i]
            done3 = True

        if done1 and done2 and done3:
            break
            
    return AUC, epsilon1, epsilon2, epsilon3
