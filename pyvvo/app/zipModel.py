'''
Created on Apr 27, 2018

@author: thay838

Original R code written by Dave Engel. R code adapted to Python by Bin Zheng.
Final adoption into application by Brandon Thayer.

Notes from Dave:
Augmented Lagrangian Adaptive Barrier Minimization
'''
import numpy as np
import pandas as pd
import math
import mystic.solvers as my
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Constant for ZIP coefficients. ORDER MATTERS!
ZIPTerms = ['impedance', 'current', 'power']

# List of available solvers for zipFit
SOLVERS = ['fmin_powell', 'SLSQP']

# Constants for convergence tolerance
FTOL = 1e-8
GTOL = 5 # Number of iterations without change for fmin_powell

# Define default initial guess for ZIP models.
# We'll take the Oscillating Fan from the CVR report: 
# https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-19596.pdf
# Z%: 73.32, I%: 25.34, P%: 1.35
# Zpf: 0.97, Ipf: 0.95, Ppf: -1.0
PAR0 = np.array([
    0.7332 * 0.97,
    0.2534 * 0.95,
    0.0135 * -1,
    0.7332 * math.sin(math.acos(0.97)),
    0.2534 * math.sin(math.acos(0.95)),
    0.0135 * math.sin(math.acos(-1))
    ])
# Dave was initially using this:
# PAR0 = np.ones(6)*(1/18)

def zipFit(V, P, Q, Vn=240.0, solver='fmin_powell', par0=PAR0):
    """Solve for ZIP coefficients usable by GridLAB-D.
    
    V: voltage magnitude array
    P: real power array
    Q: reactive power array
    Vn: nominal voltage
    solver: either 'fmin_powell' to use mystic's modified scipy fmin_powell 
        solver, or 'SLSQP' to use scipy's sequential least squares programming
        solver.
    par0: Initial guess. Should be array of a1, a2, a3, b1, b2, b3
    """
    # Estimate nominal power
    Sn = estimateNominalPower(P=P, Q=Q)
    
    # Massage into standard polynomial format
    Vbar = V/Vn
    Pbar = P/Sn
    Qbar = Q/Sn
    
    # Solve.
    if solver == 'fmin_powell':
        sol = my.fmin_powell(Object, args=(Vbar, Pbar, Qbar), x0=par0,
                             contraint=Constrain, disp=False, gtol=GTOL,
                             ftol=FTOL, full_output=True)
        # Extract the polynomial coefficients
        p = sol[0][0:3]
        q = sol[0][3:6]
        
        # Get the value of the objective function (so the squared error)
        err = sol[1]
    elif solver == 'SLSQP':
        sol = minimize(Object, par0, args=(Vbar, Pbar, Qbar), method='SLSQP',
                       constraints={'type':'eq', 'fun': Constrain},
                       bounds=None, options={'ftol': FTOL})
        
        # Extract the polynomial coefficients
        p = sol.x[0:3]
        q = sol.x[3:6]
        
        # Get the value of the objective function (so the squared error)
        err = sol.fun
    else:
        raise UserWarning('Given solver, {}, is not implemented.'.format(solver))
        
        # Some code Bin had in?
        #cons2 = {'type':'eq', 'fun': Constrain2}
        #cons3 = {'type':'eq', 'fun': Constrain3}
        """
        if abs(Constrain(opti.x))>0.75:
            opti = minimize(Object, par0, args=(V1bar, P1bar, Q1bar), method='SLSQP', constraints = cons2, bounds = None)
                        
        if abs(Constrain(opti.x))>2.0:
            opti = minimize(Object, par0, args=(V1bar, P1bar, Q1bar), method='SLSQP', constraints = cons3, bounds = None)
        """
        
    # Convert the polynomial coefficients to GridLAB-D format (fractions and
    # power factors)
    coeff = polyToGLD(p, q)
    
    # Collect other useful information
    coeff['base_power'] = Sn
    coeff['error'] = err
    coeff['poly'] = (*p, *q)
    
    return coeff

def estimateNominalPower(P, Q):
    """Given a set of apparent power measurements, estimate nominal power.
    
    For now, we'll simply use the median of the apparent power.
    """
    Sn = np.median(np.sqrt(np.multiply(P,P) + np.multiply(Q,Q)))
    
    # If the median P value is negative, flip Sn.
    #if np.median(P) < 0:
    #    Sn *= -1
    
    return Sn

def Object(Params, Vbar, Pbar, Qbar):
    """Objective function for minimization. Minimize squared error."""
    a1, a2, a3, b1, b2, b3 = Params
    return sum( (Pbar - (a1*(Vbar*Vbar)+a2*Vbar+a3))**2
               + (Qbar - (b1*(Vbar*Vbar)+b2*Vbar+b3))**2 )/len(Vbar)
    
def Constrain(Params):
    """Constraint for ZIP modeling - """
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 1.0

# These functions aren't currently being used.
'''
def Constrain2(Params):
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 2.0

def Constrain3(Params):
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 3.0
'''


def polyToGLD(p, q):
    """Takes polynomial ZIP coefficients and converts them to GridLAB-D format.
    
    GridLAB-D takes in ZIP fractions and 'power factors' (cosine of the angle).
    Additionally, a negative power factor is used for leading, and a positive
    power factor is used for lagging. Essentially, a negative PF is a signal
    to flip the imaginary component of apparent power.
    
    NOTE: we're counting on coefficients to come in in 'a, b, c' order, AKA
    impedance, current, power.
    
    So:
        p = (a1, a2, a3)
        q = (b1, b2, b3)
        a1 = Z%cos(thetaZ), b1 = Z%sin(thetaZ)
        a2 = I%cos(thetaI), b2 = I%sin(thetaI)
        a3 = P%cos(thetaP), b4 = P%sin(thetaP)
    """
    # Initialize return
    out = {}
    
    # Track index. Note we're 
    i = 0
    for k in ZIPTerms:
        # Initialize the fraction. Note that this reduces correctly, but loses
        # sign information:
        # a1 = Z%*cos(thetaZ), b1 = Z%*sin(thetaZ) and so on.
        fraction = math.sqrt(p[i]*p[i]+q[i]*q[i])
        
        # Derive the power-factor:
        try:
            pf = abs(p[i]/fraction)
        except ZeroDivisionError:
            # If we divided by zero, simply make the power factor 1
            pf = 1
            
        # match what is done in Gridlab-D
        if p[i] > 0 and q[i] < 0:
            # Leading power factor
            pf *= -1
        elif p[i] < 0 and q[i] < 0:
            # Negative load, flip the fraction
            fraction *= -1
        elif p[i] < 0 and q[i] > 0:
            # Negative load and leading power factor, flip both.
            pf *= -1
            fraction *= -1
        
        # Assign to return.
        out[k + '_fraction'] = fraction
        out[k + '_pf'] = pf
        
        # Increment index counter
        i += 1
    
    return out
    
def gldZIP(V, coeff, Vn):
    """Computes P and Q from ZIP coefficients and voltage as GridLAB-D does.
    
    This is not meant to be optimal/efficient, but rather a rewrite of how
    GridLAB-D performs it for testing purposes.
    
    Check out the 'triplex_load_update_fxn()' in:
    https://github.com/gridlab-d/gridlab-d/blob/master/powerflow/triplex_load.cpp
    """
    # GridLAB-D adjusts coefficients if they don't exactly add to one. Here's a
    # line from triplex_load.cpp:
    # 
    # power_fraction[0] = 1 - current_fraction[0] - impedance_fraction[0];
    
    '''
    # GridLAB-D forces the coefficients to sum to 1 if they don't exactly. This screws things up. 
    # TODO: The application should run a custom GLD build which doesn't do this.
    coeffSum = (coeff['power_fraction'] + coeff['current_fraction']
                + coeff['impedance_fraction'])
    
    if (coeffSum) != 1:
    #if not np.isclose(coeffSum, 1, atol=0.02):
        
        #print('Sum of coefficients is {}, which != 1. Correcting as GridLAB-D does.'.format(coeffSum))
        if coeffSum < 1:
            print('debug.')
        coeff['power_fraction'] = 1 - coeff['current_fraction'] - coeff['impedance_fraction']
    '''
    
    # Loop over the ZIP coefficients and compute each 
    d = {}
    for k in ZIPTerms:
        real = coeff['base_power']*coeff[k+'_fraction']*abs(coeff[k+'_pf'])
        imag = real * math.sqrt(1/coeff[k+'_pf']**2 - 1)
        
        if coeff[k +'_pf'] < 0:
            imag *= -1
            
        d[k] = (real, imag)
    
    # Compute P and Q
    P_z = (V**2/Vn**2) * d['impedance'][0]
    P_i = (V/Vn) * d['current'][0]
    P_p = d['power'][0]
    P = P_z + P_i + P_p
    
    Q_z = (V**2/Vn**2) * d['impedance'][1]
    Q_i = (V/Vn) * d['current'][1]
    Q_p = d['power'][1]
    Q = Q_z + Q_i + Q_p
    
    return P, Q

def cluster(data, nCluster=None, minClusterElements=4, maxTInCluster=2):
    """Cluster data to aid in ZIP modeling.
    
    Clustering will performed on active power and time. Essentially, this
        function decides whether there should be a seperate ZIP fit for each 
        time interval in 'data,' or if some time intervals should be combined
        for ZIP fitting.
        
    INPUTS
    data: pandas dataframe with T, P, Q, and V columns. The T column should be
        datetime64 and the index of the array.
        P: real power
        Q: reactive power
        V: voltage magnitude
        
    nCluster: number of cluster to break the data up. If nCluster is None, the 
        initial number of clusters will be computed based on the number of
        unique 'minute timestamps' in data.
        
    minClusterElements: Minimum number of elements allowed in a cluster. If a
        cluster has less than minClusterElements, the number of clusters will 
        be reduced and clustering performed again.
        
    maxTInCluster: Maximum 'minute timestamps' allowed to be in a cluster. W
    
    OUTPUTS
    labels: Returns the cluster "labels" which is a list of numbers indicating
        the cluster number for each data element. Labels is guaranteed to be
        one to one with data.
    """
    # Ensure our input data is adequately large.
    if data.shape[0] < minClusterElements:
        raise ValueError(('Rows in data ({}) is less than the minimum ' 
                          + 'cluster size ({})').format(data.shape[0],
                                                        minClusterElements))
    
    # Get minutes of each measurement
    M = data.index.minute
    
    if nCluster is None:
        nCluster = len(np.unique(M))

    # Normalize P and M. This tends to force clustering to go along with the
    # minutes.
    
    # normP = 1.0 ; normM = 1.0
    normP = 1 / data['P'].max()
    normM = 1 / max(M)
    
    # Create a pandas dataframe with normalized P and M
    dClust = pd.DataFrame({'P': data['P'] * normP, 
                           'M': M * normM})
    
    # Loop from nCluster down to 1 and perform K Means clustering. Note that
    # we'll force a recluster if there are less than minClusterElements and if
    # there are too many different times in a cluster
    for k in range(nCluster, 0, -1):
        # Initialize K means object.
        KM = KMeans(n_clusters=k)
        
        # Perform the fit.
        KM.fit(dClust)
        
        # Force recluster if there are too few elements in any cluster
        if np.min(np.bincount(KM.labels_)) < minClusterElements:
            # Move on to the next iteration of the loop (less clusters)
            continue
        
        # Determine how many 'minute timestamps' fall into each cluster
        tInClust = np.zeros(k)
        for j in range(k):
            tInClust[j] = len(np.unique(M[KM.labels_ == j]))
        
        # If we don't have too many times in a cluster, we're satisfied: break
        # the loop.
        if np.max(tInClust) <= maxTInCluster:
            break
    
    # At the moment, we're using time as our primary predictor. If the same
    # time interval is spread across multiple clusters, we need to recluster.
    if np.max(tInClust) > maxTInCluster:
        raise UserWarning('Failed to cluster: too many times in a cluster.')
    
    # At the cluster labels to the data and return it.
    labels = np.array(KM.labels_)
    
    return labels
    

if __name__ == '__main__':
    # Connect to the database.
    import db
    import matplotlib.pyplot as plt
    dbObj = db.db(password='')
    # We'll use the 'helper' for times
    from helper import clock
    # Times for "training" data
    st = '2016-01-01 00:00:00'
    et = '2016-01-15 00:00:00'
    # timezone
    tz = 'PST+8PDT'
    # hour we'll use for fitting and predicting
    hour = 9
    
    # Times for "prediction" data
    stP = '2016-01-15 01:00:00'
    etP = '2016-01-29 00:00:00'
    
    # Initialize a clock object for "traning" datetimes.
    clockObjTrain = clock(startStr=st, finalStr=et,
                     interval=3600,
                     tzStr=tz)
    
    clockObjPred = clock(startStr=stP, finalStr=etP, 
                         interval=3600,
                         tzStr=tz)
    
    # nominal voltage
    Vn=240
    
    # Table data is in
    table = 'r2_12_47_2_ami_triplex_15_min'
    # Node name
    nodes = ['tpm0_R2-12-47-2_tm_1_R2-12-47-2_tn_193',
             'tpm0_R2-12-47-2_tm_6_R2-12-47-2_tn_198',
             'tpm0_R2-12-47-2_tm_11_R2-12-47-2_tn_203',
             'tpm4_R2-12-47-2_tm_80_R2-12-47-2_tn_272',
             'tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379',
             'tpm0_R2-12-47-2_tm_7_R2-12-47-2_tn_199',
             'tpm6_R2-12-47-2_tm_32_R2-12-47-2_tn_224',
             'tpm0_R2-12-47-2_tm_4_R2-12-47-2_tn_196',
             'tpm1_R2-12-47-2_tm_22_R2-12-47-2_tn_214',
             'tpm0_R2-12-47-2_tm_145_R2-12-47-2_tn_337',
             'tpm2_R2-12-47-2_tm_29_R2-12-47-2_tn_221',
             'tpm0_R2-12-47-2_tm_152_R2-12-47-2_tn_344',
             'tpm1_R2-12-47-2_tm_136_R2-12-47-2_tn_328',
             'tpm0_R2-12-47-2_tm_135_R2-12-47-2_tn_327',
             'tpm2_R2-12-47-2_tm_137_R2-12-47-2_tn_329',
             'tpm0_R2-12-47-2_tm_168_R2-12-47-2_tn_360'
             ]
    for node in nodes:
        print('*'*80)
        print('Node {}'.format(node))
        # Get data for the node
        d = dbObj.getTPQVForNode(table=table, node=node,
                                 starttime=clockObjTrain.start_dt,
                                 stoptime=clockObjTrain.final_dt)
        
        # Filter out data to get only weekdays at at 9a.m. hour
        dFiltered = d.loc[(d.index.dayofweek >= 0) & (d.index.dayofweek <= 4) & \
                      (d.index.hour == hour), :]
        
        # Cluster data.
        labels = cluster(data=dFiltered)
        
        # Perform a ZIP fit for each cluster.
        
        # Grab unique set of labels
        uLabels = np.unique(labels)
        
        # Initialize list to hold fit for each cluster
        fits = []
        
        # Define solver to use.
        #solver = 'fmin_powell'
        solver = 'SLSQP'
        
        # Initialize arrays for tracking model estimates
        Pest = np.zeros(dFiltered.shape[0])
        Qest = np.zeros(dFiltered.shape[0])
        
        # Loop over each unique label and perform ZIP fit.
        for k in uLabels:
            # Grab labels equal to k for logical indexing
            ind = labels == k
            # Perform the ZIP fit
            fits.append(zipFit(V=dFiltered['V'][ind], P=dFiltered['P'][ind],
                               Q=dFiltered['Q'][ind], Vn=Vn, solver=solver))
            # Evaluate the ZIP fit
            Pest[ind], Qest[ind] = gldZIP(V=dFiltered['V'][ind], coeff=fits[-1],
                                          Vn=Vn)
            
        # Compute the SSE for P and Q
        SSEp = math.sqrt(np.sum((dFiltered['P'] - Pest)**2)/len(Pest))
        SSEq = math.sqrt(np.sum((dFiltered['Q'] - Qest)**2)/len(Qest))
        
        print('\nTraining Error Estimate')
        print('SSE(P) = ', SSEp, '  SSE(Q) = ', SSEq)
        
        # Plot fit
        trainPlot = pd.DataFrame({'P':dFiltered['P'], 'Q':dFiltered['Q'],
                                  'P_est':Pest, 'Q_est':Qest},
                                  index=dFiltered.index)
        trainPlot.plot(); plt.legend(loc='best'); plt.title('Training Data: {}'.format(node))
        
        # Test prediction
        
        # Get data for the node
        d = dbObj.getTPQVForNode(table=table, node=node,
                                 starttime=clockObjPred.start_dt,
                                 stoptime=clockObjPred.final_dt)
        
        # Filter out data to get only weekdays at at 9a.m. hour
        dFiltered = d.loc[(d.index.dayofweek >= 0) & (d.index.dayofweek <= 4) & \
                      (d.index.hour == hour), :]
        
        # Initialize arrays for tracking model estimates
        Pest = np.zeros(dFiltered.shape[0])
        Qest = np.zeros(dFiltered.shape[0])
        
        # Loop over each unique label and perform ZIP fit.
        for k in uLabels:
            # Grab labels equal to k for logical indexing
            ind = labels == k
            # Evaluate the ZIP fit
            Pest[ind], Qest[ind] = gldZIP(V=dFiltered['V'][ind], coeff=fits[-1],
                                          Vn=Vn)
            
        # Compute the SSE for P and Q
        SSEp = math.sqrt(np.sum((dFiltered['P'] - Pest)**2)/len(Pest))
        SSEq = math.sqrt(np.sum((dFiltered['Q'] - Qest)**2)/len(Qest))
        
        print('\nPrediction Error Estimate')
        print('SSE(P) = ', SSEp, '  SSE(Q) = ', SSEq)
        
        # Plot fit
        predPlot = pd.DataFrame({'P':dFiltered['P'], 'Q':dFiltered['Q'],
                                  'P_est':Pest, 'Q_est':Qest},
                                  index=dFiltered.index)
        predPlot.plot(); plt.legend(loc='best'); plt.title('Prediction Data {}'.format(node))
        
    plt.show()