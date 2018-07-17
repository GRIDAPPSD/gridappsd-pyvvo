'''
Created on Apr 27, 2018

@author: thay838

Original R code written by Dave Engel. R code adapted to Python by Bin Zheng.
Final adoption into application by Brandon Thayer.

Notes from Dave:
Augmented Lagrangian Adaptive Barrier Minimization
'''
# Standard library:
import math
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from queue import Empty as EmptyQueue

# Installed packages:
import numpy as np
import pandas as pd
import mystic.solvers as my
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Constant for ZIP coefficients. ORDER MATTERS!
ZIPTerms = ['impedance', 'current', 'power']

# List of available solvers for zipFit
SOLVERS = ['fmin_powell', 'SLSQP']

# Constants for convergence tolerance
FTOL = 1e-8
# GTOL = 5 # Number of iterations without change for fmin_powell

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

# Force all polynomial terms to be between -1.5 and 1.5 Here are the polynomial
# terms:
#
# a1 = Z%cos(thetaZ), b1 = Z%sin(thetaZ)
# a2 = I%cos(thetaI), b2 = I%sin(thetaI)
# a3 = P%cos(thetaP), b4 = P%sin(thetaP)
#
# Note that sin and cos are always between -1 and 1, and our ZIP fractions
# shouldn't normally exceed 1. Since it's possible to get negative fraction
# terms, we should allow terms to go slightly higher than -1, 1, hence the -1.5
# to 1.5
BOUNDS = [(-1.5, 1.5) for x in range(6)]

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
        '''
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS, contraints=ZIPConstraint, disp=False,
                             gtol=GTOL, ftol=FTOL, full_output=True)
        '''
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS,
                             contraints={'type':'eq', 'fun': ZIPConstraint},
                             disp=False, ftol=FTOL, full_output=True)
        '''
        # Penalty doesn't seem to work well (vs constraint).
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS, penalty=ConstrainMystic,
                             disp=False, ftol=FTOL, full_output=True)
        '''
        # Extract the polynomial coefficients
        p = sol[0][0:3]
        q = sol[0][3:6]
        
        # Get the value of the objective function (so the squared error)
        err = sol[1]
        
        # Check warnings.
        # TODO: handle failurs.
        if sol[4] == 1:
            print('fmin_powell failed: maximum number of function iterations.')
        elif sol[4] == 2:
            print('fmin_powell failed: maximum number of iterations.')
            
    elif solver == 'SLSQP':
        sol = minimize(ZIPObjective, par0, args=(Vbar, Pbar, Qbar), method='SLSQP',
                       constraints={'type':'eq', 'fun': ZIPConstraint},
                       bounds=BOUNDS, options={'ftol': FTOL})
        
        # Extract the polynomial coefficients
        p = sol.x[0:3]
        q = sol.x[3:6]
        
        # Get the value of the objective function (so the squared error)
        err = sol.fun
        
        if not sol.success:
            # Failed to solve. For now, just print.
            # TODO: handle failures.
            print('SLSQP failed: {}'.format(sol.message))
    else:
        raise UserWarning('Given solver, {}, is not implemented.'.format(solver))
        
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

def ZIPObjective(Params, Vbar, Pbar, Qbar):
    """Objective function for minimization.
    
    Minimize squared error of the ZIP polynomial.
    """
    a1, a2, a3, b1, b2, b3 = Params
    return sum( (Pbar - (a1*(Vbar*Vbar)+a2*Vbar+a3))**2
               + (Qbar - (b1*(Vbar*Vbar)+b2*Vbar+b3))**2 )/len(Vbar)
    
def ZIPConstraint(Params):
    """Constraint for ZIP modeling. Ensure "fractions" add up to one.
    
    a1, b1 = Z%cos(thetaZ), Z%sin(thetaZ)
    a2, b2 = I%cos(thetaI), I%sin(thetaI)
    a3, b3 = P%cos(thetaP), P%sin(thetaP)
    """
    # Extract parameters from tuple.
    a1, a2, a3, b1, b2, b3 = Params
    
    # Derive power factors/fractions from the polynomial coefficients.
    coeff = polyToGLD((a1, a2, a3), (b1, b2, b3))
    
    # Return the sum of the fractions, minus 1 (optimization solvers call this
    # function as a constraint, and consider it "satisfied" if it returns 0).
    return coeff['impedance_fraction'] + coeff['current_fraction'] + \
        coeff['power_fraction'] - 1
        
    """
    NOTE: code below is what we were originally doing. After switching to the 
    code above (call polyToGLD, get fractions, sum), we saw the optimization 
    routines doing a much better job meeting the constraint.
    # Use sin^2(theta) + cos^2(theta) = 1 identity to extract fractions, sum
    # them up, subtract 1.
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + \
        math.sqrt(a3*a3 + b3*b3) - 1.0
    """
    

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
            
        # match what is done in Gridlab-D.
        # TODO: update so we aren't flipping the fraction here? We should
        # probably instead use a negative Sn if the "load" is exporting power.
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

def featureScale(x, xRef=None):
    """Helper function to perform feature scaling.
    
    INPUTS:
    x: pandas DataFrame or Series.
    xRef: reference pandas DataFrame.
    
    If only x is provided, x will be normalized against itself.
    If xRef is additionally supplied, x will be normalized against xRef
    
    OUTPUTS:
    xPrime: pandas DataFrame (or Series, depending on type of x). Each column
        is scaled to that all values fall in the range [0, 1]
    """
    if xRef is None:
        xRef = x
    
    xPrime = (x - xRef.min()) / (xRef.max() - xRef.min())
    
    # If an entire column is NaN, zero it out.
    if len(xPrime.shape) > 1:
        # Pandas Dataframe
        NaNSeries = xPrime.isnull().all()
    elif len(xPrime.shape) == 1:
        # Pandas Series
        NaNSeries = xPrime.isnull()
    
    # Loop and zero out.    
    for index, value in NaNSeries.iteritems():
        if value:
            xPrime[index] = 0
    
    return xPrime

def findBestClusterFit(data, presentConditions, minClusterSize=4, Vn=240,
                       solver='SLSQP', randomState=None, poly=None):
    """
    
    INPUTS:
    data: pandas DataFrame containing the data to be used for clustering.
    presentConditions: pandas Series containing data for selecting a cluster
    minClusterSize: integer defining the smallest number of data points allowed
        in a cluster that will be used to perform a ZIP fit.
    Vn: nominal voltage 
    solver: solver (in SOLVERS) to use
    randomState: numpy random.randomState object for reproducable experiments.
    poly: polynomial to use for starting conditions for the ZIP fit.
    """
    # Compute maximum possible clusters:
    n = np.floor(data.shape[0] / minClusterSize).astype(int)
    
    # Match up columns in data and presentConditions for in determining which
    # cluster to use.
    useCol = []
    excludeColInd = []
    for colInd, colName in enumerate(data.columns):
        if colName not in presentConditions.index:
            # This column shouldn't be used when determining which cluster to
            # use for fitting.
            excludeColInd.append(colInd)
        else:
            useCol.append(colName)
            
    # Ensure presentConditions is in the same order as 'data'
    presentConditions = presentConditions[useCol]
    
    # Normalize 'data'
    dNorm = featureScale(x=data)
    
    # Normalize presentConditions for finding the right cluster.
    pc = featureScale(x=presentConditions, xRef=data.loc[:, useCol])
    
    # Initialize variables for tracking our best fit.
    bestCoeff = None
    minRMSD = np.inf
    
    # Loop over cluster counts from highest to lowest.
    for k in range(n, 0, -1):
        # Initalize K Means cluster object.
        # TODO: Set this up to run in a multi-threaded manner.
        # https://stackoverflow.com/questions/38601026/easy-way-to-use-parallel-options-of-scikit-learn-functions-on-hpc
        # https://github.com/scikit-learn/scikit-learn/blob/ed5e127b2460b94dbf3398d97990cb54f188d360/sklearn/externals/joblib/parallel.py
        KM = KMeans(n_clusters=k, random_state=randomState)
    
        # Perform the clustering.
        KM.fit(dNorm)
        
        # Grab cluster centers.
        centers = KM.cluster_centers_
        
        # Remove the P and Q information.
        centers = np.delete(centers, excludeColInd, axis=1)
        
        # Use squared Euclidean distance to pick a center.
        minDistance = np.inf
        bestLabel = None
        
        for rowInd in range(centers.shape[0]):
            sqDist = ((pc - centers[rowInd])**2).sum()
            
            # Update min as necessary
            if sqDist < minDistance:
                minDistance = sqDist.sum()
                bestLabel = rowInd
            
        # If this cluster doesn't have enough data in it, move along.
        u, c = np.unique(KM.labels_, return_counts=True)
        if c[u == bestLabel] < minClusterSize:
            continue
        
        # Extract data to perform fit.
        fitData = data.loc[KM.labels_ == bestLabel, ['P', 'Q', 'V']]
        
        fitOutputs = fitAndEvaluate(fitData=fitData, Vn=Vn, solver=solver, 
                                    poly=poly)
        
        # Should we consider the sum of these errors? Only look at P? 
        rmsd = fitOutputs['rmsdP'] + fitOutputs['rmsdQ']
        
        # Track if this is the best so far.
        if rmsd < minRMSD:
            minRMSD = rmsd
            minRMSDP = fitOutputs['rmsdP']
            minRMSDQ = fitOutputs['rmsdQ']
            # Pout = Pest.copy(deep=True)
            # Qout = Qest.copy(deep=True)
            # bestCoeff = copy.deepcopy(coeff)
            bestCoeff = fitOutputs['coeff']
            
    return bestCoeff, minRMSDP, minRMSDQ

def fitAndEvaluate(fitData, Vn, solver, poly=None):
    """Helper to perform and evaluate ZIP fit.
    
    INPUTS: 
    fitData: pandas DataFrame with P, Q, and V columns
    Vn: nominal voltage
    solver: solver to use
    poly: starting condition polynomial values for the fit. If None, the 
        constant PAR0 will be used.
    """
    if poly is None:
        poly = PAR0
    
    # Perform ZIP fit.
    coeff = zipFit(V=fitData['V'], P=fitData['P'],
                    Q=fitData['Q'], Vn=Vn, solver=solver, par0=poly)
        
    # Evaluate the ZIP fit
    Pest, Qest = gldZIP(V=fitData['V'], coeff=coeff, Vn=Vn)
        
    # Compute the root mean square deviation
    rmsdP = computeRMSD(fitData['P'], Pest)
    rmsdQ = computeRMSD(fitData['Q'], Qest)
    
    return {'coeff': coeff, 'Pest': Pest, 'Qest': Qest, 'rmsdP': rmsdP, 
            'rmsdQ': rmsdQ}

def fitForNode(dbObj, dataIn, randomState=None):
    """Perform a cluster (optional) and ZIP fit for a given node and times.
    
    INPUTS:
    dbObj: db.db object for managing MySQL database interactions.
    dataIn: Dictionary. Fields described below.
    
        REQUIRED FIELDS:
        table: table in database to use.
        node: name of node to pull from database.
        starttime: aware datetime indicating inclusive left time bound.
        stoptime: aware datetime indicating inclusive right time bound.
        cluster: boolean flag. If True, data will be clustered and a ZIP
            fit will be computed for the appropriate cluster. If False,
            all the data (after being filtered by timeFilter) is used in
            the ZIP fit.
        mode: 'test' or 'predict.' In 'predict' mode, P, Q, and V are not
            known for the next timestep. In 'test' mode, they are.
        timeFilter: boolean array used to filter data obtained from the
            database. Note that climateData (see below) should have used
            this timeFilter.
        Vn: nominal voltage for the given node.
        solver: solver to use for performing the ZIP fit. Should be in
            the SOLVERS constant.
        poly: polynomial for previous fit for this node. If None, PAR0 constant
            will be used (set later down the function chain).
            
        OPTIONAL/DEPENDENT FIELDS:
        
        temperature_forecast: forecasted temperature for next time. Only
            used if mode is 'predict' and cluster is True.
        solar_flux_forecast: "" solar_flux ""
        climateData: pandas dataframe, indexed by time. Columns are 
            'temperature' and 'solar_flux.' Note that climateData should
            be pre-filtered by the timeFilter. Only used if 'cluster' is
            True.
        minClusterSize: minimum number of datapoints a cluster must have in
            order to be used for fitting. Only used if 'cluster' is True.
            
    randomState: numpy random.RandomState object or None. Used in clustering.
    
    OUTPUTS:
    outDict: Dictionary with the following fields:
        node: repeat of dataIn['node']
        rmsdP_train = Root mean square deviation on the training dataset for P.
        rmsdQ_train = "" for Q.
        coeff = Dictionary of ZIP coefficients from zipFit function
        
        FIELDS IFF dataIn['mode'] == 'test':
        V = Voltage used for the test.
        P_actual = Expected (actual) P value
        Q_actual = "" Q value
        P_estimate = Prediction (estimate) of P given V and ZIP coefficients
        Q_estimate = "" of Q ""
    """
    # Get data from database.
    d = dbObj.getTPQVForNode(table=dataIn['table'], node=dataIn['node'],
                             starttime=dataIn['starttime'],
                             stoptime=dataIn['stoptime'])
    
    # Ensure our filter matches our data
    if len(dataIn['timeFilter']) != d.shape[0]:
        raise ValueError('Given bad time filter or start/stop times!')
    
    # Filter data by time.
    d = d.loc[dataIn['timeFilter'], :]
    
    # If we're clustering, associate the climateData.
    if dataIn['cluster']:
        # Associate climate data. Note the climate data has already been 
        # filtered.
        d = d.merge(dataIn['climateData'], how='outer', on='T')
    
    # Mode will be 'predict' if we don't actually know what P, Q, and V are
    # for the next time step. Mode will be 'test' if we do know, and want
    # are 'testing' our prediction powers. 
    #
    # In 'test' mode, the last row of data is assumed to be the real data
    # for the period which we're testing.
    if dataIn['mode'] == 'test':
        # Grab the data for which we're trying to predict.
        dActual = d.iloc[-1]
        
        # Drop it from the dataframe so we don't include it in our fitting.
        d = d.drop(dActual.name)
        
        # Use the actual data for our "present conditions"
        if dataIn['cluster']:
            # We'll have out climate data.
            fields = ['V', 'temperature', 'solar_flux']
        else:
            # No climate data
            fields = ['V']
            
        pc = dActual.loc[fields]
        
    elif dataIn['mode'] == 'predict':
        pc = pd.Series()
        
        if dataIn['cluster']:
            # Use forecast values for temperature and solar_flux, if they
            # are given. Otherwise, use last available value.
            for f in ['temperature', 'solar_flux']:
                try: 
                    # Use the forecasted value.
                    pc[f] = dataIn[f + '_forecast']
                except KeyError:
                    # Not given a forecast, use last available value.
                    pc[f] = d.iloc[-1][f]
                    
        # Use the last measured voltage.
        # TODO: This is the last field measured voltage. We can probably
        # include the last modeled voltage in here.
        pc['V'] = d.iloc[-1]['V']
        
    else:
        # Mode other than 'test' or 'predict'
        raise ValueError("Unexpected mode, {}".format(dataIn['mode']))
    
    # Initialize return.
    outDict = {}
    outDict['node'] = dataIn['node']
    
    # Either cluster and perform fit or just perform fit.
    if dataIn['cluster']:
        # Cluster by P, Q, V, temp, and solar flux, then perform a ZIP fit.
        coeff, rmsdP, rmsdQ = \
            findBestClusterFit(data=d, presentConditions=pc,
                               minClusterSize=dataIn['minClusterSize'],
                               Vn=dataIn['Vn'], solver=dataIn['solver'],
                               randomState=randomState, poly=dataIn['poly'])
        
        # Put outputs in the outDict.
        outDict['rmsdP_train'] = rmsdP
        outDict['rmsdQ_train'] = rmsdQ
        outDict['coeff'] = coeff
    
    else:
        # No clustering, just fit.
        fitOutputs = fitAndEvaluate(fitData=d, Vn=Vn, solver=solver, 
                                    poly=dataIn['poly'])
        # Put outputs in the outDict.
        outDict['rmsdP_train'] = fitOutputs['rmsdP']
        outDict['rmsdQ_train'] = fitOutputs['rmsdQ']
        outDict['coeff'] = fitOutputs['coeff']

    # If we're testing, perform the test.
    if dataIn['mode'] == 'test':
        # Use these coefficients to predict the next time interval.
        Pest, Qest = gldZIP(V=dActual['V'], coeff=coeff, Vn=dataIn['Vn'])
        
        outDict['V'] = dActual['V']
        outDict['P_actual'] = dActual['P']
        outDict['Q_actual'] = dActual['Q']
        outDict['P_estimate'] = Pest
        outDict['Q_estimate'] = Qest
    
    # All done.
    return outDict

    
def fitForNodeWorker(dbInputs, inQ, outQ, randomSeed=None):
    """Function designed to perform ZIP fits in a parallel manner (on a
        worker). This should work for either a thread or process. Since this
        is CPU bound, processes make more sense, threads may not provide much
        parallelization.
    
    INPUTS:
        dbInputs: dictionary of keyword arguments for db.db constructor.
        inQ: JoinableQueue which will be have data needed to perform the fit
            inserted into it. Each item will be a dictionary. See comments for
            the 'dataIn' input to the 'fitForNode' function to see all the 
            fields.
        outQ: Queue which will have results put in it.
        randomSeed: integer for seeding random number generator.
        
    OUTPUTS:
        A dictionary is placed into the outQ. To see all the fields, see the
        comments for the returned dictionary in the 'fitForNode' function. This
        function adds a 'processTime' field which tracks how long it took to
        call fitForNode.
    """
    from db import db
    from time import process_time
    
    # Get database object.
    dbObj = db(**dbInputs)
    
    # Get random state (used for clustering).
    randomState = np.random.RandomState(seed=randomSeed)
    
    # Enter loop which continues until signal received to terminate.
    while True:
        # Pull data out of the input queue.
        dataIn = inQ.get()
            
        # None will signal termination of the process.
        if dataIn is None:
            break
        
        # Time this run.
        t0 = process_time()
        
        # Perform ZIP fitting.
        outDict = fitForNode(dbObj=dbObj, dataIn=dataIn,
                             randomState=randomState)
        
        # Assign timing.    
        outDict['processTime'] = process_time() - t0
        
        # Put the dictionary in the output queue.
        outQ.put(outDict)
        
        # Mark this task as done.
        inQ.task_done()
        
        # Continue to next loop iteration.
        
def computeRMSD(actual, predicted):
    """Root-mean-square deviation.
    """
    out = math.sqrt(np.sum((actual - predicted)**2)/len(actual))
    return out

def getTimeFilter(clockObj, datetimeIndex, interval, numInterval=2,
                  clockField='start'):
    """Create time filter to use before ZIP fitting.
    
    INPUTS:
    clockObj: helper.clock object.
    datetimeIndex: Pandas Series/DataFrame DatetimeIndex.  
    interval: size of interval in seconds. The filter will include times
        +/ numInterval * interval.
    numInterval: Number of intervals to include in the filter.
    clockField: First input to clockObj.timeDiff. Defaults to 'start'
    """
    # Grab given hour +/- numInterval intervals.
    lowerTime = clockObj.timeDiff(clockField, -numInterval*interval).time()
    upperTime = clockObj.timeDiff(clockField, numInterval*interval).time()

    # Get logical for all days which are of the same type (weekday vs
    # weekend)
    DoWBool = (datetimeIndex.dayofweek >= dayRange[0]) & \
        (datetimeIndex.dayofweek <= dayRange[1])
    # Get logical to ensure we're in the time bounds.
    upperBool = datetimeIndex.time <= upperTime
    lowerBool = datetimeIndex.time >= lowerTime
    
    # Determine how to combine the upper and lower booleans.
    if lowerTime > upperTime:
        # Our times are crossing over the day boundary, need to use 'or.'
        timeFilter = upperBool | lowerBool
    elif lowerTime < upperTime:
        # Times do not cross day boundary. Use 'and.'
        timeFilter = upperBool & lowerBool
    else:
        raise UserWarning('Unexpected behavior... Times are equal...')
    
    # Construct overall time filter.
    overallFilter = DoWBool & timeFilter

    return overallFilter

if __name__ == '__main__':
    # Connect to the database.
    import db
    dbInputs = {'password': '', 'pool_size': 1}
    dbObj = db.db(**dbInputs)
    # We'll use the 'helper' for times
    from helper import clock
    # Times for performing fitting.
    st = '2016-06-01 12:00:00'
    et = '2016-06-01 14:00:00'
    # timezone
    tz = 'PST+8PDT'
    
    # Set random seed.
    seed = 42
    
    # Define our data interval (15 minutes).
    intervalMinute = 15
    intervalSecond = intervalMinute*60
    
    # Use a two week window for grabbing historic data
    window = 60*60*24*7*2
    
    # Initialize a clock object for "traning" datetimes.
    clockObj = clock(startStr=st, finalStr=et,
                     interval=intervalSecond,
                     tzStr=tz, window=window)
    
    # nominal voltage
    Vn=240
    
    # solver to use 
    # solver='fmin_powell'
    solver = 'SLSQP'
    
    # Table data is in
    table = 'r2_12_47_2_ami_triplex_15_min'
    climateTable = 'r2_12_47_2_ami_climate_1_min'
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
    
    # Initialize some data frames for holding results.
    outDf = pd.DataFrame
        
    # Hard-code output names
    outDfName = 'cluster' + '_' + solver
    
    # Initialize queues for parallelization.
    inQ = JoinableQueue()
    outQ = Queue()
    
    # Define how many processers/processes to use
    PROCESSES = cpu_count()
    # PROCESSES = 1
    print('Using {} cores.'.format(str(PROCESSES)))
    
    # Start workers.
    for _ in range(PROCESSES):
        Process(target=fitForNodeWorker,
                args=(dbInputs, inQ, outQ, seed)).start()
    
    # Flag for writing headers to file.
    headerFlag=True
    
    # Initialize dictionary for tracking previous fits to use as a starting
    # condition for the next fit.
    prevPoly = {}
    
    # Loop over time to perform fits and predictions.
    while clockObj.stillTime():
        # Grab times to use for this interval
        windowStart, windowEnd = clockObj.getWindow()
        clockStart, clockStop = clockObj.getStartStop()
        
        # Get climate data for the time in question. We need to reach back by
        # one intervalSecond so that things are correct when we resample.
        climateStart = clockObj.timeDiff(tStr='windowStart',
                                         delta=(-1*intervalSecond))
        '''
        # Call below grabs only historic climate data, what we actually want
        # is to include present conditions for testing.
        climateStop = clockObj.timeDiff(tStr='windowStop',
                                        delta=(-1*intervalSecond))
        '''
        climateStop = clockObj.timeDiff(tStr='start',
                                        delta=(-1*intervalSecond))
        
        climateData = dbObj.getTempAndFlux(table=climateTable,
                                           starttime=climateStart,
                                           stoptime=climateStop)
        
        # Resample to get mean climate data over each 15 minute intervalSecond. This
        # should match up with how AMI averages are computed (data before the
        # stamped time is used to create the average)
        climateData = climateData.resample((str(intervalMinute) + 'Min'),
                                           label='right').mean()
                                           
        # Determine whether we're in a weekday or weekend, grab an inclusive 
        # range to use.
        dayRange = clockObj.dayOfWeekRange()
        
        # Get boolean filters for time.
        
        timeFilter = getTimeFilter(clockObj=clockObj,
                                   datetimeIndex=climateData.index,
                                   interval=intervalSecond,
                                   numInterval=2, clockField='start')
        
        # Filter the climate data.
        climateData = climateData[timeFilter]
        
        # Loop over the nodes.
        for node in nodes:
            # Grab previous polynomial (if it's been set)
            try:
                poly = prevPoly[node]
            except KeyError:
                poly = None
            
            # Put dictionary in the queue for processing.
            inQ.put({'table': table, 'node': node, 'starttime': windowStart,
                     'stoptime': clockStart, 'cluster': True, 'mode': 'test',
                     'timeFilter': timeFilter, 'climateData': climateData,
                     'minClusterSize': 4, 'Vn': Vn, 'solver': solver,
                     'poly': poly})
            
        # Wait for the work to be done.
        inQ.join()
        
        # Initialize list for dumping queue data to.
        qList = []
        
        # Get data out of the output queue and into the output dataframe.
        while True:
            try:
                # Grab data from the queue.
                thisData = outQ.get_nowait()
            except EmptyQueue:
                # Queue is empty, so we have all the data.
                break
            else:
                # Queue isn't empty.
                thisData['T'] = clockObj.times['start']['str']
                # Flatten the 'coeff' return, exclude stuff we don't want
                for key, item in thisData['coeff'].items():
                    if key == 'poly':
                        # Track the previous polynomial.
                        prevPoly[node] = thisData['coeff']['poly']
                    elif key == 'error':
                        # No need to track optimization error.
                        continue
                    
                    # Add the item.
                    thisData[key] = item
                
                # Remove the 'coeff' item.
                # TODO: Track previous 'poly' to use as initial conditions for
                # the next fit.
                thisData.pop('coeff')
                
                # Add this dictionary to the list.
                qList.append(thisData)
            
        # Create a dataframe for this timestep, write it to file.
        pd.DataFrame(qList).to_csv(outDfName + '.csv', mode='a',
                                   header=headerFlag)
        
        # Ensure we only write headers the first time.
        headerFlag = False
        
        print('Done with time {}.'.format(clockStart))

        # Advance clock.
        clockObj.advanceTime()
        
    # Send the process the "kill signal."
    for _ in range(PROCESSES):
        inQ.put(None)
        
    print('All done.')