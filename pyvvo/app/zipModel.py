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
import multiprocessing as mp
from queue import Empty, Queue
from time import process_time
import threading

# Installed packages:
import numpy as np
import pandas as pd
import mystic.solvers as my
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.externals.joblib.parallel import parallel_backend

# pyvvo imports
from db import db

# Make numpy error any time we get a floating point error.
# At the moment, workers catch exceptions and so we'll just get NaNs in the
# output file.
np.seterr(all='raise')

# Constant for ZIP coefficients. ORDER MATTERS!
ZIPTerms = ['impedance', 'current', 'power']

# List of available solvers for zipFit
SOLVERS = ['fmin_powell', 'SLSQP']

# Constants for convergence tolerance. My understanding from the scipy docs is
# that once we stop getting FTOL improvement between iterations, we're done. 
FTOL = 5e-5

# Maximum allowed iterations for SLSQP solver.
MAXITER = 500

# GTOL = 5 # Number of iterations without change for fmin_powell

# We'll be running KMeans in a multithreaded manner. Define number of threads.
KMTHREADS = 8

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
    Vbar = V / Vn
    Pbar = P / Sn
    Qbar = Q / Sn

    # Solve.
    if solver == 'fmin_powell':
        '''
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS, contraints=ZIPConstraint, disp=False,
                             gtol=GTOL, ftol=FTOL, full_output=True)
        '''
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS,
                             contraints={'type': 'eq', 'fun': ZIPConstraint},
                             disp=False, ftol=FTOL, full_output=True)
        '''
        # Penalty doesn't seem to work well (vs constraint).
        sol = my.fmin_powell(ZIPObjective, args=(Vbar, Pbar, Qbar), x0=par0,
                             bounds=BOUNDS, penalty=ConstrainMystic,
                             disp=False, ftol=FTOL, full_output=True)
        '''

        # Track the polynomial solution for assignment later.
        poly = sol[0]

        # Get the value of the objective function (so the squared error)
        err = sol[1]

        # Check warnings.
        # TODO: handle failures.
        if sol[4] == 1:
            print('fmin_powell failed: maximum number of function iterations.')
        elif sol[4] == 2:
            print('fmin_powell failed: maximum number of iterations.')

    elif solver == 'SLSQP':
        sol = minimize(ZIPObjective, par0, args=(Vbar, Pbar, Qbar),
                       method='SLSQP',
                       constraints={'type': 'eq', 'fun': ZIPConstraint},
                       bounds=BOUNDS, options={'ftol': FTOL,
                                               'maxiter': MAXITER})

        # Track the polynomial solution for assignment later.
        poly = sol.x

        # Get the value of the objective function (so the squared error)
        err = sol.fun

        if not sol.success:
            # Failed to solve. For now, just print.
            # TODO: handle failures.
            print('SLSQP failed: {}'.format(sol.message))
    else:
        raise UserWarning(
            'Given solver, {}, is not implemented.'.format(solver))

    # Extract the polynomial coefficients
    p = np.array(poly[0:3])
    q = np.array(poly[3:6])

    # Convert the polynomial coefficients to GridLAB-D format (fractions and
    # power factors)
    coeff = polyToGLD(p, q)

    # Collect other useful information
    coeff['base_power'] = Sn
    coeff['error'] = err
    coeff['poly'] = poly

    return coeff


def estimateNominalPower(P, Q):
    """Given a set of apparent power measurements, estimate nominal power.
    
    For now, we'll simply use the median of the apparent power.
    """
    Sn = np.median(np.sqrt(np.multiply(P, P) + np.multiply(Q, Q)))

    # TODO: Should we grab the voltage associated with this Sn to use as the 
    # nominal voltage? The ZIP load model is designed such that at nominal 
    # voltage we get nominal power, and not using the voltage associated with
    # the nominal power breaks away from that.

    # If the median P value is negative, flip Sn.
    # if np.median(P) < 0:
    #    Sn *= -1

    return Sn


def ZIPObjective(Params, Vbar, Pbar, Qbar):
    """Objective function for minimization.
    
    Minimize squared error of the ZIP polynomial.
    
    INPUTS:
    Params: tuple: a1, a2, a3, b1, b2, b3
    Vbar: numpy array of voltage divided by nominal voltage
    Pbar: numpy array of real power divided by nominal apparent power
    Qbar: numpy array of reactive power divided by nominal apparent power
    
    OUTPUT:
    sum((Pbar - (a1*Vbar^2 + a2*Vbar +a3))^2 
        + (Qbar - (b1*Vbar^2 + b2*Vbar + b3)))/length(Vbar)
    """
    # Pre-compute Vbar^2
    Vs = np.square(Vbar)

    # Compute sum of squared error.
    e = np.sum(
        np.square(Pbar - (Params[0] * Vs + Params[1] * Vbar + Params[2])) \
        + np.square(Qbar - (Params[3] * Vs + Params[4] * Vbar + Params[5])))

    # Return squared error normalized by length of elements.
    return e / Vbar.shape[0]


def ZIPConstraint(Params):
    """Constraint for ZIP modeling. Ensure "fractions" add up to one.
    
    a1, b1 = Z%cos(thetaZ), Z%sin(thetaZ)
    a2, b2 = I%cos(thetaI), I%sin(thetaI)
    a3, b3 = P%cos(thetaP), P%sin(thetaP)
    """
    # Extract parameters from tuple.
    # a1, a2, a3, b1, b2, b3 = Params

    # Derive fractions (and power factors, but not using those) from the
    # polynomial coefficients.
    f, _ = getFracAndPF(np.array(Params[0:3]), np.array(Params[3:]))

    # Return the sum of the fractions, minus 1 (optimization solvers call this
    # function as a constraint, and consider it "satisfied" if it returns 0).
    return np.sum(f) - 1

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
    
    INPUTS:
        p: numpy array holding a1, a2, a3 (in order)
        q: numpy array holding b1, b2, b3 (in order)
        
    OUTPUTS:
        dictionary with the following fields:
        impedance_fraction, current_fraction, power_fraction: fraction/pct for
            ZIP coefficients.
        impedance_pf, current_pf, power_pf: signed power factor (cos(theta)) 
            for ZIP coefficients.
    
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
    # Get fractions and power factors
    f, pf = getFracAndPF(p, q)

    # Initialize return
    out = {}

    # Get fractions and power factors into GridLAB-D named parameters. NOTE:
    # this depends on the array elements from getFracAndPF being in the correct
    # order: impedance, current, power. This order needs to match up with the
    # order of ZIPTerms.
    for i, k in enumerate(ZIPTerms):
        # Assign to return.
        out[k + '_fraction'] = f[i]
        out[k + '_pf'] = pf[i]

    # Done. Return.
    return out


def getFracAndPF(p, q):
    """Helper to get ZIP fractions and powerfactors from polynomial terms.
    
    INPUTS:
    p: numpy array holding a1, a2, a3 (in order)
    q: numpy array holding b1, b2, b3 (in order)
    
    OUTPUTS:
    f: numpy array holding Z, I, and P fractions
    pf: numpy array holding impedance power factor, current "", power ""
    """

    # Initialize the fractions. Note that this reduces correctly, but loses
    # sign information:
    #
    # a1 = Z%*cos(thetaZ), b1 = Z%*sin(thetaZ) and so on.
    f = np.sqrt(np.square(p) + np.square(q))

    # Initialize power factors. Using divide's 'out' and 'where' arguments, we 
    # ensure that division by zero results in a 1 for the power factor.
    pf = np.absolute(np.divide(p, f, out=np.ones_like(p), where=(f != 0)))

    # Get boolean arrays for where p and q are positive
    try:
        posP = p > 0
        posQ = q > 0
    except FloatingPointError:
        # This can happen if the optimization totally fails... Maybe if given
        # bad starting point?
        raise

    # To meet GridLAB-D conventions, we need to make the power factor negative
    # if it's leading. We also need to flip the fraction if p is negative.

    # p > 0 and q < 0: leading power factor, flip the pf
    b = posP & (~posQ)
    pf[b] = pf[b] * -1

    # p < 0 and q < 0: negative load, flip fraction
    b = (~posP) & (~posQ)
    f[b] = f[b] * -1

    # p < 0 and q > 0: negative load and leading power factor, flip both
    b = (~posP) & posQ
    f[b] = f[b] * -1
    pf[b] = pf[b] * -1

    return f, pf


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

    # Loop over the ZIP coefficients and compute real and imaginary components
    # for the characteristic (impedance, current, power). 
    d = {}
    for k in ZIPTerms:
        real = coeff['base_power'] * coeff[k + '_fraction'] * abs(
            coeff[k + '_pf'])
        imag = real * math.sqrt(1 / coeff[k + '_pf'] ** 2 - 1)

        # Flip the imaginary sign if the power factor is less than 0 (leading).
        if coeff[k + '_pf'] < 0:
            imag *= -1

        d[k] = (real, imag)

    # Compute P and Q
    P_z = (V ** 2 / Vn ** 2) * d['impedance'][0]
    P_i = (V / Vn) * d['current'][0]
    P_p = d['power'][0]
    P = P_z + P_i + P_p

    Q_z = (V ** 2 / Vn ** 2) * d['impedance'][1]
    Q_i = (V / Vn) * d['current'][1]
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
        # Pandas DataFrame
        NaNSeries = xPrime.isnull().all()
    elif len(xPrime.shape) == 1:
        # Pandas Series
        NaNSeries = xPrime.isnull()

    # Loop and zero out.    
    for index in NaNSeries.index[NaNSeries]:
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

    '''
    if dNorm.isnull().any().any():
        pass
    '''

    # Normalize presentConditions for finding the right cluster.
    pc = featureScale(x=presentConditions, xRef=data.loc[:, useCol])

    # Initialize variables for tracking our best fit.
    bestCoeff = None
    minRMSD = np.inf

    # Loop over cluster counts from highest to lowest.
    for k in range(n, 0, -1):
        '''
        # Initialize K Means cluster object, perform clustering in
        # multi-threaded manner.
        # https://stackoverflow.com/questions/38601026/easy-way-to-use-parallel-options-of-scikit-learn-functions-on-hpc
        # https://github.com/scikit-learn/scikit-learn/blob/ed5e127b2460b94dbf3398d97990cb54f188d360/sklearn/externals/joblib/parallel.py
        with parallel_backend('threading'):
        
            KM = KMeans(n_clusters=k, random_state=randomState,
                        n_jobs=KMTHREADS)
        '''

        # Initialize K Means cluster object.
        KM = KMeans(n_clusters=k, random_state=randomState)

        try:
            # Perform the clustering.
            KM.fit(dNorm)
        except Exception:
            # If the clustering failed in some way, just move on to the
            # next possibility.
            # TODO: what if all fail?
            continue

        # Grab cluster centers.
        centers = KM.cluster_centers_

        # Remove the P and Q information.
        centers = np.delete(centers, excludeColInd, axis=1)

        # Use squared Euclidean distance to pick a center.
        minDistance = np.inf
        bestLabel = None

        for rowInd in range(centers.shape[0]):
            sqDist = ((pc - centers[rowInd]) ** 2).sum()

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


def fitForNode(dataIn, randomState=None):
    """Perform a cluster (optional) and ZIP fit for a given node and times.
    
    INPUTS:
    dbObj: db.db object for managing MySQL database interactions.
    dataIn: Dictionary. Fields described below.
    
        REQUIRED FIELDS:
        table: table in database to use.
        node: name of node to pull from database.
        node_data: pandas DataFrame with data for this node. Data should come
            from call to db.getTPQVForNode
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
    # Ensure our filter matches our data
    if len(dataIn['timeFilter']) != dataIn['node_data'].shape[0]:
        raise ValueError('Given bad time filter or start/stop times!')

    # Filter data by time.
    d = dataIn['node_data'].loc[dataIn['timeFilter'], :]

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

        # Drop it from the DataFrame so we don't include it in our fitting.
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


def fitForNodeWorker(inQ, outQ, tx=None, randomSeed=None):
    """Function designed to perform ZIP fits in a parallel manner (on a
        worker). This should work for either a thread or process. Since this
        is CPU bound, processes make more sense, threads may not provide much
        parallelization.
    
    INPUTS:
        inQ: multiprocessing JoinableQueue which will be have data needed to
            perform the fit inserted into it. Each item will be a dictionary.
            See comments for the 'dataIn' input to the 'fitForNode' function to
            see all the fields.
        outQ: multiprocessing Queue which will have results put in it.
        tx: multiprocessing Pipe connection for sending.
        randomSeed: integer for seeding random number generator.
        
    OUTPUTS:
        A dictionary is placed into the outQ. To see all the fields, see the
        comments for the returned dictionary in the 'fitForNode' function. This
        function adds a 'processTime' field which tracks how long it took to
        call fitForNode.
    """

    # Get random state (used for clustering).
    random_state = np.random.RandomState(seed=randomSeed)

    # Enter loop which continues until signal received to terminate.
    while True:
        # Pull data out of the input queue.
        data_in = inQ.get()

        # None will signal termination of the process.
        if data_in is None:
            break

        # Time this run.
        t0 = process_time()

        # Perform ZIP fitting.
        try:
            fit_data = fitForNode(dataIn=data_in, randomState=random_state)
        except Exception as e:
            # Something went wrong, simply put the node name in the queue.
            fit_data = data_in['node']
        else:
            # Assign timing, put dictionary in queue.
            fit_data['processTime'] = process_time() - t0
            fit_data['database_time'] = data_in['database_time']

        finally:

            # Always put data in the output queue.
            outQ.put(fit_data)

            # Always mark the task as done so the program doesn't hang while
            # we wait.
            inQ.task_done()

            # Always notify that we've finished with this node.
            if tx is not None:
                tx.send(data_in['node'])

        # Continue to next loop iteration.


def computeRMSD(actual, predicted):
    """Root-mean-square deviation for two numpy arrays. These should be nx1.
    """
    out = math.sqrt(np.sum(np.square(actual - predicted)) / actual.shape[0])
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
    lowerTime = clockObj.timeDiff(clockField, -numInterval * interval).time()
    upperTime = clockObj.timeDiff(clockField, numInterval * interval).time()

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
        # Times are equal. This can happen for the "fall back" portion of DST.
        # I suppose we'll use 'or'?
        timeFilter = upperBool | lowerBool
        print('Lower and upper times are equal. This is likely DST.')

    # Construct overall time filter.
    overallFilter = DoWBool & timeFilter

    return overallFilter


def database_worker(db_obj, thread_queue, process_queue):
    """Function for threads to get node data from the database.
    """
    while True:
        # Grab data from the thread_queue.
        data_in = thread_queue.get()

        # None will signal termination of the thread.
        if data_in is None:
            break

        # Time database access.
        t0 = process_time()

        # Get data for this node from the database.
        data_in['node_data'] = \
            db_obj.getTPQVForNode(table=data_in['table'], node=data_in['node'],
                                  starttime=data_in['starttime'],
                                  stoptime=data_in['stoptime'])

        # Assign timing.
        data_in['database_time'] = process_time() - t0

        # Put the dictionary in the process_queue. NOTE: this will block until
        # a free slot is available.
        process_queue.put(data_in)

        # Mark this task as complete.
        thread_queue.task_done()

        # Continue to next loop iteration.


def update_progress(receivers, nodes_in_progress):
    """Helper function to update what nodes we're waiting on for fitting."""
    # Wait until a receiver (or receivers) has a node for us.
    # ready_rx = mp.connection.wait(receivers)
    #for rx in ready_rx:
    for rx in receivers:
        # Attempt to check if data is ready.
        try:
            data_ready = rx.poll()
        except BrokenPipeError:
            # What's going on here?
            # TODO: remove this catch
            pass

        if data_ready:
            # Remove this node from the in progress list.
            try:
                nodes_in_progress.remove(rx.recv())
            except ValueError:
                # No idea how this is happening...
                # TODO: WHY?!
                pass

    return nodes_in_progress


def get_and_start_processes(num_processes, pipe, process_in_queue,
                            process_out_queue, seed):
    """Start processes. Returns multiprocessing Process objects.

    INPUTS:
    num_processes: number of processes to use.
    pipe: boolean. True --> give process transmitting end of a pipe.
    process_in_queue: multiprocessing JoinableQueue for input to Processes
    process_out_queue: multiprocessing Queue for output from Processes
    seed: random seed to use in Processes
    """

    # Initialize list to hold process objects.
    process_objects = []

    # Initialize list to hold pipe connections (receiving end).
    if pipe:
        receivers = []

    # Initialize key word arguments for fitForNodeWorker function
    func_args = {'inQ': process_in_queue, 'outQ': process_out_queue,
                 'tx': None, 'randomSeed': seed}

    # Create pipe for worker, start each worker.
    for _ in range(num_processes):

        # Get connections for unidirectional pipe.
        if pipe:
            rx, tx = mp.Pipe(duplex=False)
            receivers.append(rx)
            func_args['tx'] = tx

        # Initialize process.
        this_process = mp.Process(target=fitForNodeWorker,
                                  kwargs=func_args)

        # Start process
        this_process.start()

        # Track.
        process_objects.append(this_process)

    # Return.
    if pipe:
        return process_objects, receivers
    else:
        return process_objects, None


def get_and_start_threads(num_threads, db_obj, thread_queue,
                          process_in_queue):
    """Helper to start threads. Returns list of thread objects.

    INPUTS:
    num_threads: number of threads to use.
    db_obj: db.db object. NOTE: it's pool size should be >= to num_threads
        in order for multi-threaded database access to be effective.
    thread_queue: threading Queue for passing data in to the thread.
    process_in_queue: multiprocessing JoinableQueue for passing data to
        Processes
    """
    # Generate keyword arguments for the database_worker function
    database_worker_args = {'db_obj': db_obj, 'thread_queue': thread_queue,
                            'process_queue': process_in_queue}
    # Start and track threads.
    thread_objects = []
    for _ in range(num_threads):
        # Initialize thread.
        this_thread = threading.Thread(target=database_worker,
                                       kwargs=database_worker_args)

        # Start thread.
        this_thread.start()

        # Track thread.
        thread_objects.append(this_thread)

    return thread_objects

if __name__ == '__main__':
    # We'll use the 'helper' for times
    from helper import clock

    # Times for performing fitting.
    #st = '2016-11-06 00:45:00'
    #et = '2016-11-06 02:15:00'
    st = '2016-01-01 05:00:00'
    et = '2016-01-01 06:00:00'
    #st = '2016-02-01 00:00:00'
    #et = '2016-08-01 00:00:00'
    # timezone
    tz = 'PST+8PDT'

    # Set random seed.
    seed = 42

    # Define our data interval (15 minutes).
    intervalMinute = 15
    intervalSecond = intervalMinute * 60

    # Use a two week window for grabbing historic data
    window = 60 * 60 * 24 * 7 * 2

    # Initialize a clock object for "training" datetimes.
    clockObj = clock(startStr=st, finalStr=et,
                     interval=intervalSecond,
                     tzStr=tz, window=window)

    # nominal voltage
    Vn = 240

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
    '''

    nodes = ['tpm0_R2-12-47-2_tm_1_R2-12-47-2_tn_193',
             'tpm0_R2-12-47-2_tm_6_R2-12-47-2_tn_198']
    '''

    # Initialize some data frames for holding results.
    outDf = pd.DataFrame

    # Hard-code output names
    outDfName = 'cluster' + '_' + solver

    # We'll use threads to pull node data to feed the processes.
    THREADS = 8
    print('Using {} threads for database access'.format(THREADS))

    # Define how many processors/processes to use
    # PROCESSES = mp.cpu_count() - 1
    PROCESSES = 7
    print('Using {} cores.'.format(PROCESSES))

    # Connect to the database.
    dbInputs = {'password': '', 'pool_size': THREADS+1}
    db_obj = db(**dbInputs)

    # Initialize queue for threads.
    thread_queue = Queue()

    # Initialize queues for processes. We'll limit it's size so we don't pull
    # in too much data and blow up our memory needs. For now, we're working
    # with only 16 nodes, so we'll hard-code cap it there.
    process_in_queue = mp.JoinableQueue(maxsize=16)
    process_out_queue = mp.Queue()

    # Start and track threads.
    thread_objects = get_and_start_threads(num_threads=THREADS, db_obj=db_obj,
                                           thread_queue=thread_queue,
                                           process_in_queue=process_in_queue)

    # Start and track processes.
    process_objects, _ = \
        get_and_start_processes(num_processes=PROCESSES, pipe=False,
                                process_in_queue=process_in_queue,
                                process_out_queue=process_out_queue, seed=seed)
    '''
    process_objects, receivers = \
        get_and_start_processes(num_processes=PROCESSES, pipe=True,
                                process_in_queue=process_in_queue,
                                process_out_queue=process_out_queue, seed=seed)
    '''

    # Flag for writing headers to file.
    headerFlag = True

    # Initialize dictionary for tracking previous fits to use as a starting
    # condition for the next fit.
    prevPoly = {}

    # Loop over time to perform fits and predictions.
    while clockObj.stillTime():
        # Get list of nodes which are in progress (haven't had a fit performed
        # yet)
        #nodes_in_progress = list(nodes)

        # Grab times to use for this interval
        windowStart, windowEnd = clockObj.getWindow()
        clockStart, clockStop = clockObj.getStartStop()

        # Get the climate data for window up to start time. This will include
        # present conditions for testing purposes.
        climateData = db_obj.getTempAndFlux(table=climateTable,
                                            starttime=windowStart,
                                            stoptime=clockStart)

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
            thread_queue.put({'table': table, 'node': node,
                              'starttime': windowStart, 'stoptime': clockStart,
                              'cluster': True, 'mode': 'test',
                              'timeFilter': timeFilter,
                              'climateData': climateData, 'minClusterSize': 4,
                              'Vn': Vn, 'solver': solver, 'poly': poly})

        # Wait for database work to be done.
        thread_queue.join()
        print('Database fetching complete.')

        # Spin up another process to help, now that we've freed up a core from
        # performing database access.
        #
        # TODO: Why does spinning up this process (then trying to kill one)
        # cause the program to hang (and only on the second time through the
        # loop)?
        '''
        mp.Process(target=fitForNodeWorker, args=(process_in_queue,
                                                  process_out_queue,
                                                  seed)).start()
        '''

        '''
        # Loop until all nodes have a fit. This is for debugging, but could be
        # useful to log status as we go.
        while process_out_queue.qsize() < len(nodes):
            # Track what nodes are in progress
            nodes_in_progress = \
                update_progress(receivers=receivers,
                                nodes_in_progress=nodes_in_progress)

        # Track progress once more.
        nodes_in_progress = \
            update_progress(receivers=receivers,
                            nodes_in_progress=nodes_in_progress)
        '''

        # Wait for multiprocessing work to finish.
        process_in_queue.join()

        '''        
        # Kill a single process (recall we just spun up and extra one).
        process_in_queue.put(None)
        '''

        # Initialize list for dumping queue data to.
        qList = []

        # Get data out of the output queue and into the output DataFrame.
        while True:
            try:
                # Grab data from the queue.
                thisData = process_out_queue.get_nowait()
            except Empty:
                # Queue is empty, so we have all the data.
                break

            # Queue isn't empty.

            # If we received a string, then the optimization failed in some
            # way.
            if type(thisData) is str:
                # Make a simple dictionary and put it in the list. Pandas is
                # smart enough to null out all the other data.
                qList.append({'node': thisData})

                # Move on to the next iteration of the loop.
                continue

            # If we got here, the optimization didn't totally fail. Augment
            # dictionary with timing information.
            thisData['T'] = clockObj.times['start']['str']

            # Check the sum of the fractions.
            fraction_sum = thisData['coeff']['impedance_fraction'] + \
                thisData['coeff']['current_fraction'] + \
                thisData['coeff']['power_fraction']

            # If the sum isn't reasonably close to 1, then we can get
            # failures in our solvers by giving it an invalid starting
            # point.
            #
            # TODO: it'd be nice to know what tolerances we should use...
            coeff_close_to_one = np.isclose(fraction_sum, 1, atol=FTOL)

            # Flatten the 'coeff' return, exclude stuff we don't want.
            for key, item in thisData['coeff'].items():
                if key == 'poly' and coeff_close_to_one:
                    # Track the previous polynomial.
                    prevPoly[thisData['node']] = thisData['coeff']['poly']
                elif key == 'error':
                    # No need to track optimization error.
                    continue

                # Add the item.
                thisData[key] = item

            # Remove the 'coeff' item.
            thisData.pop('coeff')

            # Add this dictionary to the list.
            qList.append(thisData)

        # Create a DataFrame for this timestep, write it to file.
        pd.DataFrame(qList).to_csv(outDfName + '.csv', mode='a',
                                   header=headerFlag)

        # Ensure we only write headers the first time.
        headerFlag = False

        print('Done with time {}.'.format(clockStart))

        # Advance clock.
        clockObj.advanceTime()

    # Send the process the "kill signal."
    for _ in range(PROCESSES):
        process_in_queue.put(None)

    # Send the threads the "kill signal."
    for _ in range(THREADS):
        thread_queue.put(None)

    print('All done.')
