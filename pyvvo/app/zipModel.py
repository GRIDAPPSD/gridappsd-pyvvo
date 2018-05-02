'''
Created on Apr 27, 2018

@author: thay838

Original R code written by Dave Engel. R code adapted to Python by Bin Zheng.
Final adoption into application by Brandon Thayer.

Notes from Dave:
Augmented Lagrangian Adaptive Barrier Minimization
'''
import numpy as np
import math
import mystic.solvers as my
from scipy.optimize import minimize

# Constant for ZIP coefficients. ORDER MATTERS!
ZIPTerms = ['impedance', 'current', 'power']

# List of available solvers for zipFit
SOLVERS = ['fmin_powell', 'SLSQP']

# Constants for convergence tolerance
FTOL = 1e-8
GTOL = 5 # Number of iterations without change for fmin_powell

def zipFit(V, P, Q, Vn=240.0, solver='fmin_powell'):
    """Solve for ZIP coefficients usable by GridLAB-D.
    
    V: voltage magnitude array
    P: real power array
    Q: reactive power array
    Vn: nominal voltage
    solver: either 'fmin_powell' to use mystic's modified scipy fmin_powell 
        solver, or 'SLSQP' to use scipy's sequential least squares programming
        solver.
    """
    # Estimate nominal power
    Sn = estimateNominalPower(P=P, Q=Q)
    
    # Massage into standard polynomial format
    Vbar = V/Vn
    Pbar = P/Sn
    Qbar = Q/Sn
    
    # Initial parameters for ZIP model
    # TODO: initialize from previously computed coefficients to start.
    # Why are we multiply by 1/18?
    par0 = np.ones(6)*(1/18)
    
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
    d = {}
    for k in ZIPTerms:
        real = coeff['base_power']*coeff[k+'_fraction']*abs(coeff[k+'_pf'])
        imag = real * math.sqrt(1/coeff[k+'_pf']**2 - 1)
        
        if coeff[k +'_pf'] < 0:
            imag *= -1
            
        d[k] = (real, imag)
    
    # Compute P and Q
    P = (
        (V**2/Vn**2) * d['impedance'][0]
        + (V/Vn) * d['current'][0]
        + d['power'][0]
        )
    
    Q = (
        (V**2/Vn**2) * d['impedance'][1]
        + (V/Vn) * d['current'][1]
        + d['power'][0]
        )
    
    return P, Q

if __name__ == '__main__':
    # Get voltage array
    V = np.arange(0.95*240, 1.05*240)
    Vn = 240
    #**************************************************************************
    # Constant current
    I = 1+1j
    S = V * np.conjugate(I)
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zipFit(V, P_constI, Q_constI, solver='fmin_powell')
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zipFit(V, P_constI, Q_constI, solver='SLSQP')
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant current test.')
    #**************************************************************************
    # Constant impedance
    Z = 1+1j
    I = V / Z
    S = V * np.conjugate(I)
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zipFit(V, P_constI, Q_constI, solver='fmin_powell')
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zipFit(V, P_constI, Q_constI, solver='SLSQP')
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant impedance test.')
    #**************************************************************************
    # Constant power
    S = np.ones_like(V) * 1+1j
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zipFit(V, P_constI, Q_constI, solver='fmin_powell')
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zipFit(V, P_constI, Q_constI, solver='SLSQP')
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant power test.')
    #**************************************************************************
    # Mixture
    # Constant impedance:
    Z = 1+1j
    I_z = V /Z
    S_z = V * np.conjugate(I_z)
    # Constant current:
    I = 1+1j
    S_i = V * np.conjugate(I)
    # Constant power.
    S_p = np.ones_like(V) * 1+1j
    # Combine
    S_tot = S_z + S_i + S_p
    P_tot = np.real(S_tot)
    Q_tot = np.imag(S_tot)
    m = zipFit(V, P_tot, Q_tot, solver='fmin_powell')
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zipFit(V, P_tot, Q_tot, solver='SLSQP')
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant impedance test.')
    
    # TODO