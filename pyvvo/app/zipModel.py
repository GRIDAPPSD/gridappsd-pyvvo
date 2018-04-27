'''
Created on Apr 27, 2018

@author: thay838

Original R code written by Dave Engel. R code adapted to Python by Bin Zheng.
Final adoption into application by Brandon Thayer.
'''
import numpy as np
import math
import mystic.solvers as my
from scipy.optimize import minimize

# Constant for ZIP coefficients. ORDER MATTERS!
ZIPTerms = ['impedance', 'current', 'power']

def zip_mystic(V, P, Q, Vn=240.0):
    # constants
    par0 = np.ones(6)*(1/18)  # initial parameters for ZIP model
    # scale voltage by nominal volt. (Vn) and the power by the base power (Sn) 
    Sn = np.median(np.sqrt(np.multiply(P,P) + np.multiply(Q,Q)))
    Vbar = V/Vn
    Pbar = P/Sn
    Qbar = Q/Sn

    sol = my.fmin_powell(Object, args=(Vbar, Pbar, Qbar), x0=par0, constraint=Constrain, disp=False,
                         gtol=3, ftol=1e-8, full_output=True)
    # Another algorithm:
    #sol = my.fmin(Object, args=(Vbar, Pbar, Qbar), x0=par0, constraint=Constrain, disp=False,
    #              gtol=3, ftol=1e-8, full_output=True)
          
    xopt = sol[0]
    fopt = sol[1]
    p = xopt[0:3]
    q = xopt[3:6]

    coeff = polyToGLD(p, q)
    coeff['base_power'] = Sn
    coeff['error'] = fopt
    coeff['poly'] = (p, q)
    
    return coeff

def zip_scipy(V, P, Q, Vn=240.0):
    
    par0 = np.ones(6)*(1/18)  # initial parameters for ZIP model
    # scale voltage by nominal volt. (Vn) and the power by the base power (Sn) 
    Sn = np.median(np.sqrt(np.multiply(P,P) + np.multiply(Q,Q)))
    Vbar = V/Vn
    Pbar = P/Sn
    Qbar = Q/Sn
    
    # fit ZIP model
    # Minimize a scalar function of one or more variables using 
    # Sequential Least SQuares Programming (SLSQP).
    cons = {'type':'eq', 'fun': Constrain}
    cons2 = {'type':'eq', 'fun': Constrain2}
    cons3 = {'type':'eq', 'fun': Constrain3}
    
    #print(args[0])
    opti = minimize(Object, par0, args=(Vbar, Pbar, Qbar), method='SLSQP', constraints = cons, bounds = None)
    
    if abs(Constrain(opti.x))>0.75:
        opti = minimize(Object, par0, args=(V1bar, P1bar, Q1bar), method='SLSQP', constraints = cons2, bounds = None)
                        
    if abs(Constrain(opti.x))>2.0:
        opti = minimize(Object, par0, args=(V1bar, P1bar, Q1bar), method='SLSQP', constraints = cons3, bounds = None)
        
    # convert back to original scale
    # opti$par[1] = Z%*cos(Zo), opti$par[2] = I%*cos(Io), opti$par[3] = P%*cos(Po)
    # opti$par[4] = Z%*sin(Zo), opti$par[5] = I%*sin(Io), opti$par[6] = P%*sin(Po)
    p = opti.x[0:3]
    q = opti.x[3:6]
    sci_opt = opti.fun
    
    coeff = polyToGLD(p, q)
    coeff['base_power'] = Sn
    coeff['error'] = sci_opt
    coeff['poly'] = (p, q)
    
    return coeff

def Object(Params, Vbar, Pbar, Qbar):
    a1, a2, a3, b1, b2, b3 = Params
    return sum( (Pbar - (a1*(Vbar*Vbar)+a2*Vbar+a3))**2
               + (Qbar - (b1*(Vbar*Vbar)+b2*Vbar+b3))**2 )/len(Vbar)
    
def Constrain(Params):
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 1.0

def Constrain2(Params):
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 2.0

def Constrain3(Params):
    a1, a2, a3, b1, b2, b3 = Params
    return math.sqrt(a1*a1 + b1*b1) + math.sqrt(a2*a2 + b2*b2) + math.sqrt(a3*a3 + b3*b3) - 3.0


def polyToGLD(p, q):
    """Takes polynomial ZIP coefficients and converts them to GridLAB-D format.
    
    GridLAB-D takes in ZIP fractions and 'power factors' (cosine of the angle).
    Additionally, a negative power factor is used for leading, and a positive
    power factor is used for lagging. Essentially, a negative PF is a signal
    to flip the imaginary component of apparent power.
    
    NOTE: we're counting on coefficients to come in in 'a, b, c' order, AKA
    impedance, current, power
    """
    # Initialize return
    out = {}
    
    # Track index. Note we're 
    i = 0
    for k in ZIPTerms:
        fraction = math.sqrt(p[i]*p[i]+q[i]*q[i])
        pf = 1.0
        if fraction!=0:
            pf = abs(p[i]/fraction)
            # match what is done in Gridlab-D
        if p[i]>0 and q[i]<0:
            # Leading power factor
            pf = (-1.0)*pf
        elif p[i]<0 and q[i]<0:
            # Negative load, flip the fraction
            fraction = (-1.0)*fraction
        elif p[i]<0 and q[i]>0:
            # Negative load and leading power factor, flip both.
            pf = (-1.0)*pf
            fraction = (-1.0)*fraction
        
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
        
        if coeff[k+'_pf'] < 0:
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
    Z = V / I
    S = V * np.conjugate(I)
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zip_mystic(V, P_constI, Q_constI)
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zip_scipy(V, P_constI, Q_constI)
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant current test.')
    #**************************************************************************
    # Constant impedance
    Z = 1+1j
    I = V / Z
    S = V * np.conjugate(I)
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zip_mystic(V, P_constI, Q_constI)
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zip_scipy(V, P_constI, Q_constI)
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant impedance test.')
    #**************************************************************************
    # Constant power
    S = np.ones_like(V) * 1+1j
    P_constI = np.real(S)
    Q_constI = np.imag(S)
    m = zip_mystic(V, P_constI, Q_constI)
    P_M, Q_M = gldZIP(V, m, Vn)
    s = zip_scipy(V, P_constI, Q_constI)
    P_S, Q_S = gldZIP(V, s, Vn)
    print('Finished constant power test.')
    #**************************************************************************
    # Mixture
    # TODO