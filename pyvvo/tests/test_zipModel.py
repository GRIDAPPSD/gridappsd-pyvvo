'''
Created on May 1, 2018

@author: thay838
'''
# Get the parent directory on the path:
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import unittest
import numpy as np

import zipModel

# Get simple voltage array
V = np.arange(0.9*240, 1.1*240)
# Nominal voltage is 240V
Vn = 240

# Tolerance for fit. Let's go with 1%
RTOL=0.01

# Initialize array for testing.
testData = []

# Create constant impedance, constant current, constant power, and mixed test
# cases.
#******************************************************************************
# CONSTANT IMPEDANCE TEST 1
Z = 1+1j
I = V / Z
S = V * np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Impedance Test 1'))
# CONSTANT IMPEDANCE TEST 2
Z = 0.2-15j
I = V / Z
S = V * np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Impedance Test 2'))
# CONSTANT IMPEDANCE TEST 3 (exporting power)
Z = -1200+70j
I = V / Z
S = V * np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Impedance Test 3'))
#******************************************************************************
# CONSTANT CURRENT TEST 1
I = 1+1j
S = V*np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Current Test 1'))
# CONSTANT CURRENT TEST 2
I = 75-4j
S = V*np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Current Test 2'))
# CONSTANT CURRENT TEST 3 (exporting power)
I = -213.542+12.15j
S = V*np.conjugate(I)
testData.append((np.real(S), np.imag(S), 'Constant Current Test 3'))
#******************************************************************************
# CONSTANT POWER TEST 1
S = 1+1j
testData.append((np.real(S), np.imag(S), 'Constant Power Test 1'))
# CONSTANT POWER TEST 2
S = 15000.12-230.25j
testData.append((np.real(S), np.imag(S), 'Constant Power Test 2'))
# CONSTANT POWER TEST 3 (exporting power)
S = -234+42j
testData.append((np.real(S), np.imag(S), 'Constant Power Test 3'))
#******************************************************************************
# MIXED TEST 1
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
S = S_z + S_i + S_p
testData.append((np.real(S), np.imag(S), 'Mixed Test 1'))
# MIXED TEST 2
# Constant impedance:
Z = 120-12j
I_z = V /Z
S_z = V * np.conjugate(I_z)
# Constant current:
I = 72+10j
S_i = V * np.conjugate(I)
# Constant power.
S_p = np.ones_like(V) * 1300-1700j
# Combine
S = S_z + S_i + S_p
testData.append((np.real(S), np.imag(S), 'Mixed Test 2'))
# MIXED TEST 3
# Constant impedance:
Z = 0.06-30j
I_z = V /Z
S_z = V * np.conjugate(I_z)
# Constant current:
I = -120-15j
S_i = V * np.conjugate(I)
# Constant power.
S_p = np.ones_like(V) * 15+3j
# Combine
S = S_z + S_i + S_p
testData.append((np.real(S), np.imag(S), 'Mixed Test 3'))
#******************************************************************************

class Test(unittest.TestCase):


    def test_zipFit(self):
        """Test the zipFit function by giving it inputs from polynomials."""
        # Loop over test cases
        for t in testData:
            # Loop over solvers
            for s in zipModel.SOLVERS:
                with self.subTest('Test Case: {}, Solver: {}'.format(t[2], s)):
                    # Get coefficients
                    coeff = zipModel.zipFit(V=V, P=t[0], Q=t[1], Vn=Vn,
                                            solver=s)
                    # Get GridLAB-D's understanding of P and Q based on V and
                    # the provided coefficients
                    P, Q = zipModel.gldZIP(V, coeff, Vn=Vn)
                    
                    # Test closeness
                    resultP = np.allclose(t[0], P, rtol=RTOL)
                    resultQ = np.allclose(t[1], Q, rtol=RTOL)
                    result = resultP and resultQ
                    
                    # Craft a message.
                    if not (resultP or resultQ):
                        # Both failed.
                        msg = 'Neither P nor Q match for {} with the {} solver.'
                    elif not resultP:
                        # P failed.
                        msg = 'P does not match for {} with the {} solver.'
                    elif not resultQ:
                        # Q failed.
                        msg = 'Q does not match for {} with the {} solver.'
                    else:
                        # Success.
                        msg = 'Success for {} with the {} solver.'
                        print(msg.format(t[2], s))
                    
                    # Execute the test assertion.
                    self.assertTrue(result, msg.format(t[2], s))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()