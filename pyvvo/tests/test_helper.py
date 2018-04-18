'''
Created on Apr 18, 2018

@author: thay838
'''
# Get the parent directory on the path:
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import unittest
import cmath
import math
import helper

# Two times pi for conversions.
TPI = 2 * cmath.pi
# Dictionary of complex numbers and their string representations.
COMPLEXDICT = {
    '1+1j': 1+1j,
    '+348863+13.716d VA': 348863 * cmath.exp(13.716/360*TPI*1j),
    '-12.2+13d I': -12.2 * cmath.exp(13/360*TPI*1j),
    '+3.258-2.14890r kV': 3.258 * cmath.exp(-2.14890*1j),
    '-1+2j VAr': -1+2j,
    '+1.2e-003+1.8e-2j d': 1.2e-003 + (1.8e-2)*1j,
    '-1.5e02+12d f': -1.5e02 * cmath.exp(12/360*TPI*1j)
}
# Dictionary of complex numbers and their associated power factors.
# I love how python lets you use complex numbers as dictionary keys.
PFDICT = {
    1+1j: (0.7071067811865475, 'lag'),
    1-1j: (0.7071067811865475, 'lead'),
    -1+1j: (0.7071067811865475, 'lag'),
    -1-1j: (0.7071067811865475, 'lead')
}
# Dictionary for testing the binary width function
BINWIDTHDICT = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 4, 
    11: 4,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 5,
    17: 5
}

# Dictionary for converting lists into binary
BINLISTDICT = {
    0: [0,],
    1: [1,],
    2: [1, 0],
    3: [1, 1],
    4: [1, 0, 0],
    5: [1, 0, 1],
    6: [1, 1, 0],
    7: [1, 1, 1],
    8: [1, 0, 0, 0],
    9: [1, 0, 0, 1],
    10: [1, 0, 1, 0],
    11: [1, 0, 1, 1],
    20: [1, 0, 1, 0, 0],
    170: [0, 1, 0, 1, 0, 1, 0, 1, 0]
}

class Test(unittest.TestCase):

    def test_getComplex(self):
        """getComplex method converts strings to complex numbers."""        
        for k, v in COMPLEXDICT.items():
            with self.subTest(complexStr = k):
                # Grab complex number. Note getComplex also returns units.
                cplx = helper.getComplex(k)[0]
                self.assertTrue(cmath.isclose(cplx, v))
                
    def test_powerFactor(self):
        """powerFactor method gives a power factor from a complex number."""
        for cplx, pair in PFDICT.items():
            with self.subTest(complexNum=cplx):
                pf, direction = helper.powerFactor(cplx)
                self.assertTrue(math.isclose(pf, pair[0]))
                self.assertEqual(direction, pair[1])
                
    def test_binaryWidth(self):
        """binaryWidth computes length of binary number for an integer."""
        for num, width in BINWIDTHDICT.items():
            with self.subTest(integer=num):
                width2 = helper.binaryWidth(num)
                self.assertEqual(width, width2)
                
    def test_bin2int(self):
        """bin2int converts a list representation of binary to an integer."""
        for num, binList in BINLISTDICT.items():
            with self.subTest(binList=binList):
                num2 = helper.bin2int(binList)
                self.assertEqual(num, num2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()