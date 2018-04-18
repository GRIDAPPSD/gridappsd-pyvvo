'''
Created on Apr 18, 2018

@author: thay838
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import cmath
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

class Test(unittest.TestCase):

    def test_getComplex(self):
        """Test the getComplex method, which handles different complex formats.
        """        
        for k, v in COMPLEXDICT.items():
            with self.subTest(complexStr = k):
                # Grab complex number. Note getComplex also returns units.
                cplx = helper.getComplex(k)[0]
                self.assertTrue(cmath.isclose(cplx, v))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()