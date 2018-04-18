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
import gld

# List for testing translate taps. Each element should be in the form:
# [lowerTaps, position, expected Output]
TRANSLATETAPSLIST = [
    # 16 lower taps, position 0 leads to -16
    [16, 0, -16],
    [16, 5, -11],
    [8, 8, 0],
    [4, 8, 4],
    [7, 1, -6]
]

class Test(unittest.TestCase):


    def test_translateTaps(self):
        """translateTaps translates tap position on [0, t + t] to [-t, t]"""
        for el in TRANSLATETAPSLIST:
            with self.subTest(tapList=el):
                out = gld.translateTaps(el[0], el[1])
                self.assertEqual(el[2], out)
    
    def test_inverseTranslateTaps(self):
        """inverseTranslateTaps translates tap on [-t, t] to [0, t + t]"""
        for el in TRANSLATETAPSLIST:
            with self.subTest(tapList=el):
                out = gld.inverseTranslateTaps(el[0], el[2])
                self.assertEqual(el[1], out)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_translateTaps']
    unittest.main()