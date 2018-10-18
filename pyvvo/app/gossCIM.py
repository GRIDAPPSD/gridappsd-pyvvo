'''
Created on Apr 25, 2018

@author: thay838
'''
import sys
import os
import simplejson as json
import time
#*******************************************************************************
# Import gridappsd-python
# Get this directory
F = os.path.abspath(__file__)
D = os.path.dirname(F)

# We need to import from gridappsd-python, which should be two levels
# above the 'app' directory.

# Walk up until we're in 'app' (we should be to start)
while os.path.basename(D) != 'app':
    D = os.path.dirname(D)
    if D == '/':
        # TODO: this won't work on Windows.
        raise UserWarning('Walked up to root, no directory named "app"')

# Strip two entries from the 'D' to effectively move up to the top of the
# repository.
for _ in range(2):
    D = os.path.dirname(D)
    
# Add gridappsd-python to the path.
sys.path.insert(0, os.path.join(D, 'gridappsd-python'))

from gridappsd import GridAPPSD
#*******************************************************************************
# CONSTANTS
# Topic for simulation log?
SIM_LOG = "/topic/goss.gridappsd.process.log.simulation"
SIM_OUT = "/topic/goss.gridappsd.fncs.output"
#*******************************************************************************
# Class for listening to the GOSS bus and taking action 
class GOSSListener(object):
    def __init__(self, simID, measDict, appName):
        """Initialize a GOSSListener."""
        # Set properties from inputs
        self.measDict = measDict
        self.simID = simID
        self.appName = appName
        
        # Create a GridAPPSD object and subscribe to simulation output.
        self.gossObj = GridAPPSD(simulation_id=self.simID, source=self.appName,
                                 base_simulation_status_topic=SIM_LOG)
        self.gossObj.subscribe(SIM_OUT, self)
        
        # Initialize dictionary for storing simulation data_ls
        # TODO: we probably shouldn't stack this up in memory... recipe for 
        # disaster.
        self.data = {}
    
    def on_message(self, headers, msg):
        """Forward relevant information along."""
        # Get the message as a dictionary.
        msg = json.loads(msg)
        
        # Grabe the timestamp
        t = int(msg['timestamp'])
        
        # Ensure we don't have this timestamp already.
        try:
            self.data[t]
        except KeyError:
            # Good, we don't have the key. Create it.
            self.data[t] = {}
        else:
            # This key already exists? We're in trouble...
            raise UserWarning("We already have data_ls for time {}".format)
        
        # Get the simulation output as a dictionary.
        # May need to wrap this in a try-catch, TBD
        simOut = json.loads(msg['output'])
        
        # Loop over our measurements, extract info from the sim output
        for k in self.measDict:
            self.data[t][k] = simOut[self.simID][k]
            print(simOut[self.simID][k])
            
if __name__ == '__main__':
    obj = GOSSListener(simID='715107754',
                       measDict={"cap_capbank2b": 'stuff'},
                       appName='pyvvo')
    msg = {"command": "nextTimeStep", "currentTime": 0}
    for x in range(8):
        time.sleep(1)
        msg["currentTime"] += 1
        print('Incremented time, sending to FNCS.')
        obj.gossObj.send("goss.gridappsd.fncs.input", json.dumps(msg))
        
    # Let's inspect the the object's data_ls dictionary.
    print('Do it!')
    
    # Just wait...
    time.sleep(60)
    