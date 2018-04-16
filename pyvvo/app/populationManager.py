'''
Created on Feb 16, 2018

@author: thay838
'''
from queue import Queue
import threading

class populationManager:
    
    def __init__(self, dbObj, numInd):
        """The population manager will truncate tables when an individual is
        'killed off' during natural selection and manage the list of unique
        ids for the population.
        
        INPUTS:
            dbObj: initialized util/db object.
            numInd: number of individuals in a population.
        """
        # Initialize a queues for handling uids and cleanup
        self.uidQ = Queue()
        self.cleanupQ = Queue()
        
        # Fill up the uidQ. To avoid blocked queues, we'll have double the UIDs
        # available.
        for i in range(numInd*2):
            self.uidQ.put(i)
            
        # Fire up a single thread for cleanup. We could use more in the future
        # if it's necessary.
        self.cleanupThread = threading.Thread(target=cleanupThread,
                                              args=(self.cleanupQ,
                                                    self.uidQ,
                                                    dbObj))
        
    def getUID(self, timeout=None):
        """Simple method to grab a UID from the queue
        """
        uid = self.uidQ.get(block=True, timeout=timeout)
        return uid
    
    def clean(self, tableSuffix, uid, kill):
        """Simple method to put tableSuffix and uid into the cleanupQ.
        """
        self.cleanupQ.put_nowait({'tableSuffix': tableSuffix,
                                  'uid': uid,
                                  'kill': kill})
        
    def wait(self):
        """Simple method to wait until cleanup is done."""
        self.cleanupQ.join()
        
def cleanupThread(cleanupQ, uidQ, dbObj):
    """Function to cleanup individuals in the cleanupQ, and when complete, put
    the freed up UID in the uidQ.
    
    Thread is terminated when a 'None' object is put in the cleanupQ
    
    INPUTS:
        cleanupQ: queue for cleanup. Each element placed in the queue should 
            be a dict with fields 'tableSuffix,' 'uid,' and 'kill'
        uidQ: queue to put freed up UIDs in when they're available
        dbObj: initialized util/db object to handle database interactions.
    """
    while True:
        # Grab a dictionary from the queue.
        inDict = cleanupQ.get()
        
        # Check input. If None, we're done here.
        if inDict is None:
            cleanupQ.task_done()
            break
        
        # Truncate the individual's tables.
        dbObj.truncateTableBySuffix(suffix=inDict['tableSuffix'])
        
        # If the individual is being killed, make their uid available.
        if inDict['kill']:
            uidQ.put_nowait(inDict['uid'])
        
        # Mark the cleanup task as complete
        cleanupQ.task_done()
