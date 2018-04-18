'''
Created on Feb 16, 2018

@author: thay838
'''
from queue import Queue
import threading
import logging

class populationManager:
    
    def __init__(self, dbObj, numInd, log=None):
        """The population manager will truncate tables when an individual is
        'killed off' during natural selection and manage the list of unique
        ids for the population.
        
        INPUTS:
            dbObj: initialized util/db object.
            numInd: number of individuals in a population.
            log: logging.Logger instance or None.
        """
        # Set up the log
        if log is not None:
            # Use the given logger.
            self.log = log
        else:
            # Create a basic logger.
            self.log = logging.getLogger()
            
        # Initialize a queues for handling uids and cleanup
        self.uidQ = Queue()
        self.cleanupQ = Queue()
        
        # Fill up the uidQ. To avoid blocked queues, we'll have double the UIDs
        # available.
        for i in range(numInd*2):
            self.uidQ.put(i)
            
        self.log.debug('{} UIDs put in the UID queue'.format(numInd*2))
            
        # Fire up a single thread for cleanup. We could use more in the future
        # if it's necessary.
        self.cleanupThread = threading.Thread(target=cleanupThread,
                                              args=(self.cleanupQ,
                                                    self.uidQ,
                                                    dbObj, self.log,))
        self.cleanupThread.start()
        
        self.log.debug("Cleanup thread started.")
        
    def getUID(self, timeout=None):
        """Simple method to grab a UID from the queue
        """
        uid = self.uidQ.get(block=True, timeout=timeout)
        self.log.debug('UID {} pulled from the UID queue.'.format(uid))
        return uid
    
    def clean(self, tableSuffix, uid, kill):
        """Simple method to put tableSuffix and uid into the cleanupQ."""
        inDict = {'tableSuffix': tableSuffix,'uid': uid, 'kill': kill}
        self.cleanupQ.put_nowait(inDict)
        self.log.debug('Dict put in cleanup queue: {}'.format(inDict))
        
    def wait(self):
        """Simple method to wait until cleanup is done."""
        self.log.debug('Waiting for cleanup to finish...')
        self.cleanupQ.join()
        self.log.debug('Cleanup complete.')
        
def cleanupThread(cleanupQ, uidQ, dbObj, log):
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
            log.debug('Task marked as done. Breaking while loop to terminate thread.')
            break
        
        # Truncate the individual's tables.
        dbObj.truncateTableBySuffix(suffix=inDict['tableSuffix'])
        log.debug('Tables with suffix {} truncated.'.format(inDict['tableSuffix']))
        
        # If the individual is being killed, make their uid available.
        if inDict['kill']:
            uidQ.put_nowait(inDict['uid'])
            log.debug('UID {} put back in the UID queue.'.format(inDict['uid']))
        
        # Mark the cleanup task as complete
        cleanupQ.task_done()
        log.debug('Task marked as done.')
