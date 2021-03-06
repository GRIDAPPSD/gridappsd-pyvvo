'''
Created on Aug 15, 2017

@author: thay838
'''
# Standard library:
import math
import random
import os
from queue import Queue
import threading
import sys
import copy
import logging
import time

# pyvvo
from individual import individual, CAPSTATUS
import populationManager
import helper

# Installed
import simplejson as json

class population:

    def __init__(self, strModel, numInd, numGen, inPath, outDir, reg, cap,
                 starttime, stoptime, timezone, dbObj, recorders,
                 numModelThreads=os.cpu_count(),
                 costs={'realEnergy': 0.00008,
                        'powerFactorLead': {'limit': 0.99, 'cost': 0.1},
                        'powerFactorLag': {'limit': 0.99, 'cost': 0.1},
                        'tapChange': 0.5, 'capSwitch': 2,
                        'undervoltage': 0.05, 'overvoltage': 0.05},
                 probabilities={'top': 0.2, 'weak': 0.8, 'mutate': 0.2,
                                'cross': 0.7, 'capMutate': 0.1,
                                'regMutate': 0.05},
                 baseControlFlag=None,
                 randomSeed=None,
                 gldInstall=None,
                 log=None):
        """Initialize a population of individuals.
        
        INPUTS:
            numInd: Number of individuals to create
            numGen: Number of generations to run
            inPath: Path and filename to the base GridLAB-D model to be 
                modified.
            outDir: Directory for new models to be written.
            reg: Dictionary as described in gld.py docstring
            cap: Dictionary as described in gld.py docstring
            starttime: datetime object representing start of simulation
            stoptime: "..." end "..."
            timezone: timezone string
            dbObj: Initialized util/db.db object.
                TODO: we need to make sure that the number of available
                connections is not less than the number of model threads.
            recorders: dictionary defining recorders which individuals use for
                adding recorders to their models. Check out genetic.individual
                documentation for more details.
            numModelThreads: number of threads for running models. Since the
                threads start subprocesses, this corresponds to number of
                cores used for simulation.
            costs: Dictionary describing costs associated with model fitness.
                energy: price of energy, $/Wh
                tapChange: cost to move one tap one position, $
                capSwitch: cost to switch a single capacitor phase, $
                undervoltage: cost of undervoltage violations, $.
                overvoltage: cost of overvoltage violations, $.
            probabilities: Dictionary describing various probabilities
                associated with the genetic algorithm.
                
                top: THIS IS NOT A PROBABILITY. Decimal representing how many
                    of the top individuals to keep between generations. [0, 1)
                weak: probability of 'killing' an individual which didn't make
                    the 'top' cut for the next generation
                mutate: probability to mutate a given offspring or
                    individual.
                cross: While replenishing population, chance to cross two 
                    individuals (rather than just mutate one individual)
                capMutate: probability each capacitor chromosome gene gets
                    mutated.
                regMutate: probability each regulator chromosome gene gets
                    mutated.
            baseControlFlag: control flag for baseline individual. See inputs
                to an individual's constructor for details.
            randomSeed: integer to seed the random number generator. Python is
                smart enough to seed for all modules below.
            gldInstall: dict with two fields, 'DIR' and 'LD_LIBRARY_PATH'
                - 'DIR' should point to GridLAB-D installation to use.
                - 'LD_LIBRARY_PATH' should be None on Windows, but should point
                    to the necessary lib folder on Linux (/usr/local/mysql/lib)
            log: logging.Logger instance. If none, a simple default log will 
                be used.
        """
        # Set up the log
        if log is not None:
            # Use the given logger.
            self.log = log
        else:
            # Create a basic logger.
            self.log = logging.getLogger()

        # Seed the random number generator.
        random.seed(randomSeed)

        # Set timezone
        self.timezone = timezone
        # Set database object
        self.dbObj = dbObj

        # Set recorders
        self.recorders = recorders

        # Get a population manager for dealing out UIDs and cleaning up the
        # database.
        self.popMgr = populationManager.populationManager(dbObj=dbObj,
                                                          numInd=numInd,
                                                          log=self.log)
        self.log.info('Population manager initialized.')

        # Initialize list to hold all individuals in population.
        self.individualsList = []

        # Assign costs.
        self.costs = costs

        # Assign probabilites.
        self.probabilities = probabilities

        # Set the number of generations and individuals.
        self.numGen = numGen
        self.numInd = numInd

        # Set inPath and outDir
        self.inPath = inPath
        self.outDir = outDir

        # Track the baseControlFlag
        self.baseControlFlag = baseControlFlag

        # GridLAB-D path
        self.gldInstall = gldInstall

        # Initialize queues and threads for running GLD models in parallel and
        # cleaning up models we're done with.
        self.modelThreads = []
        self.modelQueue = Queue()

        # Call the 'prep' function which sets several object attributes AND
        # initializes the population.
        self.prep(starttime=starttime, stoptime=stoptime, strModel=strModel,
                  cap=cap, reg=reg)

        # Start the threads to be used for running GridLAB-D models. These 
        # models are run in a seperate subprocess, so we need to be sure this
        # is limited to the number of available cores.
        for _ in range(numModelThreads):
            t = threading.Thread(target=writeRunEval, args=(self.modelQueue,
                                                            self.costs,
                                                            self.log,))
            self.modelThreads.append(t)
            t.start()

        self.log.info(('Model threads started, population initialization '
                       + 'complete.'))

    def prep(self, starttime, stoptime, strModel, cap, reg, keep=0.1):
        """Method to 'prepare' a population object. This method has two uses:
        initializing the population, and updating it for the next run.
        
        TODO: More inputs (like costs and probabilities) should be added here
        when desired.
        
        INPUTS:
            starttime, stoptime, strModel, cap, and reg are described in 
            __init__.
            
            keep is for keeping individuals between time periods.
                Essentially, we'll be seeding this population with 'keep' of 
                the best individuals.
        """
        self.log.debug('Starting the "prep" function.')
        # Set times.
        self.starttime = starttime
        self.stoptime = stoptime

        # Set population base model
        self.strModel = strModel

        # Set regulators and capacitors as property. Since the population
        # object will modify reg and cap, make deep copies.
        self.reg = copy.deepcopy(reg)
        self.cap = copy.deepcopy(cap)

        # Define some common inputs for individuals
        self.indInputs = {'reg': self.reg, 'cap': self.cap,
                          'starttime': self.starttime,
                          'stoptime': self.stoptime,
                          'timezone': self.timezone,
                          'dbObj': self.dbObj,
                          'recorders': self.recorders,
                          'gldInstall': self.gldInstall}

        # If the population includes a 'baseline' model, we need to track it.
        # TODO: May want to update this to track multiple baseline individuals
        self.baselineIndex = None
        self.baselineData = None

        # Track the best scores for each generation.
        self.generationBest = []

        # Track the sum of fitness - used to compute roulette wheel weights
        self.fitSum = 0

        # Track weights of fitness for "roulette wheel" method
        self.rouletteWeights = None

        # If there are individuals in the list, keep some.
        if len(self.individualsList) > 0:
            # Determine how many to keep
            numKeep = round(len(self.individualsList) * keep)

            # Kill individuals we don't want to keep.
            for ind in self.individualsList[numKeep:]:
                self.popMgr.clean(tableSuffix=ind.tableSuffix, uid=ind.uid,
                                  kill=True)

            # Truncate the list to kill the individuals.
            self.individualsList = self.individualsList[0:numKeep]
            self.log.info('Individual list pruned for the next optimization.')

            # Cleanup and prep the remaining individuals. We'll truncate their
            # tables rather than deleting to save a tiny bit of time.
            for ind in self.individualsList:
                # Truncate tables.
                self.popMgr.clean(tableSuffix=ind.tableSuffix, uid=ind.uid,
                                  kill=False)
                # Prep.
                ind.prep(starttime=self.starttime, stoptime=self.stoptime,
                         reg=self.reg, cap=self.cap)
            self.log.info('Remaining individuals cleaned and prepped.')

        # Initialize the population.
        self.initializePop()

        # Wait for cleanup to be complete.
        self.popMgr.wait()

        self.log.info("Prep function complete.")

    def initializePop(self):
        """Method to initialize the population.
        
        TODO: Make more flexible.
        """
        self.log.debug('Starting the "initializePop" function.')
        # Create baseline individual.
        if self.baseControlFlag is not None:
            # Set regFlag and capFlag
            if self.baseControlFlag:
                # Non-zero case --> volt or volt_var control
                regFlag = capFlag = 3
            else:
                # Manual control, use 'newState'
                regFlag = capFlag = 4

            # Add a baseline individual with the given control flag   
            self.individualsList.append(individual(**self.indInputs,
                                                   uid=self.popMgr.getUID(),
                                                   regFlag=regFlag,
                                                   capFlag=capFlag,
                                                   controlFlag=self.baseControlFlag))

            # Track the baseline individual's index.
            self.baselineIndex = len(self.individualsList) - 1
            self.log.debug('Baseline individual created and added to list.')

        # Create 'extreme' individuals - all caps in/out, regs maxed up/down
        # Control flag of 0 for manual control
        for n in range(len(CAPSTATUS)):
            for regFlag in range(2):
                ind = individual(**self.indInputs,
                                 uid=self.popMgr.getUID(),
                                 regFlag=regFlag,
                                 capFlag=n,
                                 controlFlag=0
                                 )
                self.individualsList.append(ind)
        self.log.debug("'Extreme' individuals created.")

        # Create individuals with biased regulator and capacitor positions
        # TODO: Stop hard-coding the number.
        for _ in range(4):
            ind = individual(**self.indInputs,
                             uid=self.popMgr.getUID(),
                             regFlag=2,
                             capFlag=2,
                             controlFlag=0
                             )
            self.individualsList.append(ind)
        self.log.debug("'Biased' individuals created.")

        # Randomly create the rest of the individuals.
        while len(self.individualsList) < self.numInd:
            # Initialize individual.
            ind = individual(**self.indInputs,
                             uid=self.popMgr.getUID(),
                             regFlag=5,
                             capFlag=5,
                             controlFlag=0
                             )
            self.individualsList.append(ind)
        self.log.debug("Random individuals created.")

        self.log.info('Population of {} individuals initialized.'.format(
            len(self.individualsList)))

    def ga(self):
        """Main function to run the genetic algorithm.
        """
        g = 0
        t0 = time.time()
        model_count = len(self.individualsList)
        # Put all individuals in the queue for processing.
        for ind in self.individualsList:
            self.modelQueue.put_nowait({'individual': ind,
                                        'strModel': self.strModel,
                                        'inPath': self.inPath,
                                        'outDir': self.outDir})
        self.log.info('All individuals put in modeling queue.')
        # Loop over the generations
        while g < self.numGen:
            # Wait until all models have been run and evaluated.
            self.modelQueue.join()
            t1 = time.time()
            self.log.info(('{} model runs complete in {:.0f} seconds for '
                           'generation {}.').format(model_count, t1-t0, g+1))

            # If this is the first generation and we're tracking a baseline, 
            # save the requisite information.
            if (g == 0) and (self.baselineIndex is not None):
                # Get a reference to the individual
                bInd = self.individualsList[self.baselineIndex]
                # Clear the index (individual will get sorted
                self.baselineIndex = None
                # Save information.
                self.baselineData = {'costs': copy.deepcopy(bInd.costs),
                                     'cap': copy.deepcopy(bInd.cap),
                                     'reg': copy.deepcopy(bInd.reg)}
                # Get a well formatted string representation
                self.baselineData['str'] = \
                    helper.getSummaryStr(costs=self.baselineData['costs'],
                                         reg=self.baselineData['reg'],
                                         cap=self.baselineData['cap'])
                self.log.debug('Baseline individual data assigned.')
                self.log.debug('Baseline costs:\n{}'.format(
                    json.dumps(bInd.costs, indent=4)))

            # Sort the individualsList by score.
            self.individualsList.sort(key=lambda x: x.costs['total'])

            # Track best score for this generation.
            self.generationBest.append(self.individualsList[0].costs['total'])
            self.log.info('Lowest cost for this generation: {:.2f}'.format(
                self.generationBest[-1]))

            # Increment generation counter.
            g += 1

            # This could probably be refactored, but anyways...
            # perform natural selection, crossing, mutation, and model runs if
            # we're not in the last generation.
            if g < self.numGen:
                # Select the fittest individuals and some unfit ones.
                self.naturalSelection()
                msg = 'Natural selection complete for generation {}'.format(g)
                self.log.info(msg)

                # Measure diversity
                # regDiff, capDiff = self.measureDiversity()

                # Replenish the population by crossing and mutating individuals
                # then run their models.
                t0 = time.time()
                model_count = self.crossMutateRun()
                msg = 'Cross and mutate complete for generation {}.'.format(g)
                self.log.info(msg)
                msg = (' All models should now be running for generation {'
                       '}.').format(g + 1)
                self.log.info(msg)

        # Done.
        self.log.info('Genetic algorithm complete.')
        self.log.info('Lowest cost: {:.2f}'.format(self.generationBest[-1]))
        # Return the best individual.
        return self.individualsList[0]

    def addToModelQueue(self, individual):
        """Helper function to put an individual and relevant inputs into a
            dictionary to run a model.
        """
        self.modelQueue.put_nowait({'individual': individual,
                                    'strModel': self.strModel,
                                    'inPath': self.inPath,
                                    'outDir': self.outDir})
        uid = individual.uid
        self.log.debug(
            'Individual with UID {} put in model queue.'.format(uid))

    def naturalSelection(self):
        """Determines which individuals will be used to create next generation.
        """
        # Determine how many individuals to keep for certain.
        k = math.ceil(self.probabilities['top'] * len(self.individualsList))
        self.log.debug('Keeping a minimum of {} individuals.'.format(k))

        # Loop over the unfit individuals, and either delete or keep based on
        # the weakProb
        i = 0
        while i < len(self.individualsList):
            # If we are past the k'th individual and the random draw mandates
            # it, kill it.
            if (i >= k) and (random.random() < self.probabilities['weak']):
                # Remove indiviual from individualsList, cleanup.
                ind = self.individualsList.pop(i)
                self.log.debug(('Killing individual {} via natural '
                                + 'selection').format(ind.uid))
                self.popMgr.clean(tableSuffix=ind.tableSuffix, uid=ind.uid,
                                  kill=True)
                # No need to increment the index since we removed the 
                # individual.
                continue

            # Add the cost to the fit sum, increment the index. Note that the 
            # fit sum gets zeroed out in the 'prep' function
            self.fitSum += self.individualsList[i].costs['total']
            i += 1

        self.log.debug(('Death by natural selection complete. There are {} '
                        + 'surviving '
                        + 'individuals.').format(len(self.individualsList)))

        # Create weights by standard cost weighting.
        self.rouletteWeights = []
        for ind in self.individualsList:
            self.rouletteWeights.append((ind.costs['total'] / self.fitSum))
            '''
            self.rouletteWeights.append(1 / (individual.costs['total'] 
                                             / self.fitSum))
            '''
        self.log.debug('Roulette weights assigned for each individual.')

    def crossMutateRun(self):
        """Crosses traits from surviving individuals to regenerate population,
            then runs the new individuals to evaluate their cost.
        """
        count = 0
        # Loop until population has been replenished.
        # Extract the number of individuals.
        n = len(self.individualsList)
        # chooseCount = []
        while len(self.individualsList) < self.numInd:
            if random.random() < self.probabilities['cross']:
                # Since we're crossing over, we won't force a mutation.
                forceMutate = False

                # Prime loop to select two unique individuals. Loop ensures
                # unique individuals are chosen.
                _individualsList = [0, 0]
                while _individualsList[0] == _individualsList[1]:
                    # Pick two individuals based on cumulative weights.
                    _individualsList = random.choices(
                        self.individualsList[0:n],
                        weights= \
                            self.rouletteWeights,
                        k=2)
                # Keep track of who created these next individuals.
                parents = (_individualsList[0].uid, _individualsList[1].uid)
                self.log.debug(('Individuals {} and {} selected for '
                                + 'crossing').format(parents[0], parents[1]))

                # Cross the regulator chromosomes
                regChroms = crossChrom(chrom1=_individualsList[0].regChrom,
                                       chrom2=_individualsList[1].regChrom)
                self.log.debug('Regulator chromosomes crossed.')

                # Cross the capaictor chromosomes
                capChroms = crossChrom(chrom1=_individualsList[0].capChrom,
                                       chrom2=_individualsList[1].capChrom)
                self.log.debug('Capacitor chromosomes crossed.')

            else:
                # We're not crossing over, so force mutation.
                forceMutate = True
                # Draw an individual.
                _individualsList = random.choices(self.individualsList[0:n],
                                                  weights=self.rouletteWeights,
                                                  k=1)

                # Track parents
                parents = (_individualsList[0].uid,)
                self.log.debug(('No crossing, just mutation of individual '
                                + '{}'.format(parents[0])))

                # Grab the necessary chromosomes, put in a list
                regChroms = [_individualsList[0].regChrom]
                capChroms = [_individualsList[0].capChrom]

            # Track chosen individuals.
            """
            for i in _individualsList:
                uids = [x[1] for x in chooseCount]
                if i.uid in uids:
                    ind = uids.index(i.uid)
                    # Increment the occurence count
                    chooseCount[ind][2] += 1
                else:
                    chooseCount.append([i.fitness, i.uid, 1])
            """

            # Possibly mutate individual(s).
            if forceMutate or (random.random() < self.probabilities['mutate']):
                # Mutate regulator chromosome:
                regChroms = mutateChroms(c=regChroms,
                                         prob=self.probabilities['regMutate'])
                self.log.debug('Regulator chromosome(s) mutated.')
                # Mutate capacitor chromosome:
                capChroms = mutateChroms(c=capChroms,
                                         prob=self.probabilities['capMutate'])
                self.log.debug('Capacitor chromosome(s) mutated.')

            # Create individuals based on new chromosomes, add to list, put
            # in queue for processing.
            for i in range(len(regChroms)):
                # Initialize new individual
                uid = self.popMgr.getUID()
                ind = individual(**self.indInputs,
                                 uid=uid,
                                 regChrom=regChroms[i],
                                 capChrom=capChroms[i],
                                 parents=parents,
                                 )
                self.log.debug('New individual, {}, initialized'.format(uid))
                # Put individual in the list and the queue.
                self.individualsList.append(ind)
                self.addToModelQueue(individual=ind)
                count += 1
                self.log.debug(('Individual {} put in the model '
                                + 'queue.').format(uid))

        return count

        """
        # Sort the chooseCount by number of occurences
        chooseCount.sort(key=lambda x: x[2])
        print('Fitness, UID, Occurences', flush=True)
        for el in chooseCount:
            print('{:.2f},{},{}'.format(el[0], el[1], el[2]))
        """

    def measureDiversity(self):
        """Function to loop over chromosomes and count differences between
        individuals. This information is useful in a histogram.
        """
        # Compute diversity
        n = 0
        regDiff = []
        capDiff = []
        # Loop over all individuals in the list
        for ind in self.individualsList:
            n += 1
            # Loop over all individuals later in the list
            for i in range(n, len(self.individualsList)):
                # Loop over reg chrom, count differences.
                regCount = 0
                for g in range(0, len(ind.regChrom)):
                    if ind.regChrom[g] != self.individualsList[i].regChrom[g]:
                        regCount += 1

                regDiff.append(regCount)

                # Loop over cap chrom, count differences.
                capCount = 0
                for g in range(0, len(ind.capChrom)):
                    if ind.capChrom[g] != self.individualsList[i].capChrom[g]:
                        capCount += 1

                capDiff.append(capCount)

        return regDiff, capDiff

    def stopThreads(self, timeout=10):
        """Function to gracefully stop the running threads.
        """
        # Signal to threads that we're done by putting 'None' in the queue.
        for _ in self.modelThreads: self.modelQueue.put_nowait(None)
        for t in self.modelThreads: t.join(timeout=timeout)
        # print('Threads terminated.', flush=True)


def writeRunEval(modelQueue, costs, log):
    # , cnxnpool):
    # tEvent):
    """Write individual's model, run the model, and evaluate costs. This is
    effectively a wrapper for individual.writeRunUpdateEval()
    
    NOTE: will take no action if an individual's model has already been
        run.
    
    NOTE: This function is static due to the threading involved. This feels
        memory inefficient but should save some headache.
        
    NOTE: This function is specifically formatted to be run via a thread
        object which is terminated when a 'None' object is put in the 
        modelQueue.
        
    INPUTS:
        modelQueue: queue which will have dictionaries inserted into it.
            dictionaries should contain individual, strModel, inPath, 
            and outDir fields from a population object.
    """
    while True:
        try:
            # Extract an individual from the queue.
            inDict = modelQueue.get()

            # Check input.
            if inDict is None:
                # If None is returned, we're all done here.
                modelQueue.task_done()
                break

            uid = inDict['individual'].uid
            log.debug('Pulled individual {} from model queue.'.format(uid))
            # Write, run, update, and evaluate the individual.
            inDict['individual'].writeRunUpdateEval(
                strModel=inDict['strModel'],
                inPath=inDict['inPath'],
                outDir=inDict['outDir'],
                costs=costs)

            # Denote task as complete.
            modelQueue.task_done()

            log.debug(('Completed running individual {}. There are {} '
                       + 'individuals left in the model '
                       + 'queue.').format(uid, modelQueue.qsize()))

        except:
            print('Exception occurred!', flush=True)
            error_type, error, traceback = sys.exc_info()
            print(error_type, flush=True)
            print(error, flush=True)
            print(traceback, flush=True)


def mutateChroms(c, prob):
    """Take a chromosome and randomly mutate it.
    
    INPUTS:
        c: list of chromsomes, which are tuples of 1's and 0's. Ex: 
            (1, 0, 0, 1, 0)
        prob: decimal in set [0.0, 1.0] to determine chance of
            mutating (bit-flipping) an individual gene
    """
    out = []
    for chrom in c:
        newC = list(chrom)
        # count = 0
        for ind in range(len(c)):
            if random.random() < prob:
                # Flip the bit!
                newC[ind] = 1 - newC[ind]
                # count += 1

        # Convert to tuple, put in output list
        out.append(tuple(newC))

    return out


def crossChrom(chrom1, chrom2):
    """Take two chromosomes and create two new ones.
    
    INPUTS:
        chrom1: tuple of 1's and 0's, same length as chrom2
        chrom2: tuple of 1's and 0's, same length as chrom1
        
    OUTPUTS:
        c1: new list of 1's and 0's, same length as chrom1 and 2
        c2: ""
    """
    # Force chromosomes to be same length
    assert len(chrom1) == len(chrom2)

    # Randomly determine range of crossover
    r = range(random.randint(0, len(chrom1)), len(chrom1))

    # Initialize the two chromosomes to be copies of 1 and 2, respectively
    c1 = list(chrom1)
    c2 = list(chrom2)

    # Loop over crossover range
    for k in r:
        # Note that random() is on interval [0.0, 1.0). Thus, we'll
        # consider [0.0, 0.5) and [0.5, 1.0) for our intervals. 
        # Note that since we initialized chrom to be a copy of chrom1, 
        # there's no need for an else case.
        if random.random() < 0.5:
            # Crossover.
            c1[k] = chrom2[k]
            c2[k] = chrom1[k]

    # Return the new chromosomes
    return [tuple(c1), tuple(c2)]


if __name__ == "__main__":
    pass
"""
if __name__ == "__main__":
    import time
    #import matplotlib.pyplot as plt
    n = 1
    f = open('C:/Users/thay838/Desktop/vvo/output.txt', 'w')
    for k in range(n):
        print('*' * 80, file=f)
        print('Generation {}'.format(k), file=f)
        t0 = time.time()
        popObj = population(numInd=100, numGen=10,
                            modelIn='C:/Users/thay838/Desktop/R2-12.47-2.glm',
                            reg={'R2-12-47-2_reg_1': {
                                                    'raise_taps': 16, 
                                                    'lower_taps': 16,
                                                    'taps': [
                                                        'tap_A',
                                                        'tap_B',
                                                        'tap_C'
                                                    ]
                                                   },
                              'R2-12-47-2_reg_2': {
                                                    'raise_taps': 16,
                                                    'lower_taps': 16,
                                                    'taps': [
                                                        'tap_A',
                                                        'tap_B',
                                                        'tap_C'
                                                    ]
                                                   }
                            },
                            cap={
                                'R2-12-47-2_cap_1': ['switchA', 'switchB', 'switchC'],
                                'R2-12-47-2_cap_2': ['switchA', 'switchB', 'switchC'],
                                'R2-12-47-2_cap_3': ['switchA', 'switchB', 'switchC'],
                                'R2-12-47-2_cap_4': ['switchA', 'switchB', 'switchC']
                            },
                            outDir='C:/Users/thay838/Desktop/vvo'
                            )
        popObj.ga()
        t1 = time.time()
        print('Runtime: {:.0f} s'.format(t1-t0), file=f)
        print('Scores: ', file=f)
        for s in popObj.generationBest:
            print('{:.4g}'.format(s), end=', ', file=f)
            
        print(file=f)
        print('Best Individual:', file=f)
        bestUID = popObj.indFitness[0][UIDIND]
        for ix in popObj.individualsList:
            if ix.uid == bestUID:
                print('\tCapacitor settings:', file=f)
                for capName, capDict in ix.cap.items():
                    print('\t\t' + capName + ':', file=f)
                    for switchName, switchDict in capDict.items():
                        print('\t\t\t' + switchName + ': ' 
                              + switchDict['status'], file=f)
                print(file=f)
                
                print('\tRegulator settings:', file=f)
                for regName, regDict in ix.reg.items():
                    print('\t\t' + regName + ':', file=f)
                    for tapName, tapDict in regDict['taps'].items():
                        print('\t\t\t' + tapName + ': ' + str(tapDict['pos']),
                              file=f)
                    
                break
            else:
                pass
        print('*' * 80, file=f, flush=True)
        #x = list(range(len(popObj.generationBest)))
        #plt.plot(x, popObj.generationBest)
        #plt.xlabel('Generations')
        #plt.ylabel('Best Score')
        #plt.title('Best Score for Each Generation')
        #plt.grid(True)
        #plt.savefig('C:/Users/thay838/Desktop/vvo/run_{}.png'.format(k))
        #plt.close()
        #plt.show()
        """
