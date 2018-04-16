"""
required packages: pyodbc

Created on Aug 9, 2017

@author: thay838
"""
import random
from glm import modGLM
import util.gld
import util.helper
import util.constants
import os
import copy

# Define cap status, to be accessed by binary indices.
CAPSTATUS = ['OPEN', 'CLOSED']
# Define the percentage of the tap range which sigma should be for drawing
# tap settings. Recall the 68-95-99.7 rule for normal distributions -
# 1 std dev, 2 std dev, 3 std dev probabilities
TAPSIGMAPCT = 0.1
# Define mode of triangular distribution for biasing capacitors. NOTE: By
# making the mode greater than 0.5, we're more likely to draw a number which
# rounds to one. Therefore, we should maintain the previous state if the draw
# rounds to one.
CAPTRIANGULARMODE = 0.8
# reg, cap control settings for each value of individual's controlFlag input.
CONTROL = [('MANUAL', 'MANUAL'), ('OUTPUT_VOLTAGE', 'VOLT'),
           ('OUTPUT_VOLTAGE', 'VAR'), ('OUTPUT_VOLTAGE', 'VARVOLT'),
           ('MANUAL', 'MANUAL')]
 
class individual:
    
    def __init__(self, uid, starttime, stoptime, timezone, dbObj, recorders,
                 reg=None, regFlag=5, cap=None, capFlag=5, regChrom=None, 
                 capChrom=None, parents=None, controlFlag=0, gldPath=None):
        """An individual contains information about Volt/VAR control devices
        
        Individuals can be initialized in two ways: 
        1) From scratch: Provide self, uid, reg, and cap inputs. Positions
            will be randomly generated
            
        2) From a chromosome: Provide self, reg, cap, uid, regChrom, and 
            capChrom. After crossing two individuals, a new chromosome will
            result. Use this chromosome to update reg and cap
        
        INPUTS:
            starttime: datetime object for model start
            stoptime: "..." end 
            timezone: string representing timezone, as it would appear in 
                GridLAB-D's tzinfo.txt. Example: 'PST+8PDT'
            dbObj: Initialized object of util/db.db class
            reg: Dictionary as described in the docstring for the gld module.
            
                Note possible tap positions be in interval [-lower_taps,
                raise_taps].
            
            regFlag: Flag for special handling of regulators.
                0: All regulator taps set to their minimum tap position
                1: All regulator taps set to their maximum tap position
                2: Regulator tap positions will be biased (via a Gaussian 
                    distribution and the TAPSIGMAPCT constant) toward the
                    previous tap positions
                3: Regulator state unchanged - simply reflects what's in the 
                    reg input's 'prevState'
                4: Regulator state given in reg input's 'newState' - just need
                    to generate chromosome and count tap changes
                5: Regulator tap positions will be determined randomly 
                    
            cap: Dictionary describing capacitors as described in the docstring
                for the gld module.
                
            capFlag: Flag for special handling of capacitors.
                0: All capacitors set to CAPSTATUS[0] (OPEN)
                1: All capacitors set to CAPSTATUS[1] (CLOSED)
                2: Capacitor states will be biased (via a triangular
                    distribution with the mode at CAPTRIANGULARMODE) toward the
                    previous states
                3: Capacitor state unchanged - simply reflects what's in the
                    'cap' input's 'prevState'.
                4: Capacitor state given in cap input's 'newState' - just need
                    to generate chromosome and count switching events
                5: Capacitor positions will be determined randomly.
                
            uid: Unique ID of individual. Should be an int.
            
            regChrom: Regulator chromosome resulting from crossing two
                individuals. List of ones and zeros representing tap positions. 
                
            capChrom: Capacitor chromosome resulting from crossing two
                individuals. List of ones and zeros representing switch status.
                
            parents: Tuple of UIDs of parents that created this individual.
                None if individual was randomly created at population init.
                
            controlFlag: Flag to determine the control scheme used in an 
                individual's model. This will also affect how evaluation is
                performed. For example, a model with volt_var_control has to
                calculate it's tap changes and capacitor switching events after
                a model has been run, while a manual control scheme computes
                tap changing/switching before running its model.
                Possible inputs (more to come later, maybe):
                0: Manual control of regulators and capacitors
                1: Output voltage control of regulators and capacitors
                2: Output voltage control of regulators, VAR control of
                    capacitors
                3: Output voltage control of regulators, VARVOLT control of
                    capacitors
                4: volt_var_control of regulators and capacitors. This will
                    create a volt_var_control object, and set regs and caps to
                    MANUAL to avoid letting them switch at initialization
                
            NOTE: If the controlFlag is not 0, the regFlag and capFlag must
                be set to 3.
                
            recorders: dictionary of recorder objects. Top level objects 
				should be a list of dictionaries or a dictionary. Each lower
				level dictionary should have two keys: 'properties' and
				'objType.' objType can be 'recorder' or 'group_recorder'. 
				properties should be a dictionary and will be passed directly
				to modGLM.addMySQLRecorder or modGLM.addMySQLGroup_Recorder.
                
                Possible keys:
                    energy: dictionary describing recorder used to measure
                        total feeder energy
                    power: dictionary describing recorder used to measure
                        total feeder power.
                    triplexVoltage: dictionary describing group_recorder for
                        recording triplex voltages.
                        *NOTE: we'll be recording 'measured_voltage_12' which
                            at present is only valid for triplex_load
                            objects.
                            
                *NOTE: regulators and capacitors recorders will be created by
                    reading the cap and reg inputs. The interval will be taken
                    from the 'energy' table.    
                *NOTE: regulators will NOT be recorded if the controlFlag is 0,
                    as there is no need to track position/tap count.
                *NOTE: all regulators should use the same table. 
                *NOTE: capacitors will NOT be recorded if the controlFlag is 0,
                    as there is no need to track status/switch count.
                *NOTE: all capacitors should use the same table.
        """
        # Ensure flags are compatible.
        if controlFlag:
            assert capFlag == regFlag == 3
        
        # Initialize some attributes that also need reset when re-using an
        # individual.
        self.prep(starttime=starttime, stoptime=stoptime)
        
        # Set the timezone (this isn't done in prep function as it's assumed
        # to never change for an individual - feeders don't get up and move, 
        # and timezones don't often change.
        self.timezone = timezone
        
        # Assign gldPath
        self.gldPath = gldPath
        
        # set database object
        self.dbObj = dbObj
        
        # Set the control flag
        self.controlFlag = controlFlag
        
        # Assign reg and cap dicts. Perform a deepcopy, since an individual
        # will modify its own copy of the given reg and cap definitions.
        self.reg = copy.deepcopy(reg)
        self.cap = copy.deepcopy(cap)
        
        # Assign the unique identifier.
        self.uid = uid
        # We'll use the prefix to uniquely identify tables.
        self.tableSuffix = '_' + str(uid)
        
        # Parent tracking is useful to see how well the GA is working
        self.parents = parents
        
        # When writing model, output directory will be saved.
        self.outDir = None
        
        # Set recorders property
        self.recorders = copy.deepcopy(recorders)
        
        # If not given a regChrom or capChrom, generate them.
        if (regChrom is None) and (capChrom is None):
            # Generate regulator chromosome:
            self.genRegChrom(flag=regFlag)
            # Generate capacitor chromosome:
            self.genCapChrom(flag=capFlag)
        else:
            # Use the given chromosomes to update the dictionaries.
            self.regChrom = regChrom
            self.modifyRegGivenChrom()
            self.capChrom = capChrom
            self.modifyCapGivenChrom()
            
    def prep(self, starttime, stoptime, reg=None, cap=None):
        """Method to get an individual ready for use/re-use - in the genetic
            algorithm, the population for the next time interval should be
            seeded with the best individuals from the previous time interval.
            
        During a call to the constructor, this gets attributes initialized.
        
        Pass in 'reg' and 'cap' ONLY to get an individual ready to be used in
        a new population.
        """
        # Assing times.
        self.starttime = starttime
        self.start_str = starttime.strftime(util.constants.DATE_TZ_FMT)
        self.stoptime = stoptime
        self.stop_str = stoptime.strftime(util.constants.DATE_TZ_FMT)
        # Full path to output model.
        self.modelPath = None
        
        # When the model is run, output will be saved.
        self.modelOutput = None

        # The evalFitness method assigns costs
        self.costs = None
        
        # Update the 'prevState' of the individuals reg and cap dictionaries.
        if reg and cap:
            out = util.helper.updateVVODicts(regOld=self.reg, capOld=self.cap,
                                             regNew=reg, capNew=cap)
            
            self.reg = out['reg']
            self.cap = out['cap']
            self.tapChangeCount = out['tapChangeCount']
            self.capSwitchCount = out['capSwitchCount']
        else:
            # Track tap changes and capacitor switching
            self.tapChangeCount = 0
            self.capSwitchCount = 0
            
    def __eq__(self, other):
        """Compare individuals by looping over their chromosomes
        
        TODO: This isn't very sophisticated. It doesn't check if chromosomes
        are the same length, exist, etc.
        """
        # Start with regulator
        for k in range(len(self.regChrom)):
            if self.regChrom[k] != other.regChrom[k]:
                return False
            
        # On to capacitor
        for k in range(len(self.capChrom)):
            if self.capChrom[k] != other.capChrom[k]:
                return False
            
        return True
        
            
    def __str__(self):
        """Individual's string should include fitness and reg/cap info.
        
        This is a simple wrapper to call helper.getSummaryStr since the
        benchmark system should be displayed in the same way.
        """
        s = util.helper.getSummaryStr(costs=self.costs, reg=self.reg,
                                      cap=self.cap, regChrom=self.regChrom,
                                      capChrom=self.capChrom,
                                      parents=self.parents)
        
        return s
        
    def genRegChrom(self, flag):
        """Method to randomly generate an individual's regulator chromosome
        
        INPUTS:
            flag: dictates how regulator tap positions are created.
                0: All regulator taps set to their minimum tap position
                1: All regulator taps set to their maximum tap position
                2: Regulator tap positions will be biased (via a Gaussian 
                    distribution and the TAPSIGMAPCT constant) toward the
                    previous tap positions
                3: Regulator state unchanged - simply reflects what's in the 
                    reg input's 'prevState'
                4: Regulator state given in reg input's 'newState' - just need
                    to generate chromosome and count tap changes
                5: Regulator tap positions will be determined randomly
                
        NOTE: the individual's controlFlag will be used to determine whether or
            not the individual's tapChangeCount should be updated.
        """
        # Initialize chromosome for regulator and dict to store list indices.
        self.regChrom = ()
         
        # Intialize index counters.
        s = 0;
        e = 0;
        
        # Loop through the regs and create binary representation of taps.
        for r, v in self.reg.items():
            
            # Define the upper tap bound (tb).
            tb = v['raise_taps'] + v['lower_taps']
            
            # Compute the needed field width to represent the upper tap bound
            # Use + 1 to account for 2^0
            width = util.helper.binaryWidth(tb) 
            
            # Define variables as needed based on the flag. I started to try to
            # make micro-optimizations for code factoring, but let's go for
            # readable instead.
            if flag== 0:
                newState = 0
            elif flag == 1:
                newState = tb
            elif flag == 2:
                # If we're biasing from the previous position, get a sigma for
                # the Gaussian distribution.
                tapSigma = round(TAPSIGMAPCT * (tb + 1))
            elif flag == 3:
                state = 'prevState'
            elif flag == 4:
                state = 'newState'
            
            # Loop through the phases
            for phase, phaseData in v['phases'].items():
                
                # If we're biasing new positions based on previous positions:
                if flag == 2:
                    # Randomly draw tap position from gaussian distribution.
                    
                    # Translate previous position to integer on interval [0,tb]
                    prevState = \
                        util.gld.inverseTranslateTaps(lowerTaps=v['lower_taps'],
                                                 pos=phaseData['prevState'])
                        
                    # Initialize the newState for while loop.
                    newState = -1
                    
                    # The standard distribution runs from (-inf, +inf) - draw 
                    # until position is valid. Recall valid positions are
                    # [0, tb]
                    while (newState < 0) or (newState > tb):
                        # Draw the tap position from the normal distribution.
                        # Here oure 'mu' is the previous value
                        newState = round(random.gauss(prevState, tapSigma))
                        
                elif (flag == 3) or (flag == 4):
                    # Translate position to integer on interval [0, tb]
                    newState = \
                        util.gld.inverseTranslateTaps(lowerTaps=v['lower_taps'],
                                                      pos=phaseData[state])
                        
                elif flag == 5:
                    # Randomly draw.
                    newState = random.randint(0, tb)
                
                # Express tap setting as binary list with consistent width.
                binTuple = tuple([int(x) for x in "{0:0{width}b}".format(newState,
                                                                  width=width)])
                
                # Extend the regulator chromosome.
                self.regChrom += binTuple
                
                # Increment end index.
                e += len(binTuple)
                
                # Translate newState for GridLAB-D.
                self.reg[r]['phases'][phase]['newState'] = \
                    util.gld.translateTaps(lowerTaps=v['lower_taps'], pos=newState)
                    
                # Increment the tap change counter (previous pos - this pos) if
                # this individual is using MANUAL control. Otherwise, tap
                # changes must be computed after the model run.
                if self.controlFlag == 0:
                    self.tapChangeCount += \
                        abs(self.reg[r]['phases'][phase]['prevState']
                            - self.reg[r]['phases'][phase]['newState'])
                
                # Ensure the indices for this phase match what was originally
                # built.
                if self.reg[r]['phases'][phase]['chromInd'] != (s, e):
                    raise UserWarning(('The original regulator chromosome '
                                       'indices do not match what individual '
                                       'calculated!'))
                
                # Increment start index.
                s += len(binTuple)
    
    # Function below is a DRAFT!
    '''
    def genRegChromVoltage(self, regulation=0.1, nominalVoltage=7200,
                           bandWidthDec=0.2):
        """Alternative to genRegChrom that plays with voltage set points rather
        than tap positions.
        
        TODO: regulation and nominalVoltage should be part of the 'reg' 
        dictionary that is input to the individual/application
        
        TODO: Handle different regulator modes (specifically remote_node)
        
        TODO: dwell_time?
        
        NOTE: in the future, we could seed the population via heuristics. Like
        during times of high PV increase the band_width and decrease the
        band_center, for example
        """
        # Initialize chromosome for regulator and dict to store list indices.
        self.regChrom = ()
        # Track chromosome index
        ind = 0
        
        # Loop over the regulators
        for r, d in self.reg.items():
            # Compute the voltage bounds.
            # NOTE: there's a key assumption here: we'll never be setting  
            # setpoints beyond the 'nominal' range of the regulator. In
            # reality, if the regulator is operating at a lower/higher voltage
            # than nominal, it can regulate lower/higher than the nominal
            # regulation range.
            # TODO: do we want to allow setpoints outside of the nominal range?
            
            # Compute voltage change per tap. Note that GridLAB-D documentation
            # says: "per tap change ratio equals regulation / raise taps"
            # This bakes in a symmetric assumption - that's okay.
            changePerTapDec = regulation / d['raise_taps']
            voltPerTap = changePerTapDec * nominalVoltage
            # Compute upper and lower voltage bounds
            upperBound = nominalVoltage + (nominalVoltage * changePerTapDec
                                           * d['raise_taps'])
            lowerBound = nominalVoltage - (nominalVoltage * changePerTapDec
                                           * d['lower_taps'])
            # Randomly select a band_center (voltage set-point) 
            band_center = ((upperBound - lowerBound)*random.random()
                           + lowerBound)
            
            # For now, fix band_width. Note GridLAB-D 
            # TODO: do we want to randomly pick the band_width? This would
            # increase the search space, but could help drive toward a more
            # optimal solution (there may be times of the day where you want
            # more or less sensitivity)
            band_width = voltPerTap * (1 + 2*bandWidthDec)
            
            # Assign the band_width and band_center.
            self.reg[r]['band_center'] = band_center
            self.reg[r]['band_width'] = band_width
            
            # TODO: don't hard-code (we may support 'REMOTE_NODE' in the
            # future).
            self.reg[r]['Control'] = 'OUTPUT_VOLTAGE'
            
            # Update the chromosome
    '''
            
                
    def genCapChrom(self, flag):
        """Method to generate an individual's capacitor chromosome.
        
        INPUTS:
            flag:
                0: All capacitors set to CAPSTATUS[0] (OPEN)
                1: All capacitors set to CAPSTATUS[1] (CLOSED)
                2: Capacitor states will be biased (via a triangular
                    distribution with the mode at CAPTRIANGULARMODE) toward the
                    previous states
                3: Capacitor state unchanged - simply reflects what's in the
                    'cap' input's 'prevState'.
                4: Capacitor state given in cap input's 'newState' - just need
                    to generate chromosome and count switching events
                5: Capacitor positions will be determined randomly.
                
        NOTE: the individual's controlFlag will be used to determine whether or
            not the individual's capSwitchCount should be updated.
            
        OUTPUTS:
            modifies self.cap, sets self.capChrom
        """
        # If we're forcing all caps to the same status, determine the binary
        # representation.
        if flag < 2:
            capBinary = flag
            capStatus = CAPSTATUS[flag]
        elif flag == 3:
            state = 'prevState'
        elif flag == 4:
            state = 'newState'
        
        # Initialize chromosome for capacitors and dict to store list indices.
        self.capChrom = ()

        # Keep track of chromosome index
        ind = 0
        
        # Loop through the capacitors, randomly assign state for each phase
        for c, capData in self.cap.items():
            
            # Loop through the phases and randomly decide state
            for phase in capData['phases']:
                
                # Take action based on flag.
                if flag == 2:
                    # Use triangular distribution to bias choice
                    draw = round(random.triangular(mode=CAPTRIANGULARMODE))
                    
                    # Extract the previous state
                    prevState = self.cap[c]['phases'][phase]['prevState']
                    
                    # If the draw rounded to one, use the previous state.
                    # If the draw rounded to zero, use the opposite state
                    if draw:
                        capStatus = prevState
                        capBinary = CAPSTATUS.index(capStatus)
                    else:
                        # Flip the bit
                        capBinary = 1 - CAPSTATUS.index(prevState) 
                        capStatus = CAPSTATUS[capBinary]
                        
                elif (flag == 3) or (flag == 4):
                    # Use either 'prevState' or 'newState' for this state
                    capStatus = self.cap[c]['phases'][phase][state]
                    capBinary = CAPSTATUS.index(capStatus)
                elif flag == 5:
                    # Randomly determine state
                    capBinary = round(random.random())
                    capStatus = CAPSTATUS[capBinary]
                    
                # Assign to the capacitor
                self.capChrom += (capBinary,)
                self.cap[c]['phases'][phase]['newState'] = capStatus
                
                # Ensure indices are lining up.
                if self.cap[c]['phases'][phase]['chromInd'] != ind:
                    raise UserWarning(('Pre-determined chromosome index '
                                       'does not match index determined by '
                                       'individual!'))
                
                # Increment the switch counter if applicable
                if (self.controlFlag == 0
                    and ((self.cap[c]['phases'][phase]['newState']
                          != self.cap[c]['phases'][phase]['prevState']))):
                    
                    self.capSwitchCount += 1
                
                # Increment the chromosome counter
                ind += 1
                
    def modifyRegGivenChrom(self):
        """Modifiy self.reg based on self.regChrom
        """
        # Loop through self.reg and update 'newState'
        for r, regData in self.reg.items():
            for phase, phaseData in regData['phases'].items():
                
                # Extract the binary representation of tap position.
                tapBin = \
                    self.regChrom[phaseData['chromInd'][0]:\
                                  phaseData['chromInd'][1]]
                    
                # Convert the binary to an integer
                posInt = util.helper.bin2int(tapBin)
                
                # Convert integer to tap position and assign to new position
                self.reg[r]['phases'][phase]['newState'] = \
                    util.gld.translateTaps(lowerTaps=self.reg[r]['lower_taps'],
                                      pos=posInt)
                    
                # Increment the tap change counter (previous pos - this pos)
                self.tapChangeCount += \
                    abs(self.reg[r]['phases'][phase]['prevState']
                        - self.reg[r]['phases'][phase]['newState'])
    
    def modifyCapGivenChrom(self):
        """Modify self.cap based on self.capChrom
        """
        # Loop through the capDict and assign 'newState'
        for c, capData in self.cap.items():
            for phase in capData['phases']:
                # Read chromosome and assign newState
                self.cap[c]['phases'][phase]['newState'] = \
                    CAPSTATUS[self.capChrom[self.cap[c]['phases'][phase]\
                                            ['chromInd']]]
                
                # Bump the capacitor switch count if applicable
                if self.cap[c]['phases'][phase]['newState'] != \
                        self.cap[c]['phases'][phase]['prevState']:
                    
                    self.capSwitchCount += 1
                
    def writeModel(self, strModel, inPath, outDir):
        """Create a GridLAB-D .glm file for the given individual by modifying
        setpoints for controllable devices (capacitors, regulators, eventually
        DERs) and adding all requisite recorders. Everything else in the model
        is considered up to date and ready to go.
        
        INPUTS:
            self: constructed individual
            strModel: string of .glm file found at inPath
            inPath: path to model to modify control settings
            outDir: directory to write new model to. Filename will be inferred
                from the inPath, and the individuals uid preceded by an
                underscore will be added
                
        OUTPUTS:
            Writes model to file
        """
        # Assign output directory.
        self.outDir = outDir
        
        # Get the filename of the original model and create output path
        modelPath = \
            modGLM.modGLM.addFileSuffix(inPath=os.path.basename(inPath),
                                            suffix=str(self.uid))
        
        # Track the output path for running the model later.
        self.modelPath = modelPath
        
        # Instantiate a modGLM object.
        writeObj = modGLM.modGLM(strModel=strModel, pathModelIn=inPath,
                                pathModelOut=(outDir + '/' + modelPath))
        
        # Set control for regulators and capacitors.
        regControl, capControl = CONTROL[self.controlFlag]
        
        for r in self.reg:
            # Modify control setting.
            self.reg[r]['Control'] = regControl

        for c in self.cap:
            # Modify control setting.
            self.cap[c]['control'] = capControl
            
        # Change capacitor and regulator statuses/positions and control.
        writeObj.commandRegulators(reg=self.reg)
        writeObj.commandCapacitors(cap=self.cap)
        
        # If we're using GridLAB-D's volt_var_controller, add it to the model
        if self.controlFlag == 4:
            # TODO: This is SUPER HARD-CODED to only work for the
            # R2-12-47-2 feeder. Eventually, this needs made more flexible.
            # NOTE: this method creates a player file... annoying.
            self.vvoPlayer = writeObj.addVVO(starttime=self.start_str)
            
        # Add the power and energy recorders, track their tables.
        self.powerTable = self.addRecorder(recordDict=self.recorders['power'],
                                           writeObj=writeObj)
        self.energyTable = self.addRecorder(recordDict=self.recorders['energy'],
                                            writeObj=writeObj)
        
        # Add voltage recorder(s):
        self.triplexTable = \
            self.addRecorder(recordDict=self.recorders['triplexVoltage'],
                             writeObj=writeObj)
            
        # Add capacitor and regulator recorders if the controlFlag > 0
        if self.controlFlag > 0:
            # If we're not in manual control, we need to record counts and
            # state for regulators and capacitors.
            
            # Grab the time interval from the energy table.
            tInt = self.recorders['energy']['properties']['interval']
            
            # Loop over regulators and add recorders.
            for reg in self.reg:
                # Initialize dictionary
                recordDict = {'objType': 'recorder'}
                # Get the dictionary of properties
                recordDict['properties'] = \
                    util.gld.getRegOrCapRecordDict(objName=reg,
                                                   table='reg',
                                                   objType='reg',
                                                   timeInterval=tInt,
                                                   limit=-1)
                
                # Add the recorder.
                tr = self.addRecorder(recordDict=recordDict, writeObj=writeObj)
            
            # Track the regulator table.
            self.regTable = tr
                
            # Loop over capacitors and add recorders.
            for cap in self.cap:
                # Initialize dictionary
                recordDict = {'objType': 'recorder'}
                # Get the dictionary of properties
                recordDict['properties'] = \
                    util.gld.getRegOrCapRecordDict(objName=cap,
                                                   table='cap',
                                                   objType='cap',
                                                   timeInterval=tInt,
                                                   limit=-1)
                
                # Add the recorder.
                tc = self.addRecorder(recordDict=recordDict, writeObj=writeObj)
                
            # Track the capacitor table
            self.capTable = tc
        
        # Write the modified model to file.
        writeObj.writeModel()
        
    def addRecorder(self, recordDict, writeObj):
        """Helper function to add a recorder object from self.recorders to a
        model. Returns table name (modified based on UID) and list of columns.
        """
        # Make a copy of the recordDict (otherwise, if the individual is reused
        # we end up continually tacking the tableSuffix)
        rD = copy.deepcopy(recordDict)
        # Add '_<uid>' to table name.
        rD['properties']['table'] = (rD['properties']['table']
                                     + self.tableSuffix)
        
        # recorders and group_recorders need handled differently.
        if rD['objType'] == 'recorder':
            # Add the recorder.
            writeObj.addMySQLRecorder(**rD['properties'])
            # In this case, we can grab the columns directly from the
            # rD.
            cols = rD['properties']['propList']
            # At the time of writing, regular mysql recorders cannot specify a
            # complex part.
            complex_part = []
        elif rD['objType'] == 'group_recorder':
            # Add the recorder.
            writeObj.addMySQLGroup_Recorder(**rD['properties'])
            # Currently, with a group_recorder we don't care about the columns,
            # since there's one or more columns for each object within the
            # group we're recording.
            cols = []
            complex_part = rD['properties']['complex_part']
        
        # Return. 
        return {'table': rD['properties']['table'],
                'columns': cols,
                'complex_part': complex_part,
                'type': rD['objType']}
        
    def runModel(self):
        """Function to run GridLAB-D model.
        """
        self.modelOutput = util.gld.runModel(modelPath=(self.outDir + '/'
                                                        + self.modelPath),
                                             gldPath=self.gldPath)
        # TODO: Handle a failed GridLAB-D run (catch a CalledProcessError)
        # If a model failed to run, print to the console.
        if self.modelOutput.returncode:
            print("FAILURE! Individual {}'s model gave non-zero returncode.".format(self.uid))
        
    def update(self, stoptime=None):
        """Function to update regulator tap operations and positions, and 
        capacitor switch operations and states. This function should be called
        after running a model (runModel), and before evaluating fitness
        (evalFitness)
        
        NOTE: This function does nothing if the individual's controlFlag is 0,
            since these updates occur before running a model.
            
        INPUTS:
            stoptime: Provide if you dont' want to use self.stoptime
        """
        # Do nothing is the controlFlag is 0
        if self.controlFlag == 0:
            return
        
        # Determine times
        if not stoptime:
            stoptime=self.stoptime
        
        # For other control schemes, we need to get the state change and state
        # information from the database.
        # Update the regulator tap change count. Note we've already ensured
        # the change properties are recorded.
        # NOTE: passing stoptime for both times to ensure we ONLY read the 
        # change count at the end - otherwise we might double count.
        self.tapChangeCount = \
            self.dbObj.sumMatrix(table=self.regTable['table'],
                                 cols=util.gld.REG_CHANGE_PROPS,
                                 starttime=stoptime,
                                 stoptime=stoptime)
        # Update the capacitor switch count
        self.capSwitchCount = \
            self.dbObj.sumMatrix(table=self.capTable['table'],
                                 cols=util.gld.CAP_CHANGE_PROPS,
                                 starttime=stoptime,
                                 stoptime=stoptime)
        
        # The 'newState' properties of 'reg' and 'cap' need updated. Note that
        # we've already ensure the state properties were recorded.
        self.reg = \
            self.dbObj.updateStatus(inDict=self.reg, dictType='reg',
                                    table=self.regTable['table'],
                                    phaseCols=util.gld.REG_STATE_PROPS,
                                    t=stoptime)
        
        self.cap = \
            self.dbObj.updateStatus(inDict=self.cap, dictType='cap',
                                    table=self.capTable['table'],
                                    phaseCols=util.gld.CAP_STATE_PROPS,
                                    t=stoptime)
        
        # Update the regulator and capacitor chromosomes.
        self.genRegChrom(flag=4)
        self.genCapChrom(flag=4)
                            
    def evalFitness(self, costs, tCol='t', starttime=None, stoptime=None):
        """Function to evaluate fitness of individual. This is essentially a
            wrapper to call util.gld.computeCosts
        
        INPUTS:
            costs: dictionary with the following fields:
               see util/gld.py for full definition.
            tCol: name of time column(s)
            starttime: starttime to evaluate fitness for. If None, uses
                self.starttime
            stoptime: stoptime "..." self.stoptime
            voltFlag: Flag for whether to compute voltage violations or not.
                This should be removed once the mysql group recorder is ready.
                This is here to accomodate pmaps/experiment.py
        """
        # Establish times if they aren't given explicitely
        if starttime is None:
            starttime = self.starttime
            
        if stoptime is None:
            stoptime = self.stoptime

        # Compute costs.
        self.costs = util.gld.computeCosts(dbObj=self.dbObj,
                                           energyTable=self.energyTable,
                                           powerTable=self.powerTable,
                                           triplexTable=self.triplexTable,
                                           tapChangeCount=self.tapChangeCount,
                                           capSwitchCount=self.capSwitchCount,
                                           costs=costs,
                                           starttime=starttime,
                                           stoptime=stoptime,
                                           tCol=tCol,
                                           )
    
    def writeRunUpdateEval(self, strModel, inPath, outDir, costs):
        """Function to write and run model, update individual, and evaluate
        the individual's fitness.
        
        INPUTS:
            strModel: see writeModel()
            inPath: see writeModel()
            outDir: see writeModel()
            costs: costs for fitness evaluation. See evalFitness
            
        OUTPUTS:
            list of tables
        """
        # Write the model.
        self.writeModel(strModel=strModel, inPath=inPath, outDir=outDir)
        # Run the model.
        self.runModel()
        # Update tap/cap states and change counts if necessary.
        self.update()
        # Evaluate costs.
        self.evalFitness(costs=costs)