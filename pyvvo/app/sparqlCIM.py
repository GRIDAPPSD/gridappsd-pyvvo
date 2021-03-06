'''
Created on Mar 1, 2018

@author: thay838
'''
from SPARQLWrapper import SPARQLWrapper2, SPARQLWrapper
from collections import OrderedDict

# pyvvo imports
from helper import binaryWidth
from constants import LOADS, NOMVFACTOR

# Define query prefix
PREFIX = """
PREFIX r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX c: <http://iec.ch/TC57/2012/CIM-schema-cim17#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
#FDRID = '_9CE150A8-8CC5-A0F9-B67E-BBD8C79D3095' #R2-12.47-2
#FDRID = '_0663ADF2-FC00-45BE-858E-50B3D1D01696' #R2-12.47-2
FDRID = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3' # ieee 8500 node

class sparqlCIM:
    
    def __init__(self, host="localhost", port=9999, namespace="kb"):
        """Create a sparql wrapper instance to connect to the triple store
        """
        # Assign.
        self.host = host
        self.port = port
        self.namespace = namespace
        self.connStr = ("http://" + host + ":" + str(port) + 
                        "/blazegraph/namespace/" + namespace + "/sparql")
        
        # Get sparql connection running
        self.sparql = SPARQLWrapper2(self.connStr)
        
    def getRegs(self, fdrid):
        """Get voltage regulators, massage into desired format.
        
        INPUTS: fdrid: feeder ID. Pull only from specific feeder
        
        TODO: we need the mrid for commands.
        """
        # feeder selection options - if all commented out, query matches
        # all feeders
        # VALUES ?fdrid {"_C1C3E687-6FFD-C753-582B-632A27E28507"}  # 123 bus
        # VALUES ?fdrid {"_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"}  # 13 bus
        # VALUES ?fdrid {"_5B816B93-7A5F-B64C-8460-47C17D6E4B0F"}  # 13 bus assets
        # VALUES ?fdrid {"_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3"}  # 8500 node
        # VALUES ?fdrid {"_67AB291F-DCCD-31B7-B499-338206B9828F"}  # J1
        # VALUES ?fdrid {"_9CE150A8-8CC5-A0F9-B67E-BBD8C79D3095"}  # R2 12.47 3
    
        # Build the regulator query.
        # TODO: clean up query so it is only pulling the data_ls we care about.
        regQ = (PREFIX + 
                "SELECT ?rname ?pname ?tname ?wnum ?phs ?incr ?mode ?enabled "
                "?highStep ?lowStep ?neutralStep ?normalStep ?neutralU ?step "
                "?initDelay ?subDelay ?ltc ?vlim ?vset ?vbw ?ldc ?fwdR ?fwdX "
                "?revR ?revX ?discrete ?ctl_enabled ?ctlmode ?monphs "
                "?ctRating ?ctRatio ?ptRatio ?id ?fdrid "
                "WHERE {{ "
                    'VALUES ?fdrid {{"{fdrid}"}} '
                    "?pxf c:Equipment.EquipmentContainer ?fdr. "
                    "?fdr c:IdentifiedObject.mRID ?fdrid. "
                    "?rtc r:type c:RatioTapChanger. "
                    "?rtc c:IdentifiedObject.name ?rname. "
                    "?rtc c:RatioTapChanger.TransformerEnd ?end. "
                    "?end c:TransformerEnd.endNumber ?wnum. "
                    "OPTIONAL {{ "
                        "?end c:TransformerTankEnd.phases ?phsraw. "
                        'bind(strafter(str(?phsraw),"PhaseCode.") as ?phs)'
                    "}} " 
                    "?end c:TransformerTankEnd.TransformerTank ?tank. " 
                    "?tank c:TransformerTank.PowerTransformer ?pxf. "
                    "?pxf c:IdentifiedObject.name ?pname. "
                    "?pxf c:IdentifiedObject.mRID ?id. "
                    "?tank c:IdentifiedObject.name ?tname. "
                    "?rtc c:RatioTapChanger.stepVoltageIncrement ?incr. "
                    "?rtc c:RatioTapChanger.tculControlMode ?moderaw. "
                    'bind(strafter(str(?moderaw),"TransformerControlMode.")'
                        " as ?mode) "
                    "?rtc c:TapChanger.controlEnabled ?enabled. "
                    "?rtc c:TapChanger.highStep ?highStep. "
                    "?rtc c:TapChanger.initialDelay ?initDelay. "
                    "?rtc c:TapChanger.lowStep ?lowStep. "
                    "?rtc c:TapChanger.ltcFlag ?ltc. "
                    "?rtc c:TapChanger.neutralStep ?neutralStep. "
                    "?rtc c:TapChanger.neutralU ?neutralU. "
                    "?rtc c:TapChanger.normalStep ?normalStep. "
                    "?rtc c:TapChanger.step ?step. "
                    "?rtc c:TapChanger.subsequentDelay ?subDelay. "
                    "?rtc c:TapChanger.TapChangerControl ?ctl. "
                    "?ctl c:TapChangerControl.limitVoltage ?vlim. "
                    "?ctl c:TapChangerControl.lineDropCompensation ?ldc. "
                    "?ctl c:TapChangerControl.lineDropR ?fwdR. "
                    "?ctl c:TapChangerControl.lineDropX ?fwdX. "
                    "?ctl c:TapChangerControl.reverseLineDropR ?revR. "
                    "?ctl c:TapChangerControl.reverseLineDropX ?revX. "
                    "?ctl c:RegulatingControl.discrete ?discrete. "
                    "?ctl c:RegulatingControl.enabled ?ctl_enabled. "
                    "?ctl c:RegulatingControl.mode ?ctlmoderaw. "
                    'bind(strafter(str(?ctlmoderaw),'
                        '"RegulatingControlModeKind.") as ?ctlmode) '
                    "?ctl c:RegulatingControl.monitoredPhase ?monraw. "
                    'bind(strafter(str(?monraw),"PhaseCode.") as ?monphs) '
                    "?ctl c:RegulatingControl.targetDeadband ?vbw. "
                    "?ctl c:RegulatingControl.targetValue ?vset. "
                    "?asset c:Asset.PowerSystemResources ?rtc. "
                    "?asset c:Asset.AssetInfo ?inf. "
                    "?inf c:TapChangerInfo.ctRating ?ctRating. "
                    "?inf c:TapChangerInfo.ctRatio ?ctRatio. "
                    "?inf c:TapChangerInfo.ptRatio ?ptRatio. "
                "}} "
                "ORDER BY ?pname ?tname ?rname ?wnum "
                ).format(fdrid=fdrid)
        
        # Set and execute the query.
        self.sparql.setQuery(regQ)
        ret = self.sparql.query()
        
        # Initialize dict to store regulator information. It's ordered to we 
        # can count on all individuals having the same chromosome indices.
        reg = OrderedDict()
        
        # We'll be assigning chromosome positions as we go. This ensures that
        # all individuals have chromosomes which line up. Intialize counters.
        s = 0
        e = 0
        
        # Loop over the regulators. Note that we'll get an object per phase, so
        # be cognizant of that.
        for el in ret.bindings:
            # Extract the regulator's name. Note that names are prefixed by
            # 'reg_'
            name = 'reg_' + el['pname'].value
            
            # If we haven't initialized this regulator's dict, do so now
            try:
                reg[name]
            except KeyError:
                # Initialize to dictionary
                reg[name] = {}
                
            # Compute 'raise_taps' and 'lower_taps'
            raise_taps = (int(el['highStep'].value)
                          - int(el['neutralStep'].value)
                          )
            lower_taps = (int(el['neutralStep'].value)
                          - int(el['lowStep'].value)
                         )
            
            # Compute the number of binary values needed to represent the
            # number of taps.
            numTaps = int(el['highStep'].value) - int(el['lowStep'].value)
            width = binaryWidth(numTaps)
            
            # Increment the ending index
            e += width
            
            # Grab the step voltage increment
            stepVoltageIncrement = float(el['incr'].value)
            
            # Put top-level properties in the dictionary
            for nameVal in [('raise_taps', raise_taps),
                            ('lower_taps', lower_taps),
                            ('stepVoltageIncrement', stepVoltageIncrement),
                            ('id', el['id'].value),
                            ]:
                try:
                    # Attempt to access the key.
                    sameVal = (reg[name][nameVal[0]] == nameVal[1])
                except KeyError:
                    # Key doesn't exist, assign it.
                    reg[name][nameVal[0]] = nameVal[1]
                else:
                    # Key exists, ensure that this phase has the same value as
                    # was assigned previously.
                    if not sameVal:
                        raise ValueError('Regulator {} does not '.format(name)
                                         + 'have the same '
                                         + '{} on all '.format(nameVal[0])
                                         + 'phases.')
            
            # Compute the nominal tap position for GridLAB-D.
            # TODO: this should probably be obtained from some sort of 
            # measurement object.
            prevState = round(((float(el['step'].value) - 1) * 100)
                              / stepVoltageIncrement)
            
            # Set the 'prevState' for this phase.
            try:
                # Attempt to access phases ('phases')
                reg[name]['phases']
            except KeyError:
                # phases hasn't been initialized.
                reg[name]['phases'] = OrderedDict()
                
            # Build dict for this phase
            reg[name]['phases'][el['phs'].value.upper()] = \
                {'prevState': prevState, 'chromInd': (s, e)}
                
            # Increment the starting index
            s += width
        
        """
        for el in ret.bindings:
            print('*'*60)
            for el2 in el:
                print(el2 + ": " + str(el[el2].value))
        """
        
        # We're all done. Return.
        return reg
    
    def getCaps(self, fdrid):
        """Get capacitors, massage into desired format.
        
        INPUTS: fdrid: feeder ID. Pull only from specific feeder
        
        """
        capQ = (PREFIX +
                "SELECT ?name ?basev ?nomu ?bsection ?bus ?conn ?grnd ?phs "
                "?ctrlenabled ?discrete ?mode ?deadband ?setpoint ?delay "
                "?monclass ?moneq ?monbus ?monphs ?id ?fdrid "
                "WHERE {{ "
                    "?s r:type c:LinearShuntCompensator. "
                    'VALUES ?fdrid {{"{fdrid}"}} '
                    "?s c:Equipment.EquipmentContainer ?fdr. "
                    "?fdr c:IdentifiedObject.mRID ?fdrid. "
                    "?s c:IdentifiedObject.name ?name. "
                    "?s c:ConductingEquipment.BaseVoltage ?bv. "
                    "?bv c:BaseVoltage.nominalVoltage ?basev. "
                    "?s c:ShuntCompensator.nomU ?nomu. " 
                    "?s c:LinearShuntCompensator.bPerSection ?bsection. " 
                    "?s c:ShuntCompensator.phaseConnection ?connraw. "
                    'bind(strafter(str(?connraw),"PhaseShuntConnectionKind.")'
                    " as ?conn) "
                    "?s c:ShuntCompensator.grounded ?grnd. "
                    "OPTIONAL {{ "
                        "?scp c:ShuntCompensatorPhase.ShuntCompensator ?s. "
                        "?scp c:ShuntCompensatorPhase.phase ?phsraw. "
                        'bind(strafter(str(?phsraw),"SinglePhaseKind.")'
                        " as ?phs) "
                    "}} "
                    "OPTIONAL {{ "
                        "?ctl c:RegulatingControl.RegulatingCondEq ?s. "
                        "?ctl c:RegulatingControl.discrete ?discrete. "
                        "?ctl c:RegulatingControl.enabled ?ctrlenabled. "
                        "?ctl c:RegulatingControl.mode ?moderaw. "
                        'bind(strafter(str(?moderaw),'
                            '"RegulatingControlModeKind.")'
                            " as ?mode) "
                        "?ctl c:RegulatingControl.monitoredPhase ?monraw. "
                        'bind(strafter(str(?monraw),"PhaseCode.") as ?monphs) '
                        "?ctl c:RegulatingControl.targetDeadband ?deadband. "
                        "?ctl c:RegulatingControl.targetValue ?setpoint. "
                        "?s c:ShuntCompensator.aVRDelay ?delay. "
                        "?ctl c:RegulatingControl.Terminal ?trm. "
                        "?trm c:Terminal.ConductingEquipment ?eq. "
                        "?eq a ?classraw. "
                        'bind(strafter(str(?classraw),"cim17#") as ?monclass) '
                        "?eq c:IdentifiedObject.name ?moneq. "
                        "?trm c:Terminal.ConnectivityNode ?moncn. "
                        "?moncn c:IdentifiedObject.name ?monbus. "
                    "}} "
                    "?s c:IdentifiedObject.mRID ?id. " 
                    "?t c:Terminal.ConductingEquipment ?s. "
                    "?t c:Terminal.ConnectivityNode ?cn. " 
                    "?cn c:IdentifiedObject.name ?bus "
            "}} "
            "ORDER by ?name "
            ).format(fdrid=fdrid)
            
        # Set and execute the query.
        self.sparql.setQuery(capQ)
        ret = self.sparql.query()
        
        # Initialize capacitor return. Ordered dict is so individuals are
        # guaranteed to have the same chromosome ordering.
        cap = OrderedDict()
        
        # We'll be tracking chromosome indices to ensure consistency between
        # individuals.
        ind = 0
        
        # Loop over the bindings.
        for el in ret.bindings:
            # If this capacitor isn't controllable, we don't want to include
            # it in our dictionary.
            # TODO: what's the best way to check if it's controllable?
            
            try:
                el['ctrlenabled']
            except KeyError:
                # No control, move on.
                continue
            
            # Unlike regulators, we'll get one return per element in the 
            # gridlabd model. Note that names are prefixed by 'cap_'
            name = 'cap_' + el['name'].value
            
            # Apparently the absence of phase (phs) indicates that all three
            # phases are present? 
            #
            # http://gridappsd.readthedocs.io/en/latest/developer_resources/index.html
            #
            # TODO: confirm.
            
            # Figure out phases
            try:
                # try to grab the phase.
                p = el['phs'].value
            except KeyError:
                # the phase doesn't exist. We use all 3.
                phaseTuple = ('A', 'B', 'C')
            else:
                # phase exists.
                phaseTuple = (p,)
                
            # To get the state, we'll need to query measurement objects....
            # TODO: get state from measurements.
            
            # For now, assume all caps start open.
            # Build dict of phases.
            phases = OrderedDict()
            for p in phaseTuple:
                phases[p] = {'prevState': 'OPEN', 'chromInd': ind}
                ind += 1
                
            # Build dictionary for this capacitor
            cap[name] = {'phases': phases, 'id': el['id'].value}
        
        """
        for el in ret.bindings:
            print('*'*60)
            for el2 in el:
                print(el2 + ": " + str(el[el2].value))
        """
            
        # All done. Return.
        return cap
    
    def getLoadNomV(self, fdrid):
        """Method to get nominal voltages from each 'EnergyConsumer.'
        
        Query source is Powergrid-Models/blazegraph/queries.txt, and it was 
        modified from there.
        
        For now, we'll just get the 'bus' (which is really the object name in
        the GridLAB-D model) and nominal voltage. Nominal voltage is given
        as 3-phase phase-to-phase, even if we're talking about a split-phase.
        I suppose that's what you get when you apply a transmission standard
        to the secondary side of a distribution system...
        
        TODO: Later, we may want to act according to the load connection/phases
        """
        
        query = \
            (PREFIX +
            r'SELECT ?name ?bus ?basev ?conn (group_concat(distinct ?phs;separator=",") as ?phases) '
            "WHERE {{ "
                "?s r:type c:EnergyConsumer. "
                'VALUES ?fdrid {{"{fdrid}"}} '
                "?s c:Equipment.EquipmentContainer ?fdr. "
                "?fdr c:IdentifiedObject.mRID ?fdrid. " 
                "?s c:IdentifiedObject.name ?name. "
                "?s c:ConductingEquipment.BaseVoltage ?bv. "
                "?bv c:BaseVoltage.nominalVoltage ?basev. "
                "?s c:EnergyConsumer.phaseConnection ?connraw. "
                'bind(strafter(str(?connraw),"PhaseShuntConnectionKind.") as ?conn) '
                "OPTIONAL {{ "
                    "?ecp c:EnergyConsumerPhase.EnergyConsumer ?s. "
                    "?ecp c:EnergyConsumerPhase.phase ?phsraw. "
                    'bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) '
                "}} "
                "?t c:Terminal.ConductingEquipment ?s. "
                "?t c:Terminal.ConnectivityNode ?cn. " 
                "?cn c:IdentifiedObject.name ?bus "
            "}} "
            "GROUP BY ?name ?bus ?basev ?p ?q ?conn "
            "ORDER by ?name "
            ).format(fdrid=fdrid)
        
        # Set and execute the query.
        self.sparql.setQuery(query)
        ret = self.sparql.query()
        
        # Initialize output
        out = {'triplex': {'v': 208 / NOMVFACTOR, 'meters': []},
               '480V': {'v': 480 / NOMVFACTOR, 'meters': []}
               }
        
        # Loop over the return
        for el in ret.bindings:
            # grab variables
            v = float(el['basev'].value)
            phs = el['phases'].value
            name = el['bus'].value
            if v == LOADS['triplex']['v'] and ('s1' in phs or 's2' in phs):
                # Triplex (split-phase) load
                if name not in out['triplex']['meters']:
                    out['triplex']['meters'].append(name)
                
            elif v == LOADS['480V']['v']:
                # Industrial 480V load
                if name not in out['480V']['meters']:
                    out['480V']['meters'].append(name)
            else:
                raise UserWarning(\
                    ('Unexpected load from blazegraph: '
                    '  name: {}\n  voltage: {}\n  phases: {}'.format(name, v,
                                                                     phs)
                     )
                                  )
            
        return out
    
    def getLoadMeasurements(self, fdrid):
        """Get listing of measurement objects attached to EnergyConsumers.
        """
        query = \
            (PREFIX +
             "SELECT ?class ?type ?name ?bus ?phases ?eqname ?eqid ?trmid ?id "
             "WHERE {{ "
                'VALUES ?fdrid {{"{fdrid}"}} '
                "?eq c:Equipment.EquipmentContainer ?fdr. "
                "?fdr c:IdentifiedObject.mRID ?fdrid. " 
                '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
                'UNION '
                '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
                '?s c:IdentifiedObject.name ?name . '
                '?s c:IdentifiedObject.mRID ?id . '
                '?s c:Measurement.PowerSystemResource ?eq . '
                '?s c:Measurement.Terminal ?trm . '
                '?s c:Measurement.measurementType ?type . '
                '?trm c:IdentifiedObject.mRID ?trmid. '
                '?eq c:IdentifiedObject.mRID ?eqid. '
                '?eq c:IdentifiedObject.name ?eqname. '
                '?eq r:type c:EnergyConsumer. '
                '?trm c:Terminal.ConnectivityNode ?cn. '
                '?cn c:IdentifiedObject.name ?bus. '
                '?s c:Measurement.phases ?phsraw . '
                '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}} '
                '}} '
                'ORDER BY ?class ?type ?name '
            ).format(fdrid=fdrid)
            
        # Set and execute the query.
        self.sparql.setQuery(query)
        ret = self.sparql.query()
        
        # Initialize output dictionary
        out = {}
        
        # Loop over the results
        for el in ret.bindings:
            try: 
                # Get the name of the meter, which in this case is 'bus'
                meter = el['bus'].value
                # If we already have a measurement for this meter, append the
                # relevant pieces. 
                out[meter]['measClass'].append(el['class'].value)
                out[meter]['measID'].append(el['id'].value)
                out[meter]['phases'].append(el['phases'].value)
                out[meter]['measType'].append(el['type'].value)
                
            except KeyError:
                # We haven't hit this meter yet, create an entry.
                out[meter] = \
                    {'measClass': [el['class'].value,],
                     'loadID': el['eqid'].value,
                     'loadName': el['eqname'].value,
                     'measID': [el['id'].value,],
                     'measName': el['name'].value,
                     'phases': [el['phases'].value],
                     'termID': el['trmid'].value,
                     'measType': [el['type'].value,]
                    }
                    
        return out
    
    def getSwingVoltage(self, fdrid):
        """Get feeder nominal voltage at the swing node.
        
        NOTE: This returns the maximum voltage returned. 
        
        TODO: Use a more sophisticated query to actually get the voltage at the
        swing node, rather than assuming it's the highest voltage in the 
        feeder.
        """
        
        # Form the query.
        query = \
            (PREFIX + 
             'SELECT DISTINCT ?vnom '
             'WHERE {{ '
             '    ?fdr c:IdentifiedObject.mRID "{fdrid}". '
             '    ?s c:ConnectivityNode.ConnectivityNodeContainer|c:Equipment.EquipmentContainer ?fdr. '
             '    ?s c:ConductingEquipment.BaseVoltage ?lev. '
             '    ?lev c:BaseVoltage.nominalVoltage ?vnom. '
             '}} '
             'ORDER by ?vnom').format(fdrid=fdrid)
             
        # Set and execute the query.
        self.sparql.setQuery(query)
        ret = self.sparql.query()
        
        # Initialize return
        maxV = 0
        
        # Loop over the return
        for el in ret.bindings:
            v = float(el['vnom'].value) 
            if v > maxV:
                maxV = v
                
        # Convert from phase to phase to phase to neutral and return
        return maxV / NOMVFACTOR
        
    def dropAll(self):
        """Simple method to drop all
        """
        # Set and execute the query.
        q = (PREFIX + " DROP ALL")
        # Get a post connection
        sparql = SPARQLWrapper(self.connStr)
        sparql.setMethod(method='POST')
        sparql.setQuery(q)
        ret = self.sparql.query()
        return ret

if __name__ == '__main__':
    obj = sparqlCIM()
    #ret = obj.dropAll()
    #reg = obj.getRegs()
    reg = obj.getLoadMeasurements(fdrid=FDRID)
    print('yay')