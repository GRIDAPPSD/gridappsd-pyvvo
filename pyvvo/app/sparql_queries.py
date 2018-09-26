"""Collection of sparql queries for use by pyvvo."""
from collections import OrderedDict

from helper import binaryWidth
from constants import LOADS, NOMVFACTOR

# Define query prefix
PREFIX = """
PREFIX r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX c: <http://iec.ch/TC57/2012/CIM-schema-cim17#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Query for getting regulator information. Should be formatted with a
# feeder ID: .format(fdrid=fdrid)
REGULATOR_QUERY = \
    (PREFIX +
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
     )

# Query for getting capacitor information. Should be formatted with a
# # feeder ID: .format(fdrid=fdrid)
CAPACITOR_QUERY = \
    (PREFIX +
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
     )

# Query to get nominal voltage of all loads. Should be formatted with a
# feeder ID: .format(fdrid=fdrid)
LOAD_NOMINAL_VOLTAGE_QUERY = \
    (PREFIX +
     r'SELECT ?name ?bus ?basev ?conn '
     '(group_concat(distinct ?phs;separator=",") as ?phases) '
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
     )

# Query to get swing voltage.
# TODO: This is a bad hack, which sorts voltage magnitudes, and the
# corresponding function just grabs the highest.
SWING_VOLTAGE_QUERY = \
    (PREFIX +
     'SELECT DISTINCT ?vnom '
     'WHERE {{ '
     '    ?fdr c:IdentifiedObject.mRID "{fdrid}". '
     '    ?s c:ConnectivityNode.ConnectivityNodeContainer|c:Equipment.'
     'EquipmentContainer ?fdr. '
     '    ?s c:ConductingEquipment.BaseVoltage ?lev. '
     '    ?lev c:BaseVoltage.nominalVoltage ?vnom. '
     '}} '
     'ORDER by ?vnom')

# Query for getting listing of measurement objects attached to EnergyConsumers.
LOAD_MEASUREMENTS_QUERY = \
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
     )


def parse_regulator_query(bindings):
    """Parse regulator query response given bindings."""
    # Initialize dict to store regulator information. It's ordered to we
    # can count on all individuals having the same chromosome indices.
    reg = OrderedDict()

    # We'll be assigning chromosome positions as we go. This ensures that
    # all individuals have chromosomes which line up. Intialize counters.
    s = 0
    e = 0

    # Loop over the regulators. Note that we'll get an object per phase, so
    # be cognizant of that.
    for el in bindings:
        # Extract the regulator's name. Note that names are prefixed by
        # 'reg_'
        name = 'reg_' + el['pname']['value']

        # If we haven't initialized this regulator's dict, do so now
        try:
            reg[name]
        except KeyError:
            # Initialize to dictionary
            reg[name] = {}

        # Compute 'raise_taps' and 'lower_taps'
        raise_taps = (int(el['highStep']['value'])
                      - int(el['neutralStep']['value'])
                      )
        lower_taps = (int(el['neutralStep']['value'])
                      - int(el['lowStep']['value'])
                      )

        # Compute the number of binary values needed to represent the
        # number of taps.
        num_taps = int(el['highStep']['value']) - int(el['lowStep']['value'])
        width = binaryWidth(num_taps)

        # Increment the ending index
        e += width

        # Grab the step voltage increment
        step_voltage_increment = float(el['incr']['value'])

        # Put top-level properties in the dictionary
        for name_val in [('raise_taps', raise_taps),
                         ('lower_taps', lower_taps),
                         ('stepVoltageIncrement', step_voltage_increment),
                         ('id', el['id']['value']),
                         ]:
            try:
                # Attempt to access the key.
                same_val = (reg[name][name_val[0]] == name_val[1])
            except KeyError:
                # Key doesn't exist, assign it.
                reg[name][name_val[0]] = name_val[1]
            else:
                # Key exists, ensure that this phase has the same value as
                # was assigned previously.
                if not same_val:
                    raise ValueError('Regulator {} does not '.format(name)
                                     + 'have the same '
                                     + '{} on all '.format(name_val[0])
                                     + 'phases.')

        # Compute the nominal tap position for GridLAB-D.
        # TODO: this should probably be obtained from some sort of
        # measurement object.
        prev_state = round(((float(el['step']['value']) - 1) * 100)
                           / step_voltage_increment)

        # Set the 'prevState' for this phase.
        try:
            # Attempt to access phases ('phases')
            reg[name]['phases']
        except KeyError:
            # phases hasn't been initialized.
            reg[name]['phases'] = OrderedDict()

        # Build dict for this phase
        reg[name]['phases'][el['phs']['value'].upper()] = \
            {'prevState': prev_state, 'chromInd': (s, e)}

        # Increment the starting index
        s += width

    return reg


def parse_capacitor_query(bindings):
    """Parse capacitor response given bindings."""
    # Initialize capacitor return. Ordered dict is so individuals are
    # guaranteed to have the same chromosome ordering.
    cap = OrderedDict()

    # We'll be tracking chromosome indices to ensure consistency between
    # individuals.
    ind = 0

    # Loop over the bindings.
    for el in bindings:
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
        name = 'cap_' + el['name']['value']

        # Apparently the absence of phase (phs) indicates that all three
        # phases are present? 
        #
        # http://gridappsd.readthedocs.io/en/latest/developer_resources/index.html
        #
        # TODO: confirm.

        # Figure out phases
        try:
            # try to grab the phase.
            p = el['phs']['value']
        except KeyError:
            # the phase doesn't exist. We use all 3.
            phase_tuple = ('A', 'B', 'C')
        else:
            # phase exists.
            phase_tuple = (p,)

        # To get the state, we'll need to query measurement objects....
        # TODO: get state from measurements.

        # For now, assume all caps start open.
        # Build dict of phases.
        phases = OrderedDict()
        for p in phase_tuple:
            phases[p] = {'prevState': 'OPEN', 'chromInd': ind}
            ind += 1

        # Build dictionary for this capacitor
        cap[name] = {'phases': phases, 'id': el['id']['value']}

    return cap


def parse_load_nominal_voltage_query(bindings):
    """Parse load nominal voltage bindings."""
    # Initialize output
    out = {'triplex': {'v': 208 / NOMVFACTOR, 'meters': []},
           '480V': {'v': 480 / NOMVFACTOR, 'meters': []}
           }

    # Loop over the return
    for el in bindings:
        # grab variables
        v = float(el['basev']['value'])
        phs = el['phases']['value']
        name = el['bus']['value']
        if v == LOADS['triplex']['v'] and ('s1' in phs or 's2' in phs):
            # Triplex (split-phase) load
            if name not in out['triplex']['meters']:
                out['triplex']['meters'].append(name)

        elif v == LOADS['480V']['v']:
            # Industrial 480V load
            if name not in out['480V']['meters']:
                out['480V']['meters'].append(name)
        else:
            raise UserWarning( 
                ('Unexpected load from blazegraph: '
                 '  name: {}\n  voltage: {}\n  phases: {}'.format(name, v,
                                                                  phs)
                 )
            )
        
    return out


def parse_swing_voltage_query(bindings):
    # TODO: This is a bad hack which just grabs the highest voltage in
    # the system.

    # Initialize return
    max_v = 0

    # Loop over the return
    for el in bindings:
        v = float(el['vnom']['value'])
        if v > max_v:
            max_v = v

    # Convert from phase to phase to phase to neutral and return
    return max_v / NOMVFACTOR


def parse_load_measurements_query(bindings):
    # Initialize output dictionary
    out = {}

    # Loop over the results
    for el in bindings:
        # Get the name of the meter, which in this case is 'bus'
        meter = el['bus']['value']

        try:
            # If we already have a measurement for this meter, append the
            # relevant pieces. 
            out[meter]['measClass'].append(el['class']['value'])
            out[meter]['measID'].append(el['id']['value'])
            out[meter]['phases'].append(el['phases']['value'])
            out[meter]['measType'].append(el['type']['value'])

        except KeyError:
            # We haven't hit this meter yet, create an entry.
            out[meter] = \
                {'measClass': [el['class']['value'], ],
                 'loadID': el['eqid']['value'],
                 'loadName': el['eqname']['value'],
                 'measID': [el['id']['value'], ],
                 'measName': el['name']['value'],
                 'phases': [el['phases']['value']],
                 'termID': el['trmid']['value'],
                 'measType': [el['type']['value'], ]
                 }

    return out
