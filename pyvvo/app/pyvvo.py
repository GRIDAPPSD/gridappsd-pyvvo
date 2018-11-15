"""
This is the 'run' module for the pyvvo application

Created on Jan 25, 2018

@author: thay838
"""
# gridappsd-python imports
from gridappsd import GridAPPSD, topics, utils, difference_builder

# Standard library
import logging
import argparse
import sys
import time
import datetime

# Installed
import simplejson as json
from simplejson.errors import JSONDecodeError
import numpy as np
import pandas as pd
import dateutil
# TODO: just using this for debugging
import simplejson as json

# pyvvo
import sparql_queries
import modGLM
import constants as CONST
import db
import helper
import population
import individual

import re

# Logging.
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)


def read_config(config_file):
    """Helper function to read pyvvo configuration file.
    """
    with open(config_file) as c:
        config = json.load(c)

    return config


def buildRecorderDicts(energyInterval, powerInterval, voltageInterval,
                       energyPowerMeter, triplexGroup, recordMode,
                       query_buffer_limit):
    """Helper function to construct dictionaries to be used by individuals to
    add recorders to their own models.
    
    Note that the returned dictionary will more or less be directly passed to
    a genetic.individual object, and subsequently passed to the appropriate
    method in glm.modGLM.
    
    We could add custom table definitions in the future, but why?
    """
    recorders = {
        'energy': {'objType': 'recorder',
                   'properties': {'parent': energyPowerMeter,
                                  'table': 'energy',
                                  'interval': energyInterval,
                                  'propList': ['measured_real_energy', ],
                                  'limit': -1,
                                  'mode': recordMode,
                                  'query_buffer_limit': query_buffer_limit
                                  }
                   },
        'power': {'objType': 'recorder',
                  'properties': {'parent': energyPowerMeter,
                                 'table': 'power',
                                 'interval': powerInterval,
                                 'propList': ['measured_real_power',
                                              'measured_reactive_power'],
                                 'limit': -1,
                                 'mode': recordMode,
                                 'query_buffer_limit': query_buffer_limit
                                 }
                  },
        'triplexVoltage': {'objType': 'recorder',
                           'properties': {'group': triplexGroup,
                                          'propList': [
                                              'measured_voltage_1.mag',
                                              'measured_voltage_2.mag'],
                                          'interval': voltageInterval,
                                          'table': 'triplexVoltage',
                                          'limit': -1,
                                          'mode': recordMode,
                                          'query_buffer_limit': query_buffer_limit
                                          }
                           }
    }

    return recorders


def setup_log(log_config):
    """Helper to setup the log.
    
    INPUTS:
    logConfig -- sub-dictionary pulled from config['LOG'].
        keys: FILE, MODE, LEVEL, FORMAT
    
    OUTPUTS:
    logging.Logger object
    """

    # Set up the log.
    log = logging.getLogger(__name__)
    # Set log level.
    level = getattr(logging, log_config['LEVEL'].upper())
    log.setLevel(level)
    # Create file handler for log.
    if log_config['FILE']:
        # Use a file handler.
        handler = logging.FileHandler(filename=log_config['FILE'],
                                      mode=log_config['MODE'])
    else:
        # Use a stream handler.
        handler = logging.StreamHandler()

    # Set its level.
    handler.setLevel(level)
    # Create formatter for file handler. Use date format in constants module.
    formatter = logging.Formatter(log_config['FORMAT'],
                                  datefmt=log_config['DATE_FMT'])
    # Previously used format:
    # "[%(asctime)s] [thread %(thread)d] [%(module)s] [%(levelname)s]: %(message)s"
    # Attach the formatter to the file handler.
    handler.setFormatter(formatter)
    # Attach the file handler to the log
    log.addHandler(handler)

    return log


def get_model_id(gridappsd_object, model_name, log):
    """Given a model's name, get it's ID from the platform."""
    # Get model information.
    log.debug('Calling GridAPPSD.query_model_info.')
    result = gridappsd_object.query_model_info()

    #
    if not result['responseComplete']:
        s = ('GridAPPSD.query_model_info responseComplete field not return '
             'True.')
        log.error(s)
        raise UserWarning(s)
    else:
        log.debug('GridAPPSD.query_model_info call successful.')

    # Loop over the models until we find our model_name, and get its ID.
    model_id = None
    for model in result['data']['models']:
        if model['modelName'] == model_name:
            model_id = model['modelId']

    log.debug('Model ID for {} is {}'.format(model_name, model_id))

    if model_id is None:
        s = 'Could not find the model ID for {}.'.format(MODEL)
        log.error(s)
        raise UserWarning(s)

    return model_id


def get_all_model_data(gridappsd_object, model_id, log):
    """Helper to pull all requisite model data for pyvvo.

    This includes: voltage regulators, capacitors, load nominal
        voltages, swing bus voltage, and load measurement data.
    """
    # Define dictionary for each type of data.
    data_list = [
        {'type': 'voltage_regulator',
         'query_string': sparql_queries.REGULATOR_QUERY,
         'parse_function': sparql_queries.parse_regulator_query
         },
        {'type': 'capacitor',
         'query_string': sparql_queries.CAPACITOR_QUERY,
         'parse_function': sparql_queries.parse_capacitor_query
         },
        {'type': 'load_nominal_voltage',
         'query_string': sparql_queries.LOAD_NOMINAL_VOLTAGE_QUERY,
         'parse_function': sparql_queries.parse_load_nominal_voltage_query
         },
        {'type': 'swing_voltage',
         'query_string': sparql_queries.SWING_VOLTAGE_QUERY,
         'parse_function': sparql_queries.parse_swing_voltage_query
         },
        {'type': 'load_measurements',
         'query_string': sparql_queries.LOAD_MEASUREMENTS_QUERY,
         'parse_function': sparql_queries.parse_load_measurements_query
         },
    ]

    # Initialize return.
    out = {}

    # Loop over the data.
    for data_dict in data_list:
        out[data_dict['type']] = query_and_parse(
            gridappsd_object=gridappsd_object,
            query_string=data_dict['query_string'].format(fdrid=model_id),
            parse_function=data_dict['parse_function'],
            log_string=data_dict['type'].replace('_', ' '), log=log)

    return out


def query_and_parse(gridappsd_object, query_string, parse_function,
                    log_string, log):
    # Get data.
    data = gridappsd_object.query_data(query_string)
    log.info('Retrieved {} data.'.format(log_string))

    # Parse data.
    data_parsed = parse_function(data['data']['results']['bindings'])
    log.info('Parsed {} data'.format(log_string))

    return data_parsed


class DumpOutput:

    def __init__(self):
        self.count = 0

    def on_message(self, headers, message):
        if self.count < 20:
            with open('sim_out_{}.json'.format(self.count), 'w') as f:
                f.write(message)

            self.count += 1


def command_regulators(log, sim_id, reg_dict, gridappsd_object, sim_in_topic):
    """Send commands for regulators based on results of GA.
    """
    # Loop over regulators.
    for reg_name, reg in reg_dict.items():
        # Log.
        log.info('Building command for voltage regulator "{}".'.format(
            reg_name))

        # Get a diff builder. Note we could do this outside the loop, but we
        # may as well build messages for each regulator individually.
        diff_builder = difference_builder.DifferenceBuilder(sim_id)

        # Extract the voltage increment.
        # TODO: actually use this when the platform is updated.
        v_incr = reg['stepVoltageIncrement']

        # Loop over regulator phases.
        for phase, phase_dict in reg['phases'].items():

            # Translate tap positions to steps.
            # TODO: actually use this when the platform is updated.
            # old_step = \
            #   helper.reg_tap_gld_to_cim(phase_dict['prevState'], v_incr)
            # new_step = \
            #   helper.reg_tap_gld_to_cim(phase_dict['newState'], v_incr)

            # Only command regulators if we need to.
            if phase_dict['newState'] != phase_dict['prevState']:
                diff_builder.add_difference(
                    object_id=phase_dict['mrid'],
                    attribute='TapChanger.step',
                    forward_value=phase_dict['newState'],
                    reverse_value=phase_dict['prevState'])

        msg = diff_builder.get_message()
        log.info('Regulator command for {}:\n{}.'.format(
            reg_name, json.dumps(msg, indent=4)))
        gridappsd_object.send(topic=sim_in_topic, message=json.dumps(msg))
        log.info('Regulator command for {} sent.'.format(reg_name))


def command_capacitors(log, sim_id, cap_dict, gridappsd_object, sim_in_topic):
    """Send commands for capacitors based on results of GA.
    """
    log.debug('Running "command_capacitors" function.')
    # Loop over capacitors.
    for cap_name, cap in cap_dict.items():
        # Log.
        log.info('Building command for capacitor "{}".'.format(cap_name))

        # Get a diff builder. Note we could do this outside the loop, but we
        # may as well build messages for each capacitors individually.
        diff_builder = difference_builder.DifferenceBuilder(sim_id)

        # Loop over capacitor phases.
        for phase, phase_dict in cap['phases'].items():
            log.debug('Getting state for {}, phase {}'.format(cap_name, phase))

            # Only command regulators if we need to.
            if phase_dict['newState'] != phase_dict['prevState']:
                new_pos = individual.CAPSTATUS.index(phase_dict['newState'])
                old_pos = individual.CAPSTATUS.index(phase_dict['prevState'])
                diff_builder.add_difference(
                    object_id=cap['mrid'],
                    attribute='ShuntCompensator.sections',
                    forward_value=new_pos, reverse_value=old_pos)

        if len(diff_builder._forward) > 0:
            msg = diff_builder.get_message()
            log.info('Capacitor command for {}:\n{}.'.format(
                cap_name, json.dumps(msg, indent=4)))
            gridappsd_object.send(topic=sim_in_topic, message=json.dumps(msg))
            log.info('Capacitor command for {} sent.'.format(cap_name))
        else:
            log.info('No command necessary for capacitor {}'.format(cap_name))


def run(log, config, sim_id, model_id, gridappsd_address, sim_in_topic,
        sim_out_topic=None):

    # Initialize GridAPPSD object from within the platform.
    gridappsd_object = GridAPPSD(simulation_id=sim_id,
                                 address=gridappsd_address,
                                 username=utils.get_gridappsd_user(),
                                 password=utils.get_gridappsd_pass())

    # if sim_out_topic is not None:
    #     dump_output = DumpOutput()
    #     gridappsd_object.subscribe(topic=sim_out_topic, callback=dump_output)

    # Get all relevant model data from blazegraph/CIM.
    model_data = get_all_model_data(gridappsd_object, model_id, log=log)

    # Hard-code timezone for weather data.
    # TODO: when to "un-hard code?"
    tz = dateutil.tz.gettz('America/Denver')

    # Hard-code starting date in 2013, since that's what we have weather
    # data for.
    # TODO: "un-hard code" when possible
    # TODO: When UTC conversion bug is fixed with weather data, change
    # hour from 7 to 0 below.
    start_dt = datetime.datetime(year=2013, month=1, day=1, hour=7, minute=0,
                                 second=0, microsecond=0, tzinfo=tz)
    # NOTE: while this probably is slower than adding to the Unix
    # timestamp, this gives us handy dates for logging. This probably
    # isn't the way to go long term.
    end_dt = start_dt + dateutil.relativedelta.relativedelta(days=14)

    # Convert to Unix timestamps (which are in UTC)
    start_ts = datetime.datetime.timestamp(start_dt)
    end_ts = datetime.datetime.timestamp(end_dt)

    # The platform uses microseconds since the epoch, rather than
    # seconds, so be sure to convert. Also, it's taking strings, which
    # is annoying.
    start_time = '{:.0f}'.format(start_ts*1e6)
    end_time = '{:.0f}'.format(end_ts*1e6)

    # Pull weather data for the specified interval from the time series
    # database, and average it over 15 minute intervals.
    interval = 15
    interval_unit = 'min'
    weather = get_weather(gridappsd_object, start_time, end_time,
                          interval=interval, interval_unit=interval_unit)

    # Get strings for dates (logging only)
    # TODO: hard-coding date formatting... yay!
    # TODO: should probably only do this if the log level is INFO.
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S%z')
    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S%z')

    # Log it.
    log_str = \
        ('Weather data for {} through {} '.format(start_str, end_str)
         + 'pulled and averaged over {} {} '.format(interval, interval_unit)
         + 'intervals.')
    log.info(log_str)

    # Loop over the load measurements and pull historic data.

    # Grab a single MRID for experimentation
    meter_name = 'sx2673305b'
    data = get_data_for_meter(gridappsd_object, sim_id,
                              model_data['load_measurements'][meter_name])
    log.info('Data for meter {} pulled and parsed.'.format(meter_name))
    print('Measurements for meter {}:'.format(meter_name))
    print(data)

    # Get the GridLAB-D model
    # TODO: add to the Python API.
    payload = {'configurationType': 'GridLAB-D Base GLM',
               'parameters': {'model_id': model_id}}
    gld_model = gridappsd_object.get_response(topic=topics.CONFIG,
                                              message=payload,
                                              timeout=30)
    log.info('GridLAB-D model for GA use received.')

    # Remove the json remnants from the message via regular expressions.

    gld_model['message'] = re.sub('^\s*\{\s*"data"\s*:\s*', '',
                                  gld_model['message'])
    gld_model['message'] = re.sub('\s*,\s*"responseComplete".+$', '',
                                  gld_model['message'])
    log.warn('Bad json for GridLAB-D model fixed via regular expressions.')

    # Get a modGLM model to modify the base model.
    model_obj = modGLM.modGLM(strModel=gld_model['message'],
                              pathModelOut='test.glm', pathModelIn='pyvvo.glm'
                              )
    model_obj.writeModel()
    log.info('modGLM object instantiated.')

    # Set up the model to run.
    st = '2016-01-01 00:00:00'
    et = '2016-01-01 00:15:00'
    tz = 'UTC0'

    swing_meter_name = model_obj.setupModel(
        starttime=st, stoptime=et, timezone=tz, database=config['GLD-DB'],
        powerflowFlag=True, vSource=model_data['swing_voltage'],
        triplexGroup=CONST.LOADS['triplex']['group'],
        triplexList=model_data['load_nominal_voltage']['triplex']['meters']
    )
    log.info('GridLAB-D model prepped for GA use.')

    # Write the base model
    # model_obj.writeModel()
    # log.info('Base GridLAB-D model configured and written to file.')

    # Build dictionary of recorder definitions which individuals in the
    # population will add to their model. We'll use the append record mode.
    # This can be risky! If you're not careful about clearing the database out
    # between subsequent test runs, you can write duplicate rows.
    recorders = buildRecorderDicts(
        energyInterval=config['INTERVALS']['OPTIMIZATION'],
        powerInterval=config['INTERVALS']['SAMPLE'],
        voltageInterval=config['INTERVALS']['SAMPLE'],
        energyPowerMeter=swing_meter_name,
        triplexGroup=CONST.LOADS['triplex']['group'],
        recordMode='a',
        query_buffer_limit=config['GLD-DB-OTHER']['QUERY_BUFFER_LIMIT']
    )
    log.info('Recorder dictionaries created.')

    # Convert costs from fraction of nominal voltage to actual voltage
    # Get pointer to costs dict.
    costs = config['COSTS']
    costs['undervoltage']['limit'] = \
        (costs['undervoltage']['limit']
         * model_data['load_nominal_voltage']['triplex']['v'])
    costs['overvoltage']['limit'] = \
        (costs['overvoltage']['limit']
         * model_data['load_nominal_voltage']['triplex']['v'])
    log.info('Voltage fractions converted to actual voltage for costs.')

    # Initialize a clock object for datetimes.
    clockObj = helper.clock(startStr=st, finalStr=et,
                            interval=config['INTERVALS']['OPTIMIZATION'],
                            tzStr=tz)
    log.info('Clock object initialized')

    # Connect to the MySQL database for GridLAB-D simulations
    db_obj = db.db(**config['GLD-DB'],
                   pool_size=config['GLD-DB-OTHER']['NUM-CONNECTIONS'])
    log.info('Connected to MySQL for GA GridLAB-D output.')

    # Clear out the database while testing.
    # TODO: take this out?
    db_obj.dropAllTables()
    log.warning(
        'All tables dropped in {}'.format(config['GLD-DB']['database']))
        
    # Initialize a population.
    # TODO - let's get the 'inPath' outta here. It's really just being used for
    # model naming, and we may as well be more explicit about that.
    sdt, edt = clockObj.getStartStop()
    pop_obj = population.population(
        strModel=model_obj.strModel, numInd=config['GA']['INDIVIDUALS'],
        numGen=config['GA']['GENERATIONS'],
        numModelThreads=config['GA']['THREADS'], recorders=recorders,
        dbObj=db_obj, starttime=sdt, stoptime=edt,
        timezone=tz, inPath=model_obj.pathModelIn,
        outDir='/pyvvo/pyvvo/models',
        reg=model_data['voltage_regulator'], cap=model_data['capacitor'],
        costs=costs, probabilities=config['PROBABILITIES'],
        gldInstall=config['GLD-INSTALLATION'],
        randomSeed=config['RANDOM-SEED'],
        log=log, baseControlFlag=0)

    log.info('Population object initialized.')

    log.info('Starting genetic algorithm...')
    best_ind = pop_obj.ga()
    log.info('Shutting down genetic algorithm threads...')
    pop_obj.stopThreads()

    log.info('Baseline costs:\n{}'.format(json.dumps(
        pop_obj.baselineData['costs'], indent=4)))
    log.info('Best individual:\n{}'.format(best_ind))

    # Send commands.
    if sim_in_topic is not None:
        command_capacitors(log=log, sim_id=sim_id, cap_dict=best_ind.cap,
                           gridappsd_object=gridappsd_object,
                           sim_in_topic=sim_in_topic)

        log.warning('Sleeping 5 seconds before commanding regulators.')
        time.sleep(5)

        command_regulators(log=log, sim_id=sim_id, reg_dict=best_ind.reg,
                           gridappsd_object=gridappsd_object,
                           sim_in_topic=sim_in_topic)


def get_historic_measurements(gd, sim_id, mrid):
    """"""
    # TODO: Use topics from gridappsd-python when fixed.
    t = '/queue/goss.gridappsd.process.request.data.timeseries'
    payload = {'queryMeasurement': 'PROVEN_MEASUREMENT',
               'queryFilter': {'hasSimulationId': sim_id,
                               'hasMrid': mrid},
               'responseFormat': 'JSON'}
    sim_data = gd.get_response(topic=t, message=payload, timeout=30)

    meas_df = parse_historic_measurements(sim_data)
    pass


def get_data_for_meter(gd, sim_id, meter_dict):
    """NOTE: This is currently hard-coded to work for split-phase
       residences.
    """

    # TODO: We could initialize both the query dictionary and topic
    # outside of this function for efficiency.

    # Initialize query dictionary.
    qd = {'queryMeasurement': 'PROVEN_MEASUREMENT',
          'queryFilter': {'hasSimulationId': sim_id, 'hasMrid': None},
          'responseFormat': 'JSON'}

    # Hard code topic.
    # TODO: Use topic from gridappsd-python when fixed.
    topic = '/queue/goss.gridappsd.process.request.data.timeseries'

    # Loop over the various measurements in the meter dictionary.
    # Note that measClass, measID, phases, and measType should all be
    # the same length. See sparql_queries.parse_load_measurements_query.
    # TODO: We aren't currently using measClass
    # TODO: Querying the database 4x per meter is terrible... Fix this
    # when the API supports better filtering.
    for idx, meas_id in enumerate(meter_dict['measID']):

        # Update query dictionary.
        qd['queryFilter']['hasMrid'] = meas_id

        # Query the database.
        meas_data = gd.get_response(topic=topic, message=qd, timeout=60)

        # Parse the return.
        t, mag, angle = parse_historic_measurements(meas_data)

        # Create time index.
        t_index = pd.to_datetime(t, unit='s', utc=True, origin='unix',
                                 box=True)

        # On the first iteration, initialize the DataFrame.
        # TODO: should this be a try-catch instead? Or initialize
        # outside of the loop then re-index?
        if idx == 0:
            meas_df = pd.DataFrame(0+1j*0, columns=['voltage', 'power'],
                                   index=t_index)

        # Update the appropriate column of the DataFrame.
        if meter_dict['measType'][idx] == 'PNV':
            col = 'voltage'
        elif meter_dict['measType'][idx] == 'VA':
            col = 'power'
        else:
            raise UserWarning('Unexpected measurement type: {}'.format(
                meter_dict['measType'][idx]))

        # Simply sum the complex values.
        meas_df[col] += get_complex_polar(mag, angle)

    # TODO: extract V mag, P, Q
    return meas_df


def rad_to_deg(angle):
    """"""
    # TODO: this doesn't belong here
    return angle * np.pi / 180


def get_complex_polar(mag, angle):
    """Angle must be in radians"""
    # TODO: this doesn't belong here.
    return mag * np.exp(1j * angle)


def parse_historic_measurements(data):
    """"""
    t = []
    mag = []
    angle = []

    # Loop over the "rows."
    for row in data['data']['measurements'][0]['points']:
        # Loop over all the measurements, since they aren't properly
        # keyed.
        for meas_dict in row['row']['entry']:
            # Grab type and value of measurement.
            meas_type = meas_dict['key']
            meas_value = meas_dict['value']

            if meas_type == 'hasMagnitude':
                mag.append(float(meas_value))
            elif meas_type == 'hasAngle':
                angle.append(float(meas_value))
            elif meas_type == 'time':
                # TODO: should we use a float or integer? I'll use an
                # integer since fractions of a second aren't important
                # for this application.
                t.append(int(meas_value))

    # TODO: the database is returning time in seconds since the epoch,
    # but our queries are in microseconds...
    # t_index = pd.to_datetime(t, unit='s', utc=True, origin='unix', box=True)

    # Convert angles to radians (cause why would we ever use degrees?)
    angle = rad_to_deg(np.array(angle))

    # Get
    # df = pd.DataFrame({'mag': mag, 'angle': angle}, index=t_index)

    return t, np.array(mag), angle


def get_weather(gd, start_time, end_time, interval, interval_unit):
    """temp function for getting weather data."""
    payload = {'queryMeasurement': 'weather',
               'queryFilter': {'startTime': start_time,
                               'endTime': end_time},
               'responseFormat': 'JSON'}
    # NOTE: topics.TIMESERIES is:
    # '/queue/goss.gridappsd.process.request.timeseries'
    # while we need:
    # '/queue/goss.gridappsd.process.request.data.timeseries'
    t = '/queue/goss.gridappsd.process.request.data.timeseries'
    weather_data = gd.get_response(topic=t, message=payload, timeout=30)
    # Get DataFrame of weather
    df_weather = parse_weather(weather_data)

    # Resample, remove negative GHI.
    return adjust_weather(df_weather, interval=interval, interval_unit='Min')


def parse_weather(data):
    """Helper to parse the ridiculous platform weather data return.

    For the journal paper, we used "solar_flux" from GridLAB-D. It seems
    simplest here to use the "global" irradiance, which is the sum of
    direct and diffuse irradiation.
    """
    # Initialize dictionary (which we'll later convert to a DataFrame)
    wd = {'temperature': [], 'ghi': []}
    t = []

    # Loop over the "rows."
    for row in data['data']['measurements'][0]['points']:
        # Loop over all the measurements, since they aren't properly
        # keyed.
        for meas_dict in row['row']['entry']:
            # Grab type and value of measurement.
            meas_type = meas_dict['key']
            meas_value = meas_dict['value']

            if meas_type == 'TowerDryBulbTemp':
                wd['temperature'].append(float(meas_value))
            elif meas_type == 'GlobalCM22':
                wd['ghi'].append(float(meas_value))
            elif meas_type == 'time':
                # TODO: should we use a float or integer? I'll use an
                # integer since fractions of a second aren't important
                # for this application.
                t.append(int(meas_value))

    # TODO: the database is returning time in seconds since the epoch,
    # but our queries are in microseconds...
    t_index = pd.to_datetime(t, unit='s', utc=True, origin='unix',
                             box=True)
    # Convert to pandas DataFrame
    df_weather = pd.DataFrame(wd, index=t_index)
    return df_weather


def adjust_weather(data, interval, interval_unit):
    """Resample weather data, zero out negative GHI.

    data should be DataFrame from parse_weather
    interval: e.g. 15
    interval_unit: e.g. "Min"
    """
    # Get 15-minute average. Since we want historic data leading up to
    # the time in our interval, use 'left' options
    weather = data.resample('{}{}'.format(interval, interval_unit),
                            closed='right', label='right').mean()

    # Zero-out negative GHI.
    weather['ghi'][weather['ghi'] < 0] = 0

    return weather


def main():
    # Switch for app being managed by platform vs running outside of it.
    IN_PLATFORM = False

    # Read configuration file.
    config = read_config('config.json')

    # Setup log.
    log = setup_log(config['LOG'])
    log.info('Log configured.')

    if IN_PLATFORM:
        # In docker-compose, use the service name.
        MYSQL_HOST = 'mysql-pyvvo'

        # When running inside the platform, use standard MySQL port.
        MYSQL_PORT = 3306

        # Get the gridappsd_address.
        gridappsd_address = utils.get_gridappsd_address()

        # Initialize argument parser.
        parser = argparse.ArgumentParser()

        # Get simulation ID.
        parser.add_argument("sim_id", help="Simulation ID to send/receive "
                                           "data/commands")

        # Get the simulation request so we can extract the model ID.
        parser.add_argument("sim_request", help="Request sent to start "
                                                "simulation.")

        # Extract arguments.
        args = parser.parse_args()
        log.debug('Arguments parsed.')

        # Get the topic for listening to simulation output.
        sim_out_topic = topics.fncs_output_topic(args.sim_id)

        # Get the topic for sending commands to simulation.
        sim_in_topic = topics.fncs_input_topic(args.sim_id)

        # Get the simulation request into a dictionary.
        sim_request = json.loads(args.sim_request.replace("\'", ""))

        # Extract the model ID. It's silly that they've labeled the model
        # "Line_name"
        model_id = sim_request["power_system_config"]["Line_name"]
        log.debug('Model MRID: {}'.format(model_id))

        # Extract the sim_id.
        sim_id = args.sim_id

    else:
        # For development, use this machine's internal IP.
        '''
        # The code below only works when run OUTSIDE of a container.
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        INTERNAL_IP = s.getsockname()[0]
        s.close()
        '''
        INTERNAL_IP = '192.168.0.26'

        MYSQL_HOST = INTERNAL_IP

        # MYSQL_PORT is different when running outside the platform.
        MYSQL_PORT = 3307

        # Define where we can connect to the platform externally.
        gridappsd_address = (INTERNAL_IP, 61613)

        # For now, not listening to a simulation topic.
        # NOTE: the sim_id below is for an offline simulation, so I
        # can grab timeseries information.
        sim_id = "293975150"
        sim_out_topic = None
        sim_in_topic = None

        # Initialize GridAPPSD object so we can pull the model ID.
        gridappsd_object = GridAPPSD(address=gridappsd_address)

        # Get the id for the 8500 node feeder.
        model_name = 'ieee8500'
        model_id = get_model_id(gridappsd_object=gridappsd_object,
                                model_name=model_name, log=log)
        log.info('Retrieved model ID for {}.'.format(model_name))

    # Override the MySQL host and port.
    config['GLD-DB']['host'] = MYSQL_HOST
    config['GLD-DB']['port'] = MYSQL_PORT

    run(log=log, config=config, sim_id=sim_id, model_id=model_id,
        sim_in_topic=sim_in_topic, gridappsd_address=gridappsd_address,
        sim_out_topic=None)

    # Initialize GridAPPSD object.
    gridappsd_object = GridAPPSD(simulation_id=sim_id,
                                 address=gridappsd_address,
                                 username=utils.get_gridappsd_user(),
                                 password=utils.get_gridappsd_pass())

    # Grab measurement information
    load_meas = \
        query_and_parse(
            gridappsd_object,
            query_string=sparql_queries.LOAD_MEASUREMENTS_QUERY.format(fdrid=model_id),
            parse_function=sparql_queries.parse_load_measurements_query,
            log_string='Load Measurement', log=log
        )

    log.info('Terminating pyvvo...')
    sys.exit(0)


if __name__ == '__main__':
    main()
