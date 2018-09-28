"""
This is the 'main' module for the pyvvo application

Created on Jan 25, 2018

@author: thay838
"""
# gridappsd-python imports
from gridappsd import GridAPPSD, topics, utils

# Standard library
import logging

# Installed
import simplejson as json
from simplejson.errors import JSONDecodeError

# pyvvo
import sparql_queries
import modGLM
import constants as CONST
import db
import helper
import population


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


def main():
    # Read configuration file.
    config = read_config('config.json')

    # Setup log.
    log = setup_log(config['LOG'])
    log.info('Log configured.')

    # For development, hard-code this machine's internal IP.
    # MYSQL_HOST = '192.168.0.33'
    # In docker-compose, use the service name.
    MYSQL_HOST = 'mysql-pyvvo'

    # When running inside the platform, use standard MySQL port.
    MYSQL_PORT = 3306

    # MYSQL_PORT is different when running outside the platform.
    # MYSQL_PORT = 3307

    # Override the MySQL host and port.
    config['GLD-DB']['host'] = MYSQL_HOST
    config['GLD-DB']['port'] = MYSQL_PORT

    # Initialize GridAPPSD object from within the platform.
    gridappsd_object = GridAPPSD(address=utils.get_gridappsd_address(),
                                 username=utils.get_gridappsd_user(),
                                 password=utils.get_gridappsd_pass())

    # Initialize GridAPPSD object when outside the platform.
    # gridappsd_object = GridAPPSD(address=('192.168.0.33', 61613))

    # We'll be using the 8500 node feeder.
    MODEL = 'ieee8500'

    # Get ID for feeder.
    model_id = get_model_id(gridappsd_object=gridappsd_object,
                            model_name=MODEL, log=log)
    log.info('Retrieved model ID.')

    # Get all relevant model data.
    model_data = get_all_model_data(gridappsd_object, model_id, log=log)

    # Get the GridLAB-D model
    # TODO: add to the Python API.
    payload = {'configurationType': 'GridLAB-D Base GLM',
               'parameters': {'model_id': model_id}}
    gld_model = gridappsd_object.get_response(topic=topics.CONFIG,
                                              message=payload,
                                              timeout=30)
    log.info('GridLAB-D model received.')

    # HARD-CODE remove the json remnants from the message.
    gld_model['message'] = gld_model['message'][8:-43]
    log.warn('Bad json from GridLAB-D model removed via hard-code.')

    # Get a modGLM model to modify the base model.
    model_obj = modGLM.modGLM(strModel=gld_model['message'],
                              pathModelOut='test.glm', pathModelIn='pyvvo.glm'
                              )
    log.info('modGLM object instantiated.')

    # Set up the model to run.
    st = '2016-01-01 00:00:00'
    et = '2016-01-01 00:15:00'
    tz = 'UTC0'
    # db_dict = {**config['GLD-DB'], 'tz_offset': }
    swing_meter_name = model_obj.setupModel(
        starttime=st, stoptime=et, timezone=tz, database=config['GLD-DB'],
        powerflowFlag=True, vSource=model_data['swing_voltage'],
        triplexGroup=CONST.LOADS['triplex']['group'],
        triplexList=model_data['load_nominal_voltage']['triplex']['meters']
    )

    # Write the base model
    model_obj.writeModel()
    log.info('Base GridLAB-D model configured and written to file.')

    # Connect to the MySQL database for GridLAB-D simulations
    db_obj = db.db(**config['GLD-DB'],
                   pool_size=config['GLD-DB-OTHER']['NUM-CONNECTIONS'])
    log.info('Connected to MySQL database for GridLAB-D simulation output.')

    # Clear out the database while testing.
    # TODO: take this out?
    db_obj.dropAllTables()
    log.warning(
        'All tables dropped in {}'.format(config['GLD-DB']['database']))

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
    log.info('Voltage fractions converted to actual voltage.')

    # Initialize a clock object for datetimes.
    clockObj = helper.clock(startStr=st, finalStr=et,
                            interval=config['INTERVALS']['OPTIMIZATION'],
                            tzStr=tz)
    log.info('Clock object initialized')

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
        log=log)

    log.info('Population object initialized.')

    log.info('Starting genetic algorithm...')
    best_ind = pop_obj.ga()
    log.info('Shutting down genetic algorithm threads...')
    pop_obj.stopThreads()

    log.info('Best individual: {}'.format(best_ind))


if __name__ == '__main__':
    main()
