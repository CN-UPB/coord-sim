import logging
import random
import time
import os
from shutil import copyfile
from coordsim.metrics.metrics import Metrics
import coordsim.reader.reader as reader
from coordsim.simulation.flowsimulator import FlowSimulator
from coordsim.simulation.simulatorparams import SimulatorParams
import numpy
import simpy
from spinterface import SimulatorInterface
from sprinterface.action import SPRAction
from sprinterface.state import SPRState
from coordsim.writer.writer import ResultWriter
from coordsim.trace_processor.trace_processor import TraceProcessor
from coordsim.traffic_predictor.traffic_predictor import TrafficPredictor
from coordsim.traffic_predictor.lstm_predictor import LSTM_Predictor

logger = logging.getLogger(__name__)


class Simulator(SimulatorInterface):
    def __init__(self, network_file, service_functions_file, config_file, resource_functions_path="", test_mode=False,
                 test_dir=None):
        super().__init__(test_mode)
        # Number of time the simulator has run. Necessary to correctly calculate env run time of apply function
        self.run_times = int(1)
        self.network_file = network_file
        self.test_dir = test_dir
        # init network, sfc, sf, and config files
        self.network, self.ing_nodes, self.eg_nodes = reader.read_network(self.network_file)
        self.sfc_list = reader.get_sfc(service_functions_file)
        self.sf_list = reader.get_sf(service_functions_file, resource_functions_path)
        self.config = reader.get_config(config_file)
        self.metrics = Metrics(self.network, self.sf_list)
        # Assume result path is the path where network file is in.
        self.result_base_path = os.path.dirname(self.network_file)
        if 'trace_path' in self.config:
            # Quick solution to copy trace file to same path for network file as provided by calling algo.
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            copyfile(trace_path, os.path.join(self.result_base_path, os.path.basename(trace_path)))

        self.prediction = False
        # Check if future ingress traffic setting is enabled
        if 'future_traffic' in self.config and self.config['future_traffic']:
            self.prediction = True
        self.params = SimulatorParams(self.network, self.ing_nodes, self.eg_nodes, self.sfc_list, self.sf_list,
                                      self.config, self.metrics, prediction=self.prediction)
        write_schedule = False
        if 'write_schedule' in self.config and self.config['write_schedule']:
            write_schedule = True
        # Create CSV writer
        self.writer = ResultWriter(
            self.test_mode, self.test_dir, write_schedule, recording_spacings=self.params.run_duration)

        self.episode = 0
        # Load trace file
        if 'trace_path' in self.config:
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            self.trace = reader.get_trace(trace_path)

        self.lstm_predictor = None
        if 'lstm_prediction' in self.config and self.config['lstm_prediction']:
            self.lstm_predictor = LSTM_Predictor(self.trace, params=self.params,
                                                 weights_dir=self.config['lstm_weights'])

    def __del__(self):
        # write dropped flow locs to yaml
        self.writer.write_dropped_flow_locs(self.metrics.metrics['dropped_flows_locs'])

    def init(self, seed):
        # Reset predictor class at beginning of every init
        if self.prediction:
            self.predictor = TrafficPredictor(self.params, self.lstm_predictor)
        # increment episode count
        self.episode += 1
        # reset network caps and available SFs:
        reader.reset_cap(self.network)
        # Initialize metrics, record start time
        self.run_times = int(1)
        self.start_time = time.time()

        # Generate SimPy simulation environment
        self.env = simpy.Environment()

        self.params.metrics.reset_metrics()

        # Instantiate the parameter object for the simulator.
        if self.params.use_states and 'trace_path' in self.config:
            logger.warning('Two state model and traces are both activated, thi will cause unexpected behaviour!')

        if self.params.use_states:
            if self.params.in_init_state:
                self.params.in_init_state = False
            else:
                self.params.update_state()

        self.duration = self.params.run_duration
        # Get and plant random seed
        self.seed = seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        # TODO: Do not need flow arrival list
        # self.params.reset_flow_lists()
        # generate flow lists 1x here since we are in `init()`
        # self.params.generate_flow_lists()

        # Instantiate a simulator object, pass the environment and params
        self.simulator = FlowSimulator(self.env, self.params)

        # Trace handling
        if 'trace_path' in self.config:
            TraceProcessor(self.params, self.env, self.trace, self.simulator)

        # Start the simulator
        self.simulator.start()

        # Run the environment for one step to get initial stats.
        # self.env.step()
        # self.flow_trigger_list = list(self.simulator.flow_triggers.values())
        # event_info = self.env.run(until=simpy.events.AnyOf(self.env, self.flow_trigger_list))
        # # get the latest trigger list
        # self.flow_trigger_list = list(self.simulator.flow_triggers.values())
        # flow, sfc = event_info.events[0].value
        # self.simulator.flow_triggers[flow.current_node_id] = self.env.event()
        flow, sfc = self.env.run(until=self.simulator.flow_trigger)
        # Parse the NetworkX object into a dict format specified in SimulatorState. This is done to account
        # for changing node remaining capacities.
        # Also, parse the network stats and prepare it in SimulatorState format.
        self.parse_network()
        self.network_metrics()

        # Record end time and running time metrics
        self.end_time = time.time()
        self.params.metrics.running_time(self.start_time, self.end_time)
        # Check to see if traffic prediction is enabled to provide future traffic not current traffic
        if self.prediction:
            self.predictor.predict_traffic(self.env.now)
            stats = self.params.metrics.get_metrics()
            self.traffic = stats['run_total_requested_traffic']
        simulator_state = SPRState(
            flow, self.simulator.params.network,
            self.simulator.params.sf_placement,
            self.sfc_list,
            self.sf_list,
            self.traffic,
            self.network_stats
        )
        logger.debug(f"t={self.env.now}: {simulator_state}")
        # Check to see if init called in warmup, if so, set warmup to false
        # This is to allow for better prediction and better overall control
        # in the future
        self.last_apply_time = time.time()
        return simulator_state

    def apply(self, actions: SPRAction):
        alg_runtime = time.time() - self.last_apply_time
        self.writer.write_runtime(alg_runtime)
        # reset metrics for steps
        self.params.metrics.reset_run_metrics()

        # write action results
        self.writer.write_action_result(self.episode, self.env.now, actions, self.simulator.params.network)
        flow = actions.flow
        currrent_node = flow.current_node_id
        current_sf = flow.current_sf
        # Apply placement if decision is 0: process at this node and no instance is there
        if actions.destination_node_id == flow.current_node_id:
            # check if instance is already here
            available_sf = self.simulator.params.network.nodes[flow.current_node_id]['available_sf']
            if flow.current_sf not in list(available_sf.keys()):
                # If no instance exists: place instance in the node
                self.simulator.params.network.nodes[currrent_node]['available_sf'][current_sf] = {
                    'load': 0.0,
                    'last_active': self.simulator.env.now,
                    'startup_time': self.simulator.env.now
                }

        # Check active VNFs in the network
        self.update_vnf_active_status()

        # Create a placement
        sf_placement = {}
        for node in self.simulator.params.network.nodes(data=True):
            node_id = node[0]
            node_available_sf = list(node[1]['available_sf'].keys())
            sf_placement[node_id] = node_available_sf
        self.simulator.params.sf_placement = sf_placement
        self.env.process(
            self.simulator.pass_flow(
                flow, self.sfc_list[flow.sfc],
                request_decision=False,
                next_node=actions.destination_node_id
            )
        )

        # Run the simulation again until a new flow decision request
        flow, sfc = self.env.run(until=self.simulator.flow_trigger)

        # Parse the NetworkX object into a dict format specified in SimulatorState. This is done to account
        # for changing node remaining capacities.
        # Also, parse the network stats and prepare it in SimulatorState format.
        self.parse_network()
        self.network_metrics()

        # Increment the run times variable
        self.run_times += 1

        # Record end time of the apply round, doesn't change start time to show the running time of the entire
        # simulation at the end of the simulation.
        self.end_time = time.time()
        self.params.metrics.running_time(self.start_time, self.end_time)

        if self.params.use_states:
            self.params.update_state()

        # generate flow data for next run (used for prediction)
        # self.params.generate_flow_lists(now=self.env.now)

        # Check to see if traffic prediction is enabled to provide future traffic not current traffic
        if self.prediction:
            self.predictor.predict_traffic(self.env.now)
            stats = self.params.metrics.get_metrics()
            self.traffic = stats['run_total_requested_traffic']
        # Create a new SimulatorState object to pass to the RL Agent
        simulator_state = SPRState(
            flow,
            self.simulator.params.network,
            self.simulator.params.sf_placement,
            self.sfc_list,
            self.sf_list,
            self.traffic,
            self.network_stats
        )
        self.writer.write_state_results(self.episode, self.env.now, simulator_state, self.params.metrics.get_metrics())
        logger.debug(f"t={self.env.now}: {simulator_state}")

        self.last_apply_time = time.time()
        return simulator_state

    def parse_network(self) -> dict:
        """
        Converts the NetworkX network in the simulator to a dict in a format specified in the SimulatorState class.
        """
        self.network_dict = {'nodes': [], 'edges': []}
        for node in self.params.network.nodes(data=True):
            node_cap = node[1]['cap']
            # 'used_resources' here is the max usage for the run.
            node_remaining_cap = node[1]['remaining_cap']
            self.network_dict['nodes'].append({'id': node[0], 'resource': node_cap,
                                               'remaining_resource': node_remaining_cap})
        for edge in self.network.edges(data=True):
            edge_src = edge[0]
            edge_dest = edge[1]
            edge_delay = edge[2]['delay']
            edge_dr = edge[2]['cap']
            # We use a fixed user data rate for the edges here as the functionality is not yet incorporated in the
            # simulator.
            # TODO: Implement used edge data rates in the simulator.
            edge_used_dr = 0
            self.network_dict['edges'].append({
                'src': edge_src,
                'dst': edge_dest,
                'delay': edge_delay,
                'data_rate': edge_dr,
                'used_data_rate': edge_used_dr
            })

    def network_metrics(self):
        """
        Processes the metrics and parses them in a format specified in the SimulatorState class.
        """
        stats = self.params.metrics.get_metrics()
        self.traffic = stats['run_total_requested_traffic']
        self.network_stats = {
            'processed_traffic': stats['run_total_processed_traffic'],
            'total_flows': stats['generated_flows'],
            'successful_flows': stats['processed_flows'],
            'dropped_flows': stats['dropped_flows'],
            'in_network_flows': stats['total_active_flows'],
            'avg_end2end_delay': stats['avg_end2end_delay'],
            'run_avg_end2end_delay': stats['run_avg_end2end_delay'],
            'run_max_end2end_delay': stats['run_max_end2end_delay'],
            'run_avg_path_delay': stats['run_avg_path_delay'],
            'run_total_processed_traffic': stats['run_total_processed_traffic'],
            'run_dropped_flows_per_node': stats['run_dropped_flows_per_node'],
            'run_generated_flows': stats['run_generated_flows'],
            'run_dropped_flows': stats['run_dropped_flows'],
            'run_successful_flows': stats['run_processed_flows']
        }

    def get_active_ingress_nodes(self):
        """Return names of all ingress nodes that are currently active, ie, produce flows."""
        return [ing[0] for ing in self.ing_nodes if self.params.inter_arr_mean[ing[0]] is not None]

    def update_vnf_active_status(self):
        for node in self.network.nodes(data=True):
            n_id = node[0]
            now = self.simulator.env.now
            timeout = self.params.vnf_timeout
            # Using dict here to create a copy. Solves RuntimeError: dict size changed during iteration
            available_sf: dict = dict(self.simulator.params.network.nodes[n_id]['available_sf'])
            for sf, sf_params in available_sf.items():
                # Remove VNFs if not active and timeout passed
                if sf_params['load'] == 0.0:
                    # VNF is not active
                    if sf_params['last_active'] < now - timeout:
                        # VNF has not been active for `timeout` time: remove
                        del self.simulator.params.network.nodes[n_id]['available_sf'][sf]

                else:
                    # Node is active: update `last_active` time to be `now`
                    self.simulator.params.network.nodes[n_id]['available_sf'][sf]['last_active'] = now


# for debugging
if __name__ == "__main__":
    # run from project root for file paths to work
    # I changed triangle to have 2 ingress nodes for debugging
    network_file = 'params/networks/triangle.graphml'
    service_file = 'params/services/abc.yaml'
    config_file = 'params/config/sim_config.yaml'

    sim = Simulator(network_file, service_file, config_file)
    state = sim.init(seed=1234)
    dummy_action = SPRAction(placement={}, scheduling={})
    # FIXME: this currently breaks - negative flow counter?
    #  should be possible to have an empty action and just drop all flows!
    state = sim.apply(dummy_action)
