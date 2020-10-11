"""
Simulator file writer module
"""

import csv
import os
import yaml
from spinterface import SimulatorState
from sprinterface.action import SPRAction


class ResultWriter():
    """
    Result Writer
    Helper class to write results to CSV files.
    """
    def __init__(self, test_mode: bool, test_dir, write_schedule=False, recording_spacings=100):
        """
        If the simulator is in test mode, create result folder and CSV files
        """
        self.last_recording_time = 0
        self.recording_spacings = recording_spacings
        self.test_mode = test_mode
        if self.test_mode:
            self.placement_file_name = f"{test_dir}/placements.csv"
            self.resources_file_name = f"{test_dir}/node_metrics.csv"
            self.metrics_file_name = f"{test_dir}/metrics.csv"
            self.dropped_flows_file_name = f"{test_dir}/dropped_flows.yaml"
            self.rl_state_file_name = f"{test_dir}/rl_state.csv"
            self.run_flows_file_name = f"{test_dir}/run_flows.csv"
            self.flow_action_file_name = f"{test_dir}/flow_actions.csv"

            # Create the results directory if not exists
            os.makedirs(os.path.dirname(self.placement_file_name), exist_ok=True)

            self.placement_stream = open(self.placement_file_name, 'a+', newline='')
            self.resources_stream = open(self.resources_file_name, 'a+', newline='')
            self.metrics_stream = open(self.metrics_file_name, 'a+', newline='')
            self.rl_state_stream = open(self.rl_state_file_name, 'a+', newline='')
            self.run_flows_stream = open(self.run_flows_file_name, 'a+', newline='')
            self.flow_action_stream = open(self.flow_action_file_name, 'a+', newline='')

            # Create CSV writers
            self.placement_writer = csv.writer(self.placement_stream)
            self.resources_writer = csv.writer(self.resources_stream)
            self.metrics_writer = csv.writer(self.metrics_stream)
            self.rl_state_writer = csv.writer(self.rl_state_stream)
            self.run_flows_writer = csv.writer(self.run_flows_stream)
            self.flow_action_writer = csv.writer(self.flow_action_stream)

            # Write the headers to the files
            self.create_csv_headers()

    def __del__(self):
        # Close all writer streams
        if self.test_mode:
            self.placement_stream.close()
            self.resources_stream.close()
            self.metrics_stream.close()
            self.rl_state_stream.close()
            self.run_flows_stream.close()
            self.flow_action_stream.close()

    def create_csv_headers(self):
        """
        Creates statistics CSV headers and writes them to their files
        """

        # Create CSV headers
        placement_output_header = ['episode', 'time', 'node', 'sf']
        resources_output_header = ['episode', 'time', 'node', 'node_capacity', 'used_resources', 'ingress_traffic']
        metrics_output_header = ['episode', 'time', 'total_flows', 'successful_flows', 'dropped_flows',
                                 'in_network_flows', 'avg_end2end_delay']
        run_flows_output_header = ['episode', 'time', 'successful_flows', 'dropped_flows', 'total_flows']
        flow_action_output_header = ['episode', 'time', 'flow_id',
                                     'curr_node_id', 'dest_node', 'cur_node_rem_cap', 'next_node_rem_cap',
                                     'link_cap', 'link_rem_cap']

        # Write headers to CSV files
        self.placement_writer.writerow(placement_output_header)
        self.resources_writer.writerow(resources_output_header)
        self.metrics_writer.writerow(metrics_output_header)
        self.run_flows_writer.writerow(run_flows_output_header)
        self.flow_action_writer.writerow(flow_action_output_header)

    def write_action_result(self, episode, time, action: SPRAction, network):
        """
        Write simulator actions to CSV files for statistics purposes
        """
        # TODO: Add discrete action recording
        # if self.test_mode:
        #     placement = action.placement
        #     placement_output = []
        #     scheduling_output = []

        #     for node_id, sfs in placement.items():
        #         for sf in sfs:
        #             placement_output_row = [episode, time, node_id, sf]
        #             placement_output.append(placement_output_row)
        #     if self.write_schedule:
        #         scheduling = action.scheduling
        #         for node, sfcs in scheduling.items():
        #             for sfc, sfs in sfcs.items():
        #                 for sf, scheduling in sfs.items():
        #                     for schedule_node, schedule_prob in scheduling.items():
        #                         scheduling_output_row = [episode, time, node, sfc, sf, schedule_node, schedule_prob]
        #                         scheduling_output.append(scheduling_output_row)
        #         self.scheduling_writer.writerows(scheduling_output)

        #     self.placement_writer.writerows(placement_output)

        # TODO: Add discrete action recording
        cur_node_rem_cap = network.nodes[action.flow.current_node_id]['remaining_cap']
        if self.test_mode:
            if action.destination_node_id is None:
                dest_node = 'None'
                next_node_rem_cap = -1
                link_cap = -1
                rem_cap = -1
            else:
                dest_node = action.destination_node_id
                next_node_rem_cap = network.nodes[dest_node]['remaining_cap']
                if dest_node == action.flow.current_node_id:
                    link_cap = 'inf'
                    rem_cap = 'inf'
                else:
                    link_cap = network.edges[(action.flow.current_node_id, dest_node)]['cap']
                    rem_cap = network.edges[(action.flow.current_node_id, dest_node)]['remaining_cap']

            flow_action_output = [episode, time, action.flow.flow_id,
                                  action.flow.current_node_id, dest_node, cur_node_rem_cap, next_node_rem_cap,
                                  link_cap, rem_cap]
            self.flow_action_writer.writerow(flow_action_output)

    def write_state_results(self, episode, time, state: SimulatorState, metrics):
        """
        Write node resource consumption to CSV file
        """
        # TODO: UPDATE METRICS WRITING to include new ones for discrete action
        if self.test_mode and time >= self.last_recording_time + self.recording_spacings:
            self.last_recording_time = time
            network = state.network
            stats = state.network_stats

            metrics_output = [episode, time, stats['total_flows'], stats['successful_flows'], stats['dropped_flows'],
                              stats['in_network_flows'], stats['avg_end2end_delay']]

            resource_output = []
            for node in network.nodes(data=True):
                node_id = node[0]
                node_cap = node[1]['cap']
                used_resources = node_cap - node[1]['remaining_cap']
                ingress_traffic = 0
                # get all sfc
                sfcs = list(state.sfcs.keys())
                # iterate over sfcs to get traffic from all sfcs
                for sfc in sfcs:
                    ingress_sf = state.sfcs[sfc][0]
                    ingress_traffic += metrics['run_act_total_requested_traffic'].get(
                        node_id, {}).get(
                            sfc, {}).get(
                                ingress_sf, 0)
                resource_output_row = [episode, time, node_id, node_cap, used_resources, ingress_traffic]
                resource_output.append(resource_output_row)

            run_flows_output = [episode, time, metrics['run_processed_flows'], metrics['run_dropped_flows'],
                                metrics['run_generated_flows']]
            self.run_flows_writer.writerow(run_flows_output)
            self.metrics_writer.writerow(metrics_output)
            self.resources_writer.writerows(resource_output)
            # Write placement
            placement_output = []
            for node in network.nodes(data=True):
                node_id = node[0]
                for sf in node[1]['available_sf']:
                    placement_output_row = [episode, time, node_id, sf]
                    placement_output.append(placement_output_row)
            self.placement_writer.writerows(placement_output)

    def write_dropped_flow_locs(self, dropped_flow_locs):
        """Dump dropped flow counters into yaml file. Called at end of simulation"""
        if self.test_mode:
            with open(self.dropped_flows_file_name, 'w') as f:
                yaml.dump(dropped_flow_locs, f, default_flow_style=False)

    def write_rl_state(self, rl_state):
        if self.test_mode:
            self.rl_state_writer.writerow(rl_state)
