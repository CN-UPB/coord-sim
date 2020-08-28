import logging
import random
import numpy as np
from coordsim.network.flow import Flow
# from coordsim.metrics import metrics

log = logging.getLogger(__name__)

"""
Flow Simulator class
This class holds the flow simulator and its internal flow handling functions.
Flow of data through the simulator (abstract):

start() -> generate_flow() -> init_flow() -> pass_flow() -> process_flow()
and forward_flow() -> depart_flow() or pass_flow()

"""


class FlowSimulator:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.total_flow_count = 0
        # Blank event trigger for the simulator
        self.flow_trigger = self.env.event()

    def start(self):
        """
        Start the simulator.
        """
        log.info("Starting simulation")
        # Setting the all-pairs shortest path in the NetworkX network as a graph attribute
        log.info("Using nodes list {}\n".format(list(self.params.network.nodes.keys())))
        log.info("Total of {} ingress nodes available\n".format(len(self.params.ing_nodes)))
        if self.params.eg_nodes:
            log.info("Total of {} egress nodes available\n".format(len(self.params.eg_nodes)))
        for node in self.params.ing_nodes:
            node_id = node[0]
            self.env.process(self.generate_flow(node_id))

    def generate_flow(self, node_id):
        """
        Generate flows at the ingress nodes.
        """
        while self.params.inter_arr_mean[node_id] is not None:
            self.total_flow_count += 1

            if self.params.flow_list_idx is None:
                if self.params.deterministic_arrival:
                    inter_arr_time = self.params.inter_arr_mean[node_id]
                else:
                    # Poisson arrival -> exponential distributed inter-arrival time
                    inter_arr_time = random.expovariate(lambd=1.0/self.params.inter_arr_mean[node_id])
                # set normally distributed flow data rate
                flow_dr = np.random.normal(self.params.flow_dr_mean, self.params.flow_dr_stdev)
                if self.params.deterministic_size:
                    flow_size = self.params.flow_size_shape
                else:
                    # heavy-tail flow size
                    flow_size = np.random.pareto(self.params.flow_size_shape) + 1
                # Skip flows with negative flow_dr or flow_size values
                if flow_dr <= 0.00 or flow_size <= 0.00:
                    continue
            # use generated list of flow arrivals
            else:
                inter_arr_time, flow_dr, flow_size = self.params.get_next_flow_data(node_id)

            # Assign a random SFC to the flow
            flow_sfc = np.random.choice([sfc for sfc in self.params.sfc_list.keys()])
            # Get the flow's creation time (current environment time)
            creation_time = self.env.now
            # Set the egress node for the flow if some are specified in the network file
            flow_egress_node = None
            if self.params.eg_nodes:
                flow_egress_node = random.choice(self.params.eg_nodes)
            # Generate flow based on given params
            flow = Flow(str(self.total_flow_count), flow_sfc, flow_dr, flow_size, creation_time,
                        current_node_id=node_id, egress_node_id=flow_egress_node)
            # Update metrics for the generated flow
            self.params.metrics.generated_flow(flow, node_id)
            # Generate flows and schedule them at ingress node
            self.env.process(self.init_flow(flow))
            yield self.env.timeout(inter_arr_time)

    def init_flow(self, flow):
        """
        Initialize flows within the network. This function takes the generated flow object at the ingress node
        and handles it according to the requested SFC. We check if the SFC that is being requested is indeed
        within the schedule, otherwise we log a warning and drop the flow.
        The algorithm will check the flow's requested SFC, and will forward the flow through the network using the
        SFC's list of SFs based on the LB rules that are provided through the scheduler's 'flow_schedule()'
        function.
        """
        log.info(
            "Flow {} generated. arrived at node {} Requesting {} - flow duration: {}ms, "
            "flow dr: {}. Time: {}".format(flow.flow_id, flow.current_node_id, flow.sfc, flow.duration, flow.dr,
                                           self.env.now))
        sfc = self.params.sfc_list[flow.sfc]
        # Check to see if requested SFC exists
        if sfc is not None:
            # Iterate over the SFs and process the flow at each SF.
            yield self.env.process(self.pass_flow(flow, sfc))
        else:
            log.info(f"Requested SFC was not found. Dropping flow {flow.flow_id}")
            # del self.flow_triggers[flow.flow_id]
            # Update metrics for the dropped flow
            self.params.metrics.dropped_flow(flow)
            return

    def pass_flow(self, flow, sfc, request_decision=True, next_node=None):
        """
        Passes the flow to the next node to begin processing.
        The flow might still be arriving at a previous node or SF.
        This function is used in a mutual recursion alongside process_flow() function to allow flows to arrive and begin
        processing without waiting for the flow to completely arrive.
        The mutual recursion is as follows:
        pass_flow() -> process_flow() -> pass_flow() and so on...
        Breaking condition: Flow reaches last position within the SFC, then process_flow() calls depart_flow()
        instead of pass_flow(). The position of the flow within the SFC is determined using current_position
        attribute of the flow object.
        """

        # TODO: Check if another flow is requesting decisions, then wait some backoff time.  

        # Check if TTL is above zero to make sure flow is still relevant
        if flow.ttl <= 0:
            log.info(f"Flow {flow.flow_id} passed TTL! Dropping flow")
            # del self.flow_triggers[flow.flow_id]
            # Update metrics for the dropped flow
            self.params.metrics.dropped_flow(flow)
            return

        # set current sf of flow
        # Only update flow's SF when not going to egress
        if not flow.forward_to_eg:
            # We only care to set SF if flow needs one. If forward_to_eg, flow finished processing
            sf = sfc[flow.current_position]
            flow.current_sf = sf

        if flow.forward_to_eg and flow.current_node_id == flow.egress_node_id:
            # We are in an egress node, no need for further decisions
            # Wait flow duration
            yield self.env.timeout(flow.duration)
            # # Flow finished arriving: update link capacities
            # self.update_link_cap(flow, flow.last_node_id, flow.current_node_id)
            # Depart network
            self.depart_flow(flow, remove_active_flow=False)
            return

        if flow.forward_to_eg and flow.current_node_id == next_node:
            # If flow finished processing and decision is to keep at the same node: +1 delay
            yield self.env.timeout(1)
            flow.ttl -= 1
            flow.end2end_delay += 1

        # If request decision is True, trigger the event
        if request_decision:
            # Trigger an event to stop the simulator run
            self.flow_trigger.succeed(value=(flow, sfc))
            self.flow_trigger = self.env.event()
            return

        if next_node is None:
            log.info(f"No node to forward flow {flow.flow_id} to. Dropping it")
            # Update metrics for the dropped flow
            # del self.flow_triggers[flow.flow_id]
            self.params.metrics.dropped_flow(flow)
            return

        # Check if decision is to process at this node
        if next_node == flow.current_node_id:

            log.info("Flow {} STARTED ARRIVING at node {}. Time: {}"
                     .format(flow.flow_id, flow.current_node_id, self.env.now))

            # Only process at a node if not moving to egress
            if not flow.forward_to_eg:
                self.params.metrics.add_requesting_flow(flow)
                yield self.env.process(self.process_flow(flow, sfc))

            elif flow.forward_to_eg:
                # we are not at egress, keep requesting decisions
                # Pass the head of the flow
                yield self.env.process(self.pass_flow(flow, sfc))

        else:
            # Decision is to forward flow to another node

            # Forward flow to that node
            flow_forwarded = yield self.env.process(self.forward_flow(flow, next_node))

            # TODO: CHECK IF FLOW OCCUPIED PREVIOUS LINK BUT DID NOT PROCESS AT A NODE
            # IF SO, LINK MUST BE CLEANED
            # IDEA: REMOVE LINK CAP UPDATE FROM PROCESS_FLOW AND PUT IT IN A SIMPY PROCESS
            if flow_forwarded:
                # Ask for decision again after arriving at the next node
                # if done processing and forwarding to egress, set that flag
                yield self.env.process(self.pass_flow(flow, sfc))

    def forward_flow(self, flow, next_node):
        """
        Calculates the path delays occurring when forwarding a node
        Path delays are calculated using the Shortest path
        The delay is simulated by timing out for the delay amount of duration

        Returns:
        False if flow was not forwarded
        True if flow was successfully forwarded
        """
        if next_node is None:
            log.info(f"No node to forward flow {flow.flow_id} to. Dropping it")
            # Update metrics for the dropped flow
            self.params.metrics.dropped_flow(flow)
            return False
        flow.last_node_id = flow.current_node_id
        path_delay = 0
        if flow.current_node_id != next_node:
            path_delay = self.params.network.graph['shortest_paths'][(flow.current_node_id, next_node)][1]
            # Get edges resources
            edge_rem_cap = self.params.network.edges[(flow.current_node_id, next_node)]['remaining_cap']
            # calculate new remaining cap
            new_rem_cap = edge_rem_cap - flow.dr
            if new_rem_cap >= 0:
                # There is enoough capacity on the edge: send the flow
                log.info(f"Flow {flow.flow_id} started travelling on edge ({flow.current_node_id}, {next_node})")
                self.params.network.edges[(flow.current_node_id, next_node)]['remaining_cap'] = new_rem_cap
            else:
                # Not enough capacity on the edge: drop the flow
                log.info(f"No cap on edge ({flow.current_node_id}, {next_node}) to handle {flow.flow_id}. Dropping it")
                # Update metrics for the dropped flow
                self.params.metrics.dropped_flow(flow)
                return False

        # Metrics calculation for path delay. Flow's end2end delay is also incremented.
        self.params.metrics.add_path_delay(path_delay)
        flow.end2end_delay += path_delay
        # deduct path_delay from TTL
        flow.ttl -= path_delay

        if flow.current_node_id == next_node:
            assert path_delay == 0, "While Forwarding the flow, the Current and Next node same, yet path_delay != 0"
            log.info("Flow {} will stay in node {}. Time: {}.".format(flow.flow_id, flow.current_node_id, self.env.now))
        else:
            log.info("Flow {} will leave node {} towards node {}. Time {}"
                     .format(flow.flow_id, flow.current_node_id, next_node, self.env.now))
            # Wait for the path delay to reach the next node
            yield self.env.timeout(path_delay)
            # Update the current node of the flow
            flow.current_node_id = next_node
            # Store a copy of current and past node ids
            current_node_id = flow.current_node_id
            last_node_id = flow.last_node_id
            # create a non-blocking simpy process to cleanup after flow finishes arriving to this node
            self.env.process(self.cleanup_link_after_arrival(flow, last_node_id, current_node_id))
            return True

    def cleanup_link_after_arrival(self, flow, last_node_id, current_node_id):
        """
        Simpy process: wait flow.duration then cleanup link
        Used only when flow is forwarding to egress node and no flow processing done
        """
        # Wait flow duration
        yield self.env.timeout(flow.duration)
        # Update link cap
        self.update_link_cap(flow, last_node_id, current_node_id)

    def process_flow(self, flow, sfc):
        """
        Process the flow at the requested SF of the current node.
        """
        # Generate a processing delay for the SF
        current_node_id = flow.current_node_id
        # last_node_id = flow.last_node_id
        sf = sfc[flow.current_position]
        flow.current_sf = sf

        log.info("Flow {} STARTED PROCESSING at node {} for processing. Time: {}"
                 .format(flow.flow_id, flow.current_node_id, self.env.now))

        if sf in self.params.sf_placement[current_node_id]:
            current_sf = flow.current_sf
            vnf_delay_mean = self.params.sf_list[current_sf]["processing_delay_mean"]
            vnf_delay_stdev = self.params.sf_list[current_sf]["processing_delay_stdev"]
            processing_delay = np.absolute(np.random.normal(vnf_delay_mean, vnf_delay_stdev))
            # Update metrics for the processing delay
            # Add the delay to the flow's end2end delay
            self.params.metrics.add_processing_delay(processing_delay)
            flow.end2end_delay += processing_delay
            # Deduct processing delay from flow TTL
            flow.ttl -= processing_delay

            # Calculate the demanded capacity when the flow is processed at this node
            demanded_total_capacity = 0.0
            for sf_i, sf_data in self.params.network.nodes[current_node_id]['available_sf'].items():
                if sf == sf_i:
                    # Include flows data rate in requested sf capacity calculation
                    demanded_total_capacity += self.params.sf_list[sf]['resource_function'](sf_data['load'] + flow.dr)
                else:
                    demanded_total_capacity += self.params.sf_list[sf_i]['resource_function'](sf_data['load'])

            # Get node capacities
            node_cap = self.params.network.nodes[current_node_id]["cap"]
            node_remaining_cap = self.params.network.nodes[current_node_id]["remaining_cap"]
            assert node_remaining_cap >= 0, "Remaining node capacity cannot be less than 0 (zero)!"
            if demanded_total_capacity <= node_cap:
                log.info("Flow {} started processing at sf {} at node {}. Time: {}, Processing delay: {}"
                         .format(flow.flow_id, current_sf, current_node_id, self.env.now, processing_delay))

                # Metrics: Add active flow to the SF once the flow has begun processing.
                self.params.metrics.add_active_flow(flow, current_node_id, current_sf)

                # Add load to sf
                self.params.network.nodes[current_node_id]['available_sf'][sf]['load'] += flow.dr
                # Set remaining node capacity
                self.params.network.nodes[current_node_id]['remaining_cap'] = node_cap - demanded_total_capacity
                # Set max node usage
                self.params.metrics.calc_max_node_usage(current_node_id, demanded_total_capacity)
                # Just for the sake of keeping lines small, the node_remaining_cap is updated again.
                node_remaining_cap = self.params.network.nodes[current_node_id]["remaining_cap"]

                # Wait for the VNF to finish processing the head of the flow
                yield self.env.timeout(processing_delay)
                log.info("Flow {} started departing sf {} at node {}. Time {}"
                         .format(flow.flow_id, current_sf, current_node_id, self.env.now))

                # Check if flow is currently in last SF, if so, then:
                # - Check if the flow has some Egress node set or not. If not then just depart. If Yes then:
                #   - check if the current node is the egress node. If Yes then depart. If No then forward the flow to
                #     the egress node using the shortest_path

                if flow.current_position == len(sfc) - 1:
                    # Still increase position by 1 to indicate fully processed in passed state
                    flow.current_position += 1
                    if flow.current_node_id == flow.egress_node_id:
                        # Flow is processed and resides at egress node: depart flow
                        yield self.env.timeout(flow.duration)
                        self.depart_flow(flow)
                    elif flow.egress_node_id is None:
                        # Flow is processed and no egress node specified: depart flow
                        log.info(f'Flow {flow.flow_id} has no egress node, will depart from'
                                 f' current node {flow.current_node_id}. Time {self.env.now}.')
                        yield self.env.timeout(flow.duration)
                        self.depart_flow(flow)
                    else:
                        # Remove the active flow from the SF after it departed the SF on current node towards egress
                        self.params.metrics.remove_active_flow(flow, current_node_id, current_sf)
                        # Forward flow to the egress node and then depart from there
                        # yield self.env.process(self.forward_flow(flow, flow.egress_node_id))
                        # Set flow flag to forward to egress
                        flow.forward_to_eg = True
                        # request decision to route to egress
                        yield self.env.process(self.pass_flow(flow, flow.sfc))
                        # yield self.env.timeout(flow.duration)
                        # In this situation the last sf was never active for the egress node,
                        # so we should not remove it from the metrics
                        # self.depart_flow(flow, remove_active_flow=False)
                else:
                    # Increment the position of the flow within SFC
                    flow.current_position += 1
                    self.env.process(self.pass_flow(flow, sfc))
                    yield self.env.timeout(flow.duration)

                    # before departing the SF.
                    # print(metrics.get_metrics()['current_active_flows'])
                    log.info("Flow {} FINISHED ARRIVING at SF {} at node {} for processing. Time: {}"
                             .format(flow.flow_id, current_sf, current_node_id, self.env.now))
                    # Remove the active flow from the SF after it departed the SF
                    self.params.metrics.remove_active_flow(flow, current_node_id, current_sf)

                # update link capacities
                # self.update_link_cap(flow, last_node_id, current_node_id)

                # Remove load from sf
                self.params.network.nodes[current_node_id]['available_sf'][sf]['load'] -= flow.dr
                assert self.params.network.nodes[current_node_id]['available_sf'][sf]['load'] >= 0, \
                    'SF load cannot be less than 0!'
                # Check if SF is not processing any more flows AND if SF is removed from placement. If so the SF will
                # be removed from the load recording. This allows SFs to be handed gracefully.
                if (self.params.network.nodes[current_node_id]['available_sf'][sf]['load'] == 0) and (
                        sf not in self.params.sf_placement[current_node_id]):
                    del self.params.network.nodes[current_node_id]['available_sf'][sf]

                # Recalculation is necessary because other flows could have already arrived or departed at the node
                used_total_capacity = 0.0
                for sf_i, sf_data in self.params.network.nodes[current_node_id]['available_sf'].items():
                    used_total_capacity += self.params.sf_list[sf_i]['resource_function'](sf_data['load'])
                # Set remaining node capacity
                self.params.network.nodes[current_node_id]['remaining_cap'] = node_cap - used_total_capacity
                # Just for the sake of keeping lines small, the node_remaining_cap is updated again.
                node_remaining_cap = self.params.network.nodes[current_node_id]["remaining_cap"]

                # We assert that remaining capacity must at all times be less than the node capacity so that
                # nodes dont put back more capacity than the node's capacity.
                assert node_remaining_cap <= node_cap, "Node remaining capacity cannot be more than node capacity!"
            else:
                log.info(f"Not enough capacity for flow {flow.flow_id} at node {flow.current_node_id}. Dropping flow.")
                # update link capacities
                # self.update_link_cap(flow, last_node_id, current_node_id)
                # Update metrics for the dropped flow
                self.params.metrics.dropped_flow(flow)
                return
        else:
            log.info(f"SF {sf} was not found at {current_node_id}. Dropping flow {flow.flow_id}")
            # update link capacities
            # self.update_link_cap(flow, last_node_id, current_node_id)
            self.params.metrics.dropped_flow(flow)
            return

    def depart_flow(self, flow, remove_active_flow=True):
        """
        Process the flow at the requested SF of the current node.
        """

        # Update metrics for the processed flow
        self.params.metrics.completed_flow()
        self.params.metrics.add_end2end_delay(flow.end2end_delay)
        if remove_active_flow:
            self.params.metrics.remove_active_flow(flow, flow.current_node_id, flow.current_sf)
        log.info("Flow {} was processed and departed the network from {}. Time {}"
                 .format(flow.flow_id, flow.current_node_id, self.env.now))

    def update_link_cap(self, flow, last_node_id, current_node_id):
        # return the used capacity to the edge
        # Add the used cap back to the edge
        self.params.network.edges[(last_node_id, current_node_id)]['remaining_cap'] += flow.dr
        remaining_edge_cap = self.params.network.edges[(last_node_id, current_node_id)]['remaining_cap']
        edge_cap = self.params.network.edges[(last_node_id, current_node_id)]['cap']
        assert remaining_edge_cap <= edge_cap, "Edge rem. cap can't be > actual cap"
