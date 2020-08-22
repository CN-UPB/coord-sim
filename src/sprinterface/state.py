from coordsim.network.flow import Flow


class SPRState:
    def __init__(self, flow: Flow, network: dict, placement: dict, sfcs: dict, service_functions: dict,
                 traffic: dict, network_stats: dict):
        """initializes all properties since this is a data class

        Parameters
        ----------
        flow : Flow Object of the request
        network : dict
            {
                'nodes': [{
                    'id': str,
                    'resource': [float],
                    'remaining_resource': [float]
                }],
                'edges': [{
                    'src': str,
                    'dst': str,
                    'delay': int (ms),
                    'data_rate': int (Mbit/s),
                    'used_data_rate': int (Mbit/s),
                }],
            }
        network: networkx object
        placement : dict
            {
                'node id' : [list of SF ids]
            }
        sfcs : dict
            {
                'sfc_id': list
                    ['ids (str)']
            },

        service_functions : dict
            {
                'sf_id (str)' : dict
                {
                    'processing_delay_mean': int (ms),
                    'processing_delay_stdev': int (ms)
                },
            }


        << traffic: aggregated data rates of flows arriving at node requesting >>
        traffic : dict
            {
                'node_id (str)' : dict
                {
                    'sfc_id (str)': dict
                    {
                        'sf_id (str)': data_rate (int) [Mbit/s]
                    },
                },
            },

        network_stats : dict
            {
                'total_flows' : int,
                'successful_flows' : int,
                'dropped_flows' : int,
                'in_network_flows' : int
                'avg_end_2_end_delay' : int (ms)
            }
        """
        self.flow = flow
        self.network = network
        self.placement = placement
        self.sfcs = sfcs
        self.service_functions = service_functions
        self.traffic = traffic
        self.network_stats = network_stats
