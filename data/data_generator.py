"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Only run as main
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported!")

# Parse imports
from typing import Tuple, Generator, Dict, Any, List
import numpy as np
import tensorflow as tf
from itertools import permutations
from re import sub
import argparse
import random
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#添加命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir",default='E:\GNNET\GNNetworkingChallenge-2023_RealNetworkDT\GNNetworkingChallenge-2023_RealNetworkDT\data' ,type=str, required=True)
parser.add_argument("--output-dir", default='E:/GNNET/GNNetworkingChallenge-2023_RealNetworkDT/GNNetworkingChallenge-2023_RealNetworkDT/data/result',type=str, required=True)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-folds", type=int, default=5)
parser.add_argument(
    "--test",
    action="store_true",
    help="If true, assume that the dataset is a test dataset. If so, n-folds will be "
    + "equal to 1, there will no shuffling, and the dataset will not check for "
    + "unvalid delays.",
)
args = parser.parse_args()

# import datasets's DataNet API
sys.path.insert(0, args.input_dir)
from datanetAPI import DatanetAPI, TimeDist, Sample


def _get_network_decomposition(sample: Sample) -> Tuple[dict, list]:
    """Given a sample from the DataNet API, it returns it as a sample for the model.

    Parameters
    ----------
    sample : DatanetAPI.Sample
        Sample from the DataNet API

    Returns
    -------
    Tuple[dict, list]
        Tuple with the inputs of the model and the target variable to predict

    Raises
    ------
    ValueError
        Raised if one of the links is of an unknown type
    """

    # Read values from the DataNet API
    network_topology = sample.get_physical_topology_object()
    max_link_load = sample.get_max_link_load()
    global_delay = sample.get_global_delay()
    global_losses = sample.get_global_losses()
    traffic_matrix = sample.get_traffic_matrix()
    physical_path_matrix = sample.get_physical_path_matrix()
    performance_matrix = sample.get_performance_matrix()
    packet_info_matrix = sample.get_pkts_info_object()
    # Process sample id (for debugging purposes and for then evaluating the model)
    sample_file_path, sample_file_id = sample.get_sample_id()
    sample_file_name = sample_file_path.split("/")[-1]
    # Obtain links and nodes
    # We discard all links that start from the traffic generator
    links = dict()
    for edge in network_topology.edges:  # src, dst, port
        # We identify all traffic generators as the same port
        edge_id = sub(r"t(\d+)", "tg", network_topology.edges[edge]["port"])
        if edge_id.startswith("r") or edge_id.startswith("s"):
            links[edge_id] = {
                "capacity": float(network_topology.edges[edge].get("bandwidth", 1e9))
                / 1e9,  # original value is in bps, we change it to Gbps
            }
        elif edge_id.startswith("tg"):
            continue
        else:
            raise ValueError(f"Unknown edge type: {edge_id}")

    # In this scenario assume that flows can either follow CBR or MB distributions
    flows = dict()
    used_links = set()  # Used later so we only consider used links
    # Add flows
    for src, dst in filter(
        lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
        and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
        permutations(range(len(traffic_matrix)), 2),
    ):
        for local_flow_id in range(len(traffic_matrix[src, dst]["Flows"])):
            flow_packet_info = packet_info_matrix[src, dst][0][local_flow_id]
            flow = traffic_matrix[src, dst]["Flows"][local_flow_id]
            # Size distribution is always determinstic
            # Obtain and clean the path followed the flow
            # We must also clean up the name of the traffic generator
            clean_og_path = [
                sub(r"t(\d+)", "tg", link)
                for link in physical_path_matrix[src, dst][2::2]
            ]

            packet_timestamps = np.array([float(x[0]) for x in flow_packet_info])
            ipg = packet_timestamps[1:] - packet_timestamps[-1:]

            flow_id = f"{src}_{dst}_{local_flow_id}"
            flows[flow_id] = {
                "source": src,
                "destination": dst,
                "flow_id": flow_id,
                "length": len(clean_og_path),
                "og_path": clean_og_path,
                "traffic": flow["AvgBw"],  # in bps
                "packets": flow["PktsGen"],
                "flow_variance": flow["VarPktSize"],
                "flow_tos": flow["ToS"],
                "flow_p10PktSize": flow["p10PktSize"],
                "flow_p20PktSize": flow["p20PktSize"],
                "flow_p50PktSize": flow["p50PktSize"],
                "flow_p80PktSize": flow["p80PktSize"],
                "flow_p90PktSize": flow["p90PktSize"],
                "packet_size": flow["SizeDistParams"]["AvgPktSize"],
                "rate": flow["TimeDistParams"]["Rate"] if flow["TimeDist"] == TimeDist.CBR_T else 0,
                "ibg": flow["TimeDistParams"]["IBG"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_bitrate_per_burst": flow["TimeDistParams"]["On_Rate"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_pkts_per_burst": flow["TimeDistParams"]["Pkts_per_burst"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_type": (
                    float(flow["TimeDist"] == TimeDist.CBR_T),
                    float(flow["TimeDist"] == TimeDist.MULTIBURST_T),
                ),
                "delay": performance_matrix[src, dst]["Flows"][local_flow_id]["AvgDelay"] * 1000,  # in ms
                "ipg_mean" : np.mean(ipg) if len(ipg) > 0 else 0,
                "ipg_var": np.var(ipg) if len(ipg) > 0 else 0,  
            }

            # Add edges to the used_links set
            used_links.update(set(clean_og_path))

    # Purge unused links
    links = {kk: vv for kk, vv in links.items() if kk in used_links}

    # Normalize flow naming
    # We give the indices in such a way that flows states are concatanated as [CBR, MB]
    ordered_flows = list()
    flow_mapping = dict()
    for idx, (flow_id, flow_params) in enumerate(flows.items()):
        flow_mapping[flow_id] = idx
        ordered_flows.append(flow_params)
    n_f = len(ordered_flows)

    # Normalize link naming
    ordered_links = list()
    link_mapping = dict()
    for idx, (link_id, link_params) in enumerate(links.items()):
        link_mapping[link_id] = idx
        ordered_links.append(link_params)
    n_l = len(ordered_links)

    # Obtain list of indices representing the topology
    # link_to_path: two dimensional array, first dimension are the paths, second dimension are the link indices
    link_to_path = list()
    # We define link_pos_in_flows that will later help us build path_to_link
    link_pos_in_flows = list()
    for og_path in map(lambda x: x["og_path"], ordered_flows):
        # This list will contain the link indices in the original path,in order
        local_list = list()
        # This dict indicates for each link which are the positions in the original path, if any
        local_dict = dict()
        for link_id in og_path:
            # Transform link_id into a link index
            link_idx = link_mapping[link_id]
            local_dict.setdefault(link_idx, list()).append(len(local_list))
            local_list.append(link_idx)
        link_to_path.append(local_list)
        link_pos_in_flows.append(local_dict)

    # path_to_link: two dimensional array, first dimension are the links, second dimension are tuples.
    # Each tuple contains the path index and the link's position in the path
    # Note that a link can appear in multiple paths and multiple times in the same path
    path_to_link = list()
    for link_idx in range(n_l):
        local_list = list()
        for flow_idx in range(n_f):
            if link_idx in link_pos_in_flows[flow_idx]:
                local_list += [
                    (flow_idx, pos) for pos in link_pos_in_flows[flow_idx][link_idx]
                ]
        path_to_link.append(local_list)
    # Many of the features must have expanded dimensions so they can be concatenated
    sample = (
        {
            "sample_file_name": [sample_file_name] * n_f,
            "sample_file_id": [sample_file_id] * n_f,
            "flow_id": [flow["flow_id"] for flow in ordered_flows],
            "global_losses": np.expand_dims([global_losses], axis=1),
            "global_delay": np.expand_dims([global_delay], axis=1),
            "max_link_load": np.expand_dims([max_link_load], axis=1),
            "flow_traffic": np.expand_dims([flow["traffic"] for flow in ordered_flows], axis=1),
            "flow_bitrate_per_burst": np.expand_dims([flow["flow_bitrate_per_burst"] for flow in ordered_flows], axis=1),
            "flow_tos": np.expand_dims([flow["flow_tos"] for flow in ordered_flows], axis=1),
            "flow_p10PktSize": np.expand_dims([flow["flow_p10PktSize"] for flow in ordered_flows], axis=1),
            "flow_p20PktSize": np.expand_dims([flow["flow_p20PktSize"] for flow in ordered_flows], axis=1),
            "flow_p50PktSize": np.expand_dims([flow["flow_p50PktSize"] for flow in ordered_flows], axis=1),
            "flow_p80PktSize": np.expand_dims([flow["flow_p80PktSize"] for flow in ordered_flows], axis=1),
            "flow_p90PktSize": np.expand_dims([flow["flow_p90PktSize"] for flow in ordered_flows], axis=1),
            "rate": np.expand_dims([flow["rate"] for flow in ordered_flows], axis=1),
            "ibg": np.expand_dims([flow["ibg"] for flow in ordered_flows], axis=1),
            "flow_variance": np.expand_dims([flow["flow_variance"] for flow in ordered_flows], axis=1),
            "flow_pkts_per_burst": np.expand_dims([flow["flow_pkts_per_burst"] for flow in ordered_flows], axis=1),
            "flow_packets": np.expand_dims([flow["packets"] for flow in ordered_flows], axis=1),
            "flow_packet_size": np.expand_dims([flow["packet_size"] for flow in ordered_flows], axis=1),
            "flow_type": [flow["flow_type"] for flow in ordered_flows],
            "flow_length": [flow["length"] for flow in ordered_flows],
            "flow_ipg_mean": np.expand_dims([flow["ipg_mean"] for flow in ordered_flows], axis=1),
            "flow_ipg_var": np.expand_dims([flow["ipg_var"] for flow in ordered_flows], axis=1),
            "link_capacity": np.expand_dims([link["capacity"] for link in ordered_links], axis=1),
            "link_to_path": tf.ragged.constant(link_to_path, dtype=tf.int32),
            "path_to_link": tf.ragged.constant(path_to_link, dtype=tf.int32, ragged_rank=1),
        },
        [flow["delay"] for flow in ordered_flows],
    )
    return sample


def _generator(
    data_dir: str, shuffle: bool, verify_delays: bool
) -> Generator[Tuple[Dict[str, Any], List[float]], None, None]:
    """Returns processed samples from the given dataset.
    定义一个生成器,用于加载数据集并使用第一个函数逐个处理其样本,并过滤不合适的样品
    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool
        True to shuffle the samples, False otherwise
    verify_delays: bool, optional
        True so that samples with unvalid delay values are discarded

    Yields
    ------
    Generator[Tuple[Dict[str, Any], List[float]], None, None]
        Returns a generator of tuples, where the first element is a dictionary with the sample's features
        and the second element is a list of the sample's labels (in this case, the flow's delay)
    """
    try:
        data_dir = data_dir.decode("UTF-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    for sample in iter(tool):
        ret = _get_network_decomposition(sample)
        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
        if verify_delays and not all(x > 0 for x in ret[1]):
            continue
        yield ret

    #yield ret 如果上述条件不为真，则执行 yield ret。
    #yield 会返回当前的 ret 值，并暂停函数的执行。下一次调用生成器的 __next__() 方法时，将从此处继续执行。


def input_fn(data_dir: str, shuffle: bool = False, verify_delays:bool = True) -> tf.data.Dataset:
    """Returns a tf.data.Dataset object with the dataset stored in the given path

    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool, optional
        True to shuffle the samples, False otherwise, by default False
    verify_delays: bool, optional
        True so that samples with unvalid delay values are discarded, by default True

    Returns
    -------
    tf.data.Dataset
        The processed dataset
    """
    signature = (
        {
            "sample_file_name": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "sample_file_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "flow_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "global_losses": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "global_delay": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "max_link_load": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_tos": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_p10PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_p20PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_p50PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_p80PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_p90PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_bitrate_per_burst": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "rate": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "ibg": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_variance": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_pkts_per_burst": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packet_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_type": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            "flow_length": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "flow_ipg_mean": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_ipg_var": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
            "path_to_link": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32, ragged_rank=1),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator,
        args=[data_dir, shuffle, verify_delays],
        output_signature=signature,
    )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


# MAIN: generate the dataset

# Set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

# Parse dataset

tf.data.Dataset.save(
    input_fn(
        args.input_dir,
        shuffle=args.shuffle and not args.test,
        verify_delays=not args.test,
    ),
    args.output_dir,
    compression="GZIP",
)



if not args.test and args.n_folds > 1:
    # Split dataset into n folds
    # NOTE: the reason why we store the dataset and load it again is because otherwise
    # tensorflow will try regenerate the dataset using the generator for each fold
    # which results in taking a lot of time
    ds = tf.data.Dataset.load(args.output_dir, compression="GZIP")
    ds_split = [ds.shard(args.n_folds, ii) for ii in range(args.n_folds)]
    dataset_name = (
        args.output_dir if args.output_dir[-1] != "/" else args.output_dir[:-1]
    )

    for ii in range(args.n_folds):
        # Split dataset into train and validation
        tr_splits = [jj for jj in range(args.n_folds) if jj != ii]
        ds_val = ds_split[ii]
        ds_train = ds_split[tr_splits[0]]
        for jj in tr_splits[1:]:
            ds_train = ds_train.concatenate(ds_split[jj])

        # Save datasets
        tf.data.Dataset.save(
            ds_train, f"{dataset_name}_cv/{ii}/training", compression="GZIP"
        )
        tf.data.Dataset.save(
            ds_val, f"{dataset_name}_cv/{ii}/validation", compression="GZIP"
        )
