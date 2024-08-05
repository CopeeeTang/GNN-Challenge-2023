# Only run as main
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported!")

# Parse imports
from typing import Tuple, Generator, Dict, Any, List
import numpy as np
import tensorflow as tf
from itertools import permutations
from re import sub
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression,RFE,SelectKBest,f_regression,VarianceThreshold
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import argparse
import random
import sys
#添加命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir",default='E:\GNNET\GNNetworkingChallenge-2023_RealNetworkDT\GNNetworkingChallenge-2023_RealNetworkDT\data' ,type=str, required=True)
#parser.add_argument("--output-dir", default='E:/GNNET/GNNetworkingChallenge-2023_RealNetworkDT/GNNetworkingChallenge-2023_RealNetworkDT/data/result',type=str, required=True)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--seed", type=int, default=42)
#parser.add_argument("--n-folds", type=int, default=5)
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
    #此函数采用数据集 API(名为 DatanetAPI)提供的格式获取样本，
    并以 tensorflow 模型接受的格式返回样本
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
    network_topology = sample.get_physical_topology_object()#网络拓扑
    max_link_load=sample.get_max_link_load()
    global_delay=sample.get_global_delay()
    global_losses=sample.get_global_losses()
    global_packets=sample.get_global_packets()
    traffic_matrix = sample.get_traffic_matrix()#流量矩阵
    physical_path_matrix = sample.get_physical_path_matrix()#路由矩阵
    performance_matrix = sample.get_performance_matrix()#性能矩阵
    packet_info_matrix = sample.get_pkts_info_object()#数据包信息矩阵(包含内部所有流的数据包跟踪)
    # Process sample id (for debugging purposes and for then evaluating the model)
    sample_file_path, sample_file_id = sample.get_sample_id()
    sample_file_name = sample_file_path.split("/")[-1]
    # Obtain links and nodes
    # We discard all links that start from the traffic generator
    links = dict()
    #count=1
    for edge in network_topology.edges:  # src, dst, port [source prot][destination port] 
        '''
        if count==1:
            print("network_topology",network_topology.edges)
            print("traffic_matrix",traffic_matrix[1][1])
            print("physical_path",physical_path_matrix[1][1])
            print("performance",performance_matrix[1][1])
            print("packet_info",packet_info_matrix[1][1])
        count+=1    
        '''
        # We identify all traffic generators as the same port
        edge_id = sub(r"t(\d+)", "tg", network_topology.edges[edge]["port"])#sub函数替换
        if edge_id.startswith("r") or edge_id.startswith("s"):
            links[edge_id] = {
                "capacity": float(network_topology.edges[edge].get("bandwidth", 1e9))
                / 1e9,  # original value is in bps, we change it to Gbps 进制转换
            }
        elif edge_id.startswith("tg"):
            continue
        else:
            raise ValueError(f"Unknown edge type: {edge_id}")
    
    # In this scenario assume that flows can either follow CBR or MB distributions
    flows = dict()
    used_links = set()  # Used later so we only consider used links
    # Add flows
    #提取新功能 #外部循环遍历路由器对 lambda确保其间有流 src始 dst结束
    for src, dst in filter(
        lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
        and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
        permutations(range(len(traffic_matrix)), 2),
    ):
        
        for local_flow_id in range(len(traffic_matrix[src, dst]["Flows"])):
            flow_packet_info=packet_info_matrix[src, dst][0][local_flow_id]
            flow = traffic_matrix[src, dst]["Flows"][local_flow_id]
            # Size distribution is always determinstic
            # Obtain and clean the path followed the flow
            # We must also clean up the name of the traffic generator
            clean_og_path = [
                sub(r"t(\d+)", "tg", link)
                for link in physical_path_matrix[src, dst][2::2]
            ]
            packet_timestamps=np.array([float(x[0]) for x in flow_packet_info])
            ipg = packet_timestamps[1:] - packet_timestamps[-1:]

            flow_id = f"{src}_{dst}_{local_flow_id}"
            #提取流特征
            flows[flow_id] = {
                "source": src,
                "destination": dst,
                "flow_id": flow_id,
                "length": len(clean_og_path),
                "og_path": clean_og_path,
                "traffic": flow["AvgBw"],  # in bps
                "packets": flow["PktsGen"],
                
                "packet_size": flow["SizeDistParams"]["AvgPktSize"],
                "flow_type": (
                    float(flow["TimeDist"] == TimeDist.CBR_T),
                    float(flow["TimeDist"] == TimeDist.MULTIBURST_T),
                ),
                "delay": performance_matrix[src, dst]["Flows"][local_flow_id][
                    "AvgDelay"
                ]
                * 1000,  # in ms
                #new change
                "ipg_mean":np.mean(ipg),
                "ipg_var":np.var(ipg),
                "ipg_percentiles":np.percentile(ipg, range(101)),
                "flow_variance": flow["VarPktSize"],
                "flow_tos": flow["ToS"],
                "flow_p10PktSize": flow["p10PktSize"],
                "flow_p20PktSize": flow["p20PktSize"],
                "flow_p50PktSize": flow["p50PktSize"],
                "flow_p80PktSize": flow["p80PktSize"],
                "flow_p90PktSize": flow["p90PktSize"],
                "rate": flow["TimeDistParams"]["Rate"] if flow["TimeDist"] == TimeDist.CBR_T else 0,
                "ibg": flow["TimeDistParams"]["IBG"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_bitrate_per_burst": flow["TimeDistParams"]["On_Rate"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
                "flow_pkts_per_burst": flow["TimeDistParams"]["Pkts_per_burst"] if flow["TimeDist"] == TimeDist.MULTIBURST_T else 0,
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
            # Identifier features
            "sample_file_name": [sample_file_name] * n_f,
            "sample_file_id": [sample_file_id] * n_f,
            #new
            "max_link_load": np.expand_dims([max_link_load], axis=1),
            "global_losses": np.expand_dims([global_losses], axis=1),
            "global_delay": np.expand_dims([global_delay], axis=1),

            "flow_id": [flow["flow_id"] for flow in ordered_flows],
            "flow_traffic": np.expand_dims(
                [flow["traffic"] for flow in ordered_flows], axis=1
            ),
            # Flow attributes
            #new
            "flow_bitrate_per_burst": np.expand_dims(
                [flow["flow_bitrate_per_burst"] for flow in ordered_flows], axis=1
            ),
            "flow_tos": np.expand_dims(
                [flow["flow_tos"] for flow in ordered_flows], axis=1
            ),
            "flow_p10PktSize": np.expand_dims(
                [flow["flow_p10PktSize"] for flow in ordered_flows], axis=1
            ),
            "flow_p20PktSize": np.expand_dims(
                [flow["flow_p20PktSize"] for flow in ordered_flows], axis=1
            ),
            "flow_p50PktSize": np.expand_dims(
                [flow["flow_p50PktSize"] for flow in ordered_flows], axis=1
            ),       
            "flow_p80PktSize": np.expand_dims(
                [flow["flow_p80PktSize"] for flow in ordered_flows], axis=1
            ),  
            "flow_p90PktSize": np.expand_dims(
                [flow["flow_p90PktSize"] for flow in ordered_flows], axis=1
            ),
            "rate": np.expand_dims(
                [flow["rate"] for flow in ordered_flows], axis=1
            ),
            "ibg": np.expand_dims(
                [flow["ibg"] for flow in ordered_flows], axis=1
            ),
            "flow_variance": np.expand_dims(
                [flow["flow_variance"] for flow in ordered_flows], axis=1
            ),
            "flow_pkts_per_burst": np.expand_dims(
                [flow["flow_pkts_per_burst"] for flow in ordered_flows], axis=1
            ),
            
            "flow_packets": np.expand_dims(
                [flow["packets"] for flow in ordered_flows], axis=1
            ),
            "flow_packet_size": np.expand_dims(
                [flow["packet_size"] for flow in ordered_flows], axis=1
            ),
            "flow_type": [flow["flow_type"] for flow in ordered_flows],
            "flow_length": [flow["length"] for flow in ordered_flows],
            #new
            "flow_ipg_mean":np.expand_dims([flow["ipg_mean"] for flow in ordered_flows], axis=1
            ),
            "flow_ipg_var":np.expand_dims([flow["ipg_var"] for flow in ordered_flows], axis=1
            ),
            

            # Link attributes
            "link_capacity": np.expand_dims(
                [link["capacity"] for link in ordered_links], axis=1
            ),
            # Topology attributes
            "link_to_path": tf.ragged.constant(link_to_path),
            "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1),
        },
        [flow["delay"] for flow in ordered_flows],
    )

    return sample

def feature_select(data_dir:str,shuffle:bool):
    try:
        data_dir=data_dir.decode("UTF-8")
    except(UnicodeDecodeError,AttributeError):
        pass
    
    tool=DatanetAPI(data_dir,shuffle=shuffle)

    chosen_features=['ibg', 'flow_packets', 'flow_pkts_per_burst', 'flow_packet_size', 
                        'rate', 'flow_bitrate_per_burst', 'flow_traffic', 'flow_ipg_mean', 
                        'flow_tos', 'flow_length', 'flow_ipg_var',
                        'flow_p10PktSize', 'flow_p20PktSize' , 'flow_p50PktSize' , 
                        'flow_p80PktSize' , 'flow_p90PktSize']
    
    features_concat=np.array([])
    delay_concat=np.array([])
    count=0

    for sample in iter(tool):
        flow_features=np.array([])
        ret = _get_network_decomposition(sample)

        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
        if not all(x > 0 for x in ret[1]):
            continue
        #yield ret

        for feature in chosen_features:
            if feature=='flow_length':
                ret[0]['flow_length']=np.expand_dims(ret[0][feature],axis=1)#需要用
            if len(flow_features)==0:
                flow_features=ret[0][feature] ## 初始化流特征数组
            else:
                flow_features=np.hstack([flow_features,ret[0][feature]])## 水平堆叠特征数组
        
        #将后续数组合并到其中
        if count==0:
            features_concat= flow_features 
            delay_concat=ret[1]
        else:
            features_concat=np.concatenate([features_concat,flow_features])
            delay_concat=np.concatenate([delay_concat,ret[1]])
        
        count+=1

        
    #--------------------Filter Multi Information Gain Method
    print("数据标准化")
    mean_label=np.mean(delay_concat)
    mean_feat=np.mean(features_concat)

    stand_label=np.std(delay_concat)
    stand_feat=np.std(features_concat)

    norm_label=(delay_concat-mean_label)/stand_label
    norm_feature=(features_concat-mean_feat)/stand_feat #做归一化

    '''
    min_label = np.min(delay_concat)
    max_label = np.max(delay_concat)

    min_feat = np.min(features_concat, axis=0)
    max_feat = np.max(features_concat, axis=0)

    norm_label = (delay_concat - min_label) / (max_label - min_label)
    norm_feature = (features_concat - min_feat) / (max_feat - min_feat)  # 做 Min-Max 归一化

    '''
    print("互信息增益法")
    mig = mutual_info_regression(norm_feature,norm_label)
    type_features_mig_scores={}

    for i in range(len(chosen_features)):
        type_features_mig_scores[chosen_features[i]]=mig[i]

    sorted_type_features=sorted(type_features_mig_scores.items(),key=lambda x:x[1],reverse=True)

    for type_feature,score in sorted_type_features:
        print("feature:",type_feature,"Score:",score)
    ##Wrapper Exhaustive
    print("Wrapper Exhaustive")
    efs1=EFS(LinearRegression(),
            min_features=4,
            max_features=10,
            scoring='neg_mean_squared_error',
            print_progress=True,
            n_jobs=1)
    efs1.fit(norm_feature,norm_label)
    best=[efs1.best_idx_]
    type_features_efs_scores={}
    for i in best[0]:
        type_features_efs_scores[chosen_features[i]]=efs1.subsets_scores_[i]
    
    sorted_type_features=sorted(type_features_efs_scores.items(),key=lambda x:x[1],reverse=True)
    for type_feature,score in sorted_type_features:
        print("feature:",type_feature,"Score:",score)
    
    
    '''
    #--------- 方差阈值法-----------
    print("方差阈值法")
    X_train,X_test,y_train,y_test=train_test_split(norm_feature,norm_label,test_size=0.2,random_state=0)

    threshold=0.1
    selector=VarianceThreshold(threshold)
    X_train_var=selector.fit_transform(X_train)
    X_test_var=selector.transform(X_test)
    print("Selected features:",selector.get_support(indices=True))
    
    #k个特征-------SelectKBest
    print("k个特征-------SelectKBest")
    k=10
    selector=SelectKBest(score_func=f_regression,k=k)
    selector.fit(norm_feature,norm_label)
    #获取选定特征索引
    selected_features=selector.get_support(indices=True)
    print("Selected features:",chosen_features[selected_features])

    #---------RFE--------
    print("RFE--------")
    model=LinearRegression()
    rfe=RFE(model,n_features_to_select=k)
    rfe.fit(norm_feature,norm_label)
    #获取选定特征索引
    selected_features=rfe.get_support(indices=True)
    print("Selected features:",chosen_features[selected_features])
    '''
#set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

feature_select(args.input_dir,args.shuffle)

