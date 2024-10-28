
from exp_lib.utils import get_logger

from .gcn_flow import GCNFlow
from .dgcn_flow import DGCNFlow
from .gat_flow import GATFlow
from .node2vec_flow import Node2VecFlow 
from .fastrp_flow import FastRPFlow 
from .simple_hgn_flow import SimpleHGNFlow 

logger = get_logger()

def load_encoder_module(
    module_name, 
    input_dim, 
    embed_dim,
    with_feat,
    config,
    train_data,
):
    if module_name=="gcn":
        flow = GCNFlow(
            input_dim, 
            embed_dim, 
            with_feat, 
            config["epoch"], 
            config["learning_rate"], 
            config["l2"], 
            config["batch_size"], 
            config["dropout"], 
            config["num_layer"], 
        )
    elif module_name=="gat":
        flow = GATFlow(
            input_dim, 
            embed_dim, 
            with_feat, 
            config["epoch"], 
            config["learning_rate"], 
            config["l2"], 
            config["batch_size"], 
            config["dropout"], 
            config["num_layer"], 
            config["num_head"], 
        )
    elif module_name=="dgcn":
        flow = DGCNFlow(
            input_dim, 
            embed_dim, 
            with_feat, 
            config["epoch"], 
            config["learning_rate"], 
            config["l2"], 
            config["batch_size"], 
            config["dropout"], 
            config["num_layer"], 
        )
    elif module_name=="node2vec":
        flow = Node2VecFlow(
            embed_dim=config["embed_dim"], 
            walk_length=config["walk_length"],
            context_size=config["context_size"], 
            walks_per_node=config["walks_per_node"],
            num_negative_samples=config["num_negative_samples"], 
            num_workers=8, 
            p=config["p"], 
            q=config["q"],
            encoder_lr=config["learning_rate"],
            encoder_l2=config["l2"],
            epoch=config["epoch"],
            batch_size=config["batch_size"],
        )
    elif module_name=="fastrp":
        flow = FastRPFlow(
            embed_dim=config["embed_dim"], 
            weight1=config["weight1"], 
            weight2=config["weight2"], 
            weight3=config["weight3"], 
            weight4=config["weight4"], 
            projection_method=config["projection_method"], 
            input_matrix=config["input_matrix"], 
            alpha=config["alpha"], 
            normalization=config["normalization"],
        )
    elif module_name=="hgn":
        num_edge_types = train_data.edge_type.max() + 1
        flow = SimpleHGNFlow(
            input_dim, 
            embed_dim, 
            with_feat,
            config["epoch"], 
            config["learning_rate"], 
            config["l2"], 
            config["batch_size"], 
            config["dropout"], 
            config["num_layer"], 
            config["num_head"], 
            config["edge_dim"], 
            num_edge_types, 
        )
    else:
        logger.info("Specify a right module name")
        flow=None

    return flow

    