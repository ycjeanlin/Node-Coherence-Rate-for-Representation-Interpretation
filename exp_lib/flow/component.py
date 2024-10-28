from exp_lib.flow import HomoIntrusionTestFlow
from exp_lib.utils import get_logger, pass_rate_upper_bound, random_test 

logger = get_logger()

def homo_intrusion_test(args, config, semantic_relations, dataset, train_data, node_embed, new_test):
    epsilon, ub_intrusion_pass_rate, ub_ranking_pass_rate = random_test(
        node_embed, 
        config["topk"], 
        config["num_intruders"], 
        config["intrusion_decision_quantile"], 
    )
    intrusion_ub = pass_rate_upper_bound(node_embed, config["topk"], epsilon)
    for relation in semantic_relations:
        flow = HomoIntrusionTestFlow(
            base_folder=args.base_folder,
            dataset_name=config["dataset"],
            data_suffix=config["data_suffix"],
            topk = config["topk"],
            num_intruders = config["num_intruders"],
            num_buckets = config["num_buckets"],
            num_workers = config["num_workers"],
            num_processes = config["num_processes"],
        )
        flow.gen_test(
            relation=relation,
            sim_threshold=config["similarity_threshold"],
            model_name=config["model"],
            dataset=dataset,
            train_data=train_data,
            node_embed=node_embed,
            new_test=new_test,
            intrusion_epsilon=epsilon,
        )

        test_df = flow.run_test(
            relation=relation,
            model_name=config["model"],
            node_embed=node_embed,
            intrusion_epsilon=epsilon,
             
        )
        test_df["Cantelli Upper Bound"]=[intrusion_ub] * len(test_df)
        test_df["Intrusion Upper Bound"]=[ub_intrusion_pass_rate] * len(test_df)
        test_df["Ranking Upper Bound"]=[ub_ranking_pass_rate] * len(test_df)
        out_test_file = f"{args.base_folder}/{config['dataset']}/exp/{relation}_{args.task}_{config['dataset']}_{config['data_suffix']}_{config['model']}_{args.model_suffix}.csv"
        test_df.to_csv(out_test_file)
        logger.info(f"Output Exp File: {out_test_file}")
