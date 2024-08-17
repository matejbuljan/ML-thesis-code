# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import argparse
import datetime
import os
import re
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf
import json

from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from google.protobuf import text_format

from waymo_open_dataset.utils.sim_agents import submission_specs

from waymo_open_dataset.protos import sim_agents_metrics_pb2

# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc
rc('animation', html='jshtml')


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


# Some useful functions
def joint_scene_from_states(
        states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.JointScene:
    # States shape: (num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    states = states.numpy()
    simulated_trajectories = []
    for i_object in range(len(object_ids)):
        simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
            object_id=object_ids[i_object]
        ))
    return sim_agents_submission_pb2.JointScene(
        simulated_trajectories=simulated_trajectories)


def scenario_rollouts_from_states(
        scenario: scenario_pb2.Scenario,
        states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.ScenarioRollouts:
    # States shape: (num_rollouts, num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    joint_scenes = []
    for i_rollout in range(states.shape[0]):
        joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
    return sim_agents_submission_pb2.ScenarioRollouts(
        # Note: remember to include the Scenario ID in the proto message.
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    """Loads the `SimAgentMetricsConfig` used for the challenge."""
    # pylint: disable=line-too-long
    # pyformat: disable
    config_path = '/home/mbultc/PycharmProjects/Thesis/waymo-open-dataset/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_config.textproto'
    with open(config_path, 'r') as f:
        config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), config)
    return config


def log_memory_usage(stage):
    print(f"{stage} - Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"{stage} - Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"{stage} - Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"{stage} - Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")


def process_scenario(batch_dict, dataset, updated_frames, logger, dist_test, args, scenario_counter):
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    model.load_params_from_file(filename="/home/mbultc/Downloads/latest_model.pth", to_cpu=False, logger=logger)
    model.cuda()
    model.eval()
    # log_memory_usage("Reloading the model")
    print(f"Number of agents in the scene: {len(batch_dict['input_dict']['obj_ids'])}")

    closed_loop_trajs = []
    for closed_loop_iteration in range(80 // updated_frames):
        print(f'Doing closed loop iteration {closed_loop_iteration + 1} of {80//updated_frames}')

        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            # log_memory_usage("Done forward pass in closed loop")

            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=None)

            valid_agent_states_list = []
            updated_trajs_list = []
            for agent_dict in final_pred_dicts[0]:
                z = agent_dict['gt_trajs'][10, 2]       # This is the z-coordinate which we will keep constant to what it was at the last timestep of the input
                heading = agent_dict['gt_trajs'][10, 6]     # Initial heading direction
                dx = agent_dict['gt_trajs'][10, 3]  # Agent length in x direction
                dy = agent_dict['gt_trajs'][10, 4]  # Agent length in y direction
                dz = agent_dict['gt_trajs'][10, 5]  # Agent length in z direction
                vel_x = agent_dict['gt_trajs'][10, 7]  # Initial velocity in x direction -> needs to be calculated anew
                vel_y = agent_dict['gt_trajs'][10, 8]  # Initial velocity in y direction -> needs to be calculated anew
                valid = agent_dict['gt_trajs'][10, 9]  # All agents here are valid, right?

                xy_vector = agent_dict['pred_trajs'][0, :, :]
                heading_vector = np.full((80, 1), heading)
                z_vector = np.full((80, 1), z)
                velocity_xy = (xy_vector[1:, :] - xy_vector[:-1, :]) * 10
                new_vel_x_vector = np.full((80, 1), vel_x)
                new_vel_x_vector[1:, :] = velocity_xy[:, 0][:, np.newaxis]
                new_vel_y_vector = np.full((80, 1), vel_y)
                new_vel_y_vector[1:, :] = velocity_xy[:, 1][:, np.newaxis]
                dx_vector = np.full((80, 1), dx)
                dy_vector = np.full((80, 1), dy)
                dz_vector = np.full((80, 1), dz)
                valid_vector = np.full((80, 1), valid)

                # Heading calculations
                for timestamp_2s in range(4):
                    distance_over_2s = np.sqrt(np.power((xy_vector[10 * (timestamp_2s + 1), 0] - xy_vector[10 * timestamp_2s, 0]), 2) + np.power((xy_vector[10 * (timestamp_2s + 1), 1] - xy_vector[10 * timestamp_2s, 1]), 2))
                    if distance_over_2s > 0.3:
                        if timestamp_2s > 0:
                            delta_xy_vector = xy_vector[(timestamp_2s * 20):(timestamp_2s + 1) * 20, :] - xy_vector[(timestamp_2s * 20 - 1):((timestamp_2s + 1) * 20 - 1), :]
                            heading_vector[(timestamp_2s * 20):(timestamp_2s + 1) * 20] = np.arctan2(delta_xy_vector[:, 1][:, np.newaxis], delta_xy_vector[:, 0][:, np.newaxis])
                            for timestamp in range(timestamp_2s * 20, (timestamp_2s + 1) * 20):
                                heading_delta = heading_vector[timestamp, :] - heading_vector[timestamp - 1, :]
                                if heading_delta > 0.3:
                                    heading_vector[timestamp, :] = heading_vector[timestamp - 1, :]
                        else:
                            delta_xy_vector = xy_vector[1:20, :] - xy_vector[0:19, :]
                            heading_vector[1:20] = np.arctan2(delta_xy_vector[:, 1][:, np.newaxis], delta_xy_vector[:, 0][:, np.newaxis])
                            for timestamp in range(1, 20):
                                heading_delta = heading_vector[timestamp, :] - heading_vector[timestamp - 1, :]
                                if heading_delta > 0.3:
                                    heading_vector[timestamp, :] = heading_vector[timestamp - 1, :]

                # Aggregating data for alla center agents in the scenario
                agent_states = np.concatenate((xy_vector, z_vector, heading_vector), axis=1)
                valid_agent_states_list.append(agent_states)
                agent_updated_trajs = np.concatenate((xy_vector, z_vector, dx_vector, dy_vector, dz_vector, heading_vector, new_vel_x_vector, new_vel_y_vector, valid_vector), axis=1)
                updated_trajs_list.append(agent_updated_trajs)

            # Preparing scenario-specific data
            single_rollout = np.stack(valid_agent_states_list)
            rollouts_list = [single_rollout for _ in range(32)]
            rollouts = np.stack(rollouts_list)                  # Shape (32, #center_agents, 80, 4)
            updated_trajs = np.stack(updated_trajs_list)        # Shape (#center_agents, 80, 10)
            closed_loop_trajs.append(rollouts[:, :, :updated_frames, :])

            # log_memory_usage("Just before updating the dataset in the closed loop")
            # Update the dataset
            test_set, test_loader, sampler = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                batch_size=1,
                dist=dist_test, workers=args.workers, logger=logger, training=False,
                updated_trajs=updated_trajs, updated_frames=updated_frames, updated_scenario_id=batch_dict['input_dict']['scenario_id'][0]
            )
            # log_memory_usage("Just after updating the dataset int the closed loop")
            for i, batch_dict in enumerate(test_loader):
                if i >= scenario_counter:
                    break
            # log_memory_usage("Found the scenario again at the end of the closed loop")

    closed_loop_trajs_stacked = np.concatenate(closed_loop_trajs, axis=2)
    object_ids = batch_pred_dicts['input_dict']['center_objects_id']

    return final_pred_dicts[0][0]['scenario_id'], object_ids, closed_loop_trajs_stacked[0, :, :, :]
'''
# Module for saving individual scenario info to individual files
    scenario_data = {
        "scenario_id": final_pred_dicts[0][0]['scenario_id'],
        "object_ids": object_ids.tolist(),
        "closed-loop-trajs-tolist": closed_loop_trajs_stacked.tolist()
    }
    folder_name = "/home/mbultc/PycharmProjects/MTR/data/saved_tracks_json/20frames/"
    json_file_name = folder_name + final_pred_dicts[0][0]['scenario_id'] + ".json"
    # Save data to a JSON file
    with open(json_file_name, "w") as json_file:
        json.dump(scenario_data, json_file)
'''



def main():
    # log_memory_usage("Beginning of the program")
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
          
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id)
    else:
        epoch_id = None
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    scenario_counter = 0

    scenario_id_list_json = []
    object_ids_list_json = []
    closed_loop_trajs_list_json = []

    number_of_scenarios_for_validation = 1
    updated_frames = 80
    assert 80 % updated_frames == 0, "Update frequency must be a divisor of 80!"

    all_original_batch_dicts = []
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=1,
        dist=dist_test, workers=args.workers, logger=logger, training=False,
    )
    dataset = test_loader.dataset
    for i, batch_dict in enumerate(test_loader):
        all_original_batch_dicts.append(batch_dict)
        print(f"Scenario number {i} with id {batch_dict['input_dict']['scenario_id'][0]}")
        if i >= number_of_scenarios_for_validation:
            break

    # log_memory_usage("Saved all scenarios to a list")
    while scenario_counter < number_of_scenarios_for_validation:
        batch_dict = all_original_batch_dicts[scenario_counter]
        print("-------------------------------------------------------------------------------------------------------")
        print(f"Loaded scenario {scenario_counter} with id {batch_dict['input_dict']['scenario_id'][0]}")
        # log_memory_usage("New scenario")

        scenario_id, obj_ids, closed_loop_trajs = process_scenario(batch_dict, dataset, updated_frames, logger, dist_test, args, scenario_counter)

        # Aggregating data across multiple scenarios
        scenario_id_list_json.append(scenario_id)
        object_ids_list_json.append(obj_ids)
        closed_loop_trajs_list_json.append(closed_loop_trajs)

        scenario_counter += 1

    # Preparing things for saving to json
    nested_object_ids_list = [thing.tolist() for thing in object_ids_list_json]
    nested_closed_loop_trajs_list = [thing.tolist() for thing in closed_loop_trajs_list_json]
    data = {
        "object_ids_tensors": nested_object_ids_list,
        "scenario_ids": scenario_id_list_json,
        "closed_loop_trajs": nested_closed_loop_trajs_list
    }
    # Save data to a JSON file
    with open("/home/mbultc/PycharmProjects/MTR/data/saved_tracks_json/500_scenarios_80_frames_single_rollout.json", "w") as json_file:
        json.dump(data, json_file)

'''
     # Loading data

    DATASET_FOLDER = '/home/mbultc/womd1_1/'
    VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'mini_womd1_1/validation/*')

    # Define the dataset from the TFRecords.
    filenames = tf.io.matching_files(VALIDATION_FILES)
    print(f'Number of tfrecord files read: {filenames.shape}')
    dataset = tf.data.TFRecordDataset(filenames)
    # Since these are raw Scenario protos, we need to parse them in eager mode.
    dataset_iterator = dataset.as_numpy_iterator()
    # print(f'Number of scenarios in dataset iterator: {dataset.__len__()}')

    print(scenario_id_list_json)
    print(f'Number of predictions: {len(object_ids_list_json)}')

    # Iterating over scenarios and predictions and doing evaluation
    counter = 0
    for bytes_example in dataset_iterator:
        scenario = scenario_pb2.Scenario.FromString(bytes_example)
        assert scenario.scenario_id == scenario_id_list_json[counter], "Mismatch between scenario ids"
        print(f'Doing evaluation for scenario: {scenario.scenario_id}')

        eval_tracks_and_difficulty = {
            'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
        }
        print(eval_tracks_and_difficulty)

        simulated_states = tf.convert_to_tensor(closed_loop_trajs_list_json[counter])
        object_ids = tf.convert_to_tensor(object_ids_list_json[counter], dtype=tf.int32)

        scenario_rollouts = scenario_rollouts_from_states(scenario, simulated_states, object_ids)
        # As before, we can validate the message we just generate.
        submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

        # Load the test configuration.
        config = load_metrics_config()

        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)
        print(scenario_metrics)

        counter += 1
        if counter >= len(closed_loop_trajs_list_json):
            break
'''




if __name__ == '__main__':
    main()
