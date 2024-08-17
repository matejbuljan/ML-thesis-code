# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import json

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils
from time import perf_counter
from google.protobuf import text_format
import csv


### Some useful functions
def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.JointScene:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  # states = states.numpy()
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


def simulate_with_extrapolation(
        scenario: scenario_pb2.Scenario,
        print_verbose_comments: bool = True) -> tf.Tensor:
    vprint = print if print_verbose_comments else lambda arg: None

    # To load the data, we create a simple tensorized version of the object tracks.
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    # Using `ObjectTrajectories` we can select just the objects that we need to
    # simulate and remove the "future" part of the Scenario.
    # vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
    logged_trajectories = logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    logged_trajectories = logged_trajectories.slice_time(
        start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
    # vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')

    # We can verify that all of these objects are valid at the last step.
    # vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')

    # We extract the speed of the sim agents (in the x/y/z components) ready for
    # extrapolation (this will be our policy).
    states = tf.stack([logged_trajectories.x, logged_trajectories.y,
                       logged_trajectories.z, logged_trajectories.heading],
                      axis=-1)
    n_objects, n_steps, _ = states.shape
    # print(states[:3, -1, :])
    last_velocities = states[:, -1, :3] - states[:, -2, :3]

    # print(last_velocities)
    # We also make the heading constant, so concatenate 0. as angular speed.
    last_velocities = tf.concat(
        [last_velocities, tf.zeros((n_objects, 1))], axis=-1)
    # It can happen that the second to last state of these sim agents might be
    # invalid, so we will set a zero speed for them.
    # vprint(f'Is any 2nd to last state invalid: {tf.reduce_any(tf.logical_not(logged_trajectories.valid[:, -2]))}')
    # vprint(f'This will result in either min or max speed to be really large: {tf.reduce_max(tf.abs(last_velocities))}')
    valid_diff = tf.logical_and(logged_trajectories.valid[:, -1],
                                logged_trajectories.valid[:, -2])
    # `last_velocities` shape: (n_objects, 4).
    last_velocities = tf.where(valid_diff[:, tf.newaxis],
                               last_velocities,
                               tf.zeros_like(last_velocities))
    # vprint(f'Now this should be back to a normal value: {tf.reduce_max(tf.abs(last_velocities))}')

    # Now we carry over a simulation. As we discussed, we actually want 32 parallel
    # simulations, so we make this batched from the very beginning. We add some
    # random noise on top of our actions to make sure the behaviours are different.
    # To properly scale the noise, we get the max velocities (average over all
    # objects, corresponding to axis 0) in each of the dimensions (x/y/z/heading).
    NOISE_SCALE = 0.01
    # `max_action` shape: (4,).
    max_action = tf.reduce_max(last_velocities, axis=0)
    # We create `simulated_states` with shape (n_rollouts, n_objects, n_steps, 4).
    simulated_states = tf.tile(states[tf.newaxis, :, -1:, :], [submission_specs.N_ROLLOUTS, 1, 1, 1])
    # vprint(f'Shape: {simulated_states.shape}')

    for step in range(submission_specs.N_SIMULATION_STEPS):
        current_state = simulated_states[:, :, -1, :]
        # Random actions, take a normal and normalize by min/max actions
        action_noise = tf.random.normal(
            current_state.shape, mean=0.0, stddev=NOISE_SCALE)
        actions_with_noise = last_velocities[None, :, :] + (action_noise * max_action)
        next_state = current_state + actions_with_noise
        simulated_states = tf.concat(
            [simulated_states, next_state[:, :, None, :]], axis=2)

    # We also need to remove the first time step from `simulated_states` (it was
    # still history).
    # `simulated_states` shape before: (n_rollouts, n_objects, 81, 4).
    # `simulated_states` shape after: (n_rollouts, n_objects, 80, 4).
    simulated_states = simulated_states[:, :, 1:, :]
    # vprint(f'Final simulated states shape: {simulated_states.shape}')

    return logged_trajectories, simulated_states



DATASET_FOLDER = '/home/mbultc/womd1_1/'
TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'mini_womd1_1/validation/*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'testing/*')

# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(VALIDATION_FILES)
print(filenames.shape)
dataset = tf.data.TFRecordDataset(filenames)
# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()

results = []
scenario_number = 1000
for i in range(scenario_number):
    time1 = perf_counter()
    bytes_example = next(dataset_iterator)
    scenario = scenario_pb2.Scenario.FromString(bytes_example)
    # print(f'Checking type: {type(scenario)}')
    print(f'Loaded scenario {i} with ID: {scenario.scenario_id}')
    eval_tracks_and_difficulty = {
        'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
        'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
    }
    time2 = perf_counter()
    print(f'Loaded scenario and info time: {time2 - time1}')
    logged_trajectories, simulated_states = simulate_with_extrapolation(scenario, print_verbose_comments=True)
    time3 = perf_counter()
    print(f'Constant velocity plus noise simulation time: {time3 - time2}')
    object_ids = logged_trajectories.object_id.numpy()
    scenario_rollouts = scenario_rollouts_from_states(scenario, simulated_states, object_ids)
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

    # Load the test configuration.
    config = load_metrics_config()

    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)
    time4 = perf_counter()
    print(f'Calculating metrics time: {time4 - time3}')
    metrics_featurest_list = str(scenario_metrics).split()

    metrics_featurest_list.append('number_of_agents')
    metrics_featurest_list.append(str(len(object_ids)))

    metrics_featurest_list.append('difficulties')
    difficulties_str = ''
    for diff in eval_tracks_and_difficulty['difficulty']:
        difficulties_str += (str(diff) + ' ')
    metrics_featurest_list.append(difficulties_str)
    print(metrics_featurest_list)
    results.append(metrics_featurest_list)
    time5 = perf_counter()
    print(f'Adding results to list time: {time5 - time4}')

# Define the output CSV file path
output_csv_file = "evaluation_1000_scenarios_constant_velocity.csv"

# Write the data to the CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ["scenario_id", "metametric", "average_displacement_error", "linear_speed_likelihood",
                  "linear_acceleration_likelihood", "angular_speed_likelihood", "angular_acceleration_likelihood",
                  "distance_to_nearest_object_likelihood", "collision_indication_likelihood",
                  "time_to_collision_likelihood", "distance_to_road_edge_likelihood", "offroad_indication_likelihood",
                  "min_average_displacement_error", "number_of_agents", "difficulties"]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row with field names
    csv_writer.writeheader()

    # Write the data rows
    for data_list in results:
        row_data = {}
        for i in range(0, len(data_list), 2):
            # Remove ":" from field names and remove quotes from values
            field = data_list[i].replace(":", "").strip()
            value = data_list[i + 1].replace('"', "").strip()
            row_data[field] = value
        csv_writer.writerow(row_data)
