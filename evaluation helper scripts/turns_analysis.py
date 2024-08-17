### Imports
import csv
import os
import tarfile
import numpy as np
import tensorflow as tf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import tqdm
import json
from time import perf_counter

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from google.protobuf import text_format
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils

# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc
rc('animation', html='jshtml')


### Some useful functions

def calculate_residuals(x, y):
    slope, intercept, _, _, _ = linregress(x, y)
    fitted_y = slope * np.array(x) + intercept
    residuals = np.array(y) - fitted_y
    return residuals

def calculate_path_length(x, y):
    points = np.array([x, y]).T
    path_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    direct_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    return path_length, direct_distance

def classify_track(track, print_verbose_comments = True):
    vprint = print if print_verbose_comments else lambda arg: None
    valids = np.array([state.valid for state in track.states])
    if np.any(valids):
        x = np.array([state.center_x for state in track.states])
        y = np.array([state.center_y for state in track.states])
        x = x[x != 0]
        y = y[y != 0]
        # print(np.stack((x, y), axis=1))
        vprint(x.shape)
        path_length, direct_distance = calculate_path_length(x, y)
        vprint(f'Path length: {path_length}')
        if path_length < 4:  # Needs to be calibrated
            return 'stationary'
        # Compare path length and direct distance
        turning1 = False
        relative_diff = path_length / direct_distance
        vprint(f'Difference between direct path and total path: {relative_diff}')
        if relative_diff > 1.04:  # Needs calibration
            turning1 = True
        # Look at average residuals
        turning2 = False
        residuals = calculate_residuals(x, y)
        avg_residuals = np.average(np.abs(residuals))
        vprint(f'Average residuals: {avg_residuals}')
        if avg_residuals > 1:   # Needs to be calibrated
            turning2 = True
        if turning1 and turning2:
            return 'turning'
        else:
            return 'going straight'
    return 'invalid'

def classify_traj(x, y, print_verbose_comments = True):
    vprint = print if print_verbose_comments else lambda arg: None
    x = x[x != 0]
    y = y[y != 0]
    # print(np.stack((x, y), axis=1))
    vprint(x.shape)
    path_length, direct_distance = calculate_path_length(x, y)
    vprint(f'Path length: {path_length}')
    if path_length < 4:  # Needs to be calibrated
        return 'stationary'
    # Compare path length and direct distance
    turning1 = False
    relative_diff = path_length / direct_distance
    vprint(f'Difference between direct path and total path: {relative_diff}')
    if relative_diff > 1.04:  # Needs calibration
        turning1 = True
    # Look at average residuals
    turning2 = False
    residuals = calculate_residuals(x, y)
    avg_residuals = np.average(np.abs(residuals))
    vprint(f'Average residuals: {avg_residuals}')
    if avg_residuals > 1:  # Needs to be calibrated
        turning2 = True
    if turning1 and turning2:
        return 'turning'
    else:
        return 'going straight'

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

def make_constant_velocity(x, y):
    delta_x = (x[10] - x[9]) / 0.1
    delta_y = (y[10] - y[9]) / 0.1
    for pos in range(11, 91):
        x[pos] = x[pos - 1] + delta_x
        y[pos] = y[pos - 1] + delta_y
    return x, y

### Loading data

DATASET_FOLDER = '/home/mbultc/womd1_1/'
TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'mini_womd1_1/validation/*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'testing/*')

# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(VALIDATION_FILES)
print(f'Number of tfrecord files read: {filenames.shape}')
dataset = tf.data.TFRecordDataset(filenames)
# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()
#print(f'Number of scenarios in dataset iterator: {dataset.__len__()}')


all_json_files_data = []
scenario_id_list = []
nested_object_ids_list = []
nested_rollouts_list = []
updated_frames = 80
# Load data from the JSON file
if updated_frames == 1:
    with open("/home/mbultc/Downloads/0_to_50_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/50_to_60_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/60_to_70_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/70_to_80_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/80_to_200_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/200_to_400_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/400_to_500_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/500_to_800_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/800_to_1000_scenarios_1_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 2:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_2_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_400_scenarios_2_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/400_to_800_scenarios_2_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/800_to_1000_scenarios_2_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 5:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_5_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_500_scenarios_5_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/500_to_800_scenarios_5_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/800_to_1000_scenarios_5_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 10:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_10_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_600_scenarios_10_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/600_to_1000_scenarios_10_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 20:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_20_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_600_scenarios_20_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/600_to_1000_scenarios_20_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 40:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_40_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_600_scenarios_40_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/600_to_1000_scenarios_40_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == 80:
    with open("/home/mbultc/Downloads/0_to_300_scenarios_80_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/300_to_600_scenarios_80_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
    with open("/home/mbultc/Downloads/600_to_1000_scenarios_80_frames_single_rollout.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))
elif updated_frames == "constant velocity":
    with open("/home/mbultc/PycharmProjects/MTR/data/saved_tracks_json/1000_scenarios_constant_velocity.json", "r") as json_file:
        all_json_files_data.append(json.load(json_file))


for loaded_data in all_json_files_data:
    scenario_id_list += loaded_data['scenario_ids']
    nested_object_ids_list += loaded_data['object_ids_tensors']
    nested_rollouts_list += loaded_data['closed_loop_trajs']    # Updated

object_ids_list = [tf.convert_to_tensor(nested_list, dtype=tf.int32) for nested_list in nested_object_ids_list]
rollouts_tensors_list_single_rollout = [tf.convert_to_tensor(nested_list) for nested_list in nested_rollouts_list]
xy_pred_trajs_list = []
evaluated_agents_ids = []

# This block is for MTR-generated trajectories
'''
counter = 0
for bytes_example in dataset_iterator:
    scenario = scenario_pb2.Scenario.FromString(bytes_example)
    if scenario.scenario_id != scenario_id_list[counter]:
        continue
    object_ids = list(object_ids_list[counter])
    rollouts = rollouts_tensors_list_single_rollout[counter].numpy()
    complete_trajs_list = []
    traj_counter = 0
    evaluated_objects = submission_specs.get_evaluation_sim_agent_ids(scenario)
    evaluated_agents_ids.append(evaluated_objects)
    for track in scenario.tracks:
        if track.id in evaluated_objects:
        # if track.id in submission_specs.get_sim_agent_ids(scenario):
            x = np.array([state.center_x for state in track.states])
            y = np.array([state.center_y for state in track.states])
            x = x[:11]
            y = y[:11]
            xy_past = np.stack((x, y), axis=1)
            xy_future = rollouts[traj_counter, :, :2]
            xy_complete = np.concatenate((xy_past, xy_future), axis=0)
            complete_trajs_list.append(xy_complete)
            traj_counter += 1
    xy_pred_trajs_list.append(complete_trajs_list)
    counter += 1
    print(f'Merged {counter} scenarios')
    if counter >= 1000:
        break
    time5 = perf_counter()
'''
# This block is for constant velocity
counter = 0
for bytes_example in dataset_iterator:
    scenario = scenario_pb2.Scenario.FromString(bytes_example)
    if scenario.scenario_id != scenario_id_list[counter]:
        continue
    complete_trajs_list = []
    traj_counter = 0
    evaluated_objects = submission_specs.get_evaluation_sim_agent_ids(scenario)
    evaluated_agents_ids.append(evaluated_objects)
    for track in scenario.tracks:
        if track.id in evaluated_objects:
        # if track.id in submission_specs.get_sim_agent_ids(scenario):
            x = np.array([state.center_x for state in track.states])
            y = np.array([state.center_y for state in track.states])
            x, y = make_constant_velocity(x, y)
            xy_complete = np.stack((x, y), axis=1)
            complete_trajs_list.append(xy_complete)
            traj_counter += 1
    xy_pred_trajs_list.append(complete_trajs_list)
    counter += 1
    print(f'Made {counter} constant velocity scenarios')
    if counter >= 1000:
        break
    time5 = perf_counter()

# Now we have a list of lists where the outer list is for each scenario and the inner lists contain numpy arrays of
# size (91,2) which can individually be sent to the classify trajs function
# We need to connect each scenario with the scenario_id

output_csv_file = "evaluated_objects_turns_analysis_constant_velocity.csv"

# Write the data to the CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ["scenario_id", "evaluated_objects", "going straight", "stationary", "turning"]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row with field names
    csv_writer.writeheader()

    for i in range(1000):
        trajs = xy_pred_trajs_list[i]
        movements = {
            'scenario_id': scenario_id_list[i],
            'evaluated_objects': str(evaluated_agents_ids[i]),
            'going straight': 0,
            'stationary': 0,
            'turning': 0
        }
        for traj in trajs:
            movement = classify_traj(traj[:,0], traj[:,1], False)
            movements[movement] += 1
        csv_writer.writerow(movements)
        print(f'Classified {i} scenarios')

'''
# Define the output CSV file path
output_csv_file = "turns_analysis_ground_truth.csv"

# Write the data to the CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    fieldnames = ["scenario_id", "going straight", "stationary", "turning"]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row with field names
    csv_writer.writeheader()

    ### Iterating over scenarios and predictions and doing evaluation
    counter = 0
    for bytes_example in dataset_iterator:
        time1 = perf_counter()
        scenario = scenario_pb2.Scenario.FromString(bytes_example)
        time2 = perf_counter()
        # print(f'Loading the scenario from bytes_example: {time2 - time1}')
        if scenario.scenario_id != scenario_id_list[counter]:
            continue
        print(f'Doing evaluation for scenario number {counter} with id: {scenario.scenario_id}')

        movements = {
            'scenario_id': scenario.scenario_id,
            'going straight': 0,
            'stationary': 0,
            'turning': 0
        }
        for track in scenario.tracks:
            if track.id in submission_specs.get_sim_agent_ids(scenario):
                movement = classify_track(track, False)
                movements[movement] += 1
        csv_writer.writerow(movements)
        counter += 1
        if counter >= len(rollouts_tensors_list):
            break
'''


