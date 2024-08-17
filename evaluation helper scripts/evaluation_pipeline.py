### Imports
import csv
import os
import tarfile
import numpy as np
import tensorflow as tf
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
updated_frames = 2
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
rollouts_tensors_list = []
number_of_agents = []

for rollout_tensor in rollouts_tensors_list_single_rollout:
    print(rollout_tensor.shape)
    if updated_frames != "constant velocity":
        number_of_agents.append(rollout_tensor.shape[0])
        rollout_multiplied = [rollout_tensor for _ in range(32)]
        rollouts_tensors_list.append(np.stack(rollout_multiplied))
    else:
        number_of_agents.append(rollout_tensor.shape[1])
print(number_of_agents[380:390])
bin_edges = list(range(0, 240, 10))
counts, bins, patches = plt.hist(number_of_agents, bins=bin_edges, edgecolor='black')
# Add labels to each bin
for count, x in zip(counts, bins):
    plt.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom')
plt.title('Number of agents in scenarios. Updated frames: ' + str(updated_frames))
plt.show()

print(f'Number of predictions: {len(object_ids_list)}')

results = []

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

    eval_tracks_and_difficulty = {
        'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
        'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
    }
    # print(eval_tracks_and_difficulty)

    simulated_states = rollouts_tensors_list[counter]
    object_ids = object_ids_list[counter]

    scenario_rollouts = scenario_rollouts_from_states(scenario, simulated_states, object_ids)
    time3 = perf_counter()
    # print(f'Loading simulated states, track difficulties, making scenario_rollouts: {time3 - time2}')
    # As before, we can validate the message we just generate.
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)
    time4 = perf_counter()
    # print(f'Validating scenario rollout: {time4 - time3}')

    # Load the test configuration.
    config = load_metrics_config()
    time5 = perf_counter()
    # print(f'Loading metrics config: {time5 - time4}')

    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)
    time6 = perf_counter()
    # print(f'Computing metrics for scenario: {time6 - time5}')
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
    time7 = perf_counter()
    # print(f'Adding results to list: {time7 - time6}')
    print(f'Total time for one scenario: {time7 - time1}')

    counter += 1
    if counter >= len(rollouts_tensors_list):
        break

# Define the output CSV file path
output_csv_file = "evaluation_1000_scenarios_2_frames.csv"

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
