import pandas as pd
import numpy as np

from waymo_open_dataset.protos import scenario_pb2
# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc
rc('animation', html='jshtml')


def calculate_average(numbers_str):
    numbers = list(map(int, numbers_str.split()))
    return sum(numbers) / len(numbers) if numbers else 0

def expand_columns(df, field_name, subfield_names):
    sub_df = df.apply(lambda x: pd.Series(x[field_name]), axis=1)
    sub_df.columns = [f'{field_name}_{sub}' for sub in subfield_names]
    return sub_df

def plot_track_trajectory(track: scenario_pb2.Track) -> None:
  valids = np.array([state.valid for state in track.states])
  if np.any(valids):
    x = np.array([state.center_x for state in track.states])
    y = np.array([state.center_y for state in track.states])
    ax.plot(x[valids], y[valids], linewidth=5)


parent_folder = "/home/mbultc/Desktop/results_metrics/"
evaluation_modes = ["1_frames", "2_frames", "5_frames", "10_frames", "20_frames", "40_frames", "80_frames", "constant_velocity"]

fieldnames = ["scenario_id", "metametric", "average_displacement_error", "linear_speed_likelihood",
                  "linear_acceleration_likelihood", "angular_speed_likelihood", "angular_acceleration_likelihood",
                  "distance_to_nearest_object_likelihood", "collision_indication_likelihood",
                  "time_to_collision_likelihood", "distance_to_road_edge_likelihood", "offroad_indication_likelihood",
                  "min_average_displacement_error", "number_of_agents", "difficulties"]

evaluation_mode = evaluation_modes[0]

df = pd.read_csv(parent_folder + "evaluation_1000_scenarios_" + evaluation_mode + ".csv")
sorted_df = df.sort_values(by='metametric')
print(df)
print(sorted_df)

"""

DATASET_FOLDER = '/home/mbultc/womd1_1/'
TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training/*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'mini_womd1_1/validation/*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'testing/*')

# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(VALIDATION_FILES)
print(filenames.shape)

scenario_number = 898

dataset = tf.data.TFRecordDataset(filenames)
# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()

for _ in range(scenario_number):
    bytes_example = next(dataset_iterator)

scenario = scenario_pb2.Scenario.FromString(bytes_example)
print(f'Checking type: {type(scenario)}')
print(f'Loaded scenario with ID: {scenario.scenario_id}')

updated_frames = 1

all_json_files_data = []
scenario_id_list = []
nested_object_ids_list = []
nested_rollouts_list = []

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


for loaded_data in all_json_files_data:
    scenario_id_list += loaded_data['scenario_ids']
    nested_object_ids_list += loaded_data['object_ids_tensors']
    nested_rollouts_list += loaded_data['closed_loop_trajs']    # Updated

object_ids_list = [tf.convert_to_tensor(nested_list, dtype=tf.int32) for nested_list in nested_object_ids_list]
rollouts_tensors_list_single_rollout = [tf.convert_to_tensor(nested_list) for nested_list in nested_rollouts_list]
rollouts_tensors_list = []
number_of_agents = []

for rollout_tensor in rollouts_tensors_list_single_rollout:
    # print(rollout_tensor.shape)
    if updated_frames != "constant velocity":
        number_of_agents.append(rollout_tensor.shape[0])
        rollout_multiplied = [rollout_tensor for _ in range(32)]
        rollouts_tensors_list.append(np.stack(rollout_multiplied))
    else:
        number_of_agents.append(rollout_tensor.shape[1])
        
updated_frames = 1

all_json_files_data = []
scenario_id_list = []
nested_object_ids_list = []
nested_rollouts_list = []

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


for loaded_data in all_json_files_data:
    scenario_id_list += loaded_data['scenario_ids']
    nested_object_ids_list += loaded_data['object_ids_tensors']
    nested_rollouts_list += loaded_data['closed_loop_trajs']    # Updated

object_ids_list = [tf.convert_to_tensor(nested_list, dtype=tf.int32) for nested_list in nested_object_ids_list]
rollouts_tensors_list_single_rollout = [tf.convert_to_tensor(nested_list) for nested_list in nested_rollouts_list]
rollouts_tensors_list = []
number_of_agents = []

for rollout_tensor in rollouts_tensors_list_single_rollout:
    # print(rollout_tensor.shape)
    if updated_frames != "constant velocity":
        number_of_agents.append(rollout_tensor.shape[0])
        rollout_multiplied = [rollout_tensor for _ in range(32)]
        rollouts_tensors_list.append(np.stack(rollout_multiplied))
    else:
        number_of_agents.append(rollout_tensor.shape[1])
        
# Visualize scenario.
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

visualizations.add_map(ax, scenario)

all_track_ids = []
for track in scenario.tracks:
  plot_track_trajectory(track)
  all_track_ids.append(track.id)

print(all_track_ids)
plt.show();

print(f'Objects to be resimulated: {submission_specs.get_sim_agent_ids(scenario)}')
print(f'Total objects to be resimulated: {len(submission_specs.get_sim_agent_ids(scenario))}')

# Plot their tracks.
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
visualizations.add_map(ax, scenario)



for track in scenario.tracks:
  if track.id in submission_specs.get_sim_agent_ids(scenario):
    plot_track_trajectory(track)

plt.show();

"""
