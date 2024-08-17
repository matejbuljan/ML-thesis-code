import pandas as pd
import csv

metrics = ["metametric", "average_displacement_error", "linear_speed_likelihood",
                  "linear_acceleration_likelihood", "angular_speed_likelihood", "angular_acceleration_likelihood",
                  "distance_to_nearest_object_likelihood", "collision_indication_likelihood",
                  "time_to_collision_likelihood", "distance_to_road_edge_likelihood", "offroad_indication_likelihood"]
home_dir = '/home/mbultc/Desktop/results_metrics/'
frame_rates = [1, 2, 5, 10, 20, 40, 80, "constant_velocity"]
iterations = []
new_field_names = ['evaluation mode']
df = pd.read_csv(home_dir + "expanded_evaluation_1000_scenarios_" + str(1) + ".csv")
fields = df.columns.tolist()
for metric in metrics:
    for field in fields:
        if (field not in metrics) and (field not in ['scenario_id', 'difficulties', 'min_average_displacement_error',
                                                     'moving_perc_rel_to_gt', 'turning_perc_rel_to_gt']):
            new_field_names.append(metric + "_to_" + field)

for frame_rate in frame_rates:
    row_correlations = {}
    row_correlations['evaluation mode'] = str(frame_rate)
    df = pd.read_csv(home_dir + "expanded_evaluation_1000_scenarios_" + str(frame_rate) + ".csv")
    fields = df.columns.tolist()
    for metric in metrics:
        for field in fields:
            if (field not in metrics) and (field not in ['scenario_id', 'difficulties', 'min_average_displacement_error',
                                                         'moving_perc_rel_to_gt', 'turning_perc_rel_to_gt']):
                row_correlations[metric + "_to_" + field] = df[[field, metric]].corr().iloc[0, 1]
    iterations.append(row_correlations)

# Define the CSV file name
csv_file = 'correlations.csv'

# Open the CSV file in write mode first to write the headers
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=new_field_names)
    writer.writeheader()
    for iteration in iterations:
        writer.writerow(iteration)

print(f"Data has been written to {csv_file}")