import pandas as pd
import csv

home_dir = '/home/mbultc/Desktop/results_metrics/'
frame_rates = [1, 2, 5, 10, 20, 40, 80, "constant_velocity", "ground_truth"]
iterations = []
fieldnames = ["eval_mode", "avg_moving_perc_mean", "avg_moving_perc_std", "avg_turning_perc_mean", "avg_turning_perc_std"]

for frame_rate in frame_rates:
    row_data = {"eval_mode": str(frame_rate)}
    df = pd.read_csv(home_dir + "expanded_turns_analysis_" + str(frame_rate) + ".csv")
    row_data["avg_moving_perc_mean"] = df["moving_perc"].mean()
    row_data["avg_moving_perc_std"] = df["moving_perc"].std()
    row_data["avg_turning_perc_mean"] = df["turning_perc"].mean()
    row_data["avg_turning_perc_std"] = df["turning_perc"].std()
    iterations.append(row_data)

# Define the CSV file name
csv_file = 'avg_moving_and_turning_perc.csv'

# Open the CSV file in write mode first to write the headers
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for iteration in iterations:
        writer.writerow(iteration)

print(f"Data has been written to {csv_file}")
