import pandas as pd
import csv

home_dir = '/home/mbultc/Desktop/results_metrics/'
frame_rates = [1, 2, 5, 10, 20, 40, 80, "constant_velocity"]
df = pd.read_csv(home_dir + "expanded_evaluation_1000_scenarios_" + str(1) + ".csv")
fields = df.columns.tolist()
scenario_numbers = [1, 14, 15, 23]
iterations = []

for scenario in scenario_numbers:
    for frame_rate in frame_rates:
        df = pd.read_csv(home_dir + "expanded_evaluation_1000_scenarios_" + str(frame_rate) + ".csv")
        iterations.append(df.iloc[scenario - 1].to_dict())

# Define the CSV file name
csv_file = 'scenarios_to_visualise_metrics.csv'

# Open the CSV file in write mode first to write the headers
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for iteration in iterations:
        writer.writerow(iteration)

print(f"Data has been written to {csv_file}")
