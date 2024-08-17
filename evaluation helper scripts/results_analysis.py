import pandas as pd


def calculate_average(numbers_str):
    numbers = list(map(int, numbers_str.split()))
    return sum(numbers) / len(numbers) if numbers else 0

def expand_columns(df, field_name, subfield_names):
    sub_df = df.apply(lambda x: pd.Series(x[field_name]), axis=1)
    sub_df.columns = [f'{field_name}_{sub}' for sub in subfield_names]
    return sub_df



parent_folder = "/home/mbultc/Desktop/results_metrics/"
evaluation_modes = ["1_frames", "2_frames", "5_frames", "10_frames", "20_frames", "40_frames", "80_frames", "constant_velocity"]

fieldnames = ["scenario_id", "metametric", "average_displacement_error", "linear_speed_likelihood",
                  "linear_acceleration_likelihood", "angular_speed_likelihood", "angular_acceleration_likelihood",
                  "distance_to_nearest_object_likelihood", "collision_indication_likelihood",
                  "time_to_collision_likelihood", "distance_to_road_edge_likelihood", "offroad_indication_likelihood",
                  "min_average_displacement_error", "number_of_agents", "difficulties"]

results = {}
results["evaluation_modes"] = evaluation_modes
for fieldname in fieldnames:
    if fieldname not in ["scenario_id", "number_of_agents", "difficulties"]:
        results[fieldname] = []

for evaluation_mode in evaluation_modes:
    df = pd.read_csv(parent_folder + "evaluation_1000_scenarios_" + evaluation_mode + ".csv")
    df['avg_difficulty'] = df['difficulties'].apply(calculate_average)

    for fieldname in fieldnames:
        if fieldname not in ["scenario_id", "number_of_agents", "difficulties", "avg_difficulty"]:
            field_mean = df[fieldname].mean()
            field_std = df[fieldname].std()
            correlation_to_number_of_agents = df[[fieldname, "number_of_agents"]].corr().iloc[0, 1]
            correlation_to_avg_difficulty = df[[fieldname, "avg_difficulty"]].corr().iloc[0, 1]
            results[fieldname].append([field_mean, field_std, correlation_to_number_of_agents, correlation_to_avg_difficulty])


df = pd.DataFrame.from_dict(results)
print(df)

subfields = ["mean", "std", "correlation_to_number_of_agents", "correlation_to_avg_difficulty"]
expanded_columns = [df["evaluation_modes"]]
for fieldname in fieldnames:
    if fieldname not in ["scenario_id", "number_of_agents", "difficulties", "avg_difficulty"]:
        expanded_columns.append(expand_columns(df, fieldname, subfields))
expanded_df = pd.concat(expanded_columns, axis=1)

print(expanded_df)

expanded_df.to_csv('expanded_results.csv', index=False)


