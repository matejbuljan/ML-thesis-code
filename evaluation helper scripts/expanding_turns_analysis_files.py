import pandas as pd


home_dir = '/home/mbultc/Desktop/results_metrics/'
frame_rates = [1, 2, 5, 10, 20, 40, 80, "constant_velocity"]
turns_data = ['going straight', 'stationary', 'turning', 'moving', 'moving_perc', 'turning_perc']

'''
for frame_rate in frame_rates:
    if frame_rate != "ground_truth":
        df = pd.read_csv(home_dir + 'evaluated_objects_turns_analysis_' + str(frame_rate) + '.csv')

        df['moving'] = df['going straight'] + df['turning']
        df['moving_perc'] = df['moving'] / (df['going straight'] + df['stationary'] + df['turning'])
        df['turning_perc'] = df['turning'] / df['moving']

        df.to_csv('expanded_evaluated_objects_turns_analysis_' + str(frame_rate) + '.csv', index=False)

    if frame_rate not in ['constant_velocity', 'ground_truth']:
        df = pd.read_csv(home_dir + 'turns_analysis_' + str(frame_rate) + '_frames.csv')
    else:
        df = pd.read_csv(home_dir + 'turns_analysis_' + frame_rate + '.csv')

    df['moving'] = df['going straight'] + df['turning']
    df['moving_perc'] = df['moving'] / (df['going straight'] + df['stationary'] + df['turning'])
    df['turning_perc'] = df['turning'] / df['moving']
    df.to_csv('expanded_turns_analysis_' + str(frame_rate) + '.csv', index=False)

'''

gt_turns_df = pd.read_csv(home_dir + 'expanded_turns_analysis_ground_truth.csv')
gt_turns_df['moving'] = gt_turns_df['going straight'] + gt_turns_df['turning']
gt_turns_df['moving_perc'] = gt_turns_df['moving'] / (gt_turns_df['going straight'] + gt_turns_df['stationary'] + gt_turns_df['turning'])
gt_turns_df['turning_perc'] = gt_turns_df['turning'] / gt_turns_df['moving']

for frame_rate in frame_rates:
    if frame_rate != "constant_velocity":
        main_df = pd.read_csv(home_dir + 'evaluation_1000_scenarios_' + str(frame_rate) +'_frames.csv')
        all_turns_df = pd.read_csv(home_dir + 'expanded_turns_analysis_' + str(frame_rate) + '.csv')
        eval_turns_df = pd.read_csv(home_dir + 'expanded_evaluated_objects_turns_analysis_' + str(frame_rate) + '.csv')
    else:
        main_df = pd.read_csv(home_dir + 'evaluation_1000_scenarios_constant_velocity.csv')
        all_turns_df = pd.read_csv(home_dir + 'expanded_turns_analysis_constant_velocity.csv')
        eval_turns_df = pd.read_csv(home_dir + 'expanded_evaluated_objects_turns_analysis_constant_velocity.csv')

    for option in turns_data:
        main_df['all_' + option] = all_turns_df[option]
        main_df['eval_' + option] = eval_turns_df[option]

    main_df['moving_perc_rel_to_gt'] = main_df['all_moving_perc'] / gt_turns_df['moving_perc']
    main_df['turning_perc_rel_to_gt'] = main_df['all_turning_perc'] / gt_turns_df['turning_perc']

    print(main_df.head())
    main_df.to_csv('expanded_evaluation_1000_scenarios_' + str(frame_rate) + '.csv', index=False)
