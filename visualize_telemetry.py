import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
df = pd.read_csv("build/flight_telemetry.csv")

track_counts = df['Track_ID'].value_counts()
valid_tracks = track_counts[track_counts > 10].index
filtered_df = df[df['Track_ID'].isin(valid_tracks)]

plt.figure(figsize=(12, 8))

for track_id in valid_tracks:
    track_data = filtered_df[filtered_df['Track_ID'] == track_id]
    
    plt.plot(track_data['X_Center'], track_data['Y_Center'], 
             marker='o', markersize=3, linestyle='-', linewidth=2, 
             label=f'Target ID: {track_id}')
    
    start_point = track_data.iloc[0]
    plt.scatter(start_point['X_Center'], start_point['Y_Center'], 
                color='red', s=100, zorder=5, edgecolors='black')

plt.gca().invert_yaxis()

plt.title('AlphaTracker: Spatial Flight Trajectories', fontsize=16, fontweight='bold')
plt.xlabel('X Coordinate (Pixels)', fontsize=12)
plt.ylabel('Y Coordinate (Pixels)', fontsize=12)
plt.legend(title='Unique Entities')
plt.tight_layout()

plt.savefig("flight_paths.png", dpi=300)
plt.show()