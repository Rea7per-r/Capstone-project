import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_distance(x1, y1, z1, x2, y2, z2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_path_length(df):
    """
    Calculate path length: L = Σ |pos(t+Δt) - pos(t)|
    Sum of all incremental distances along the path.
    """
    length = 0.0
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        dist = calculate_distance(
            prev['field.pose.pose.position.x'],
            prev['field.pose.pose.position.y'],
            prev['field.pose.pose.position.z'],
            curr['field.pose.pose.position.x'],
            curr['field.pose.pose.position.y'],
            curr['field.pose.pose.position.z']
        )
        length += dist
    
    return length

def calculate_straight_line_distance(df):
    """
    Calculate straight-line distance from start to goal.
    Direct Euclidean distance between first and last point.
    """
    if len(df) < 2:
        return 0.0
    
    start = df.iloc[0]
    end = df.iloc[-1]
    
    return calculate_distance(
        start['field.pose.pose.position.x'],
        start['field.pose.pose.position.y'],
        start['field.pose.pose.position.z'],
        end['field.pose.pose.position.x'],
        end['field.pose.pose.position.y'],
        end['field.pose.pose.position.z']
    )

def calculate_mission_time(df):
    """
    Calculate mission time: T = end_time - start_time
    Convert from nanoseconds to seconds.
    """
    if len(df) < 2:
        return 0.0
    
    start_time = df.iloc[0]['%time']
    end_time = df.iloc[-1]['%time']
    
    # Convert nanoseconds to seconds
    return (end_time - start_time) / 1e9

def calculate_max_deviation(df_nominal, df_actual):
    """
    Calculate maximum lateral deviation between paths.
    Compares point-by-point distances.
    """
    min_length = min(len(df_nominal), len(df_actual))
    max_dev = 0.0
    
    for i in range(min_length):
        nom = df_nominal.iloc[i]
        act = df_actual.iloc[i]
        
        dev = calculate_distance(
            nom['field.pose.pose.position.x'],
            nom['field.pose.pose.position.y'],
            nom['field.pose.pose.position.z'],
            act['field.pose.pose.position.x'],
            act['field.pose.pose.position.y'],
            act['field.pose.pose.position.z']
        )
        max_dev = max(max_dev, dev)
    
    return max_dev

def calculate_smoothness(df):
    """
    Calculate path smoothness (total orientation change).
    Sum of quaternion differences between consecutive poses.
    Lower values indicate smoother paths.
    """
    total_change = 0.0
    
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        angular_change = np.sqrt(
            (curr['field.pose.pose.orientation.x'] - prev['field.pose.pose.orientation.x'])**2 +
            (curr['field.pose.pose.orientation.y'] - prev['field.pose.pose.orientation.y'])**2 +
            (curr['field.pose.pose.orientation.z'] - prev['field.pose.pose.orientation.z'])**2 +
            (curr['field.pose.pose.orientation.w'] - prev['field.pose.pose.orientation.w'])**2
        )
        total_change += angular_change
    
    return total_change

def analyze_paths(nominal_csv, actual_csv):
    """
    Main function to analyze and compare two paths.
    
    Args:
        nominal_csv: Path to nominal (baseline/straight) path CSV
        actual_csv: Path to actual (with avoidance) path CSV
    
    Returns:
        Dictionary containing all performance metrics
    """
    print(f"\n{'='*70}")
    print("PATH PERFORMANCE METRICS CALCULATOR")
    print(f"{'='*70}\n")
    
    # Load CSV files
    print(f"Loading nominal path: {nominal_csv}")
    df_nominal = pd.read_csv(nominal_csv)
    df_nominal.columns = df_nominal.columns.str.strip()  # Remove whitespace from headers
    
    print(f"Loading actual path: {actual_csv}")
    df_actual = pd.read_csv(actual_csv)
    df_actual.columns = df_actual.columns.str.strip()
    
    print(f"\nNominal path samples: {len(df_nominal)}")
    print(f"Actual path samples: {len(df_actual)}\n")
    
    # 1. PATH LENGTH ANALYSIS
    print(f"{'='*70}")
    print("1. PATH LENGTH ANALYSIS")
    print(f"{'='*70}")
    
    L_nominal = calculate_path_length(df_nominal)
    L_actual = calculate_path_length(df_actual)
    L_straight_nominal = calculate_straight_line_distance(df_nominal)
    L_straight_actual = calculate_straight_line_distance(df_actual)
    
    print(f"L_nominal (Baseline path length):           {L_nominal:.4f} m")
    print(f"  └─ Formula: Σ |pos(t+Δt) - pos(t)|")
    print(f"L_actual (Actual path length):              {L_actual:.4f} m")
    print(f"L_straight (Nominal start to goal):         {L_straight_nominal:.4f} m")
    print(f"L_straight (Actual start to goal):          {L_straight_actual:.4f} m")
    
    # 2. DETOUR DISTANCE
    print(f"\n{'='*70}")
    print("2. DETOUR DISTANCE ANALYSIS")
    print(f"{'='*70}")
    
    detour = L_actual - L_nominal
    detour_percentage = (detour / L_nominal) * 100
    max_deviation = calculate_max_deviation(df_nominal, df_actual)
    
    print(f"Detour = L_actual - L_nominal:              {detour:.4f} m")
    print(f"Detour percentage:                          {detour_percentage:.2f}%")
    print(f"Maximum lateral deviation:                  {max_deviation:.4f} m")
    
    # 3. PATH EFFICIENCY RATIO
    print(f"\n{'='*70}")
    print("3. PATH EFFICIENCY RATIO")
    print(f"{'='*70}")
    
    efficiency = L_nominal / L_actual
    efficiency_percentage = efficiency * 100
    efficiency_vs_straight_nominal = L_straight_nominal / L_nominal
    efficiency_vs_straight_actual = L_straight_actual / L_actual
    
    print(f"Efficiency = L_nominal / L_actual:          {efficiency:.4f}")
    print(f"Efficiency percentage:                      {efficiency_percentage:.2f}%")
    print(f"  └─ 1.0 = optimal, <1.0 = inefficient")
    print(f"Nominal path vs straight line:              {efficiency_vs_straight_nominal:.4f}")
    print(f"Actual path vs straight line:               {efficiency_vs_straight_actual:.4f}")
    
    if efficiency >= 0.95:
        efficiency_rating = "EXCELLENT"
    elif efficiency >= 0.85:
        efficiency_rating = "GOOD"
    else:
        efficiency_rating = "SUBOPTIMAL"
    print(f"Path efficiency rating:                     {efficiency_rating}")
    
    # 4. TIME TO GOAL
    print(f"\n{'='*70}")
    print("4. TIME TO GOAL ANALYSIS")
    print(f"{'='*70}")
    
    T_nominal = calculate_mission_time(df_nominal)
    T_actual = calculate_mission_time(df_actual)
    delay = T_actual - T_nominal
    delay_percentage = (delay / T_nominal) * 100 if T_nominal > 0 else 0
    
    print(f"T_nominal (Baseline mission time):          {T_nominal:.4f} s")
    print(f"  └─ Formula: mission_end_time - mission_start_time")
    print(f"T_actual (Actual mission time):             {T_actual:.4f} s")
    print(f"Delay = T_actual - T_nominal:               {delay:.4f} s")
    print(f"Delay percentage:                           {delay_percentage:.2f}%")
    
    # 5. VELOCITY ANALYSIS
    print(f"\n{'='*70}")
    print("5. VELOCITY ANALYSIS")
    print(f"{'='*70}")
    
    avg_velocity_nominal = L_nominal / T_nominal if T_nominal > 0 else 0
    avg_velocity_actual = L_actual / T_actual if T_actual > 0 else 0
    velocity_diff = avg_velocity_actual - avg_velocity_nominal
    
    print(f"Average velocity (nominal):                 {avg_velocity_nominal:.4f} m/s")
    print(f"Average velocity (actual):                  {avg_velocity_actual:.4f} m/s")
    print(f"Velocity difference:                        {velocity_diff:.4f} m/s")
    
    # 6. PATH SMOOTHNESS
    print(f"\n{'='*70}")
    print("6. PATH SMOOTHNESS ANALYSIS")
    print(f"{'='*70}")
    
    smoothness_nominal = calculate_smoothness(df_nominal)
    smoothness_actual = calculate_smoothness(df_actual)
    smoothness_diff = smoothness_actual - smoothness_nominal
    
    print(f"Nominal path smoothness:                    {smoothness_nominal:.4f}")
    print(f"Actual path smoothness:                     {smoothness_actual:.4f}")
    print(f"Additional maneuvering:                     {smoothness_diff:.4f}")
    print(f"  └─ Lower values indicate smoother paths")
    
    # SUMMARY
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Path Efficiency:                          {efficiency_percentage:.2f}% ({efficiency_rating})")
    print(f"✓ Extra Distance Due to Avoidance:          {detour:.2f} m (+{detour_percentage:.2f}%)")
    print(f"✓ Time Delay Due to Avoidance:              {delay:.2f} s (+{delay_percentage:.2f}%)")
    print(f"{'='*70}\n")
    
    # Return metrics as dictionary
    return {
        'path_length': {
            'nominal': L_nominal,
            'actual': L_actual,
            'straight_nominal': L_straight_nominal,
            'straight_actual': L_straight_actual
        },
        'detour': {
            'absolute': detour,
            'percentage': detour_percentage,
            'max_deviation': max_deviation
        },
        'efficiency': {
            'ratio': efficiency,
            'percentage': efficiency_percentage,
            'vs_straight_nominal': efficiency_vs_straight_nominal,
            'vs_straight_actual': efficiency_vs_straight_actual,
            'rating': efficiency_rating
        },
        'time': {
            'nominal': T_nominal,
            'actual': T_actual,
            'delay': delay,
            'delay_percentage': delay_percentage
        },
        'velocity': {
            'nominal': avg_velocity_nominal,
            'actual': avg_velocity_actual,
            'difference': velocity_diff
        },
        'smoothness': {
            'nominal': smoothness_nominal,
            'actual': smoothness_actual,
            'difference': smoothness_diff
        },
        'samples': {
            'nominal': len(df_nominal),
            'actual': len(df_actual)
        }
    }

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python path_metrics.py <nominal_path.csv> <actual_path.csv>")
        print("\nExample:")
        print("  python path_metrics.py straight_path.csv deviated_path.csv")
        print("\nWhat this script does:")
        print("  1. Loads two CSV files (nominal/baseline and actual/deviated paths)")
        print("  2. Calculates path lengths using position data")
        print("  3. Computes detour distance (extra distance due to avoidance)")
        print("  4. Calculates path efficiency ratio")
        print("  5. Measures time delays and velocity differences")
        print("  6. Analyzes path smoothness and deviations")
        sys.exit(1)
    
    nominal_csv = sys.argv[1]
    actual_csv = sys.argv[2]
    
    # Check if files exist
    if not Path(nominal_csv).exists():
        print(f"Error: File not found - {nominal_csv}")
        sys.exit(1)
    
    if not Path(actual_csv).exists():
        print(f"Error: File not found - {actual_csv}")
        sys.exit(1)
    
    # Analyze paths
    try:
        metrics = analyze_paths(nominal_csv, actual_csv)
        
        # Optionally save results to file
        print("\nDo you want to save results to a file? (y/n): ", end='')
        # For automated runs, comment out the input and set to 'n'
        # save_choice = 'n'
        save_choice = input().strip().lower()
        
        if save_choice == 'y':
            output_file = 'performance_metrics_results.txt'
            print(f"\nResults saved to: {output_file}")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
