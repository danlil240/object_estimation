#!/usr/bin/env python3
"""
Plot Results for AIMM-CS-DUCMKF Tracking Simulation
Matches the functionality of the C++ version
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
import sys
import os

# Add the parent directory to the path to import simulation_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_simulation_data(csv_file: str = "simulation_data.csv") -> dict:
    """Load simulation data from CSV file"""
    try:
        df = pd.read_csv(csv_file)

        # Extract data
        data = {
            "time": df["Time"].values,
            "true_positions": [],
            "estimated_positions": [],
            "position_errors": df["Position_Error"].values,
            "active_models": df["Active_Model"].values,
            "maneuver_detected": df["Maneuver_Detected"].values.astype(bool),
            "wind_detected": df["Wind_Detected"].values.astype(bool),
        }

        # Parse position vectors
        for i in range(len(df)):
            # True positions
            true_pos = np.array(
                [
                    float(df.iloc[i]["True_X"]),
                    float(df.iloc[i]["True_Y"]),
                    float(df.iloc[i]["True_Z"]),
                ]
            )
            data["true_positions"].append(true_pos)

            # Estimated positions
            est_pos = np.array(
                [
                    float(df.iloc[i]["Est_X"]),
                    float(df.iloc[i]["Est_Y"]),
                    float(df.iloc[i]["Est_Z"]),
                ]
            )
            data["estimated_positions"].append(est_pos)

        return data
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run the simulation first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_results(data: dict):
    """Create comprehensive plots for simulation results"""

    # Create figure with 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("AIMM-CS-DUCMKF Tracking Results", fontsize=16, fontweight="bold")

    # Extract position data
    true_x = [pos[0] for pos in data["true_positions"]]
    true_y = [pos[1] for pos in data["true_positions"]]
    true_z = [pos[2] for pos in data["true_positions"]]

    est_x = [pos[0] for pos in data["estimated_positions"]]
    est_y = [pos[1] for pos in data["estimated_positions"]]
    est_z = [pos[2] for pos in data["estimated_positions"]]

    time = data["time"]
    pos_err = data["position_errors"]

    # --- Plot 1: 2D Trajectory (X-Y) ---
    ax1 = axes[0, 0]
    ax1.plot(true_x, true_y, "b-", linewidth=2, label="True Trajectory")
    ax1.plot(est_x, est_y, "r--", linewidth=2, label="Estimated Trajectory")
    ax1.scatter(true_x[0], true_y[0], c="green", s=100, label="Start", zorder=5)
    ax1.scatter(true_x[-1], true_y[-1], c="red", s=100, label="End", zorder=5)
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("2D Trajectory (X-Y)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis("equal")

    # --- Plot 2: 2D Trajectory (X-Z) ---
    ax2 = axes[0, 1]
    ax2.plot(true_x, true_z, "b-", linewidth=2, label="True Trajectory")
    ax2.plot(est_x, est_z, "r--", linewidth=2, label="Estimated Trajectory")
    ax2.scatter(true_x[0], true_z[0], c="green", s=100, label="Start", zorder=5)
    ax2.scatter(true_x[-1], true_z[-1], c="red", s=100, label="End", zorder=5)
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Z Position (m)")
    ax2.set_title("2D Trajectory (X-Z)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis("equal")

    # --- Plot 3: Position Error Over Time ---
    ax3 = axes[0, 2]
    ax3.plot(time, pos_err, "b-", linewidth=1.5, label="Position Error")

    # Highlight maneuver detection points
    maneuver_mask = data["maneuver_detected"]
    if np.any(maneuver_mask):
        ax3.scatter(
            time[maneuver_mask],
            [1.0] * np.sum(maneuver_mask),
            c="red",
            s=30,
            label="Maneuver Detected",
            alpha=0.7,
        )

    # Highlight wind detection points
    wind_mask = data["wind_detected"]
    if np.any(wind_mask):
        ax3.scatter(
            time[wind_mask],
            pos_err[wind_mask],
            c="green",
            s=30,
            label="Wind Detected",
            alpha=0.7,
        )

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title("Position Error Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # --- Plot 4: Model Selection Timeline ---
    ax4 = axes[1, 0]
    model_names = {0: "CV", 1: "CT", 2: "CA"}
    colors = {0: "blue", 1: "orange", 2: "green"}

    for model_id in [0, 1, 2]:
        mask = data["active_models"] == model_id
        if np.any(mask):
            ax4.scatter(
                time[mask],
                [model_id] * np.sum(mask),
                c=colors[model_id],
                s=40,
                label=model_names[model_id],
                alpha=0.7,
            )

    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Model Type")
    ax4.set_title("Model Selection Timeline")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["CV", "CT", "CA"])
    ax4.legend()

    # --- Plot 5: Detection Timeline ---
    ax5 = axes[1, 1]

    # Plot maneuver detection
    if np.any(maneuver_mask):
        ax5.scatter(
            time[maneuver_mask],
            [1.0] * np.sum(maneuver_mask),
            c="red",
            s=50,
            label="Maneuver",
            alpha=0.7,
        )

    # Plot wind detection
    if np.any(wind_mask):
        ax5.scatter(
            time[wind_mask],
            [0.0] * np.sum(wind_mask),
            c="green",
            s=50,
            label="Wind",
            alpha=0.7,
        )

    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Detection Type")
    ax5.set_title("Detection Timeline")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.5, 1.5)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(["Wind", "Maneuver"])
    ax5.legend()

    # --- Plot 6: Performance Summary ---
    ax6 = axes[1, 2]

    # Position error histogram
    ax6.hist(pos_err, bins=20, alpha=0.7, color="skyblue", edgecolor="black")

    # Add mean and median lines
    mean_error = np.mean(pos_err)
    median_error = np.median(pos_err)

    ax6.axvline(
        mean_error,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_error:.2f}m",
    )
    ax6.axvline(
        median_error, color="orange", linewidth=2, label=f"Median: {median_error:.2f}m"
    )

    ax6.set_xlabel("Position Error (m)")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Position Error Distribution")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("tracking_results.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'tracking_results.png'")

    # Show the plot
    plt.show()


def print_performance_summary(data: dict):
    """Print performance metrics"""
    pos_err = data["position_errors"]

    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Mean Position Error: {np.mean(pos_err):.2f} m")
    print(f"Median Position Error: {np.median(pos_err):.2f} m")
    print(f"Std Position Error: {np.std(pos_err):.2f} m")
    print(f"Max Position Error: {np.max(pos_err):.2f} m")
    print(f"Min Position Error: {np.min(pos_err):.2f} m")

    # Model usage statistics
    model_counts = np.bincount(data["active_models"])
    model_names = {0: "CV", 1: "CT", 2: "CA"}
    print(f"\nModel Usage:")
    for i, count in enumerate(model_counts):
        if count > 0:
            percentage = (count / len(data["active_models"])) * 100
            print(f"  {model_names[i]}: {count} times ({percentage:.1f}%)")

    # Detection statistics
    maneuver_count = np.sum(data["maneuver_detected"])
    wind_count = np.sum(data["wind_detected"])
    total_time = len(data["time"])

    print(f"\nDetection Statistics:")
    print(
        f"  Maneuver Detected: {maneuver_count} times ({(maneuver_count/total_time)*100:.1f}%)"
    )
    print(f"  Wind Detected: {wind_count} times ({(wind_count/total_time)*100:.1f}%)")
    print("=" * 50)


def main():
    """Main function"""
    print("AIMM-CS-DUCMKF Plotting Script")
    print("=" * 40)

    # Load simulation data
    data = load_simulation_data()
    if data is None:
        return

    print(f"Loaded {len(data['time'])} data points")
    print(f"Time range: {data['time'][0]:.1f}s to {data['time'][-1]:.1f}s")

    # Create plots
    plot_results(data)

    # Print performance summary
    print_performance_summary(data)


if __name__ == "__main__":
    main()
