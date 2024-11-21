import numpy as np
import matplotlib.pyplot as plt
from lab04.rosbag2_reader_py import Rosbag2Reader
from nav_msgs.msg import Odometry


def extract_data(bag, topic_name):
    time_stamps = []
    x_positions = []
    y_positions = []
    orientations = []

    for topic, msg, t in bag:
        if topic == topic_name and isinstance(msg, Odometry):
            time_stamps.append(t * 1e-9) 
            x_positions.append(msg.pose.pose.position.x)
            y_positions.append(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            theta = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
            orientations.append(theta)

    return np.array(time_stamps), np.array(x_positions), np.array(y_positions), np.array(orientations)


def plot_localization(gt, odom, ekf):
    time_gt, x_gt, y_gt, theta_gt = gt
    time_odom, x_odom, y_odom, theta_odom = odom
    time_ekf, x_ekf, y_ekf, theta_ekf = ekf

    # Plot (x, y)
    plt.figure(figsize=(8, 6))
    plt.plot(x_gt, y_gt, label="Ground Truth", color="blue")
    plt.plot(x_odom, y_odom, label="Odometry", color="orange", linestyle="--")
    plt.plot(x_ekf, y_ekf, label="EKF", color="green", linestyle=":")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title("Traiettoria 2D")
    plt.grid()
    plt.show()

    # Plot x, y, theta e tempo
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time_gt, x_gt, label="Ground Truth", color="blue")
    axs[0].plot(time_odom, x_odom, label="Odometry", color="orange", linestyle="--")
    axs[0].plot(time_ekf, x_ekf, label="EKF", color="green", linestyle=":")
    axs[0].set_ylabel("x (m)")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time_gt, y_gt, label="Ground Truth", color="blue")
    axs[1].plot(time_odom, y_odom, label="Odometry", color="orange", linestyle="--")
    axs[1].plot(time_ekf, y_ekf, label="EKF", color="green", linestyle=":")
    axs[1].set_ylabel("y (m)")
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(time_gt, theta_gt, label="Ground Truth", color="blue")
    axs[2].plot(time_odom, theta_odom, label="Odometry", color="orange", linestyle="--")
    axs[2].plot(time_ekf, theta_ekf, label="EKF", color="green", linestyle=":")
    axs[2].set_ylabel("Theta (rad)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def calculate_rmse(true, estimate):
    """Calcola il Root Mean Square Error (RMSE)."""
    return np.sqrt(np.mean((true - estimate) ** 2))


def calculate_mae(true, estimate):
    """Calcola il Mean Absolute Error (MAE)."""
    return np.mean(np.abs(true - estimate))


def main():
    bag_path = "/home/federico/lecture_ws/rosbag2_2024_11_17-19_17_39"
    bag = Rosbag2Reader(bag_path)

    # Estrai dati dai topic
    gt_data = extract_data(bag, "/ground_truth")
    odom_data = extract_data(bag, "/odom")
    ekf_data = extract_data(bag, "/ekf")

    # Genera i plot
    plot_localization(gt_data, odom_data, ekf_data)

    #Matrici
    rmse_x = calculate_rmse(gt_data[1], ekf_data[1])
    mae_x = calculate_mae(gt_data[1], ekf_data[1])
    print(f"RMSE (x): {rmse_x:.4f}, MAE (x): {mae_x:.4f}")

if __name__ == "__main__":
    main()
