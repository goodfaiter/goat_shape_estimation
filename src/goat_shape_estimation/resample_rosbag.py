from rosbags.highlevel import AnyReader
from pathlib import Path
import pandas as pd
import numpy as np
import csv


def extract_message_data(msg):
    """Extract relevant data from ROS message - customize per your message types"""
    data = {}

    if hasattr(msg, "angular_velocity"):
        data["orientation_w"] = msg.orientation.w
        data["orientation_x"] = msg.orientation.x
        data["orientation_y"] = msg.orientation.y
        data["orientation_z"] = msg.orientation.z
        data["angular_velocity_x"] = msg.angular_velocity.x
        data["angular_velocity_y"] = msg.angular_velocity.y
        data["angular_velocity_z"] = msg.angular_velocity.z
        data["linear_acceleration_x"] = msg.linear_acceleration.x
        data["linear_acceleration_y"] = msg.linear_acceleration.y
        data["linear_acceleration_z"] = msg.linear_acceleration.z

    if hasattr(msg, "data"):
        if isinstance(msg.data, (list, np.ndarray)):
            for i, val in enumerate(msg.data):
                data[f"data_{i}"] = float(val)
        else:
            data["data"] = float(msg.data)

    # Add more message type specific extraction as needed
    return data


def read_specific_cell(csv_file, row_num, col_num):
    with open(csv_file, "r", newline="") as file:
        reader = csv.reader(file)
        for current_row, row in enumerate(reader):
            if current_row == row_num:
                try:
                    return row[col_num]
                except IndexError:
                    return None
    return None


def resample_rosbag2_and_csv(bag_path, topics=None, target_hz=10.0):
    """
    Resample ROS 2 bag data to target frequency using nearest-neighbor interpolation.

    Args:
        bag_path: Path to ROS 2 bag directory
        target_hz: Target frequency in Hz (default: 10.0)
        topics: List of topics to include (None for all topics)

    Returns:
        pd.DataFrame: Synchronized dataset with datetime index
    """
    # Create target frequency timedelta
    target_period = pd.Timedelta(seconds=1 / target_hz)

    # Mocap process
    # NOTE(VY): We assume headers are manually written correctly in the CSV.
    mocap_start_time = read_specific_cell(Path(bag_path + ".csv"), 0, 3)[5:]
    mocap_start_time = pd.to_datetime(mocap_start_time, format="%Y-%m-%d %I.%M.%S %p")
    mocap_start_time = mocap_start_time - pd.Timedelta(hours=2)
    mocap_df = pd.read_csv(Path(bag_path + ".csv"), header=7)
    for i, dt in enumerate(mocap_df["Time (Seconds)"]):
        mocap_df["Time (Seconds)"][i] = mocap_start_time + pd.Timedelta(seconds=dt)
    mocap_df.set_index("Time (Seconds)", inplace=True)

    mocap_min_ts = mocap_df.index[0].value
    mocap_max_ts = mocap_df.index[-1].value

    # Read bag file
    with AnyReader([Path(bag_path)]) as reader:
        # Get connections (topics) to process
        conns = [conn for conn in reader.connections if topics is None or conn.topic in topics]

        # First pass: collect all timestamps and pick the "narrow-est" timeframe
        rosbag_min_ts = 0
        rosbag_max_ts = float("inf")
        for conn in conns:
            timestamps = list(reader.messages(connections=[conn]))
            rosbag_min_ts = max(rosbag_min_ts, timestamps[0][1])
            rosbag_max_ts = min(rosbag_max_ts, timestamps[-1][1])

        # Create target index
        min_ts = max(mocap_min_ts, rosbag_min_ts)
        max_ts = min(mocap_max_ts, rosbag_max_ts)
        target_index = pd.date_range(
            start=pd.to_datetime(min_ts, unit="ns"),
            end=pd.to_datetime(max_ts, unit="ns"),
            freq=target_period,
        )

        # Create final data frame with required times
        result = pd.DataFrame(index=target_index)

        # Second pass: collect and resample data
        for conn in conns:
            topic_data = []
            msg_timestamps = []

            # Read all messages for this topic
            for _, timestamp, rawdata in reader.messages(connections=[conn]):
                msg = extract_message_data(reader.deserialize(rawdata, conn.msgtype))
                topic_data.append({conn.topic + '/' + key: val for key, val in msg.items()})
                msg_timestamps.append(timestamp)

            df = pd.DataFrame(topic_data)
            df["timestamp"] = pd.to_datetime(msg_timestamps, unit="ns")
            df.set_index("timestamp", inplace=True)

            # Resample
            for col in df.columns:
                result[col] = df[col].reindex(target_index, method="ffill", limit=1).interpolate(method="linear").bfill().ffill()

    for col in mocap_df.columns:
        result[col] = mocap_df[col].reindex(target_index, method="ffill", limit=1).interpolate(method="linear").bfill().ffill()

    return result


topics = [
    "/imu/data",
    "/measured_velocity",
    "/commanded_velocity",
    "/current_consumption",
    "/tendon_length_node_1/tendon_length",
    "/tendon_length_node_2/tendon_length",
    "/set_tendon_length/manual",
]

bags = [
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_58_28",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_56_30",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_54_41",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_43_08",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_41_22",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_36_32",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_53_52",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_51_55",
    "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_50_04",
    "/workspace/data/2025_07_21/rosbag2_2025_07_21-14_37_36",
    "/workspace/data/2025_07_21/rosbag2_2025_07_21-14_16_19",
]

for bag in bags:
    df = resample_rosbag2_and_csv(
        bag,
        topics=topics,
        target_hz=20.0,
    )

    # Save to CSV
    df.to_csv(bag + "_goat_training.csv")
    df.to_parquet(bag + "_goat_training.parquet", engine="pyarrow")
