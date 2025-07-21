import torch
import roma
from sklearn.model_selection import train_test_split


def create_transform_matrix(pos_w_to_b, quat_w_to_b):
    """
    Create 4x4 homogeneous transformation matrix from position and quaternion

    Args:
        pos_w_to_b: (N,3) tensor - translation component
        quat_w_to_b: (N,4) tensor - rotation as quaternion (qx, qy, qz, qw)

    Returns:
        (N,4,4) tensor - homogeneous transformation matrix
    """
    # Ensure inputs are tensors

    quat_w_to_b = roma.quat_normalize(quat_w_to_b)
    rot_w_to_b = roma.unitquat_to_rotmat(quat_w_to_b)

    # Construct 4x4 transformation matrix
    T_w_to_b = torch.zeros(rot_w_to_b.shape[0], 4, 4)
    T_w_to_b[:, :3, :3] = rot_w_to_b
    T_w_to_b[:, :3, 3] = pos_w_to_b
    T_w_to_b[:, 3, 3] = 1.0

    return T_w_to_b

def inverse_homogeneous_transform(T):
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.

    Args:
        T (torch.Tensor): 4x4 homogeneous transformation matrix.

    Returns:
        torch.Tensor: Inverse of T.
    """
    # Extract rotation and translation
    R = T[:, :3, :3]
    t = T[:, :3, 3]

    # Compute inverse
    R_inv = torch.transpose(R, dim0=1, dim1=2)  # Inverse of rotation is its transpose
    t_inv = torch.bmm(-R_inv, torch.transpose(t.unsqueeze(1), dim0=1, dim1=2)).squeeze()

    # Construct inverse transformation matrix
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device).reshape((1, 4, 4)).repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv

    return T_inv

def angular_velocities_from_quat(q_t, q_t_1, dt):
    # Adopted from https://mariogc.com/post/angular-velocity-quaternions/
    # our conv x y z w -> 0 1 2 3
    w = torch.zeros((q_t.shape[0], 3), dtype=q_t.dtype, device=q_t.device)
    w[:, 0] = 2.0 / dt * (q_t[:, 3] * q_t_1[:, 0] - q_t[:, 0] * q_t_1[:, 3] - q_t[:, 1] * q_t_1[:, 2] + q_t[:, 2] * q_t_1[:, 1])
    w[:, 1] = 2.0 / dt * (q_t[:, 3] * q_t_1[:, 1] + q_t[:, 0] * q_t_1[:, 2] - q_t[:, 1] * q_t_1[:, 3] - q_t[:, 2] * q_t_1[:, 0])
    w[:, 2] = 2.0 / dt * (q_t[:, 3] * q_t_1[:, 2] - q_t[:, 0] * q_t_1[:, 1] + q_t[:, 1] * q_t_1[:, 0] - q_t[:, 2] * q_t_1[:, 3])
    return w


def moving_average_smoothing(vector, window_size=5):
    """
    Apply moving average smoothing to a [Timesteps, 3] vector
    
    Args:
        vector: Input tensor of shape [Timesteps, 3]
        window_size: Size of the moving average window (odd number recommended)
    
    Returns:
        Smoothed tensor of same shape [Timesteps, 3]
    """
    # Create averaging kernel (normalized ones)
    kernel = torch.ones(1, 1, window_size) / window_size
    
    # Pad for same length output
    padding = window_size // 2
    
    # Apply convolution to each channel separately
    smoothed = []
    for i in range(3):  # For each of the 3 channels
        channel_data = vector[:, i].view(1, 1, -1)  # [1, 1, Timesteps]
        smoothed_channel = torch.nn.functional.conv1d(channel_data, kernel, padding=padding)
        smoothed.append(smoothed_channel)
    
    # Stack channels back together
    return torch.cat(smoothed, dim=1).squeeze(0).t()  # [Timesteps, 3]

def ema_2d_optimized(vector, alpha=0.1):
    """
    Efficient EMA implementation for [Timesteps, 3] input
    Corrected version that handles 3 channels properly
    """
    timesteps = vector.shape[0]
    device = vector.device
    
    # Create EMA weights [timesteps]
    weights = (1 - alpha) ** torch.arange(timesteps, dtype=torch.float32, device=device)
    weights = weights.flip(0)  # Reverse for proper weighting
    weights = weights / weights.sum()  # Normalize
    
    # Reshape weights for convolution [1, 1, timesteps]
    kernel = weights.view(1, 1, -1)
    
    # Prepare input: [1, 3, timesteps]
    input_ = vector.t().unsqueeze(0)
    
    # Apply depthwise convolution to handle 3 channels separately
    # We need to expand the kernel to match input channels
    kernel = kernel.expand(3, 1, -1)  # [3, 1, timesteps]
    kernel = kernel.reshape(3, 1, -1)  # [3, 1, timesteps]
    
    # Use groups=3 for channel-wise operation
    smoothed = torch.nn.functional.conv1d(
        input_,
        kernel,
        padding=timesteps-1,
        groups=3
    )[:, :, :timesteps]
    
    return smoothed.squeeze(0).t()  # [Timesteps, 3]


def world_to_robot_frame_transform(pos_point_in_w, T_world_to_robot):
    """
    Transform points from world frame to robot frame using transformation matrices

    Args:
        pos_w_to_point_in_w: (N,M,3) tensor - points in world frame
        T_world_to_robot: (N,4, 4) tensor - homegenous transformation from world to base

    Returns:
        Transformed points in robot frame
    """

    # Convert points to homogeneous coordinates
    pos_homog_w_to_point = torch.cat([pos_point_in_w, torch.ones(pos_point_in_w.shape[0], pos_point_in_w.shape[1], 1)], dim=-1).unsqueeze(-1)


    T_expanded = T_world_to_robot.unsqueeze(1).expand(-1, pos_point_in_w.shape[1], -1, -1)

    # Apply transformation
    pos_homog_robot_to_point = torch.matmul(T_expanded, pos_homog_w_to_point).squeeze(-1)

    # Convert back from homogeneous coordinates
    return pos_homog_robot_to_point[:, :, :3]

    # Only generate the points but not the full transformations
    # quat_w_to_robot = roma.quat_normalize(quat_w_to_robot)
    # quat_w_to_robot = roma.quat_wxyz_to_xyzw(quat_w_to_robot)
    # rot_w_to_robot = roma.unitquat_to_rotmat(quat_w_to_robot)
    # rot_robot_to_w = torch.transpose(rot_w_to_robot, dim0=1, dim1=2)
    # pos_robot_to_point_in_world = pos_w_to_point - pos_w_to_robot.unsqueeze(1)
    # pos_robot_to_point = torch.matmul(pos_robot_to_point_in_world, rot_robot_to_w.transpose(1, 2))
    # return pos_robot_to_point


class DataProcessorGoat:
    def __init__(self):
        self.input_shape = 21
        self.output_shape = 24
        self.input_mean = torch.tensor(
            [
                0, 0, 0, # gravity vector
                2.0762e-04,
                -3.3720e-06,
                -1.4475e-03,
                1.7342e00,
                -2.3323e-01,
                -9.0298e00,
                2.8395e-01,
                2.7578e-01,
                2.5038e-01,
                2.3396e-01,
                7.6569e-01,
                -2.5864e-02,
                5.4099e-02,
                4.6486e-02,
                -1.5180e-03,
                -9.5889e-04,
                2.1033e00,
                1.7311e00,
            ],
            dtype=torch.float,
            device=torch.device("cpu"),
        )
        self.input_std = torch.tensor(
            [
                1, 1, 1, # gavity vector
                0.2353,
                0.2135,
                0.1660,
                2.4749,
                1.6329,
                1.8700,
                3.0410,
                2.8407,
                3.0058,
                2.8082,
                11.9981,
                12.0488,
                0.1115,
                0.2306,
                0.0984,
                0.1294,
                0.3942,
                0.2806,
            ],
            dtype=torch.float,
            device=torch.device("cpu"),
        )

        self.output_mean = torch.tensor(
            [
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0,
                # 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            ],
            dtype=torch.float,
            device="cpu",
        )  # Frame points
        self.output_std = torch.tensor(
            [
                # 500, # Frame points
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 500,
                # 1, 1, 1, # Down vector
                2, 2, 2, # Base Angular Velocity rad/s
                500, 500, 500, # Base Linear Velocity mm/s
            ],
            dtype=torch.float,
            device="cpu",
        )  

    def process_input_data(self, data) -> torch.Tensor:
        num_data = data["/imu/data/orientation_w"].size
        data_tensor = torch.zeros([num_data, self.input_shape], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_quat = torch.zeros([num_data, 4], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_quat[:, 0] = torch.tensor(data["/imu/data/orientation_x"], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_quat[:, 1] = torch.tensor(data["/imu/data/orientation_y"], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_quat[:, 2] = torch.tensor(data["/imu/data/orientation_z"], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_quat[:, 3] = torch.tensor(data["/imu/data/orientation_w"], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation_rotmat = roma.unitquat_to_rotmat(robot_rot_orientation_quat)
        data_tensor[:, 0:3] = robot_rot_orientation_rotmat[:, :, 2]  # this is essentially rot_mat * [0 0 1].T

        data_tensor[:, 3] = torch.tensor(data["/imu/data/angular_velocity_x"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 4] = torch.tensor(data["/imu/data/angular_velocity_y"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 5] = torch.tensor(data["/imu/data/angular_velocity_z"], dtype=torch.float, device=torch.device("cpu"))

        data_tensor[:, 6] = torch.tensor(data["/imu/data/linear_acceleration_x"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 7] = torch.tensor(data["/imu/data/linear_acceleration_y"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 8] = torch.tensor(data["/imu/data/linear_acceleration_z"], dtype=torch.float, device=torch.device("cpu"))

        data_tensor[:, 9] = torch.tensor(data["/measured_velocity/data_0"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 10] = torch.tensor(data["/measured_velocity/data_1"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 11] = torch.tensor(data["/measured_velocity/data_2"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 12] = torch.tensor(data["/measured_velocity/data_3"], dtype=torch.float, device=torch.device("cpu"))

        data_tensor[:, 13] = torch.tensor(data["/commanded_velocity/data_0"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 14] = torch.tensor(data["/commanded_velocity/data_1"], dtype=torch.float, device=torch.device("cpu"))

        data_tensor[:, 15] = torch.tensor(data["/current_consumption/data_0"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 16] = torch.tensor(data["/current_consumption/data_1"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 17] = torch.tensor(data["/current_consumption/data_2"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 18] = torch.tensor(data["/current_consumption/data_3"], dtype=torch.float, device=torch.device("cpu"))

        data_tensor[:, 19] = torch.tensor(data["/tendon_length_node_1/tendon_length/data"], dtype=torch.float, device=torch.device("cpu"))
        data_tensor[:, 20] = torch.tensor(data["/tendon_length_node_2/tendon_length/data"], dtype=torch.float, device=torch.device("cpu"))

        return data_tensor

    def scale_input_data_tensor(self, data_tensor):
        return (data_tensor - self.input_mean) / self.input_std

    def process_output_data(self, data) -> torch.Tensor:
        num_data = data["TEST_GOAT_Rotation_X"].size
        output_tensor = torch.zeros([num_data, 3 + 3], dtype=torch.float, device=torch.device("cpu"))

        ## Frame points
        num_points = 8
        p_in_world = torch.zeros([num_data, num_points, 3], dtype=torch.float, device=torch.device("cpu"))
        robot_pos_in_world = torch.zeros([num_data, 3], dtype=torch.float, device=torch.device("cpu"))
        robot_rot_orientation = torch.zeros([num_data, 4], dtype=torch.float, device=torch.device("cpu"))
        robot_pos_in_world[:, 0] = torch.tensor(data["TEST_GOAT_Position_X"])
        robot_pos_in_world[:, 1] = torch.tensor(data["TEST_GOAT_Position_Y"])
        robot_pos_in_world[:, 2] = torch.tensor(data["TEST_GOAT_Position_Z"])
        robot_rot_orientation[:, 0] = torch.tensor(data["TEST_GOAT_Rotation_X"])
        robot_rot_orientation[:, 1] = torch.tensor(data["TEST_GOAT_Rotation_Y"])
        robot_rot_orientation[:, 2] = torch.tensor(data["TEST_GOAT_Rotation_Z"])
        robot_rot_orientation[:, 3] = torch.tensor(data["TEST_GOAT_Rotation_W"])
        for j in range(0, num_points):
            p_in_world[:, j, 0] = torch.tensor(data["MarkerSet 001:Marker" + str(j + 1) + "_Position_X"])
            p_in_world[:, j, 1] = torch.tensor(data["MarkerSet 001:Marker" + str(j + 1) + "_Position_Y"])
            p_in_world[:, j, 2] = torch.tensor(data["MarkerSet 001:Marker" + str(j + 1) + "_Position_Z"])

        # Offset positions by initial robot location
        # world_initial_pos = robot_pos_in_world[0, :] # We offset everything by a constant position. No mathematical reason, just easier to read the numbers
        # p_in_world = p_in_world - world_initial_pos
        # robot_pos_in_world = robot_pos_in_world - world_initial_pos

        # offset to match mocap to base X Y Z
        yzx_quat_offset = roma.euler_to_unitquat(convention='YZX', angles=(90, 90, 0), degrees=True).expand(num_data, -1)
        robot_rot_orientation = roma.quat_product(robot_rot_orientation, yzx_quat_offset)

        # Get transform w to robot
        T_robot_to_world = create_transform_matrix(robot_pos_in_world, robot_rot_orientation)
        T_world_to_robot = inverse_homogeneous_transform(T_robot_to_world)

        # Transform p into robot centric frame
        p_in_robot = world_to_robot_frame_transform(p_in_world, T_world_to_robot)

        sI = 0
        # output_tensor[:, sI:sI+24] = p_in_robot.reshape(num_data, num_points * 3)
        # sI += 24

        ## Downward vector
        # output_tensor[:, sI:sI+3] = -1.0 * T_world_to_robot[:, :3, 2]  # this is essentially rot_mat * [0 0 -1].T
        # sI += 3

        ## Angular Velocity following [0 w] = 2 * q_dot X q_inv
        euler_velocity_in_world = angular_velocities_from_quat(robot_rot_orientation[:-1], robot_rot_orientation[1:], 0.05)  # dt = 1/20
        euler_velocity_in_robot = torch.bmm(T_world_to_robot[1:, :3, :3], euler_velocity_in_world.unsqueeze(-1)).squeeze()
        euler_velocity_in_robot = ema_2d_optimized(euler_velocity_in_robot)
        output_tensor[1:, sI:sI+3] = euler_velocity_in_robot.squeeze()
        output_tensor[0, sI:sI+3] = output_tensor[1, sI:sI+3]  # We just fill the first velocity
        sI += 3

        ## Base Velocity following robot_vel_in_robot = rot_robot_to_world.T * (vel_in_world - angular_velocity_in_world X pos_robot_in_world)
        robot_vel_in_world = (robot_pos_in_world[1:] - robot_pos_in_world[:-1]) / 0.05  # dt = 1/20
        robot_vel_in_world = robot_vel_in_world - torch.cross(euler_velocity_in_world, T_robot_to_world[1:, :3, 3])
        robot_vel_in_robot = torch.matmul(T_world_to_robot[1:, :3, :3], robot_vel_in_world.unsqueeze(-1)).squeeze()
        robot_vel_in_robot = ema_2d_optimized(robot_vel_in_robot)
        output_tensor[1:, sI:sI+3] = robot_vel_in_robot
        output_tensor[0, sI:sI+3] = output_tensor[1, sI:sI+3]  # We just fill the first velocity

        return output_tensor

    def scale_output_data_tensor(self, data_tensor):
        return (data_tensor - self.output_mean) / self.output_std


def create_sequences(input, target, sequence_length=50, target_length=1, test_size=0.2):
    """
    Identical interface to ROSBagDataLoader.create_sequences()

    Args:
        sequence_length (int): Input sequence length
        target_length (int): Target sequence length
        test_size (float): Test set proportion

    Returns:
        X_train, X_test, y_train, y_test: Numpy arrays
    """

    # Create sequences
    num_data = len(input) - sequence_length - target_length
    sequences = torch.zeros(num_data, sequence_length, input.shape[1])
    targets = torch.zeros(num_data, target_length, target.shape[1])

    for i in range(len(input) - sequence_length - target_length):
        sequences[i] = input[i : i + sequence_length]
        targets[i] = target[i + sequence_length : i + sequence_length + target_length]

    # Train-test split (without shuffling to preserve time order)
    return train_test_split(sequences, targets, test_size=test_size, shuffle=False)
