import numpy as np


class KalmanFilter2D:
    def __init__(self, Ts, Qv, Qw_pos, Qw_vel):
        self.A = np.array(
            [[1, 0, Ts, 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.P = np.array(
            [[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]],
            dtype=np.float32,
        )
        self.Qw = np.array(
            [[Qw_pos, 0, 0, 0], [0, Qw_pos, 0, 0], [0, 0, Qw_vel, 0], [0, 0, 0, Qw_vel]]
        )
        self.Qv = np.array([[Qv, 0], [0, Qv]])
        self.K = None
        self.x = np.zeros((4), dtype=np.float32)
        pass

    def run_filter(self, y, Qv):
        x_n_n1 = self.A @ self.x
        P_n_n1 = self.A @ self.P @ self.A.T + self.Qw
        self.K = (
            P_n_n1 @ self.C.T @ (np.linalg.inv(self.C @ P_n_n1 @ self.C.T + self.Qv))
        )
        self.x = x_n_n1 + self.K @ (y - self.C @ x_n_n1)
        self.P = (np.eye(4) - self.K @ self.C) @ P_n_n1

    def get_position(self):
        return self.C @ self.x
        pass

    def get_velocity(self):
        return self.x[2:]


"""
Test the code above
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # Create a Kalman Filter
    Ts = 0.01
    Qv = 10000
    Q_pos = 100
    Q_vel = 10
    kf = KalmanFilter2D(Ts, Qv, Q_pos, Q_vel)

    # Generate the data
    t = np.arange(0, 10, Ts)
    x = np.cos(t)
    y = np.sin(t)
    z = np.vstack((x, y))
    z_noise = z + np.random.randn(2, z.shape[1]) * 0.1

    # Filter the data
    pos_x = []
    pos_y = []
    for i in range(z_noise.shape[1]):
        kf.run_filter(z_noise[:, i], 0)
        pos = kf.get_position()
        v = kf.get_velocity()
        pos_x.append(pos[0])
        pos_y.append(pos[1])

    fig, ax = plt.subplots()
    ax.plot(x, y, label="gt")
    ax.plot(pos_x, pos_y, label="kalman")
    #     ax.plot(z_noise[0,:], z_noise[1,:], label='input')
    ax.legend(loc="best")
    plt.show()
    # Plot the results
