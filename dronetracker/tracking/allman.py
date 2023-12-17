import numpy as np

class KalmanFilter2D:
    def __init__(self, Ts, R, Qp, Qv):
        self.Ts = Ts
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.P = np.array([[1000.0, 0.0, 0.0, 0.0],
                           [0.0, 1000.0, 0.0, 0.0],
                           [0.0, 0.0, 10.0, 0.0],
                           [0.0, 0.0, 0.0, 10.0]])
        self.F = np.array([[1.0, 0.0, Ts, 0.0],
                           [0.0, 1.0, 0.0, Ts],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.Q = np.array([[Qp, 0.0, 0.0, 0.0],
                           [0.0, Qp, 0.0, 0.0],
                           [0.0, 0.0, Qv, 0.0],
                           [0.0, 0.0, 0.0, Qv]])
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.array([[R, 0.0],
                           [0.0, R]])
        
        self.lifetime = 5
        self.rawBox = []
        self.floatBox = []
        

    def _predict(self):
        """
        Predict the next state and covariance matrix
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def _update(self, z):
        """
        Update the state and covariance matrix
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P
        return self.x, self.P

    def run_filter(self, z, R):
        """
        The Kalman Filter
        """
        self._predict()
        self.R = np.array([[R, 0.0],
                           [0.0, R]])
        self._update(z[:, None])
        return self.x, self.P

    def get_position(self):
        """
        Get the current position
        """
        return self.x[0:2, 0]

    def get_velocity(self):
        """
        Get the current velocity
        """
        return self.x[0, 2:4]
    
    def inc(self):
        self.lifetime += 1

    def dec(self):
        self.lifetime -= 1

"""
Test the code above
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # Create a Kalman Filter
    Ts = 0.01
    R = 5000
    Q = 50
    Qv = 100
    kf = KalmanFilter2D(Ts, R, Q, Qv)

    # Generate the data
    t = np.arange(0, 10, Ts)
    x = np.cos(t)
    y = np.sin(t)
    z = np.vstack((x, y))
    z_noise = z + np.random.randn(2, z.shape[1]) * 0.1

    # Filter the data
    x_hat = []
    y_hat = []
    v_y = []
    for i in range(z_noise.shape[1]):
        z_k = z_noise[:, i].reshape((2, 1))
        x_hat_k, P = kf.run_filter(z_k, R)
        x_hat.append(float(x_hat_k[0,0]))
        y_hat.append(float(x_hat_k[1,1]))
        v_y.append(kf.get_velocity()[0])

    # Plot the results
    plt.figure()
    plt.plot(t, z_noise[0, :], 'r.', label='Measurements')
    plt.plot(t, x_hat, 'b-', label='Kalman Filter')
    plt.plot(t, z[0, :], 'g-', label='True Value')
    plt.legend(loc='best')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
#     plt.show()

    fig, ax = plt.subplots()
    ax.plot(v_y)
    plt.show()

    plt.figure()
    plt.plot(t, z_noise[1, :], 'r.', label='Measurements')
    plt.plot(t, y_hat, 'b-', label='Kalman Filter')
    plt.plot(t, z[1, :], 'g-', label='True Value')
    plt.legend(loc='best')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.show()
