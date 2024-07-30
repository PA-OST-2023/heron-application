import numpy as np
from numpy import sin, cos, arcsin, arccos, pi, exp, arctan2
from math import degrees, radians
from scipy.spatial.transform import Rotation as R


class Beamformer:
    def __init__(self, num_mics = 32, fs=44100):
        self._num_mics = num_mics
        self._arm_angle = 0.0
        self._fs = fs
        self.update_arm_angle(self._arm_angle)
        
    def update_arm_angle(self, angle):      # degrees (0° - 90°)
        self._arm_angle = angle
        self._coords = self.calculate_umbrella_array(radians(self._arm_angle), 0.01185 - 0.0016)

    def calculate_delays(self, phi, theta, vm=340):
        angle = np.array([-radians(phi), pi / 2 - radians(theta)])      # Calculate angles for a single (phi, theta)
        mat = R.from_euler("zy", angle, degrees=False)              # Compute the rotation matrix for the given angle
        coord_t = (mat.as_matrix() @ self._coords.T).swapaxes(-1, -2)
        t_s = coord_t[:, 0].reshape((-1, 1)) / vm
        t_s -= np.min(t_s)
        return t_s
        

    # Gemoetry functions
    def convert_to_base_system(self, M, phi=+radians(22.5), r=0.095):
        rot = R.from_rotvec(phi * np.array([0, 0, 1]))
        M_t = M + np.array([r, 0, -(0.01185 - 0.0056)])
        M_tilde = rot.apply(M_t)
        return M_tilde
        
    def populate_array(self, M_tilde, plot=False):
        Mall = np.empty((32, 3), np.float32)
        rots = np.roll(np.arange(8), 2)
        for i, j in enumerate(rots):
            r = R.from_rotvec(-j * 2 * pi / 8 * np.array([0, 0, 1]))
            m_t = r.apply(M_tilde)
            Mall[4 * i : 4 * i + 4, :] = m_t
        return Mall
        
    def calculate_umbrella_array(self, gamma, z_o, plot=False):
        m_0 = np.array([0.101157, 0.039018, z_o])
        m_1 = np.array([0.145209, 0.099498, z_o])
        m_2 = np.array([0.17107, 0.177783, z_o])
        m_3 = np.array([0.173701, 0.268701, z_o])
        M = np.vstack((m_0, m_1, m_2, m_3))
        r = R.from_rotvec(gamma * np.array([0, 1, 0]))
        M_tilde = r.apply(M)
        M_tilde = self.convert_to_base_system(M_tilde)
        return self.populate_array(M_tilde, plot)
    

if __name__ == "__main__":
    beamformer = Beamformer(32, 44100)

    angle = 0.0     # degrees
    theta = 90.0     # degrees
    phi = 0.0       # degrees

    beamformer.update_arm_angle(angle)
    filter = beamformer.calculate_weights(phi, theta)
    print(filter)
