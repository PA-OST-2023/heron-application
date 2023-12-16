from numpy import pi, cos, sin, arccos, arange
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp

class BeamFormer():

    def __init__(self):

        num_pts = 32000
        r = 10
        indices = arange(0, num_pts, dtype=float)# + 0.5

        phi = arccos(1 - 2*indices[:num_pts//2]/num_pts)
        theta = pi * (1 + 5**0.5) * indices[:num_pts//2]

        x, y, z = r* cos(theta) * sin(phi), r* sin(theta) * sin(phi), r* cos(phi);

        pp.figure().add_subplot(111, projection='3d').scatter(x, y, z);
        pp.axis('equal')
        pp.show()
        pass

    def compute_filterbank(self):
        pass

    def global_beam_sweep(self, signals):
        pass

    def local_beam_sweep(self, signals, center_direction):
        pass

    def listen_at(self, signals, direction):
        pass

if __name__ == '__main__':
    beam = BeamFormer()
