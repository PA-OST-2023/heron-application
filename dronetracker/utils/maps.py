from math import radians
from numpy import sin, pi, cos

def convert_to_map(c_lon, c_lat, x, y):
    r_earth = 6371e3
    lat = y / r_earth * 180 / pi
    lon = x / (cos(radians(c_lat)) * r_earth) * 180 / pi
    lat += c_lat
    lon += c_lon

    return lat, lon


