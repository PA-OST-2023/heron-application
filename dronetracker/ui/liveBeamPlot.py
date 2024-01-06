import datetime
import numpy as np
from numpy import sin, cos
import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import degrees, radians

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from beamforming.prototypeTracker import Tracker
from beamforming.kalmanTracker import KalmanTracker, Kalman_track_object


class UI:
    def __init__(self, tracker, streamer):
        self.tracker = tracker
        self.streamer = streamer
        self.phi_beamsearch_sphere,self.theta_beamsearch_sphere  = tracker.get_sphere()
        r_flat_proj = self.theta_beamsearch_sphere
        self.x = cos(self.phi_beamsearch_sphere) * r_flat_proj
        self.y = sin(self.phi_beamsearch_sphere) * r_flat_proj
        self.block_len = 1024 * 2
        self.layout = go.Layout(
            autosize=False,
            width=900,
            height=900,
            margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        )
        self.x_sphere = sin(self.theta_beamsearch_sphere) * cos(self.phi_beamsearch_sphere)
        self.y_sphere = sin(self.theta_beamsearch_sphere) * sin(self.phi_beamsearch_sphere)
        self.z_sphere = cos(self.theta_beamsearch_sphere)
        self.x_hat = []
        self.y_hat = []
        self.max_vals = []

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        self.map_fig = None
        self.setup_map()

        self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.layout = html.Div(
            [html.H2("HERON"),
            html.Div([html.H3("INFOS")], className='one column'),
            html.Div([html.H3("Beams"),
                        dcc.Graph(id="live-beam-plots"),
                        dcc.Interval(
                        id="beam-plots",
                        interval=1 * 100,  # in milliseconds
                        n_intervals=0,
                        ),
                    ], className='five columns'),
            html.Div([html.H3("Map"),
                        dcc.Graph(figure=self.map_fig, id="live-update-graph"),
#                         dcc.Interval(
#                         id="interval-component",
#                         interval=1 * 100,  # in milliseconds
#                         n_intervals=0,)
                        ], className='five columns')
            ]

        )
        self.setup_callbacks()

    def setup_map(self):
        r_earth = 6371e3
        c_lon = 8.819
        c_lat = 47.2233
        a = 10
        d_lat = degrees(a/r_earth)
        d_lon = degrees(a/(np.sin(radians(c_lat))*r_earth))
        self.map_fig = go.Figure(go.Scattermapbox(
                        mode = "markers+lines",
                        lon = [],
                        lat = [],
                        marker = {'size': 10}))

        self.map_fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=18, mapbox_center_lat = 47.2233, mapbox_center_lon = 8.819,
            margin={"r":0,"t":0,"l":0,"b":0})

    def update_map_center(self, center_lon, center_lat):
        self.map_fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=18, mapbox_center_lat = center_lat, mapbox_center_lon = center_lon,
            margin={"r":0,"t":0,"l":0,"b":0})

    def run(self):
        self.streamer.start_stream()
        self.app.run(debug=False)

    def setup_callbacks(self):
        # Multiple components can update everytime interval gets fired.
        @callback(
            [Output(component_id="live-beam-plots", component_property="figure"),
            Output(component_id="live-update-graph", component_property="figure")],
            Input("beam-plots", "n_intervals"),
        )
        def update_graph_live(n):
            print("ENTER CALLBACK")
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print("----Done---")
                self.streamer.end_stream()
                self.streamer.start_stream()
                return None, self.map_fig
            response, meas, tracking_objects, max_val, *_ = self.tracker.track(block)
            self.max_vals.append(max_val)
            r = response
            x = self.x_sphere
            y = self.y_sphere
            z = self.z_sphere
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "mesh3d"}, {"type": "mesh3d"}],
                    #                     [{'type':'scatter'}, {'type': 'heatmap'}]]
                    [{"type": "scatter"}, {"type": "scatter"}],
                ],
            )
            fig.add_trace(
                go.Mesh3d(
                    z=(response),
                    x=(self.x),
                    y=(self.y),
                    intensity=response,
                    colorscale="Viridis",
                    showscale=False
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Mesh3d(z=z, x=x, y=y, intensity=r, colorscale="Viridis", showscale=False),
                row=1,
                col=2,
            )
            for tracking_object in tracking_objects:
                x_data = tracking_object.track[:,0]
                y_data = tracking_object.track[:,1]
                color = tracking_object.color
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines+markers",
                        name="Data",
                        line={"color": color, "width": 2},
                        marker={"color": color, "size": 2.5},
                    ),
                    row=2,
                    col=1,
                )

            self.map_fig.data = []
            c_lon = 8.8191
            c_lat = 47.22324

            for tracking_object in tracking_objects:
                x_data = tracking_object.track[:,0]
                y_data = tracking_object.track[:,1]
                color = tracking_object.color
                track_phi = np.arctan2(y_data,x_data)
                track_theta = np.sqrt(y_data**2 + x_data**2)

                fig.add_trace(
                    go.Scatter(
                        x=track_phi*180 / np.pi,
                        y=track_theta*180 / np.pi,
                        mode="lines+markers",
                        name="Data",
                        line={"color": color, "width": 2},
                        marker={"color": color, "size": 2.5},
                    ),
                    row=2,
                    col=2,
                )
                r = 30
                x_map = r * np.sin(track_theta)*np.cos(track_phi)
                y_map = r * np.sin(track_theta)*np.sin(track_phi)
                lat_map, lon_map = self.convert_to_map(c_lon, c_lat, x_map, y_map)
                self.map_fig.add_trace(go.Scattermapbox(
                        mode = "markers+lines",
                        lon = lon_map,
                        lat = lat_map,
                        line={"color": color, "width": 3},
                        marker={"color": color, "size": 3.5},))

                self.map_fig.add_trace(go.Scattermapbox(
                        mode = "markers",
                        lon = [c_lon],
                        lat = [c_lat],
                        marker={"color": "rgb(255, 0, 0)", "size": 5.5},))



#             fig.add_trace(
#                 go.Scatter(
# #                     x= [1,0,-1,0],
#                     y= self.max_vals,
#                     mode="lines+markers",
#                     name="Data",
#                     line={"color": "rgb(0, 255, 0)"},
#                     marker={"color": "rgb(0, 255, 0)", "size": 8},
#                 ),
#                 row=2,
#                 col=2,
#             )
            # fig.add_trace(go.Heatmap(z=(grid)), row=2, col=2)
            fig.update_layout(width=800, height=800, uirevision=1)
            fig.update_yaxes(
                range=[-1.7, 1.7], scaleanchor="x", scaleratio=1, row=2, col=1
            )
            fig.update_xaxes(range=[-1.7, 1.7], row=2, col=1)
            fig.update_yaxes(
                range=[0, 90], row=2, col=2
            )
            fig.update_xaxes(range=[-180, 180], row=2, col=2)
            fig.update_layout(showlegend=False)

            return fig, self.map_fig

    def convert_to_map(self,c_lon, c_lat, x, y):
        r_earth = 6371e3
        lat = y/r_earth *180 /np.pi
        lon = x/(np.sin(radians(c_lat))*r_earth) * 180 / np.pi
        lat += c_lat
        lon += c_lon

        return lat, lon


if __name__ == "__main__":
    app.run(debug=True)
