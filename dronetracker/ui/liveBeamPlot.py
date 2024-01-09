import datetime
import numpy as np
from numpy import sin, cos
import dash
from dash import Dash, dcc, html, Input, Output, callback, State
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import degrees, radians
from pathlib import Path

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from beamforming.prototypeTracker import Tracker
from beamforming.kalmanTracker import KalmanTracker, Kalman_track_object
from utils.maps import convert_to_map
from AudioInterface.waveStreamer import WavStreamer
from AudioInterface.tcpStreamer import TcpStreamer


class UI:
    def __init__(self, tracker=None, streamer=None):
        self.layout = go.Layout(
            autosize=False,
            width=900,
            height=900,
            margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        )
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        self.map_fig = None
        self._setup_map()

        self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.layout = html.Div(
            [
                html.H2("HERON"),
                html.Div(
                    [
                        html.H3("Connection"),
                        html.H4("New Connection"),
                        html.P("Connection Type"),
                        dcc.Dropdown(["IP", "WAV"], "IP", id="arr-type"),
                        html.P("IP Adress", id="arr-info-label"),
                        dcc.Input(value="192.168.33.80", id="arr-info", type="text"),
                        html.P("Port", id="arr-conf-label"),
                        dcc.Input(
                            value=6666,
                            id="arr-conf",
                            type="text",
                            style={"display": "block"},
                        ),
                        html.Button("connect", id="submit-val", n_clicks=0),
                        html.P("", id="submit-out"),
                        html.Div(style={"margin-top": "15px"}),
                        html.H4("Open Connections"),
                        html.Div(id="open-con-list"),
                        html.Div(style={"margin-top": "15px"}),
                        html.H4("Display Connections"),
                        dcc.Dropdown(["None"], id="sel-con", disabled=False),
                        html.P("Current Connection: None", id="curr-con"),
                        html.Button("disconnect", id="dis-but"),
                        html.P(id="dis-but-err"),
                    ],
                    className="two columns",
                ),
                html.Div(
                    [
                        html.H3("Beams"),
                        dcc.Graph(id="live-beam-plots"),
                        dcc.Interval(
                            id="beam-plots",
                            interval=1.75 * 100,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                    className="five columns",
                ),
                html.Div(
                    [
                        html.H3("Map"),
                        dcc.Graph(figure=self.map_fig, id="live-update-graph"),
                    ],
                    className="four columns",
                ),
            ]
        )
        self.setup_callbacks()

        self.streamers = []
        self.trackers = []
        self.tracker = tracker
        self.streamer = streamer
        self.tracker = None
        self.streamer = None

        self.phi_beamsearch_sphere, self.theta_beamsearch_sphere = tracker.get_sphere()

        r_flat_proj = self.theta_beamsearch_sphere
        self.x = cos(self.phi_beamsearch_sphere) * r_flat_proj
        self.y = sin(self.phi_beamsearch_sphere) * r_flat_proj

        self.block_len = 1024 * 2

        self.x_sphere = sin(self.theta_beamsearch_sphere) * cos(
            self.phi_beamsearch_sphere
        )
        self.y_sphere = sin(self.theta_beamsearch_sphere) * sin(
            self.phi_beamsearch_sphere
        )
        self.z_sphere = cos(self.theta_beamsearch_sphere)

        self.x_hat = []
        self.y_hat = []
        self.max_vals = []

    def _setup_map(self):
        r_earth = 6371e3
        c_lon = 8.819
        c_lat = 47.2233
        a = 10
        d_lat = degrees(a / r_earth)
        d_lon = degrees(a / (np.sin(radians(c_lat)) * r_earth))
        self.map_fig = go.Figure(
            go.Scattermapbox(mode="markers+lines", lon=[], lat=[], marker={"size": 10})
        )

        self.map_fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=18,
            mapbox_center_lat=47.2233,
            mapbox_center_lon=8.819,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

    def _update_map_center(self, center_lon, center_lat):
        self.map_fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=18,
            mapbox_center_lat=center_lat,
            mapbox_center_lon=center_lon,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

    def run(self):
        if self.streamer is not None:
            self.streamer.start_stream()
        self.app.run(debug=False)

    def setup_callbacks(self):
        @callback(
            [
                Output("arr-info-label", "children"),
                Output("arr-conf-label", "children"),
                Output("arr-conf", "style"),
                Output("arr-conf", "type"),
                Output("arr-info", "value"),
                Output("arr-conf", "value"),
            ],
            Input("arr-type", "value"),
        )
        def update_fields_cb(value):
            if value == "IP":
                #                 return "IP Adress", "", {"display": "none"}, "", ""
                return (
                    "IP Adress",
                    "Port",
                    {"display": "block"},
                    "number",
                    "192.168.33.80",
                    6666,
                )
            if value == "WAV":
                return (
                    "Wave File path",
                    "Config",
                    {"display": "block"},
                    "text",
                    "./data/random.wav",
                    str(Path(__file__).parent.parent / "configs" / "testfancy1.toml"),
                )

            return "", "", {"display": "none"}, "number", "", ""

        @callback(
            [
                Output("submit-out", "children"),
                Output("open-con-list", "children"),
                Output("sel-con", "options"),
                Output("sel-con", "value"),
                Output("curr-con", "children"),
            ],
            Input("submit-val", "n_clicks"),
            [
                State("arr-info", "value"),
                State("arr-type", "value"),
                State("arr-conf", "value"),
            ],
            prevent_initial_call=True,
        )
        def open_connection_cb(n_clicks, arr_info, arr_type, arr_conf):
            message = "Error"
            if arr_type == "IP":
                for streamer in self.streamers:
                    if streamer.name == arr_info:
                        return (
                            "IP alredy connected",
                            html.Ul(
                                [
                                    html.Li(f"{streamer.name}")
                                    for streamer in self.streamers
                                ]
                            ),
                            [
                                {"value": streamer.name, "label": streamer.name}
                                for i, streamer in enumerate(self.streamers)
                            ],
                            self.streamer.name,
                            f"Current Connection: {self.streamer.name}",
                        )

                self.streamer = TcpStreamer(arr_info, port=arr_conf)
                self.streamers.append(self.streamer)

                self.tracker = KalmanTracker()
                self.tracker.init_umbrella_array(arr_conf)
                self.trackers.append(self.tracker)
                self.streamer.start_stream()
                message = "Connected to ip"
                pass  # TODO
            if arr_type == "WAV":
                self.streamer = WavStreamer(arr_info, 1024 * 4)
                self.streamers.append(self.streamer)
                self.tracker = KalmanTracker()
                self.tracker.init_config_array(arr_conf)
                self.trackers.append(self.tracker)
                self.streamer.start_stream()
                message = "Connectet to Wav"
            curr_streamer_name = "None"
            if self.streamer is not None:
                curr_streamer_name = self.streamer.name
            return (
                message,
                html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                curr_streamer_name,
                f"Current Connection: {curr_streamer_name}")

        @callback(
            [
                Output("open-con-list", "children", allow_duplicate=True),
                Output("sel-con", "options", allow_duplicate=True),
                Output("sel-con", "value", allow_duplicate=True),
                Output("curr-con", "children", allow_duplicate=True),
                Output("dis-but-err", "children"),
            ],
            Input("dis-but", "n_clicks"),
            [
                State("sel-con", "value"),
            ],
            prevent_initial_call=True,
        )
        def close_conn_cb(n_clicks, conn):
            streamer_ind = -1
            curr_streamer_name = "none"
            for i, streamer in enumerate(self.streamers):
                if streamer.name != conn:
                    continue
                streamer_ind = i
                streamer.end_stream()
            if streamer_ind == -1:
                print('SHIT')
                return (
                        html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                        [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                        "",
                        f"Current Connection: {curr_streamer_name}",
                        "No Streamer to close")
            self.trackers.pop(streamer_ind)
            self.streamers.pop(streamer_ind)
            del self.streamer
            del self.tracker
            self.streamer = None
            self.tracker = None
            if len(self.streamers) > 0:
                self.streamer = self.streamers[0]
                curr_streamer_name = self.streamer.name
            return  (
                    html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                    [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                    "",
                    f"Current Connection: {curr_streamer_name}",
                    "Streamer closed")


        # Multiple components can update everytime interval gets fired.
        @callback(
            [
                Output(component_id="live-beam-plots", component_property="figure"),
                Output(component_id="live-update-graph", component_property="figure"),
            ],
            Input("beam-plots", "n_intervals"),
        )
        def update_graph_live(n):
            print("ENTER CALLBACK")
            # TODO Make object field
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "mesh3d"}, {"type": "mesh3d"}],
                    #                     [{'type':'scatter'}, {'type': 'heatmap'}]]
                    [{"type": "scatter"}, {"type": "scatter"}],
                ],
            )
            fig.update_layout(width=700, height=800, uirevision=1)
            if self.tracker is None and self.streamer is None:
                return fig, self.map_fig
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print("----Done---")
                self.streamer.end_stream()
                self.streamer.start_stream()
                del self.streamer
                del self.tracker
                self.tracker = None
                self.streamer = None
                return fig, self.map_fig
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
            #             fig.add_trace(
            #                 go.Mesh3d(
            #                     z=(response),
            #                     x=(self.x),
            #                     y=(self.y),
            #                     intensity=response,
            #                     colorscale="Viridis",
            #                     showscale=False,
            #                 ),
            #                 row=1,
            #                 col=1,
            #             )
            fig.add_trace(
                go.Mesh3d(
                    z=z, x=x, y=y, intensity=r, colorscale="Viridis", showscale=False
                ),
                row=1,
                col=2,
            )
            for tracking_object in tracking_objects:
                x_data = tracking_object.track[:, 0]
                y_data = tracking_object.track[:, 1]
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
                x_data = tracking_object.track[:, 0]
                y_data = tracking_object.track[:, 1]
                color = tracking_object.color
                track_phi = np.arctan2(y_data, x_data)
                track_theta = np.sqrt(y_data**2 + x_data**2)

#                 fig.add_trace(
#                     go.Scatter(
#                         x=track_phi * 180 / np.pi,
#                         y=track_theta * 180 / np.pi,
#                         mode="lines+markers",
#                         name="Data",
#                         line={"color": color, "width": 2},
#                         marker={"color": color, "size": 2.5},
#                     ),
#                     row=2,
#                     col=2,
#                 )
                r = 30
                x_map = r * np.sin(track_theta) * np.cos(track_phi)
                y_map = r * np.sin(track_theta) * np.sin(track_phi)
                lat_map, lon_map = convert_to_map(c_lon, c_lat, x_map, y_map)
                self.map_fig.add_trace(
                    go.Scattermapbox(
                        mode="markers+lines",
                        lon=lon_map,
                        lat=lat_map,
                        line={"color": color, "width": 3},
                        marker={"color": color, "size": 3.5},
                    )
                )

                self.map_fig.add_trace(
                    go.Scattermapbox(
                        mode="markers",
                        lon=[c_lon],
                        lat=[c_lat],
                        marker={"color": "rgb(255, 0, 0)", "size": 5.5},
                    )
                )

            fig.add_trace(
                go.Scatter(
#                     x= [1,0,-1,0],
                    y= self.max_vals,
                    mode="lines+markers",
                    name="Data",
                    line={"color": "rgb(0, 255, 0)"},
                    marker={"color": "rgb(0, 255, 0)", "size": 8},
                ),
                row=2,
                col=2,
            )
            # fig.add_trace(go.Heatmap(z=(grid)), row=2, col=2)
            #             fig.update_layout(uirevision=1)
            fig.update_layout(width=700, height=800, uirevision=1)
            fig.update_yaxes(
                range=[-1.7, 1.7], scaleanchor="x", scaleratio=1, row=2, col=1
            )
            fig.update_xaxes(range=[-1.7, 1.7], row=2, col=1)
            fig.update_yaxes(range=[0, 90], row=2, col=2)
            fig.update_xaxes(range=[-180, 180], row=2, col=2)
            fig.update_layout(showlegend=False)

            return fig, self.map_fig


if __name__ == "__main__":
    app.run(debug=True)
