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
from datetime import datetime

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from beamforming.prototypeTracker import Tracker
from beamforming.kalmanTracker import KalmanTracker, Kalman_track_object
from utils.maps import convert_to_map
from AudioInterface.waveStreamer import WavStreamer
from AudioInterface.tcpStreamer import TcpStreamer
from ui.layout import make_layout
from utils.communication import Communication

class UI:
    def __init__(self, streamer_settings, tracker_settings, tracker_len= 15, json_port=6667, use_compass=False, is_online=False):
        self.block_len = tracker_settings.get("block_len", 2048)
        self.use_compass = use_compass

        self.streamer_settings = streamer_settings
        self.tracker_settings = tracker_settings
        self.is_recording = False

        self.json_port = json_port

        self.layout = go.Layout(
            autosize=False,
            width=900,
            height=900,
            margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        )
#         external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
#         external_stylesheets = ["./dronetracker/ui/style.css"]
        self.map_fig = None
        if is_online:
            self._setup_map()
        else:
            self.map_fig = go.Figure(go.Scatter())
            self.map_fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},uirevision="constant", showlegend=False)

        self.is_online = is_online

#         self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app = Dash(__name__)
        self.app.layout = make_layout(self.map_fig)
        self.setup_callbacks()

        self.streamers = []
        self.trackers = []
        self.communicators = []
        self.communicator = None
        self.tracker = None
        self.streamer = None
        self.angle = None
        self.track_len = tracker_len


        self.x_hat = []
        self.y_hat = []
        self.max_vals = []


    def _update_plot_coordinates(self):
        self.phi_beamsearch_sphere, self.theta_beamsearch_sphere = self.tracker.get_sphere()

        r_flat_proj = self.theta_beamsearch_sphere
        self.x = cos(self.phi_beamsearch_sphere) * r_flat_proj
        self.y = sin(self.phi_beamsearch_sphere) * r_flat_proj


        self.x_sphere = sin(self.theta_beamsearch_sphere) * cos(
            self.phi_beamsearch_sphere
        )
        self.y_sphere = sin(self.theta_beamsearch_sphere) * sin(
            self.phi_beamsearch_sphere
        )
        self.z_sphere = cos(self.theta_beamsearch_sphere)

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
            mapbox_center_lat=center_lat,
            mapbox_center_lon=center_lon,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

    def generate_info(self, data):
        content = []
        content.append(html.P(f'Temperature: {data["sensor_temperature"]:.2f}'))
        content.append(html.P(f'Air Pressure: {data["sensor_pressure"]:.2f}'))
        content.append(html.P(f'GNSS Satelites: {data["gnss_satelite_count"]:.2f}'))
        content.append(html.P(f'Streaming Speed: {data["streaming_speed"]:.2f}'))
        content.append(html.P(f'Used Array Angle: {self.tracker.angle:.2f}'))
        content.append(html.P(f'Sound Speed: {self.tracker.v_m:.2f}'))
        return content

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
                    str(Path(__file__).parent.parent / "configs" / "umbrella0.toml"),
                )

            return "", "", {"display": "none"}, "number", "", ""

        @callback(
            [
                Output("submit-out", "children"),
                Output("open-con-list", "children"),
                Output("sel-con", "options"),
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
                            f"Current Connection: {self.streamer.name}",
                        )

                self.communicator = Communication()
                self.communicator.start(arr_info, arr_conf + 1)
                self.communicators.append(self.communicator)

                data = None
                while data is None:
                    data = self.communicator.getData()

                angle = data["sensor_angle"]

                lat, lon = (None, None)
                if data["gnss_fix"]:
                    lat = data["gnss_latitude"]
                    lon = data["gnss_longitude"]
                temperature = data["sensor_temperature"]

                compass_angle = data["sensor_heading"]

                self.vm = 331 * np.sqrt(1 + temperature / 273)

                self.tracker_settings["v_m"] = self.vm
                self.tracker = KalmanTracker(**self.tracker_settings)
                self.tracker.init_umbrella_array(angle, lat=lat, lon=lon)
                self._update_plot_coordinates()

                self.streamer = TcpStreamer(arr_info, port=arr_conf)
                self.streamers.append(self.streamer)
                self.trackers.append(self.tracker)
                self.streamer.start_stream()
                message = "Connected to ip"
                pass  # TODO
            if arr_type == "WAV":
                self.tracker = KalmanTracker(**self.tracker_settings)
                self.tracker.init_config_array(arr_conf)
                self.trackers.append(self.tracker)
                self._update_plot_coordinates()

                self.streamer = WavStreamer(arr_info, 1024 * 4)
                self.streamers.append(self.streamer)
                self.streamer.start_stream()
                self.communicator = None
                self.communicators.append(None)
                message = "Connectet to Wav"
            curr_streamer_name = "None"
            if self.streamer is not None:
                curr_streamer_name = self.streamer.name
            return (
                message,
                html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                f"Current Connection: {curr_streamer_name}")

        @callback(
            [
#                 Output("open-con-list", "children", allow_duplicate=True),
#                 Output("sel-con", "options", allow_duplicate=True),
#                 Output("sel-con", "value", allow_duplicate=True),
                Output("curr-con", "children", allow_duplicate=True),
                Output("dis-but-err", "children", allow_duplicate=True),
            ],
            Input("sel-con", "value"),
            prevent_initial_call=True,
        )
        def switch_conn(conn):
            print("/|"*200)
            print(conn)
            streamer_ind = -1
            curr_streamer_name = self.streamer.name
            for i, streamer in enumerate(self.streamers):
                if streamer.name != conn:
                    continue
                streamer_ind = i
            if streamer_ind == -1:
                return (
                        f"Current Connection: {curr_streamer_name}",
                        "Connection not available")
            self.tracker =self.trackers[streamer_ind]
            self.streamer =self.streamers[streamer_ind]
            if isinstance(self.streamer, TcpStreamer):
                self.communicator = self.communicators[streamer_ind]
            else:
                self.communicator = None
            self._update_plot_coordinates()
            return  (
                    f"Current Connection: {self.streamer.name}",
                    "Streamer changed")


        @callback(
            [
                Output("open-con-list", "children", allow_duplicate=True),
                Output("sel-con", "options", allow_duplicate=True),
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
                return (
                        html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                        [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                        f"Current Connection: {curr_streamer_name}",
                        "No Streamer to close")
            self.trackers.pop(streamer_ind)
            self.streamers.pop(streamer_ind)
            self.communicators.pop(streamer_ind)

            if isinstance(self.streamer, TcpStreamer):
                self.communicator.stop()
                del self.communicator
            self.communicator = None
            del self.streamer
            del self.tracker
            self.streamer = None
            self.tracker = None
            if len(self.streamers) > 0:
                self.streamer = self.streamers[0]
                self.tracker = self.trackers[0]
                self._update_plot_coordinates()
                curr_streamer_name = self.streamer.name
            return  (
                    html.Ul([html.Li(f"{streamer.name}") for streamer in self.streamers]),
                    [{"value": streamer.name, "label": streamer.name} for i, streamer in enumerate(self.streamers)],
                    f"Current Connection: {curr_streamer_name}",
                    "Streamer closed")

        @callback(
            Output("rec-but", "children", allow_duplicate=True),
            Input("rec-but", "n_clicks"),
            prevent_initial_call=True,
        )
        def record(n):

            if self.is_recording and self.streamer is not None:
                self.streamer.stop_recording()
                self.is_recording = False
                return ("Start Recording")
            elif self.streamer is not None:

                current_datetime = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                datetime_str = str(current_datetime)
                out_file = f'./out/{datetime_str}.wav'
                self.streamer.start_recording(out_file)
                self.is_recording = True
                return ("Stop Recording")
            return ("Start Recording")

        # Multiple components can update everytime interval gets fired.
        @callback(
            [
                Output(component_id="live-beam-plots", component_property="figure"),
                Output(component_id="live-update-graph", component_property="figure"),
                Output("div-info", "children"),
                Output("pos-div", "children"),
            ],
            Input("beam-plots", "n_intervals"),
        )
        def update_graph_live(n):
            print("ENTER CALLBACK")
            # TODO Make object field
            fig = make_subplots(
                rows=2,
                cols=1,
                specs=[
                    [{"type": "mesh3d"}],
                    [{"type": "mesh3d"}],
#                     [{"type": "scatter"}, {"type": "scatter"}],
                ],
            )
            info_div = html.Div()
            fig.update_layout(width=700, height=800, uirevision='constant')
            if self.tracker is None or self.streamer is None:
                return fig, self.map_fig, html.Div([html.P()]), html.Div([html.P()])

            if self.communicator is not None:
                self.communicator.getData()

                data = None
                while data is None:
                    data = self.communicator.getData()
                angle = data["sensor_angle"]
                if self.tracker.needs_update(angle):
                    tracker = self.tracker
                    self.tracker = None
                    tracker.update_umbrella_array(angle)
                    self.tracker = tracker

                if data["gnss_fix"]:
                    self.tracker.update_pos(data["gnss_latitude"], data["gnss_longitude"])

                if self.use_compass:
                    compass_angle = -1 * radians(data["sensor_heading"])
                else:
                    compass_angle = 0
                info_div = self.generate_info(data)
            else:
                angle = 0
                compass_angle = 0

            block = self.streamer.get_block(self.block_len)
            if block is None:
                print("----Done---")
                self.streamer.end_stream()
                self.streamer.start_stream()
                del self.streamer
                del self.tracker
                self.tracker = None
                self.streamer = None
                return fig, self.map_fig, html.Div(info_div), html.Div(id='none')
            response, meas, tracking_objects, max_val, *_ = self.tracker.track(block, compass_angle)
            self.max_vals.append(max_val)
#             r = response
#             x = self.x_sphere
#             y = self.y_sphere
#             z = self.z_sphere
            fig.add_trace(
                go.Mesh3d(
                    z=(response),
                    x=(self.x),
                    y=(self.y),
                    intensity=response,
                    colorscale="Viridis",
                    showscale=True,
                    cmin=0,
                    cmax=1
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Mesh3d(
                    z=self.z_sphere, x=self.x_sphere, y=self.y_sphere, intensity=response, colorscale="Viridis", showscale=True, cmin=0, cmax=1,
                ),
                row=2,
                col=1,
            )

            self.map_fig.data = []
#             c_lon = 8.8191
#             c_lat = 47.22324
            c_lon = 8.8189
            c_lat = 47.22321

            t_tmp = np.linspace(0, 2*np.pi, 50)
            x_circ = 31 * np.cos(t_tmp)
            y_circ = 31 * np.sin(t_tmp)
            if self.is_online:
                lat_map, lon_map = convert_to_map(self.tracker.c_lon, self.tracker.c_lat, x_circ, y_circ)
                self.map_fig.add_trace(
                    go.Scattermapbox(
                        mode="markers+lines",
                        lon=lon_map,
                        lat=lat_map,
                        line={"color": "#FF0000", "width": 0.5},
                        marker={"color": "#FF0000", "size": 0.2},
                    )
                )
                self.map_fig.add_trace(
                    go.Scattermapbox(
                        mode="markers",
                        lon=[self.tracker.c_lon],
                        lat=[self.tracker.c_lat],
                        marker={"color": "rgb(255, 0, 0)", "size": 5.5},
                    )
                )
                self._update_map_center(self.tracker.c_lon, self.tracker.c_lat)
            else:
                self.map_fig.add_trace(
                    go.Scatter(
                        mode="markers+lines",
                        y=y_circ,
                        x=x_circ,
                        line={"color": "#FF0000", "width": 0.5},
                        marker={"color": "#FF0000", "size": 0.2},
                    )
                )
                self.map_fig.add_trace(
                    go.Scatter(
                        mode="markers",
                        x=[0],
                        y=[0],
                        marker={"color": "rgb(255, 0, 0)", "size": 5.5},
                    )
                )


            positions_div_content = []
            for tracking_object in tracking_objects:
                if tracking_object.track.shape[0] < 5 or (tracking_object.track.shape[0] < 11 and tracking_object.n_predictions > 3):
                    continue
                x_data = tracking_object.track[-self.track_len:, 0]
                y_data = tracking_object.track[-self.track_len:, 1]
                color = tracking_object.color
                track_phi = np.arctan2(y_data, x_data)
                track_theta = np.sqrt(y_data**2 + x_data**2)
                positions_div_content.append(
                        html.P(f'Phi:   {degrees(track_phi[-1]):.2f}', style={"color": color}))
                positions_div_content.append(
                        html.P(f'Theta: {degrees(track_theta[-1]):.2f}', style={"color": color}))
                r = 30
                x_map = r * np.sin(track_theta) * np.cos(track_phi)
                y_map = r * np.sin(track_theta) * np.sin(track_phi)
                if self.is_online:
                    lat_map, lon_map = convert_to_map(self.tracker.c_lon, self.tracker.c_lat, x_map, y_map)
                    self.map_fig.add_trace(
                        go.Scattermapbox(
                            mode="markers+lines",
                            lon=lon_map,
                            lat=lat_map,
                            line={"color": color, "width": 2.5},
                            marker={"color": color, "size": 1.5},
                        )
                    )
                    self.map_fig.add_trace(
                        go.Scattermapbox(
                            mode="markers",
                            lon=[lon_map[-1]],
                            lat=[lat_map[-1]],
                            marker={"color": color, "size": 9.5},
                        )
                    )
                    continue
                self.map_fig.add_trace(
                    go.Scatter(
                        mode="markers+lines",
                        x=x_map,
                        y=y_map,
                        line={"color": color, "width": 2.5},
                        marker={"color": color, "size": 1.5},
                    )
                )
                self.map_fig.add_trace(
                    go.Scatter(
                        mode="markers",
                        y=[y_map[-1]],
                        x=[x_map[-1]],
                        marker={"color": color, "size": 9.5},
                    )
                )


            # fig.add_trace(go.Heatmap(z=(grid)), row=2, col=2)
            #             fig.update_layout(uirevision=1)

            camera1 = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1, y=-1, z=1.5)
            )
            camera2 = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2)
            )
            fig.update_yaxes(range=[-2.7, 2.7], row=1, col=1)
            fig.update_xaxes(range=[-2.7, 2.7], row=1, col=1)
            fig.update_scenes(zaxis_range=[0, 1.1], row=1, col=1)

            fig.update_layout(width=700, height=800, uirevision='constant')
#             self.map_fig.update_layout(uirevision='constant')
            fig.layout.scene2.camera =camera2
            fig.layout.scene1.camera =camera1
            fig.update_layout(showlegend=False)
            self.map_fig.update_yaxes(
                range=[-35, 35], scaleanchor="x", scaleratio=1)
#             self.map_fig.update_layout(showlegend=False)

            return fig, self.map_fig, html.Div(info_div, style={"border-top": "solid black", "border-bottom": "solid black"}), html.Div(positions_div_content)


if __name__ == "__main__":
    app.run(debug=True)
