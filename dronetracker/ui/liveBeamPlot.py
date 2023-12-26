import datetime
import numpy as np
from numpy import sin, cos
import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from beamforming.prototypeTracker import Tracker
class UI():
    def __init__(self, tracker, streamer):
        self.tracker = tracker
        self.streamer = streamer
        self.phi = tracker.phi
        self.theta = tracker.theta
        self.r = self.theta
        self.x = cos(self.phi) * self.r
        self.y = sin(self.phi) * self.r
        self.block_len = 1024*2
        self.layout = go.Layout(
            autosize=False,
            width=900,
            height=900,
            margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        )
        self.xx = sin(self.theta)*cos(self.phi)
        self.yy = sin(self.theta)*sin(self.phi)
        self.zz = cos(self.theta)
        self.x_hat = []
        self.y_hat = []


        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        self.app = Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.layout = html.Div(
            html.Div([
                html.H4('HERON'),
                dcc.Graph(id='live-update-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=1*100, # in milliseconds
                    n_intervals=0
                )
            ])
        )
        self.setup_callbacks()

    def run(self):
        self.streamer.start_stream()
        self.app.run(debug=True)

    def setup_callbacks(self):
    # Multiple components can update everytime interval gets fired.
        @callback(Output('live-update-graph', 'figure'),
                      Input('interval-component', 'n_intervals'))
        def update_graph_live(n):
            print("ENTER CALLBACK")
            block = self.streamer.get_block(self.block_len)
            if block is None:
                print('----Done---')
                self.streamer.end_stream()
                self.streamer.start_stream()
                return None
            response,meas,predicted, *_ = self.tracker.track(block)
            self.x_hat.append(predicted[0])
            self.y_hat.append(predicted[1])
#             r = np.log(response+1) + 10
            r = response
            x = self.xx
            y = self.yy
            z = self.zz
            fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}, {'type':'scatter'}]])
#             fig = 
#             fig = go.Figure(data=[go.Mesh3d(z=(response), x=(self.x), y=(self.y),
            fig.add_trace(go.Mesh3d(z=(response), x=(self.x), y=(self.y),
                                intensity=response, colorscale='Viridis'),
                            row=1,col=1)
            fig.add_trace(go.Mesh3d(z=z, x=x, y=y,
                                    intensity=r, colorscale='Viridis'),
                            row=1,col=2)

            fig.add_trace(go.Scatter(
                            x=self.x_hat,
                            y=self.y_hat,
                            mode="lines+markers",
                            name="Data",
                            line={"color": "rgb(0, 255, 0)"},
                            marker={"color": "rgb(0, 255, 0)", "size": 8},
#                             layout_yaxis_range=[-4,4],
                            ), row=1, col=3)
#             fig = go.Figure(data=[go.Mesh3d(z=(z), x=(x), y=(y),
#                             intensity=r, colorscale='Viridis')],
#                             layout=self.layout).set_subplots(1,1)
            fig.update_layout(width=1800, height=900, uirevision=1)
            fig.update_yaxes(range=[-1.7, 1.7], scaleanchor="x", scaleratio=1,row=1, col=3)
            fig.update_xaxes(range=[-1.7, 1.7], row=1, col=3)
#             fig.update_layout(width=1500, height=900)
#             fig2 = go.Figure(data=[go.Mesh3d(z=(response), x=(self.x), y=(self.y),
#                             intensity=r, colorscale='Viridis')],
#                             layout=self.layout).set_subplots(1,1)
            return fig


if __name__ == '__main__':
    app.run(debug=True)
