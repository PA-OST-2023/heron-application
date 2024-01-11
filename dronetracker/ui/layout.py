import dash
from dash import Dash, dcc, html, Input, Output, callback, State


def make_layout(map_fig):

    layout = html.Div(
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
                        html.Div(style={"margin-top": "15px"}),
                        html.Div([], id="div-info", ),
                        html.Div(style={"margin-top": "15px"}),
                        html.Button("Start Recording", id="rec-but"),
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
                        dcc.Graph(figure=map_fig, id="live-update-graph"),
                        html.Div([], id='pos-div')
                    ],
                    className="four columns",
                ),
            ]
        )

    return layout
