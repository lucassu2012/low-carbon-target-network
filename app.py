import json
import datetime
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import plotly
from plotly import express

import dash
from dash import dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output, State
import dash_daq as daq

from preprocessing import data_preprocessing

# %% Data Preprocessing
# data folder
DATA_PATH = 'data'

# selection_dict
df = pd.read_csv(DATA_PATH + '/03_eng_cal.csv')
df['5G_Deployment_Year'] = np.random.normal(2030, 5, len(df)).astype(int)
lon_center, lat_center = df.Site_Longitude.mean(), df.Site_Latitude.mean()
year_slider_min, year_slider_max = df['5G_Deployment_Year'].min(), df['5G_Deployment_Year'].max()
city_name_list = list(df['Site_Area'].unique())

df_power_generate = pd.read_csv(DATA_PATH + '/04_power_date.csv')
df_power_generate['Date'] = pd.to_datetime(df_power_generate['Date'])

kpi_cols_list_cr = ['KPI_TEE_2345G_Current', 'KPI_SEE_2345G_Current', 'KPI_RER_2345G_Current',
                    'KPI_NCI_2345G_Current', 'KPI_NC_2345G_Current']
kpi_cols_list_tb = ['KPI_TEE_2345G_Tobe', 'KPI_SEE_2345G_Tobe', 'KPI_RER_2345G_Tobe',
                    'KPI_NCI_2345G_Tobe', 'KPI_NC_2345G_Tobe']
table_col_list = ['Site_Name', 'Site_City', 'Site_Area', 'Site_Room', 'Site_Type', 'Site_Scene'] + kpi_cols_list_tb
supply_col_list = ['Selection_PV', 'Selection_Battery_Voltage']
power_col_list = ['Selection_Rectifier', 'Selection_Battery_HighTemp', 'Selection_Battery_Supply']
utilization_col_list = ['Selection_FCS', 'Selection_I2O']
management_col_list = ['Selection_Power_Star']
Power_Saving_Cols = [
    'Power_Saving_PV',
    'Power_Saving_Rectifier',
    'Power_Saving_Cable',
    'Power_Saving_FCS',
    'Power_Saving_I2O',
    'Power_Saving_PowerStar',
]
Power_Saving_Text = [
    'PS_PV',
    'PS_Rectifier',
    'PS_Cable',
    'PS_FCS',
    'PS_I2O',
    'PS_PowerStar',
]
Power_Consumption_Cols = [
    'Power_Consumption_Aircondition',
    'Power_Consumption_FCS',
    'Power_Consumption_Site_Radio_234G',
    'Power_Consumption_Site_Radio_5G',
    'Power_Consumption_Cable_Loss_Current',
]
Power_Consumption_Text = [
    'PC_AirCondition',
    'PC_FCS',
    'PC_234G',
    'PC_5G',
    'PC_Cable_Loss',
]
saving_col_list = ['Electricity_Saving_Total', 'Rental_Saving', 'Opex_Saving_Total', 'Carbon_Saving']

table_columns = [
    dict(id='a', name='Align left (10)', type='numeric',
         format=dash_table.Format.Format().align(dash_table.Format.Align.left).fill('-').padding_width(10)),
    dict(id='a', name='Align right (8)', type='numeric',
         format=dash_table.Format.Format(align=dash_table.Format.Align.right, fill='-', padding_width=8)),
    dict(id='a', name='Align center (6)', type='numeric',
         format=dict(specifier='-^6'))
]

table_col_list = [
    dict(id='Site_Name', name='Site_Name', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    # dict(id='Site_City', name='Site_City', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='Site_Area', name='Site_Area', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='Site_Room', name='Site_Room', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    # dict(id='Site_Type', name='Site_Type', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    # dict(id='Site_Scene', name='Site_Scene', type='string', format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='KPI_TEE_2345G_Tobe', name='KPI_TEE', type='numeric', format=Format(precision=2, scheme=Scheme.decimal)),
    dict(id='KPI_SEE_2345G_Tobe', name='KPI_SEE', type='numeric', format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='KPI_RER_2345G_Tobe', name='KPI_RER', type='numeric', format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='KPI_NCI_2345G_Tobe', name='KPI_NCI', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='KPI_NC_2345G_Tobe', name='KPI_NC', type='numeric', format=Format(precision=0, scheme=Scheme.fixed))
]

# API
mapbox_access_token = "pk.eyJ1IjoibHVjYXNzdTIwMTIiLCJhIjoiY2t3MGxsMGI0N3NiODJvczFiMjhkdHl0ZyJ9.bDUX-wn-2d9OhzvGkNgo2g"
plotly.express.set_mapbox_access_token(mapbox_access_token)

#######################################################################################################################
# %% create app
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    # external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Low-Carbon Target Network"
server = app.server

# %% Create app layout
app.layout = html.Div(
    [
        # dcc.Store stores the preprocessed datasets
        dcc.Store(id="pre_data"),
        dcc.Store(id="datasets"),
        html.Div(
            [
                html.Div(
                    html.A(
                        [
                            html.Img(
                                src=app.get_asset_url("HUAWEI-logo.png"),

                                id="plotly-image",
                                style={
                                    "height": "80px",
                                    "width": "auto",
                                    # "margin-bottom": "25px",
                                },
                            )
                        ], href="https://www.huawei.com/en/",
                    ),
                    className="one-third column",
                ),
                html.Div(
                    html.Div(
                        [
                            html.H3(
                                "Low-Carbon Target Network Online",
                                # style={"margin-bottom": "0px"},
                            ),
                            html.H5(
                                "Geographical Overview",
                                # style={"margin-top": "0px"}
                            ),
                        ]
                    ),
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        # html.A(
                        #     html.Button("Learn More", id="learn-more-button"),
                        #     href="https://www.huawei.com/en/",
                        # )
                        daq.BooleanSwitch(id='switch', on=False, label="Enable S-P-U-M Solution", labelPosition="top"),
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Filter by 5G construction date (or select range in histogram):",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=year_slider_min,
                            max=year_slider_max,
                            value=[year_slider_min, year_slider_max],
                            marks={v: v for v in range(year_slider_min, year_slider_max + 1)[::5]},
                            step=1,
                            className="dcc_control",
                            allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.P("Filter by city name:", className="control_label"),
                        dcc.RadioItems(
                            id="city_name_selector",
                            options=[
                                {"label": "All ", "value": "all"},
                                {"label": "Customize ", "value": "custom"},
                            ],
                            value="all",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="city_names",
                            options=[
                                {"label": i.title(), "value": i} for i in sorted(df['Site_Area'].unique())
                            ],
                            multi=True,
                            value=list(sorted(df['Site_Area'].unique())),
                            className="dcc_control",
                        ),
                        dcc.Checklist(
                            id="lock_selector",
                            options=[{"label": "Indoor", "value": "Indoor"},
                                     {"label": "Outdoor", "value": "Outdoor"}],
                            labelStyle={'display': 'inline-block'},
                            value=["Indoor", "Outdoor"],
                            className="dcc_control",
                        ),
                        html.P("'S-P-U-M' System:", className="control_label"),
                        dcc.RadioItems(
                            id="power_type_selector",
                            options=[
                                {"label": "'供' - Supply ", "value": "Supply"},
                                {"label": "'配' - Power ", "value": "Power"},
                                {"label": "'用' - Utilization ", "value": "Utilization"},
                                {"label": "'管' - Management ", "value": "Management"},
                                # {"label": " 自定义 ", "value": "custom"},
                                {"label": "All", "value": "all"},
                            ],
                            value="all",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="power_types",
                            options=[
                                {
                                    "label": i.replace('Selection_', ''), "value": i
                                } for i in supply_col_list + power_col_list + utilization_col_list + management_col_list
                            ]
                            ,
                            multi=True,
                            value=[
                                {
                                    "label": i.replace('Selection_', ''), "value": i
                                } for i in supply_col_list
                            ],
                            className="dcc_control",
                        ),
                        html.P("Select by kpi name:", className="control_label"),
                        dcc.Dropdown(
                            id="kpi_selector",
                            options=[
                                {"label": i, "value": i} for i in sorted(df.columns.to_list()[10:])
                            ],
                            multi=False,
                            value=df.columns.to_list()[-1],
                            placeholder="kpi name",
                            style={"border": "0px solid black"},
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container six columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        daq.LEDDisplay(
                                            id='LED-TEE',
                                            label="TEE, GB/kWh",
                                            labelPosition='top',
                                            value=0,
                                            size=32,
                                            color="#0000FF",
                                            # backgroundColor="#F1F3F5",
                                        ),
                                        html.H6(id="TEE", style={'text-align': 'center'}),
                                    ],
                                    id="tee",
                                    className="pretty_container three columns",
                                ),
                                html.Div(
                                    [
                                        daq.LEDDisplay(
                                            id='LED-RER',
                                            label="RER, %",
                                            labelPosition='top',
                                            value=0,
                                            size=32,
                                            color="#0000FF"
                                        ),
                                        html.H6(id="RER", style={'text-align': 'center'}),
                                    ],
                                    id="rer",
                                    className="pretty_container three columns",
                                ),
                                html.Div(
                                    [
                                        daq.LEDDisplay(
                                            id='LED-SEE',
                                            label="SEE, %",
                                            labelPosition='top',
                                            value=0,
                                            size=32,
                                            color="#0000FF"
                                        ),
                                        html.H6(id="SEE", style={'text-align': 'center'}),
                                    ],
                                    id="see",
                                    className="pretty_container three columns",
                                ),
                                html.Div(
                                    [
                                        daq.LEDDisplay(
                                            id='LED-NCI',
                                            label="NCI, kg CO2/TB",
                                            labelPosition='top',
                                            value=0,
                                            size=32,
                                            color="#FF5E5E"
                                        ),
                                        html.H6(id="NCI", style={'text-align': 'center'}),
                                    ],
                                    id="nci",
                                    className="pretty_container three columns",
                                ),
                                html.Div(
                                    [
                                        daq.LEDDisplay(
                                            id='LED-NC',
                                            label="NC, ton CO2",
                                            labelPosition='top',
                                            value=0,
                                            size=32,
                                            color="#FF5E5E"
                                        ),
                                        html.H6(id="NC", style={'text-align': 'center'}),
                                    ],
                                    id="nc",
                                    className="pretty_container three columns",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="count_graph",
                                          config={
                                              'displayModeBar': False
                                          },
                                          )
                            ],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="six columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="main_graph",
                                  # style={
                                  #     "height": "auto",
                                  #     "width": "auto",
                                  #     "margin-left": "0px",
                                  # },
                                  config={
                                      'displayModeBar': False
                                  }
                                  )
                    ],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [
                        dcc.Graph(id="individual_graph",
                                  # style={
                                  #     "margin-left": "0px",
                                  # },
                                  config={
                                      'displayModeBar': False
                                  }
                                  ),
                    ],
                    className="pretty_container six columns",
                )
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dash_table.DataTable(
                            id="main_table",
                            columns=table_col_list,
                            data=df.to_dict("records"),
                            page_size=10,
                            sort_action="native",
                            row_selectable="multi",
                            style_cell={"textAlign": "center", 'fontSize': 14, 'font-family': 'sans-serif'},
                            style_header={
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "border": "none",
                                # "backgroundColor": "transparent",
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white'
                            },
                            style_data={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                            style_table={"overflowX": "auto", "width": "calc(100% - 26px)", },
                            style_data_conditional=[
                                {
                                    "if": {"state": "selected"},
                                    "backgroundColor": "transparent",
                                    'color': 'black',
                                    "border": "0px solid transparent",
                                }
                            ],
                        ),
                        # html.Button("clear selection", id="clear",
                        #             style={
                        #                 "margin-top": "0px",
                        #                 "margin-bottom": "0px",
                        #             },
                        #             ),
                    ],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [
                        dcc.Graph(id="pie_graph",
                                  style={
                                      "margin-left": "0px",
                                  },
                                  config={
                                      'displayModeBar': False
                                  }
                                  )
                    ],
                    className="pretty_container six columns",
                )
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

#################################################################################################
# Create callbacks
app.clientside_callback(
    dash.dependencies.ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


# %% Create callbacks
# function to create data
@app.callback(
    Output("pre_data", "data"),
    Input("power_types", "value"),
)
def read_data(power_types):
    selection_dict = {
        'Selection_Battery_Voltage': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_Battery_Supply': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_FCS': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_I2O': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_Power_Star': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_Rectifier': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_PV': {3: 0, 2: 0, 1: 0, 0: 0},
        'Selection_Battery_HighTemp': {3: 0, 2: 0, 1: 0, 0: 0},
    }
    for col in power_types:
        selection_dict.get(col).update({3: 1, 2: 1, 1: 1, 0: 0})

    df_pre = data_preprocessing(data_path=DATA_PATH + '/02_eng_clean.csv', selection_dict=selection_dict)
    datasets = {
        'df_pre': df_pre.to_json(orient='split', date_format='iso'),
    }

    return json.dumps(datasets)


# function to filter data
@app.callback(
    Output("datasets", "data"),
    [
        Input("pre_data", "data"),
        Input("city_names", "value"),
        Input("year_slider", "value"),
        Input("lock_selector", "value"),
    ],
)
def preprocess_data(json_datasets, city_names, year_slider, lock_selector):
    datasets = json.loads(json_datasets)
    df = pd.read_json(datasets['df_pre'], orient='split')

    df_filter_site = df[
        (df["Site_Area"].isin(city_names))
        # & ((df["Battery_Type_Current"].isin(power_types)) | (df["Cooling_Type"].isin(power_types)))
        & (df["5G_Deployment_Year"] >= year_slider[0])
        & (df["5G_Deployment_Year"] <= year_slider[1])
        & (df['Site_Room'].isin(lock_selector))
        ]

    date_index = pd.date_range(start=datetime.date(year_slider[0], 1, 1),
                               end=datetime.date(year_slider[1], 1, 1),
                               freq='Y')

    df_aggregate_date = pd.DataFrame(
        {
            'Site_Name': 'Test_123',
            'Date': date_index,
            'NCI': 100 + np.cumsum(np.random.normal(10, 1, len(date_index)) - 10),
            'NC': 1000 + np.cumsum(np.random.normal(100, 1, len(date_index)) - 100),
        },
    )

    datasets = {
        'df_filter_site': df_filter_site.to_json(orient='split', date_format='iso'),
        'df_aggregate_date': df_aggregate_date.to_json(orient='split', date_format='iso'),
    }

    return json.dumps(datasets)


# Radio -> multi
@app.callback(
    Output("city_names", "value"),
    Input("city_name_selector", "value")
)
def display_status(selector):
    if selector == "all":
        return list(df['Site_Area'].unique())
    elif selector == "custom":
        return []


# Radio -> multi
@app.callback(
    Output("power_types", "value"),
    Input("power_type_selector", "value")
)
def display_type(selector):
    if selector == "all":
        return supply_col_list + power_col_list + utilization_col_list + management_col_list
    elif selector == "Supply":
        return supply_col_list
    elif selector == "Power":
        return power_col_list
    elif selector == "Utilization":
        return utilization_col_list
    elif selector == "Management":
        return management_col_list
    elif selector == "custom":
        return []


# count_graph -> year_slider
@app.callback(
    Output("year_slider", "value"),
    Input("count_graph", "selectedData")
)
def update_year_slider(count_graph_selected):
    if count_graph_selected is None:
        return [year_slider_min, year_slider_max]

    years = [int(point["x"]) for point in count_graph_selected["points"]]
    return [min(years), max(years)]


# Selectors -> kpi
@app.callback(
    [
        Output("TEE", "children"),
        Output("SEE", "children"),
        Output("RER", "children"),
        Output("NCI", "children"),
        Output("NC", "children"),
        Output('LED-TEE', 'value'),
        Output('LED-SEE', 'value'),
        Output('LED-RER', 'value'),
        Output('LED-NCI', 'value'),
        Output('LED-NC', 'value')
    ],
    [
        Input("datasets", "data"),
        Input('switch', 'on')
    ]
)
def update_text(json_datasets, switch):
    datasets = json.loads(json_datasets)
    df_filter_site = pd.read_json(datasets['df_filter_site'], orient='split')

    tee_cr, see_cr, rer_cr, nci_cr, nc_cr = (
            df_filter_site[kpi_cols_list_cr].mean() * np.array([1, 100, 100, 1, 1])).to_list()
    tee_tb, see_tb, rer_tb, nci_tb, nc_tb = (
            df_filter_site[kpi_cols_list_tb].mean() * np.array([1, 100, 100, 1, 1])).to_list()

    tee = f'{round(tee_cr, 1)} ➜ {round(tee_tb, 1)}' if switch is True else '➜'
    see = f'{int(round(see_cr, 0))} ➜ {int(round(see_tb, 0))}' if switch is True else '➜'
    rer = f'{round(rer_cr, 2)} ➜ {round(rer_tb, 2)}' if switch is True else '➜'
    nci = f'{int(round(nci_cr, 0))} ➜ {int(round(nci_tb, 0))}' if switch is True else '➜'
    nc = f'{int(round(nc_cr / 1000, 1))} ➜ {int(round(nc_tb / 1000, 1))}' if switch is True else '➜'

    led_tee = round(tee_cr, 1) if switch is False else round(tee_tb, 1)
    led_see = int(round(see_cr, 0)) if switch is False else int(round(see_tb, 0))
    led_rer = round(rer_cr, 2) if switch is False else round(rer_tb, 2)
    led_nci = int(round(nci_cr, 0)) if switch is False else int(round(nci_tb, 0))
    led_nc = int(round(nc_cr / 1000, 1)) if switch is False else int(round(nc_tb / 1000, 1))

    return tee, see, rer, nci, nc, led_tee, led_see, led_rer, led_nci, led_nc


# Selectors -> main graph
@app.callback(
    Output("main_graph", "figure"),
    [
        Input("datasets", "data"),
        Input("kpi_selector", "value"),
        Input('main_table', "derived_virtual_data"),
        Input('main_table', "derived_virtual_selected_rows")
    ],
)
def make_main_figure(json_datasets, kpi_selector, derived_virtual_data, derived_virtual_selected_rows):
    datasets = json.loads(json_datasets)
    df_filter_site = pd.read_json(datasets['df_filter_site'], orient='split')

    if len(df_filter_site) == 0:
        raise dash.exceptions.PreventUpdate

    if len(derived_virtual_selected_rows) > 0:
        df_filter_site = pd.DataFrame(derived_virtual_data).loc[derived_virtual_selected_rows]

    figure = plotly.express.scatter_mapbox(df_filter_site,
                                           lat="Site_Latitude",
                                           lon="Site_Longitude",
                                           color=kpi_selector,
                                           size="Data_Volume_234G_Month",
                                           opacity=0.6,
                                           size_max=10,
                                           zoom=9,
                                           hover_name='Site_Name',
                                           color_continuous_scale=plotly.express.colors.sequential.YlGnBu,
                                           labels={kpi_selector: "KPI"}
                                           )

    figure.update_layout(
        template=plotly.io.templates['plotly_dark'],
        autosize=True,
        margin=plotly.graph_objs.layout.Margin(l=5, r=5, t=35, b=5),
        hovermode="closest",
        # plot_bgcolor="#F9F9F9",
        # paper_bgcolor="#F9F9F9",
        showlegend=True,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            # style='carto-darkmatter',
            center=dict(lon=lon_center,
                        lat=lat_center),
            zoom=8,
        ),
        title=dict(
            text=kpi_selector,
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend=dict(
            orientation="v",
            title='kpi',
            font=dict(size=12),
            xanchor="right",
            yanchor="top",
            x=0.99,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
        ),
        updatemenus=[
            dict(
                buttons=(
                    [
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": 9,
                                    "mapbox.bearing": 0,
                                    "mapbox.center.lon": df_filter_site["Site_Longitude"].mean(),
                                    "mapbox.center.lat": df_filter_site['Site_Latitude'].mean(),
                                }
                            ],
                            label="Zoom In",
                            method="relayout",
                        )
                    ]
                ),
                direction="left",
                pad={"r": 0, "t": 0, "b": 0, "l": 0},
                showactive=False,
                type="buttons",
                x=0.005,
                y=0.90,
                xanchor="left",
                yanchor="bottom",
                borderwidth=1,
                # bgcolor="rgba(0,0,0,0)",
                # bordercolor="rgba(0,0,0,0)",
                bgcolor="#323130",
                bordercolor="#6d6d6d",
                # font=dict(color="#FFFFFF"),
            )
        ],
    )

    return figure


# Main graph -> individual graph
@app.callback(
    Output("individual_graph", "figure"),
    [
        # Input("datasets", "data"),
        Input("main_graph", "hoverData"),
        Input("year_slider", "value"),
    ]
)
def make_individual_figure(main_graph_hover, year_slider):
    date_index = pd.date_range(start=datetime.date(year_slider[0], 1, 1),
                               end=datetime.date(year_slider[1], 1, 1),
                               freq='Y')
    df_aggregate_date = pd.DataFrame(
        {
            'Site_Name': 'All',
            'Date': date_index,
            'NCI': 100 + np.cumsum(np.random.normal(10, 1, len(date_index)) - 10),
            'NC': 100 + np.cumsum(np.random.normal(100, 1, len(date_index)) - 100),
        },
    )

    if main_graph_hover is not None:
        # print(main_graph_hover)
        chosen = [point["hovertext"] for point in main_graph_hover["points"]]
        df_aggregate_date['Site_Name'] = chosen[0]
        df = df_aggregate_date[df_aggregate_date['Site_Name'].isin(chosen)]
    else:
        chosen = ['All']
        df = df_aggregate_date

    df = df_aggregate_date.groupby('Date', as_index=False).mean()

    figure = dict(
        data=[
            dict(
                type="scatter",
                mode="lines+markers",
                name="NCI, kg CO2/TB",
                x=df['Date'],
                y=df['NCI'],
                line=dict(shape="spline", smoothing=2, width=2, color="#fac1b7"),
                # line=dict(shape="spline", smoothing=2, width=2),
                marker=dict(symbol="diamond-open"),
            ),
            dict(
                type="scatter",
                mode="lines+markers",
                name="NC, kg CO2",
                x=df['Date'],
                y=df['NC'],
                line=dict(shape="spline", smoothing=2, width=2, color="#92d8d8"),
                # line=dict(shape="spline", smoothing=2, width=2),
                marker=dict(symbol="diamond-open"),
            ),
        ],
        layout=dict(
            autosize=True,
            automargin=True,
            margin=dict(l=30, r=30, b=20, t=40),
            hovermode="closest",
            # template=plotly.io.templates['plotly_dark'],
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9",
            legend=dict(
                font=dict(size=12),
                orientation="h",
                title='kpi',
                xanchor="right",
                yanchor="top",
                x=0.99,
                y=1,
                bgcolor="rgba(0,0,0,0)",
            ),
            title="KPI View of Site: {}".format(chosen[0])
        )
    )

    return figure


# Selectors, main graph -> pie graph
@app.callback(
    Output("pie_graph", "figure"),
    [
        Input("datasets", "data"),
        Input("year_slider", "value")
    ]
)
def make_pie_figure(json_datasets, year_slider):
    datasets = json.loads(json_datasets)
    df_filter_site = pd.read_json(datasets['df_filter_site'], orient='split')

    agg_ps = df_filter_site[Power_Saving_Cols].replace(0, np.nan).sum().to_list()
    agg_pc = df_filter_site[Power_Saving_Cols].replace(0, np.nan).sum().to_list()

    color_list = ['#FFEDA0', '#FA9FB5', '#A1D99B', '#67BD65', '#BFD3E6', '#B3DE69', '#FDBF6F', '#FC9272', '#D0D1E6',
                  '#ABD9E9', '#3690C0', '#F87A72', '#CA6BCC', '#DD3497', '#4EB3D3', '#FFFF33', '#FB9A99', '#A6D853',
                  '#D4B9DA', '#AEB0B8', '#CCCCCC', '#EAE5D9', '#C29A84']

    data = [
        dict(
            type="pie",
            labels=Power_Consumption_Text,
            values=agg_pc,
            name="Total Power Consumption",
            text=Power_Consumption_Text,
            hoverinfo="text+value+percent",
            textinfo="text+value+percent",
            hole=0.5,
            # marker=dict(colors=["#fac1b7", "#a9bb95", "#F9ADA0", "#849E68", "#59C3C3"]),
            # marker=dict(colors=color_list[-len(Power_Consumption_Cols):]),
            domain={"x": [0, 0.45], "y": [0.2, 0.8]},
        ),
        dict(
            type="pie",
            labels=Power_Saving_Text,
            values=agg_ps,
            name="Total Power Saving",
            text=Power_Saving_Text,
            hoverinfo="text+value+percent",
            textinfo="text+value+percent",
            hole=0.5,
            marker=dict(colors=color_list),
            domain={"x": [0.55, 0.8], "y": [0.3, 0.7]},
        ),
    ]

    layout = dict(
        autosize=True,
        automargin=True,
        margin=dict(l=30, r=30, b=20, t=40),
        hovermode="closest",
        # template=plotly.io.templates['plotly_dark'],
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        font=dict(color="#777777"),
        legend=dict(font=dict(size=12,
                              # color="#CCCCCC"
                              ),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h",
                    title='kpi',
                    xanchor="right",
                    yanchor="top",
                    x=0.99,
                    y=1,
                    # margin=dict(l=30, r=30, b=20, t=40),
                    ),
        title="Power Summary: {} to {}".format(year_slider[0], year_slider[1])
    )

    figure = dict(data=data, layout=layout)

    return figure


# Selectors -> count graph
@app.callback(
    Output("count_graph", "figure"),
    [
        Input("pre_data", "data"),
        Input("year_slider", "value"),
    ],
)
def make_count_figure(json_datasets, year_slider):
    datasets = json.loads(json_datasets)
    df_filter_site = pd.read_json(datasets['df_pre'], orient='split')

    g = df_filter_site.groupby('5G_Deployment_Year', as_index=False)[saving_col_list].sum()

    opacity = []
    for i in range(year_slider_min, year_slider_max):
        if i >= int(year_slider[0]) and i <= int(year_slider[1]):
            opacity.append(1.0)
        else:
            opacity.append(0.2)

    figure = plotly.express.bar(g,
                                x='5G_Deployment_Year',
                                y=saving_col_list,
                                # color_discrete_sequence=plotly.express.colors.diverging.Armyrose,
                                # color_discrete_sequence=plotly.express.colors.sequential.RdBu,
                                color_discrete_sequence=plotly.express.colors.diverging.RdBu[::2][::-1],
                                barmode="stack",
                                opacity=opacity,
                                )

    figure.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        # template=plotly.io.templates['plotly_dark'],
        autosize=True,
        margin=dict(l=30, r=30, b=20, t=40),
        title="Saving by Low-Carbon Target Network, USD: {} to {}".format(year_slider[0], year_slider[1]),
        dragmode='select',
        showlegend=True,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        font=dict(color="#777777"),
        legend=dict(
            font=dict(
                size=12,
                # color="#CCCCCC"
            ),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
            title=None,
            xanchor="right",
            yanchor="top",
            x=0.99,
            y=1,
        ),
    )
    # figure.write_html('test.html', auto_open=True)

    return figure


@app.callback(
    Output('main_table', "data"),
    Input("datasets", "data"),
)
def update_table(json_datasets):
    datasets = json.loads(json_datasets)
    df_filter_site = pd.read_json(datasets['df_filter_site'], orient='split')
    return df_filter_site.to_dict('records')


@app.callback(
    Output("main_table", "selected_cells"),
    Output("main_table", "active_cell"),
    Input("clear", "n_clicks"),
)
def clear(n_clicks):
    return [], None


# Main
if __name__ == "__main__":
    app.run_server(debug=False)
