#!/usr/bin/env python

import time # benchmarking
import pprint # nice output
pp = pprint.PrettyPrinter(indent=4)

import random

import pandas as pd
from ducks import Dex, FrozenDex

# generate data
data_size = 2 * (10**3)
data = [{"prop1": i, "prop2": i**2, "prop3": i**(1/2)}
        for i in range(data_size)]
random.shuffle(data)

# define functions to filter data
filtering_data_functions = []

def filter_with_filter(inp):
    full_start_time = time.time()
    start_time = time.time()
    _ = list(filter(lambda v: v['prop2'] < 200**2, inp))
    end_time = time.time()
    return end_time - start_time, end_time - full_start_time
filtering_data_functions.append(filter_with_filter)

def filter_with_list_comprehension(inp):
    full_start_time = time.time()
    start_time = time.time()
    _ = [val for val in inp if val['prop2'] < 200**2]
    end_time = time.time()
    return end_time - start_time, end_time - full_start_time
filtering_data_functions.append(filter_with_list_comprehension)

def filter_with_pandas(inp):
    full_start_time = time.time()
    df = pd.DataFrame(inp)
    start_time = time.time()
    _ = df[df['prop2'] < 200**2]
    end_time = time.time()
    return end_time - start_time, end_time - full_start_time
filtering_data_functions.append(filter_with_pandas)

def filter_with_dex(inp):
    full_start_time = time.time()
    dex = Dex(inp, [key for key in data[0]])
    start_time = time.time()
    _ = dex[{
        'prop2': {'<': 200**2},
    }]
    end_time = time.time()
    return end_time - start_time, end_time - full_start_time
filtering_data_functions.append(filter_with_dex)

def filter_with_frozendex(inp):
    full_start_time = time.time()
    dex = FrozenDex(inp, [key for key in data[0]])
    start_time = time.time()
    _ = dex[{
        'prop2': {'<': 200**2},
    }]
    end_time = time.time()
    return end_time - start_time, end_time - full_start_time
filtering_data_functions.append(filter_with_frozendex)

times = [[val * 1000 for val in filter_func(data)] + [filter_func.__name__]
         for filter_func in filtering_data_functions]

df_times = pd.DataFrame(times, columns=["short_time", "long_time", "name"])

print(df_times)

times_in_dynamic = []
base_df = pd.DataFrame(columns=["short_time", "long_time", "name", "data_size"])
for i in range(1, data_size):
    times = [[val * 1000 for val in filter_func(data[:i])] + [filter_func.__name__, i]
             for filter_func in filtering_data_functions]
    base_df = pd.concat((base_df,
                         pd.DataFrame(times,
                                      columns=["short_time", "long_time", "name", "data_size"])
                         ))

print(base_df)


### visualize

from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# generate colorscheme
# https://stackoverflow.com/questions/68081450/how-to-create-discrete-colormap-with-n-colors-using-plotly
#colorscale = "rainbow"
colorscale = "plasma"
low = 0.0
high = 1.0
n_colors = len(filtering_data_functions)
colors = px.colors.sample_colorscale(colorscale, [n/(2*n_colors -1) for n in range(n_colors)], low, high)

app = Dash(__name__)

fig_time = px.bar(df_times, x="name", y="short_time", log_y=True, barmode="group",
                  labels={
                      "short_time": "Short Time (ms)",
                      "name": "Filtering Type",
                  },
                  title="only filtering time")

fig_full_time = px.bar(df_times, x="name", y="long_time", log_y=True, barmode="group",
                       labels={
                           "long_time": "Full Time (ms)",
                           "name": "Filtering Type",
                       },
                       title="filtering + object construction time")

dynamic_time = go.Figure(
    data=[
        go.Scatter(x=base_df[base_df["name"]==func.__name__]['data_size'],
                   y=base_df[base_df["name"]==func.__name__]['short_time'],
                   name=func.__name__[12:] + " only filter",
                   opacity=0.7,
                   marker=dict(color=colors[i]))
        for i, func in enumerate(filtering_data_functions)
    ] + [
        go.Scatter(x=base_df[base_df["name"]==func.__name__]['data_size'],
                   y=base_df[base_df["name"]==func.__name__]['long_time'],
                   name=func.__name__[12:] + " + creation",
                   opacity=0.7,
                   line=dict(color=colors[i], dash='dash'),
                   marker=dict(color=colors[i]))
        for i, func in enumerate(filtering_data_functions)
    ],
    layout=go.Layout(
        title="Time comparison of filtering and construction of objects",
        xaxis=dict(
            title="Data Size",
        ),
        yaxis=dict(
            title="Time Taken (ms)",
        ),
    )
)
app.layout = html.Div(children=[
    html.H1(children='index your Python objects for fast lookup by their attributes'),
    dcc.Graph(
        id='single-try-graph-time',
        figure=fig_time
    ),
    dcc.Graph(
        id='single-try-graph-full-time',
        figure=fig_full_time
    ),
    dcc.Graph(
        id='dynamic-time',
        figure=dynamic_time
    ),
])

app.run_server(debug=True)
