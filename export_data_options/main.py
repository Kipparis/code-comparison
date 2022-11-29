import os # get file size
import pandas as pd
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
import random                   # random data generation probably
random.seed(72481, version=2)   # reproducible randomness
import string   # get list of latin characters
import time
import timeit
from functools import partial

if __name__ == "__main__":
    print("generate data")
    max_rows_qty = 10**5
    data = [{"id": i,
             "float": random.random(),
             "int": int(random.gauss(10000, 100)),
             "string": "".join(random.sample(string.ascii_lowercase + string.digits, k=7))}
            for i in range(max_rows_qty)]

    print("for every export to each data format")
    # call those methods with prefix 'to':
    # to_json, to_parquet, to_feather, etc
    export_functions = [
        ('json',    {}),
        ('csv',     {}),
        ('parquet', {}),
        ('feather', {}),
        ('hdf',     {"key": "test"})
    ]

    attrs = []
    for rows_qty in np.linspace(0, max_rows_qty, 10, dtype=int)[1:]:
        df = pd.DataFrame(data[:rows_qty])
        for fun_name, options in export_functions:
            f_attrs = dict() # собираем здесь локальные данные
            f_attrs['rows_qty'] = rows_qty
            f_attrs['name'] = fun_name

            # снимаем мерки при записи в файл
            out_filename = "out." + fun_name
            bound_function = partial(
                getattr(df, "to_" + fun_name), out_filename, **options)
            f_attrs["creation_time_ns"] = timeit.timeit(stmt=bound_function,
                                                        timer=time.process_time_ns,
                                                        number=10)
            # снимаем мерки для самого файла
            f_attrs["file_size_bytes"] = os.path.getsize(out_filename)

            # снимаем мерки при считывании с файла
            bound_function = partial(
                getattr(pd, "read_" + fun_name), out_filename, **options)
            f_attrs["read_time_ns"] = timeit.timeit(stmt=bound_function,
                                                    timer=time.process_time_ns,
                                                    number=10)
            attrs.append(f_attrs) # агрегируем
    df_attrs = pd.DataFrame(attrs)
    pp.pprint(df_attrs)

    print("create graph")
    from dash import Dash, html, dcc
    import plotly.express as px
    import plotly.graph_objs as go

    app = Dash(__name__)
    colorscale = "plasma"
    low = 0.0
    high = 1.0
    n_colors = len(export_functions)
    colors = px.colors.sample_colorscale(colorscale, [n/(2*n_colors -1) for n in range(n_colors)], low, high)
    creation_time_fig = go.Figure(
        data=[
            go.Scatter(x=df_attrs[df_attrs["name"]==func]['rows_qty'],
                       y=df_attrs[df_attrs["name"]==func]['creation_time_ns'],
                       name=func,
                       opacity=0.7,
                       marker=dict(color=colors[i]))
            for i, (func, _) in enumerate(export_functions)
        ],
        layout=go.Layout(
            title="Comparison of creation time",
            xaxis=dict(
                title="Rows Qty",
            ),
            yaxis=dict(
                title="Creation Time (ns)",
            ),
        )
    )
    read_time_fig = go.Figure(
        data=[
            go.Scatter(x=df_attrs[df_attrs["name"]==func]['rows_qty'],
                       y=df_attrs[df_attrs["name"]==func]['read_time_ns'],
                       name=func,
                       opacity=0.7,
                       marker=dict(color=colors[i]))
            for i, (func, _) in enumerate(export_functions)
        ],
        layout=go.Layout(
            title="Comparison of read time",
            xaxis=dict(
                title="Rows Qty",
            ),
            yaxis=dict(
                title="Read Time (ns)",
            ),
        )
    )
    file_size_fig = go.Figure(
        data=[
            go.Scatter(x=df_attrs[df_attrs["name"]==func]['rows_qty'],
                       y=df_attrs[df_attrs["name"]==func]['file_size_bytes'],
                       name=func,
                       opacity=0.7,
                       marker=dict(color=colors[i]))
            for i, (func, _) in enumerate(export_functions)
        ],
        layout=go.Layout(
            title="Comparison of file size",
            xaxis=dict(
                title="Rows Qty",
            ),
            yaxis=dict(
                title="File Size (bytes)",
            ),
        )
    )

    app.layout = html.Div(children=[
        html.H1(children='File Size and IO speed depending of data structure'),
        dcc.Graph(
            id='creation-time-fig',
            figure=creation_time_fig
        ),
        dcc.Graph(
            id='read-time-fig',
            figure=read_time_fig
        ),
        dcc.Graph(
            id='file-size-fig',
            figure=file_size_fig
        ),
    ])

    app.run_server(debug=True)
