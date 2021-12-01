import pandas as pd

app = dash.Dash(__name__)

# Layout of the page:
app.layout = html.Div([
    html.Div(html.Div([
        dash_table.DataTable(
            id='stats_table',
            columns=[{"name": i, "id": i} for i in df_stats_clean.columns],
            data=df_stats_clean.to_dict('records'),
            editable=False,
            style_header={
                "fontWeight": "bold",
                "textAlign": "center",
                "backgroundColor": "#666666",
                'color': 'white'
            },
            style_cell={
                "textAlign": "left",
                'fontSize': 14,
                'border': '0.5px solid gray'
            },
            style_data_conditional=[
                {
                    'if': {'state': 'active'},
                    'backgroundColor': 'white',
                    'border': '0.5px solid gray'
                },
            ],
        )
    ]),
        style={'width': '20%'}),
])


# Callbacks
@app.callback(
    Output("stats_table", "data"),
    [
        Input('stats_table', 'active_cell'),
        State('stats_table', 'data')
    ]
)
def update_datatable(cell, data):
    test = [{'低碳方案': '供-站点叠光', '推荐-高': ' ✅ 584', '推荐-中': ' ⬜ 376', '推荐-低': ' ⬜ 468'},
            {'低碳方案': '供-锂电池升压', '推荐-高': ' ✅ 1675', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 73'},
            {'低碳方案': '配-高效电源', '推荐-高': ' ✅ 1311', '推荐-中': ' ⬜ 364', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '配-高温电池', '推荐-高': ' ✅ 0', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '配-锂电池错峰', '推荐-高': ' ✅ 1675', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 73'},
            {'低碳方案': '用-射频高密', '推荐-高': ' ✅ 1895', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '用-高效MM', '推荐-高': ' ✅ 1895', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '用-天面一体', '推荐-高': ' ✅ 1895', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '用-室内改室外', '推荐-高': ' ✅ 1310', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 146'},
            {'低碳方案': '用-FCS减空调', '推荐-高': ' ✅ 0', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 0'},
            {'低碳方案': '管-智能管理系统', '推荐-高': ' ✅ 1604', '推荐-中': ' ⬜ 0', '推荐-低': ' ⬜ 291'}]

    df_test = pd.DataFrame(test).set_index('低碳方案')
    df_test.index = df_test.index.map(map_dict_inv)
    df_test.columns = df_test.columns.map({'推荐-高': 3, '推荐-中': 2, '推荐-低': 1})
    df_test = df_test.loc[df_test.index.dropna()].applymap(lambda x: 1 if '✅' in x else 0)
    {k.replace('Priority', 'Selection'): v for k, v in df_test.T.to_dict().items()}

    df_test['低碳方案'] = df_test['低碳方案'].map(map_dict_inv)
    df_test[['推荐-高', '推荐-中', '推荐-低']] = df_test[['推荐-高', '推荐-中', '推荐-低']].applymap(lambda x: 1 if '✅' in x else 0)
    df_test.dropna(inplace=True)
    df_test.set_index('低碳方案', inplace=True)
    df_test.columns = [3, 2, 1]
    {k.replace('Priority', 'Selection'): v for k, v in df_test.T.to_dict().items()}

    # df_output = pd.DataFrame(data)
    # df_stats_clean[['推荐-高', '推荐-中', '推荐-低']] = df_stats_clean[['推荐-高', '推荐-中', '推荐-低']].applymap(
    #     lambda x: 1 if '✅' in x else 0)
    # print(pd.DataFrame(data))
    print(data)
    # If there is not selection:
    if not cell:
        raise dash.exceptions.PreventUpdate
    else:
        # If the user select a box:
        # 3) takes the info for the row and column selected
        print(cell)
        row_selected = cell["row"]
        column_name = cell["column_id"]
        # 4) Change the figure of the box selected
        cell_string = data[row_selected][column_name]
        if cell_string[1] == '✅':
            data[row_selected][column_name] = cell_string.replace('✅', '⬜')
        elif cell_string[1] == '⬜':
            data[row_selected][column_name] = cell_string.replace('⬜', '✅')

        return data


if __name__ == "__main__":
    app.run_server(debug=False)



