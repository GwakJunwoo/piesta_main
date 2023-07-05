import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
from dash import dash_table
import plotly.express as px

# 데이터
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [100, 200, 300, 400]
}

df = pd.DataFrame(data)

# 앱 선언
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 사이드바 레이아웃
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links",
            className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# 메인 레이아웃
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                sidebar
            ], width=2),
            dbc.Col([
                html.H1("Hello world: Test Server", style={"textAlign": "center", "marginBottom": "50px"}),
                dcc.Input(id="input", placeholder="Enter something...", type='text'),
                html.Button(id='submit-button', n_clicks=0, children='Submit'),
                html.Div(id="output"),
                html.Hr(),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'border': '1px solid rgb(200, 200, 200)'
                        }
                    ],
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Date', 'Region']
                    ],
                    style_as_list_view=True,
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                ),
                html.Div(id='content'),
                dcc.Graph(id='graph', figure=px.bar(df, x='Category', y='Value'))
            ], width=8),
            dbc.Col([
                html.Button('Shutdown', id='shutdown-button', style={"width": "100%", "marginTop": "20px"})
            ], width=2)
        ]),
    ],
    fluid=True,
)

@app.callback(
    Output('content', 'children'),
    [Input('submit-button', 'n_clicks'), Input('table', 'active_cell')],
    [State('input', 'value'), State('table', 'data')]
)
def update_output_and_cell_clicked(n, active_cell, text, data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'submit-button':
            if n > 0:
                return f'You\'ve entered: "{text}"'
            else:
                return ''
        elif trigger_id == 'table':
            if active_cell:
                row = active_cell['row']
                col = active_cell['column_id']
                cell_value = data[row][col]
                return f"You clicked cell {cell_value} in row {row+1}, column {col}"
            else:
                return ""
        else:
            return dash.no_update

@app.callback(
    Output('graph', 'figure'),
    [Input('table', 'active_cell')]
)
def display_selected_data(active_cell):
    if active_cell:
        row_idx = active_cell["row"]
        selected_row = df.iloc[row_idx]
        return px.bar(selected_row)
    else:
        return px.bar(df, x='Category', y='Value')

@app.callback(
    Output('shutdown-button', 'children'),
    [Input('shutdown-button', 'n_clicks')]
)
def shutdown(n_clicks):
    if n_clicks:
        stop_server()
        return 'Server shutting down...'
    return 'Shutdown Server'

# 서버 종료 함수
def stop_server():
    import os
    os._exit(0)

if __name__ == '__main__':
    app.run_server(debug=True)
