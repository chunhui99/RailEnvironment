import dash
from dash import dcc, html
import plotly.graph_objs as go
import numpy as np

# 创建 Dash 应用
app = dash.Dash(__name__)

# 创建数据
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# 布局
app.layout = html.Div(children=[
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Sin wave'
                )
            ],
            'layout': go.Layout(
                title='Interactive Visualization'
            )
        }
    )
])

# 启动应用
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
