import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from ScheduleEnv import TrainSchedulingEnv
from Config import Config
# 创建 Dash 应用
app = dash.Dash(__name__)

# 加载训练调度环境的配置
config = Config()
env = TrainSchedulingEnv(config)
obs = env.reset(config)
done = False
current_step = 0

# 可视化函数，根据环境动态绘制铁轨、车站和列车
def update_figure():
    global env

    # 获取所有线路、车站和列车的状态
    lines = env.world.get_lines()
    trains = env.world.get_trains()

    # 初始化用于绘制的 traces 列表
    traces = []

    # 绘制每一条线路及其上的车站
    for line in lines:

        # 绘制铁轨（用一条线段表示）
        rail_line = go.Scatter(
            x=line.stationX,  # x 轴为车站位置
            y=[line.line_position_y] * len(line.stationX),  # y 轴为 0，表示铁轨是水平的
            mode="lines",
            line=dict(color="black", width=4),
            name=f"Railway {line.name}"
        )
        traces.append(rail_line)

        stations = line.get_all_stations()


        station_traces = [
            go.Scatter(
                x=[station.station_position_x],
                y=[station.station_position_y],
                mode="markers+text",
                marker=dict(size=20, color="blue"),
                text=f"{station.get_station_name()} ({station.waiting_passengers} waiting)",
                textposition="top center"
            ) for station in stations
        ]
        traces.extend(station_traces)

    # 绘制列车（红色矩形表示）
    train_traces = [
        go.Scatter(
            x=[train.position_x],
            y=[train.position_y],  # 列车稍微偏离铁轨，以免与车站重叠
            mode="markers+text",
            marker=dict(size=30, color="red"),
            text=f"Train {train.train_id} ({train.current_passengers} passengers)",
            textposition="top center"
        ) for train in trains
    ]
    traces.extend(train_traces)

    # 返回更新的 Plotly 图表数据
    return {
        "data": traces,
        "layout": go.Layout(
            xaxis=dict(range=[-50, 5500], title="Railway Distance"),
            yaxis=dict(range=[-2, 2], showgrid=False),  # 隐藏网格线
            showlegend=False,
            height=400
        )
    }

# 定义布局
app.layout = html.Div([
    html.H1("Train Scheduling Visualization"),
    
    # 图形展示部分
    dcc.Graph(id="train-graph", figure=update_figure()),

    # 控制按钮
    html.Div([
        html.Button("Next", id="next-button", n_clicks=0),
        html.Button("Continue", id="continue-button", n_clicks=0),
        html.Button("Stop", id="stop-button", n_clicks=0),
    ]),

    # 隐藏的定时器，用于自动执行
    dcc.Interval(id="interval-component", interval=1*1000, n_intervals=0, disabled=True)
])

# 合并了的回调函数，用于处理 'Next' 按钮和自动更新（Continue）
@app.callback(
    Output("train-graph", "figure"),
    Output("interval-component", "disabled"),
    [Input("next-button", "n_clicks"), Input("interval-component", "n_intervals"), Input("continue-button", "n_clicks"), Input("stop-button", "n_clicks")],
    [State("interval-component", "disabled")]
)
def on_next_or_continue(n_next, n_intervals, n_continue, n_stop, is_disabled):
    global env

    # 判断触发源
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # 停止按钮
    if ctx.triggered[0]["prop_id"] == "stop-button.n_clicks":
        return update_figure(), True  # 停止自动更新

    # Continue 按钮
    if ctx.triggered[0]["prop_id"] == "continue-button.n_clicks":
        return update_figure(), False  # 启动自动更新
    
    # Next 按钮或者 Interval 触发
    action = env.sample_action()  # 随机生成行动
    obs, reward, done, info = env.step(action)  # 执行一步

    return update_figure(), is_disabled  # 保持定时器的状态（禁用或启用）

# 启动应用
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)