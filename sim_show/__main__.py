from utils import read_data

import argparse
import os
import sys

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline

offline.init_notebook_mode(connected=True)


def get_args():
    parser = argparse.ArgumentParser(description='statistics properties of a file')
    parser.add_argument('-l', '--log_dir', type=str, help='sim output directory', required=True)
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')

    args = parser.parse_args()

    if os.path.isfile(args.log_dir):
        raise ValueError(args.log_dir + " is a file")
    args.log_dir = os.path.abspath(args.log_dir)

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    return args


def main():
    args = get_args()
    fills_file = os.path.join(args.log_dir, 'fills.csv')
    # orders_file = os.path.join(args.log_dir, 'orders.csv')
    pnl_file = os.path.join(args.log_dir, 'pnl.csv')
    parameters_used = open(os.path.join(args.log_dir, 'parameters_used')).readlines()[0].split(',')
    try:
        idx = parameters_used.index('--trades-file')
    except ValueError:
        idx = parameters_used.index('--files')

    trades_file = parameters_used[idx + 1].replace('%TYPE%', 'trades')

    trades: pd.DataFrame = read_data(trades_file)
    fills = read_data(fills_file)
    pnls = read_data(pnl_file)
    buys = fills.loc[fills['side'] == 'Buy'][['price']]
    sells = fills.loc[fills['side']  == 'Sell'][['price']]

    # trace = go.Candlestick(x=candles.index,
    #                        open=candles.open,
    #                        high=candles.high,
    #                        low=candles.low,
    #                        close=candles.close,
    #                        name='ohlc')

    trace = go.Scatter(
        x=trades.index,
        y=trades['price'],
        name='Tick'
        # mode='markers',
        # marker=dict(
        #     size=10,
        #     color='rgba(182, 255, 193, .9)',
        #     line=dict(
        #         width=2,
        #     )
        # )
    )

    trace2 = go.Scatter(
        x=buys.index,
        y=buys['price'],
        name='Buy',
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(182, 255, 193, .9)',
            line=dict(
                width=2,
            )
        )
    )

    trace3 = go.Scatter(
        x=sells.index,
        y=sells['price'],
        name='Sell',
        mode='markers',
        marker=dict(
            size=7,
            color='rgba(255, 182, 193, .9)',
            line=dict(
                width=1,
            )
        )
    )

    trace4 = go.Scatter(
        x=pnls.index,
        y=pnls['cum_pnl'],
        name='P&L',
        mode='lines+markers',
        marker=dict(
            size=5,
            color='rgba(193, 182, 255, .9)',
            line=dict(
                width=1,
            )
        ),
        yaxis='y2'
    )

    data = [trace, trace2, trace3, trace4]

    layout = go.Layout(
        title='Trading log',
        yaxis=dict(
            title='Price'
        ),
        yaxis2=dict(
            title='BTC',
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        ),
        xaxis2=dict()
    )

    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, auto_open=True, filename=os.path.join(args.log_dir, 'results_plot.html'))
    # offline.iplot(fig)  #

    return 0


if __name__ == "__main__":
    sys.exit(main())
