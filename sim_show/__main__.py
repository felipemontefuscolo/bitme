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
    orders_file = os.path.join(args.log_dir, 'orders.csv')
    # orders_file = os.path.join(args.log_dir, 'orders.csv')
    pnl_file = os.path.join(args.log_dir, 'pnl.csv')
    parameters_used = open(os.path.join(args.log_dir, 'parameters_used')).readlines()[0].split(',')
    try:
        idx = parameters_used.index('--trades-file')
    except ValueError:
        idx = parameters_used.index('--files')

    trades_file = parameters_used[idx + 1].replace('%TYPE%', 'trades')

    trades: pd.DataFrame = read_data(trades_file, args.begin, args.end)
    fills = read_data(fills_file, args.begin, args.end)
    orders = read_data(orders_file, args.begin, args.end)
    pnls = read_data(pnl_file, args.begin, args.end)
    buys = fills.loc[fills['side'] == 'Buy'][['price', 'order_id']]
    sells = fills.loc[fills['side']  == 'Sell'][['price', 'order_id']]
    orders = orders.loc[orders['status'] != 'Canceled']
    orders_buy = orders.loc[orders['side'] == 'buy']
    orders_sell = orders.loc[orders['side'] == 'sell']

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
        text=buys['order_id'],
        hoverinfo='text',
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
        text=sells['order_id'],
        hoverinfo='text',
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

    trace5 = go.Scatter(
        x=orders_buy.index,
        y=orders_buy['price'],
        text=orders_buy['id'],
        hoverinfo='text',
        name='Orders buy',
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=7,
            color='rgba(182, 255, 193, .9)',
            line=dict(
                width=1,
            )
        )
    )

    trace6 = go.Scatter(
        x=orders_sell.index,
        y=orders_sell['price'],
        text=orders_sell['id'],
        hoverinfo='text',
        name='Orders sell',
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=7,
            color='rgba(255, 182, 193, .9)',
            line=dict(
                width=1,
            )
        )
    )

    data = [trace, trace2, trace3, trace4, trace5, trace6]

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
