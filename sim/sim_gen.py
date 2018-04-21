import itertools
import sys

import sim


def pp(x):
    return sim.main(x)


def main():
    sim_args = 'python sim.py -f {input_data} ' \
               '-x span={span} ' \
               '-x greediness={greediness} ' \
               '-x qty_to_trade={qty_to_trade} ' \
               '-x loss_limit={loss_limit} ' \
               '-l output/{dir} ' \
               '--no-summary ' \
               '--no-output'

    input_dir = 'data/bitmex_201803.csv'
    # span = [5, 10, 20, 40, 60, 80, 100]
    # greediness = [1., .75, .5, .25]
    # qty_to_trade = [.2]
    # loss_limit = [3, 5, 7]

    span = [5, 100]
    greediness = [1.]
    qty_to_trade = [.2, 3]
    loss_limit = [3]

    pars = [span, greediness, qty_to_trade, loss_limit]

    c = []
    t = []
    for s in itertools.product(*pars):
        c += [sim_args.format(input_data=input_dir, span=s[0], greediness=s[1], qty_to_trade=s[2], loss_limit=s[3],
                              dir=len(c)).split()]
        print ' '.join(c[-1])
        t += [(s[0], s[1], s[2], s[3])]



    # pool = ThreadPool(len(c))
    # results = pool.map(sim.main, c)
    #
    # header = "span,greediness,qty_to_trade,loss_limit,profit,loss"
    # with open("sims.csv", 'w') as f:
    #     f.write(header + '\n')
    #     for tt, r in zip(t, results):
    #         f.write("{span},{greediness},{qty_to_trade},{loss_limit},{profit},{loss},{pnl}".format(
    #             span=tt[0],
    #             greediness=tt[1],
    #             qty_to_trade=tt[2],
    #             loss_limit=tt[3],
    #             profit=r.profit_total,
    #             loss=r.loss_total,
    #             pnl=r.profit_total-r.loss_total
    #         ))
    #         f.write('\n')


if __name__ == '__main__':
    sys.exit(main())
