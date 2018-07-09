#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

###### User preferences

dry_run=?false
outdir=/Users/felipe/bitme/outdir
input=/Users/felipe/bitme/data/bitmex_1day.csv

span=(5 100)
loss_limit=(3)
qty_to_trade=(0.2 3)
greediness=(1.)

###### end of user preferences

# see more on: https://www.gnu.org/software/parallel/parallel_tutorial.html

parallel --header : --files --results ${outdir}/messages --bar `[[  "$dry_run" = false ]] || echo ` python -m sim -x span={1} -x loss_limit={2} -x qty_to_trade={3} -x greediness={4} \
-l ${outdir}/data_files/{#}  \
-f "${input}"  \
::: span "${span[@]}" \
::: loss_limit "${loss_limit[@]}" \
::: qty_to_trade "${qty_to_trade[@]}" \
::: greediness "${greediness[@]}"


