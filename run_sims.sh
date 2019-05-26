#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

###### User preferences

dry_run=?true
outdir=/Users/felipe/bitcoin/output
input=/Users/felipe/bitme/data/20180701-1h-%TYPE%.csv.gz
tactics=TMV1

alpha=(100 1000 10000)
spread=(0. 0.5 5)
risk=(0.001)
max_qty=(10000)

###### end of user preferences

# see more on: https://www.gnu.org/software/parallel/parallel_tutorial.html

parallel --header : --files --results ${outdir}/messages --bar `[[  "$dry_run" = false ]] || echo ` python -m sim -x alpha={1} -x spread={2} -x risk={3} -x max_qty={4} \
--log-dir ${outdir}/data_files/{#}  \
--files "${input}"  \
--tactics "${tactics}"  \
::: alpha "${alpha[@]}" \
::: spread "${spread[@]}" \
::: risk "${risk[@]}" \
::: max_qty "${max_qty[@]}"

