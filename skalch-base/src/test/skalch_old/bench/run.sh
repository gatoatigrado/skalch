#!/bin/bash

read -p "first idx: " firstidx
read -p "second idx: " secondidx

for confidx in $(seq $firstidx $secondidx); do
    echo "
configuration ${confidx}" >> output.txt
    for i in $(seq 1 8); do
        (build_util/build.py scalac /dontcompile run_app=test.bench.BintreeInsertTest \
            run_opt_list --ui_no_gui --sy_num_solutions 1 --num_nodes "$confidx" 2>&1 \
            | tee -a all_output.txt | grep 'time taken:') >> output.txt
    done
    sleep 3 || exit 1
done
