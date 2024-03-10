#!/bin/bash

info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $@"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $@"
}

info "Run experiment for Euro28 dataset."
python3 evaluation.py --evaluation --statistic_test --dataset Euro28 --result_name experiment_euro28 --tablename experiment_euro28 2>/dev/null

if [ $? -eq 0 ]; then 
    info "Experiment has been successfuly done."
else
    error "Experiment failed."
fi

info "Run experiment for US28 dataset."
python3 evaluation.py --evaluation --statistic_test --dataset US26 --result_name experiment_us26 --tablename experiment_us26 2>/dev/null

if [ $? -eq 0 ]; then 
    info "Experiment has been successfuly done."
else
    error "Experiment failed."
fi