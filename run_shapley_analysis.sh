#!/bin/bash

info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $@"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $@"
}

check_status() {
    if [ $? -eq 0 ]; then 
        info "Experiment has been successfuly done."
    else
        error "Experiment failed."
    fi
}

info "Run shapley analysis for Euro28 dataset..."
info "Run shapley analysis for LinearRegression estimator."
python3 shapley_analysis.py --dataset Euro28 --estimator LinearRegression --image_name euro28_linear_regression 2>/dev/null
check_status

info "Run shapley analysis for SVR estimator."
python3 shapley_analysis.py --dataset Euro28 --estimator SVR --image_name euro28_svr 2>/dev/null
check_status

info "Run shapley analysis for DecisionTreeRegressor estimator."
python3 shapley_analysis.py --dataset Euro28 --estimator DecisionTreeRegressor --image_name euro28_cart 2>/dev/null
check_status

info "Run shapley analysis for KNeighborsRegression estimator."
python3 shapley_analysis.py --dataset Euro28 --estimator KNeighborsRegression --image_name euro28_knn 2>/dev/null
check_status

info "Run shapley analysis for RandomForestRegressor estimator."
python3 shapley_analysis.py --dataset Euro28 --estimator RandomForestRegressor --image_name euro28_knn 2>/dev/null
check_status


info "Run shapley analysis for US28 dataset..."
info "Run shapley analysis for LinearRegression estimator."
python3 shapley_analysis.py --dataset US26 --estimator LinearRegression --image_name euro28_linear_regression 2>/dev/null
check_status

info "Run shapley analysis for SVR estimator."
python3 shapley_analysis.py --dataset US26 --estimator SVR --image_name euro28_svr 2>/dev/null
check_status

info "Run shapley analysis for DecisionTreeRegressor estimator."
python3 shapley_analysis.py --dataset US26 --estimator DecisionTreeRegressor --image_name euro28_cart 2>/dev/null
check_status

info "Run shapley analysis for KNeighborsRegression estimator."
python3 shapley_analysis.py --dataset US26 --estimator KNeighborsRegression --image_name euro28_knn 2>/dev/null
check_status

info "Run shapley analysis for RandomForestRegressor estimator."
python3 shapley_analysis.py --dataset US26 --estimator RandomForestRegressor --image_name euro28_knn 2>/dev/null
check_status