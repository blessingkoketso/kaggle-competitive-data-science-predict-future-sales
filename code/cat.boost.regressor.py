# coding:utf-8

'''
@author: chunmin.li@ele.me
'''

# coding:utf-8

import pandas
import numpy

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from ml_metrics import rmse

cbr_min_num_round = 10
cbr_max_num_round = 500
cbr_num_round_step = 10

cbr_nthread = 2
cbr_random_seed = 2019
cbr_max_evals = 200
cbr_eval_fn = 'RMSE'
cbr_logging_level = 'Verbose'

scoring = 'neg_mean_squared_error'


def get_train_dataset():

    df = pandas.read_pickle('../features/train.pkl')

    # df = df[df.date_block_num < 34]
    df = df[(df.date_block_num > 30) & (df.date_block_num < 34)]

    df.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    df = df.fillna(0)

    features = features = [
        'date_block_num',
        'shop_id',
        'item_id',
        'city_code',
        'item_category_id',
        'type_code',
        'subtype_code',
        'item_cnt_month_lag_1',
        'item_cnt_month_lag_2',
        'item_cnt_month_lag_3',
        'item_cnt_month_lag_6',
        'item_cnt_month_lag_12',
        'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3',
        'date_item_avg_item_cnt_lag_6',
        'date_item_avg_item_cnt_lag_12',
        'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2',
        'date_shop_avg_item_cnt_lag_3',
        'date_shop_avg_item_cnt_lag_6',
        'date_shop_avg_item_cnt_lag_12',
        'date_cat_avg_item_cnt_lag_1',
        'date_shop_cat_avg_item_cnt_lag_1',
        'date_shop_type_avg_item_cnt_lag_1',
        'date_shop_subtype_avg_item_cnt_lag_1',
        'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1',
        'date_type_avg_item_cnt_lag_1',
        'date_subtype_avg_item_cnt_lag_1',
        'delta_price_lag',
        'delta_revenue_lag_1',
        'month',
        'days',
        'item_shop_last_sale',
        'item_last_sale',
        'item_shop_first_sale',
        'item_first_sale']

    df_x = df[features]
    df_y = df['item_cnt_month']

    return train_test_split(
        df_x,
        df_y,
        test_size=0.3,
        random_state=cbr_random_seed)


def objective(params):
    '''
    fmin的回调方法
    '''

    print(params)
    model = CatBoostRegressor(
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=int(params['l2_leaf_reg']),
        eval_metric=cbr_eval_fn,
        random_seed=cbr_random_seed,
        loss_function=cbr_eval_fn,
        logging_level=cbr_logging_level,
        thread_count=cbr_nthread,
        n_estimators=int(params['n_estimators']),
    )

    model.fit(train_x, train_y)
    pred = model.predict(valid_x)

    return score(pred, valid_y)


def score(pred, y):
    '''
    给最后测试结果打分，根据不同的标准，这里需要每次都改
    '''

    metric = rmse(y, pred)
    print(metric)
    return metric


if __name__ == '__main__':
    train_x, valid_x, train_y, valid_y = get_train_dataset()

    # xgb训练的超参数
    param_space_reg_catboost = {
        'l2_leaf_reg': hp.quniform(
            'l2_leaf_reg',
            1,
            10,
            1),
        'depth': hp.quniform(
            'depth',
            1,
            10,
            1),
        'learning_rate': hp.quniform(
            'learning_rate',
            0.01,
            1,
            0.01),
        'n_estimators': hp.quniform(
            'n_estimators',
            cbr_min_num_round,
            cbr_max_num_round,
            cbr_num_round_step),
        'nthread': cbr_nthread,
    }

    best = fmin(
        objective,
        param_space_reg_catboost,
        algo=partial(
            tpe.suggest,
            n_startup_jobs=1),
        max_evals=100,
        trials=Trials())
    print(best)

    best = dict(param_space_reg_catboost, **best)
    print(objective(best))
