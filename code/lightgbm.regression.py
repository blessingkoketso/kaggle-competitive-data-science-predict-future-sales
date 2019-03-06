# coding:utf-8

import pandas
import numpy

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from ml_metrics import rmse

lightgbm_min_num_round = 10
lightgbm_max_num_round = 500
lightgbm_num_round_step = 10

lightgbm_nthread = 2
lightgbm_random_seed = 2019
lightgbm_max_evals = 200
lightgbm_objective = 'regression'

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
        random_state=lightgbm_random_seed)


def objective(params):
    '''
    fmin的回调方法
    '''

    print(params)
    model = lgb.LGBMRegressor(
        objective=lightgbm_objective,
        num_leaves=int(params['num_leaves']),
        learning_rate=params['learning_rate'],
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        metric='rmse',
        min_child_weight=params['min_child_weight'],
        min_child_samples=int(params['min_child_samples']),
        bagging_fraction=params['bagging_fraction'],
        feature_fraction=params['feature_fraction'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        verbose=1,
        nthread=lightgbm_nthread
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
    param_space_reg_lightgbm_tree = {
        'objective': lightgbm_objective,
        'num_leaves': hp.quniform(
            'num_leaves',
            20,
            200,
            20),
        'learning_rate': hp.quniform(
            'learning_rate',
            0.01,
            1,
            0.01),

        'min_child_samples': hp.quniform(
            'min_child_samples',
            20,
            30,
            1),
        'feature_fraction': hp.quniform(
            'feature_fraction',
            0.5,
            1,
            0.1),
        'bagging_fraction': hp.quniform(
            'bagging_fraction',
            0.5,
            1,
            0.1),
        'reg_alpha': hp.quniform(
            'reg_alpha',
            0,
            0.5,
            0.1),
        'reg_lambda': hp.quniform(
            'reg_lambda',
            0,
            0.5,
            0.1),
        'min_child_weight': hp.quniform(
            'min_child_weight',
            0,
            10,
            1),
        'max_depth': hp.quniform(
            'max_depth',
            1,
            10,
            1),
        'n_estimators': hp.quniform(
            'n_estimators',
            lightgbm_min_num_round,
            lightgbm_max_num_round,
            lightgbm_num_round_step),
        'nthread': lightgbm_nthread,
    }

    best = fmin(
        objective,
        param_space_reg_lightgbm_tree,
        algo=partial(
            tpe.suggest,
            n_startup_jobs=1),
        max_evals=100,
        trials=Trials())
    print(best)

    best = dict(param_space_reg_lightgbm_tree, **best)
    print(objective(best))
