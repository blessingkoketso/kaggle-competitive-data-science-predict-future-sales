# coding:utf-8

import pandas
import numpy

import xgboost as xgb
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from ml_metrics import rmse

xgb_min_num_round = 10
xgb_max_num_round = 500
xgb_num_round_step = 10

xgb_nthread = 2
xgb_random_seed = 2019
xgb_max_evals = 200
xgb_objective = 'reg:linear'

scoring = 'neg_mean_squared_error'


def get_train_dataset():

    df = pandas.read_pickle('../features/train.pkl')

    # df = df[df.date_block_num < 34]
    df = df[(df.date_block_num > 30) & (df.date_block_num < 34)]

    df.replace([numpy.inf, -numpy.inf], numpy.nan,inplace=True)
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

    return train_test_split(df_x, df_y, test_size=0.3,
                            random_state=xgb_random_seed    )


def objective(params):
    '''
    fmin的回调方法
    '''

    print(params)
    model = xgb.sklearn.XGBRegressor(
        max_depth=int(params['max_depth']),
        learning_rate=params['eta'],
        n_estimators=int(params['num_round']),
        silent=False,
        objective=params['objective'],
        booster=params['booster'],
        nthread=int(params['nthread']),
        gamma=params['gamma'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        seed=params['seed']
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
    param_space_reg_xgb_tree = {
        'booster': 'gbtree',
        'objective': xgb_objective,
        'eta': hp.quniform(
            'eta',
            0.01,
            1,
            0.01),
        'gamma': hp.quniform(
            'gamma',
            0,
            2,
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
        'subsample': hp.quniform(
            'subsample',
            0.5,
            1,
            0.1),
        'colsample_bytree': hp.quniform(
            'colsample_bytree',
            0.1,
            1,
            0.1),
        'num_round': hp.quniform(
            'num_round',
            xgb_min_num_round,
            xgb_max_num_round,
            xgb_num_round_step),
        'nthread': xgb_nthread,
        'silent': 0,
        'seed': xgb_random_seed,
        "max_evals": xgb_max_evals,
    }

    best = fmin(
        objective,
        param_space_reg_xgb_tree,
        algo=partial(
            tpe.suggest,
            n_startup_jobs=1),
        max_evals=100,
        trials=Trials())
    print(best)

    best = dict(param_space_reg_xgb_tree, **best)
    print(objective(best))
