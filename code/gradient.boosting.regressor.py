# coding:utf-8

import pandas
import numpy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from ml_metrics import rmse

skl_min_n_estimators = 10
skl_max_n_estimators = 500
skl_n_estimators_step = 10

skl_random_seed = 2019
gbm_max_evals = 200

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
                            random_state=skl_random_seed)


def objective(params):

    print(params)
    model = GradientBoostingRegressor(
        n_estimators=int(params['n_estimators']),
        max_features=params['max_features'],
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        random_state=skl_random_seed,
        verbose=1
    )

    # metric = cross_val_score(model, train_x, train_y, cv=2, scoring=scoring).mean()
    # print(metric)

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

    param_space_reg_skl_gbm = {
        'n_estimators': hp.quniform(
            "n_estimators",
            skl_min_n_estimators,
            skl_max_n_estimators,
            skl_n_estimators_step),
        'learning_rate': hp.quniform(
            "learning_rate",
            0.01,
            0.5,
            0.01),
        'max_features': hp.quniform(
            "max_features",
            0.05,
            1.0,
            0.05),
        'max_depth': hp.quniform(
            'max_depth',
            1,
            15,
            1),
        'subsample': hp.quniform(
            'subsample',
            0.5,
            1,
            0.1),
        'random_state': skl_random_seed,
        "max_evals": gbm_max_evals}

    best = fmin(
        objective,
        param_space_reg_skl_gbm,
        algo=partial(
            tpe.suggest,
            n_startup_jobs=1),
        max_evals=100,
        trials=Trials())
    print(best)
    print(objective(best))
