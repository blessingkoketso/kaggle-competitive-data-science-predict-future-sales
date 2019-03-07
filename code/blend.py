# coding:utf-8

import pandas
import numpy

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from ml_metrics import rmse


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
        random_state=2019)


def objective(params):
    '''
    fmin的回调方法
    '''

    print(params)
    xgb_pred = xgb_model.predict(valid_x)
    lightgbm_pred = lightgbm_model.predict(valid_x)
    catboost_pred = cat_model.predict(valid_x)

    pred = params['xgb_part'] * xgb_pred + params['lightgbm_part'] * \
        lightgbm_pred + params['catboost_part'] * catboost_pred
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

    param_space_reg_blend = {
        'xgb_part': hp.quniform(
            'xgb_part',
            0.1,
            0.5,
            0.1),
        'lightgbm_part': hp.quniform(
            'lightgbm_part',
            0.1,
            0.5,
            0.1),
    }

    param_space_reg_blend['catboost_part'] = 1 - param_space_reg_blend['xgb_part'] - param_space_reg_blend['lightgbm_part']

    xgb_params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': 0.1,
        'gamma': 1.3,
        'min_child_weight': 5.0,
        'max_depth': 9.0,
        'subsample': 1.0,
        'colsample_bytree': 0.7000000000000001,
        'num_round': 500.0,
        'nthread': 2,
        'silent': 0,
        'seed': 2019,
        'max_evals': 200}
    xgb_model = xgb.sklearn.XGBRegressor(
        max_depth=int(xgb_params['max_depth']),
        learning_rate=xgb_params['eta'],
        n_estimators=int(xgb_params['num_round']),
        silent=False,
        objective=xgb_params['objective'],
        booster=xgb_params['booster'],
        nthread=int(xgb_params['nthread']),
        gamma=xgb_params['gamma'],
        min_child_weight=xgb_params['min_child_weight'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        seed=xgb_params['seed']
    )

    lightgbm_params = {
        'objective': 'regression',
        'num_leaves': 40.0,
        'learning_rate': 0.23,
        'min_child_samples': 21.0,
        'feature_fraction': 0.7000000000000001,
        'bagging_fraction': 1.0,
        'reg_alpha': 0.30000000000000004,
        'reg_lambda': 0.1,
        'min_child_weight': 3.0,
        'max_depth': 9.0,
        'n_estimators': 500.0,
        'nthread': 2}
    lightgbm_model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=int(lightgbm_params['num_leaves']),
        learning_rate=lightgbm_params['learning_rate'],
        n_estimators=int(lightgbm_params['n_estimators']),
        max_depth=int(lightgbm_params['max_depth']),
        metric='rmse',
        min_child_weight=lightgbm_params['min_child_weight'],
        min_child_samples=int(lightgbm_params['min_child_samples']),
        bagging_fraction=lightgbm_params['bagging_fraction'],
        feature_fraction=lightgbm_params['feature_fraction'],
        reg_alpha=lightgbm_params['reg_alpha'],
        reg_lambda=lightgbm_params['reg_lambda'],
        verbose=1,
        nthread=2
    )

    cat_params = {
        'l2_leaf_reg': 6.0,
        'depth': 10.0,
        'learning_rate': 0.32,
        'n_estimators': 430.0,
        'nthread': 2}
    cat_model = CatBoostRegressor(
        learning_rate=cat_params['learning_rate'],
        depth=cat_params['depth'],
        l2_leaf_reg=int(cat_params['l2_leaf_reg']),
        eval_metric='RMSE',
        random_seed=2019,
        loss_function='RMSE',
        logging_level='Verbose',
        thread_count=2,
        n_estimators=int(cat_params['n_estimators']),
    )

    xgb_model.fit(train_x, train_y)
    lightgbm_model.fit(train_x, train_y)
    cat_model.fit(train_x, train_y)

    best = fmin(
        objective,
        param_space_reg_blend,
        algo=partial(
            tpe.suggest,
            n_startup_jobs=1),
        max_evals=100,
        trials=Trials())
    print(best)

    best = dict(param_space_reg_blend, **best)
    best['catboost_part'] = 1 - best['xgb_part'] - best['lightgbm_part']
    print(objective(best))
