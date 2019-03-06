# coding:utf-8

import pandas
import numpy

from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from sklearn.linear_model import ElasticNetCV

skl_random_seed = 2019
lasso_max_evals = 200

scoring = 'neg_mean_squared_error'


def get_train_dataset():

    df = pandas.read_pickle('../features/train.pkl')
    df = df[(df.date_block_num < 34)]
    df = df.replace([numpy.inf, -numpy.inf], numpy.nan)
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

    return df_x, df_y


if __name__ == '__main__':
    train_x, train_y = get_train_dataset()

    # https://stackoverflow.com/questions/12283184/how-is-elastic-net-used
    model = ElasticNetCV(cv=2, random_state=0)
    model.fit(train_x, train_y)

    print(model.alpha_)
    print(model.l1_ratio_)
