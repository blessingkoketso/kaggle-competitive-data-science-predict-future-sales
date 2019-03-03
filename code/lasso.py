#coding:utf-8

import pandas
import numpy

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

skl_random_seed = 2019
lasso_max_evals = 200

scoring = 'neg_mean_squared_error'

def get_train_dataset():

  train_df = pandas.read_pickle('../features/train.pkl')
  train_df = train_df.fillna(0)

  features = features = ['date_block_num', 'shop_id', 'item_id', 'city_code',
       'item_category_id', 'type_code', 'subtype_code', 'item_cnt_month_lag_1',
       'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_lag_6',
       'item_cnt_month_lag_12', 'date_avg_item_cnt_lag_1',
       'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
       'date_item_avg_item_cnt_lag_3', 'date_item_avg_item_cnt_lag_6',
       'date_item_avg_item_cnt_lag_12', 'date_shop_avg_item_cnt_lag_1',
       'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
       'date_shop_avg_item_cnt_lag_6', 'date_shop_avg_item_cnt_lag_12',
       'date_cat_avg_item_cnt_lag_1', 'date_shop_cat_avg_item_cnt_lag_1',
       'date_shop_type_avg_item_cnt_lag_1',
       'date_shop_subtype_avg_item_cnt_lag_1', 'date_city_avg_item_cnt_lag_1',
       'date_item_city_avg_item_cnt_lag_1', 'date_type_avg_item_cnt_lag_1',
       'date_subtype_avg_item_cnt_lag_1', 'delta_price_lag',
       'delta_revenue_lag_1', 'month', 'days', 'item_shop_last_sale',
       'item_last_sale', 'item_shop_first_sale', 'item_first_sale']

  train_x = train_df[features]
  train_y = train_df['item_cnt_month']

  return train_x, train_y

train_x, train_y = get_train_dataset()

param_space_reg_skl_lasso = {
    'alpha': hp.loguniform("alpha", numpy.log(0.00001), numpy.log(0.1)),
    'random_state': skl_random_seed,
    "max_evals": lasso_max_evals,
}

def objective(params):

  print(params)
  model = Lasso(alpha=params["alpha"], normalize=True)

  metric = cross_val_score(model, train_x, train_y, cv=2, scoring=scoring).mean()
  print(metric)

  return -metric

best = fmin(objective, param_space_reg_skl_lasso, algo=partial(tpe.suggest,n_startup_jobs=1), max_evals=100, trials=Trials())
print(best)
print(objective(best))