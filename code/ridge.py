#coding:utf-8

import pandas
import numpy

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

skl_random_seed = 2019
ridge_max_evals = 200

scoring = 'neg_mean_squared_error'

def get_train_dataset():

  train_df = pandas.read_csv('../features/train.csv')
  train_df = train_df.fillna(0)

  features = features = ['item_id', 'shop_id', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
       'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17',
       'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26',
       'r27', 'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 
       'item_category_id', 'city_code', 'type_code', 'subtype_code',
       'si1', 'si2', 'si3', 'si4', 'si5', 'si6', 'si7', 'si8', 'si9',
       'si10', 'si11', 'si12', 'si13', 'si14', 'si15', 'si16', 'si17',
       'si18', 'si19', 'si20', 'si21', 'si22', 'si23', 'si24', 'si25',
       'si26', 'si27', 'si28', 'si29', 'si30', 'si31', 'si32', 'si33',
       'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11',
       'i12', 'i13', 'i14', 'i15', 'i16', 'i17', 'i18', 'i19', 'i20',
       'i21', 'i22', 'i23', 'i24', 'i25', 'i26', 'i27', 'i28', 'i29',
       'i30', 'i31', 'i32', 'i33', 's1', 's2', 's3', 's4', 's5', 's6',
       's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
       's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25',
       's26', 's27', 's28', 's29', 's30', 's31', 's32', 's33', 'ic1',
       'ic2', 'ic3', 'ic4', 'ic5', 'ic6', 'ic7', 'ic8', 'ic9', 'ic10',
       'ic11', 'ic12', 'ic13', 'ic14', 'ic15', 'ic16', 'ic17', 'ic18',
       'ic19', 'ic20', 'ic21', 'ic22', 'ic23', 'ic24', 'ic25', 'ic26',
       'ic27', 'ic28', 'ic29', 'ic30', 'ic31', 'ic32', 'ic33', 'r_m',
       'si_m', 'i_m', 's_m', 'ic_m']

  train_x = train_df[features]
  train_y = train_df['label']

  return train_x, train_y

train_x, train_y = get_train_dataset()

param_space_reg_skl_ridge = {
    'alpha': hp.loguniform("alpha", numpy.log(0.01), numpy.log(20)),
    'random_state': skl_random_seed,
    "max_evals": ridge_max_evals,
}

def objective(params):

  print(params)
  model = Ridge(alpha=param["alpha"], normalize=True)

  metric = cross_val_score(model, train_x, train_y, cv=2, scoring=scoring).mean()
  print(metric)

  return -metric

best = fmin(objective, param_space_reg_skl_ridge, algo=partial(tpe.suggest,n_startup_jobs=1), max_evals=100, trials=Trials())
print(best)
print(objective(best))