import os

# home_dir = '/content/gdrive/My Drive/AChina' 
# home_dir = '/hy-tmp'
# os.chdir(home_dir)
# pwd

# pip install tqdm 

from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

import os
import warnings

# warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# os.chdir("../../..")

#pip install scipy
#pip install torch pytorch-lightning pytorch_forecasting

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAPE, SMAPE, PoissonLoss, QuantileLoss
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

from multiprocessing import Pool, freeze_support

import optuna

import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

def main():
    # freeze_support()
    # warnings.filterwarnings("ignore")
    
    data = pd.read_csv('/hy-tmp/corn_china_pandas_onebands.csv')  # encoding= 'unicode_escape')

    data['county'] = data['county'].astype(int)

    years_list = list(data['years'].unique())

    # print(type(years_list), years_list)

    years_xtiks = []
    for ii in range(16, 512, 32):  # len(years_list)):
        years_xtiks.append(ii)

    # print(len(years_xtiks), years_xtiks)

    print(data['county'].unique())

    mean_sownareas = []
    mean_yieldvals = []
    for county in list(data['county'].unique()):
        df = data[ data['county'] ==  county]
        X = [X for X in range(0, len(df['sownareas']))]

        mean_sownarea = df['sownareas'].mean()
        mean_yieldval = df['yieldvals'].mean()
        # print(type(mean_sownarea))
        # print(type(mean_sownarea.round(2) * np.ones((len(X),1))))
        mean_sownareas.extend(mean_sownarea.round(2) * np.ones((len(X),1)))
        mean_yieldvals.extend(mean_yieldval.round(2) * np.ones((len(X),1)))

        # print(mean_sownarea, df['sownareas'])    

#         fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

#         ax1.plot(X, np.asarray(df['sownareas']), color='b', label="Actual")
#         ax1.plot(X, mean_sownarea * np.ones((len(X),1)), color='r', label="Mean")
#         ax1.set_title(f"Sown Areas in county: {county}")
#         ax1.set_xticks(years_xtiks) 
#         ax1.set_xticklabels(years_list, rotation=45, fontsize=12)

#         ax2.plot(X, np.asarray(df['yieldvals']), color='g', label="Actual")
#         ax2.plot(X, mean_yieldval * np.ones((len(X),1)), color='r', label="Mean")
#         ax2.set_title(f"Yield Value in county: {county}")
#         ax2.set_xticks(years_xtiks) 
#         ax2.set_xticklabels(years_list, rotation=45, fontsize=12)

#     plt.show()

    # print(len(mean_sownareas))

    # fn

    data.insert(4, "mean_areas", np.asarray(mean_sownareas).astype(np.float64))  
    data.insert(6, "mean_yields", np.asarray(mean_yieldvals).astype(np.float64)) 

    # data

    data['years'] = data['years'].astype(str)
    data['county'] = data['county'].astype(str)
    data['time_idx'] = data['time_idx'].astype(np.int64)
    # create the dataset from the pandas dataframe
    train_data = data[ data["years"] != "2018" ]
    valid_data = data[ data["years"] == "2018" ]

    bins_name = ["sownareas"]   #list()
    for band in tqdm(range(0, 9)):
        for bins in range(0, 512):
            bins_name.append( f'band_{band}_{bins}' )
            
    # print(bins_name)
    # fn

    encoder_length = 20

    group = ["years", "county"]
    target = "yield"   #["sownareas", "yieldvals"]

    unknown_categoricals=["years", "county", "bands"]
    static_categoricals=["years", "county"]
    known_reals=["mean_areas", "mean_yields"]    #["sownareas"]

    train_dataset_with_covariates = TimeSeriesDataSet(
        train_data,
        group_ids=group,
        target=target,
        time_idx="time_idx",
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=bins_name,  #["yield"],
        # time_varying_unknown_categoricals=["county", "bands"],
        time_varying_known_reals=known_reals,
        # time_varying_known_categoricals=["years"],
        # static_categoricals=["years", "county"],
    )

    valid_dataset_with_covariates = TimeSeriesDataSet(
        valid_data,
        group_ids=group,
        target=target,
        time_idx="time_idx",
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=bins_name,   #["yield"],
        # time_varying_unknown_categoricals=["county", "bands"],
        time_varying_known_reals=known_reals,
        # time_varying_known_categoricals=["years"],
        # static_categoricals=["years", "county"],
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset_with_covariates,
        # learning_rate=0.03,
        # hidden_size=16,
        # attention_head_size=4,
        # dropout=0.1,
        # hidden_continuous_size=8,
        # output_size=1,  # 7 quantiles by default
        # loss=RMSE(),
        loss=QuantileLoss(),
        # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        # reduce_on_plateau_patience=4,
    )

    # convert datasets to dataloaders for training
    batch_size = 42
    train_dataloader = train_dataset_with_covariates.to_dataloader(train=True,  batch_size=batch_size, num_workers=4)
    valid_dataloader = valid_dataset_with_covariates.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

#     exp_name = "2GPUs"

#     logger_name = f"TFT-exp:{exp_name}-batch_size={batch_size}-encoder_length={encoder_length}-group={group}-known_reals={known_reals}"

#     checkpoint_callback = ModelCheckpoint(dirpath='/hy-tmp/chck/'+logger_name, every_n_train_steps=1)

#     callbacks=[checkpoint_callback]

#     logger = TensorBoardLogger('/tf_logs', name=logger_name)

#     # trainer = Trainer(gpus=1, max_epochs=100, limit_train_batches=2606, logger=logger)
#     trainer = Trainer(accelerator='gpu', devices="0, 1", logger=logger, max_epochs=60, 
#                       log_every_n_steps=1, callbacks=[checkpoint_callback])

#     trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
#     # trainer.validate(model=model, dataloaders=valid_dataloaders)


    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        valid_dataloader,
        model_path="/hy-tmp",
        n_trials=100,
        max_epochs=20,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.0001, 0.01),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

    # torch.multiprocessing.set_start_method('spawn') 
    

if __name__ == "__main__":
    
    freeze_support()
    warnings.filterwarnings("ignore")
    
    main()
    
    