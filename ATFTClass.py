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
import sys

import fire

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

class ModelBase:
    
    def __init__(self, 
                 home_dir = '/hy-tmp',
                 datasetfile = 'corn_china_pandas_onebands.csv',
                 # datasetfile = 'corn_china_pandas_onebands.csv',
                 predicted_year = 2010,
                 batch_size = 1, 
                 encoder_length = 20,
                 save_checkpoint = False,
                 save_checkpoint_model = 'best-model',
                 exp_name = '',
                 crop_name = '',
                ):
    
        self.home_dir = '/hy-tmp'
        os.chdir(home_dir)
        
        print(exp_name, crop_name)
        
        if len(exp_name) == 0:
            print("exp_name is not definite")
            sys.exit(0)
        if len(crop_name) == 0:
            print("crop_name is not definite")
            sys.exit(0)
            
        self.exp_name = exp_name
        self.crop_name = crop_name
        self.batch_size = batch_size
        self.predicted_year = str(predicted_year)
        
        self.encoder_length = encoder_length
        
        # define Logger 
        self.logger_name = f"{self.exp_name}_{self.predicted_year}_batch_size={self.batch_size}"      
        self.logger_comment = f"encoder_length={encoder_length}"
        
        self.save_checkpoint = save_checkpoint
        self.save_checkpoint_model = save_checkpoint_model

        freeze_support()
        warnings.filterwarnings("ignore")

        print(f'{crop_name}_{exp_name} loading {datasetfile}')
        data = pd.read_csv(os.path.join(home_dir, datasetfile))  # encoding= 'unicode_escape')

        data['county'] = data['county'].astype(int)

        years_list = list(data['years'].unique())

        # print(data['county'].unique())

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

        data.insert(4, "mean_areas", np.asarray(mean_sownareas).astype(np.float64))  
        data.insert(6, "mean_yields", np.asarray(mean_yieldvals).astype(np.float64)) 

        data['years'] = data['years'].astype(str)
        data['county'] = data['county'].astype(str)
        data['time_idx'] = data['time_idx'].astype(np.int64)
        
        # create the dataset from the pandas dataframe
        train_data = data[ data["years"] != self.predicted_year ]
        valid_data = data[ data["years"] == self.predicted_year ]        

        bins_name = list()  # ["sownareas", "yieldvals", "yield"]   #list()
        for band in tqdm(range(0, 9)):
            for bins in range(0, 512):
                bins_name.append( f'band_{band}_{bins}' )

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

        # convert datasets to dataloaders for training
        self.train_dataloader = train_dataset_with_covariates.to_dataloader(train=True,  batch_size=self.batch_size, num_workers=4)
        self.valid_dataloader = valid_dataset_with_covariates.to_dataloader(train=False, batch_size=self.batch_size, num_workers=4)
        
        self.actuals = torch.cat([y[0] for x, y in iter(self.valid_dataloader)])
        self.predictions = self.actuals

        checkpoint_callback = ModelCheckpoint(dirpath='/hy-tmp/chck/'+self.logger_name, every_n_epochs=1)

        logger = TensorBoardLogger('/tf_logs', name=self.logger_name, comment=self.logger_comment)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        self.trainer = Trainer(accelerator='gpu', devices="0, 1", logger=logger, max_epochs=30, 
                          log_every_n_steps=1, callbacks=[checkpoint_callback, lr_monitor], fast_dev_run=True)

        learning_rate = 0.001

        self.model = TemporalFusionTransformer.from_dataset(
        train_dataset_with_covariates,
        learning_rate=learning_rate,
        # # lstm_layers=2,
        # hidden_size=16,
        # attention_head_size=4,
        # dropout=0.1,
        # hidden_continuous_size=8,
        # output_size=7,  # 7 quantiles by default
        # loss=RMSE(),
        loss=QuantileLoss(),
        # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        # reduce_on_plateau_patience=4,
        )
        
        self.best_tft = self.model
        self.checkpoint = f"{self.crop_name}_{self.exp_name}.ckpt"
        
    def train(self,):
        print('Train(): learning_rate', self.model.hparams.learning_rate)
        self.trainer = self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        self.best_tft = self.model
        # if self.save_checkpoint_model == 'best-model':
        #     print(f"{self.crop_name} {self.save_checkpoint} best-model loading...")
        #     best_model_path = self.trainer.checkpoint_callback.best_model_path
        #     print('best_model_path:', type(best_model_path), best_model_path)
        #     self.best_tft = TemporalFusionTransformer.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        #     print(f"{self.crop_name} {self.save_checkpoint} best-model loaded...")
        #     if self.save_checkpoint == True:
        #         self.best_tft.save_checkpoint(self.checkpoint)
        #         print(f"{self.crop_name} {self.save_checkpoint} best-model saved...")
        # elif self.save_checkpoint_model == 'last-model':
        #     self.trainer.save_checkpoint(self.checkpoint)
        #     self.best_tft = TemporalFusionTransformer.load_from_checkpoint(self.checkpoint)
        #     print(f"{self.crop_name} {self.save_checkpoint} last-model loaded...")
        
    def predict(self,):
        # calcualte mean absolute error on validation set
        # actuals = self.actuals     # torch.cat([y[0] for x, y in iter(self.valid_dataloader)])
        self.predictions = self.best_tft.predict(self.valid_dataloader)
        (self.actuals - self.predictions).abs().mean()
        print('self.actuals type is:', type(self.actuals))     
        
    def plot_predict(self,):
        
        X = [X for X in range(0, self.actuals.shape[0])]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        ax1.plot(X, self.actuals, color='b', label="Actual")
        ax1.plot(X, self.predictions, color='r', label="Predicted")
        ax1.set_title(self.logger_name)

        files = os.path.join(self.home_dir, f'TFTC_{self.crop_name}_{self.predicted_year}_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        # plt.show()
    
        X = [X for X in range(1, 21)]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        outs = int( self.actuals.shape[0] / 20 )

        act = []
        pred = []
        for ii in range(0,outs*20,outs):
            act.append(self.actuals[ii:ii+outs].mean())
            pred.append(self.predictions[ii:ii+outs].mean())

        ax1.plot(X, np.asarray(act), 'bo', label="Actual")
        ax1.plot(X, np.asarray(pred), 'ro', label="Predicted")
        leg = plt.legend(loc='upper center')
        plt.xticks(X)
        ax1.set_ylim([0, 1])
        plt.xlabel("counties")
        plt.ylabel("Yield")
        ax1.set_title(f"Corn yield predictions for {self.predicted_year} with Temporal Fusion Transformer")

        files = os.path.join(self.home_dir, f'TFT_{self.crop_name}_{self.predicted_year}_yield_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        # plt.show()

        X = [X for X in range(0, actuals.shape[0])]
        X = [X for X in range(1, 21)]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        act = []
        pred = []
        for ii in range(0,outs*20,outs):
            act.append(self.actuals[ii:ii+outs].mean())
            pred.append(self.predictions[ii:ii+outs].mean())

        ax1.plot(X, (1-np.abs(np.asarray(act)-np.asarray(pred))) * 100, 'bo', label="Actual")
        # ax1.plot(X, np.asarray(pred), 'r.', label="Predicted")
        ax1.set_ylim([70, 105])
        plt.xticks(X)
        plt.xlabel("counties")
        plt.ylabel("Yild Accuracy")
        ax1.set_title(f"ACCURACY for Temporal Fusion Transformer for {self.predicted_year} year for corn yield predict") 

        files = os.path.join(self.home_dir, f'TFT_{self.crop_name}_{self.predicted_year}_accuracy_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        
        
class RunTask:
    @staticmethod
    def train_TFT(exp_name, crop_name):
        model = ModelBase(exp_name=exp_name, crop_name=crop_name)
        model.train()
        model.predict()
        model.plot_predict()

if __name__ == "__main__":
    
    freeze_support()
    warnings.filterwarnings("ignore")
    
    fire.Fire(RunTask)
    
    # main()
    