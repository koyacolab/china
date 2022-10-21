import torch
from pathlib import Path

from cyp.data import MODISExporter, DataCleaner, Engineer, Engineer2
from cyp.models import ConvModel, RNNModel

import fire

import json

# imports the torch_xla package
# import torch_xla
# import torch_xla.core.xla_model as xm

# Opening TASK JSON file
TASK = dict()
# with open('data/task.json') as json_file:
#     TASK = json.load(json_file)

## for github 

TASK = { 
'YIELD_DATA_PATH' : "data/china_yield/wheat_2002-2018.csv",
'CLEAR_YIELD_DATA_PATH' : "data/china_yield/wheat_clear.csv",
'MASK_PATH' : "data/crop_yield-data_mask",
'TEMPERATURE_PATH' : "data/crop_yield-data_temperature",
'IMAGE_PATH' : "data/crop_yield-data_image",
'CLEANED_DATA_PATH' : "data/img_output",
'COUNTY_DATA_PATH' : "data/county_data.csv",
'MODELS_PATH' : "data/models",
'RANGE_DOWNLOAD' : ["2017-01-01", "2018-12-31"],
'NUM_PROC_YEARS' : 2
}

# YIELD_DATA_PATH = "data/china_yield/wheat_2002-2018.csv"
# MASK_PATH="data/crop_yield-data_mask"
# TEMPERATURE_PATH="data/crop_yield-data_temperature"
# IMAGE_PATH="data/crop_yield-data_image"
# CLEANED_DATA_PATH="data/img_output"
# COUNTY_DATA_PATH="data/county_data.csv"
# MODELS_PATH="data/models"
# RANGE_DOWNLOAD = ["2003-01-01", "2004-12-31"]
# # ALL_YEARS_LIST=[2003, 2004]
# NUM_YEARS = 2



class RunTask:

    @staticmethod
    def export(
        export_limit=None,
        major_states_only=True,
        check_if_done=True,
        download_folder=None,
        yield_data_path=TASK['YIELD_DATA_PATH'],     #yield_data.csv",
        TASK = TASK
    ):
        """
        Export all the data necessary to train the models.

        Parameters
        ----------
        export_limit: int or None, default=None
            If not none, how many .tif files to export (*3, for the image, mask and temperature
            files)
        major_states_only: boolean, default=True
            Whether to only use the 11 states responsible for 75 % of national soybean
            production, as is done in the paper
        check_if_done: boolean, default=False
            If true, will check download_folder for any .tif files which have already been
            downloaded, and won't export them again. This effectively allows for
            checkpointing, and prevents all files from having to be downloaded at once.
        download_folder: None or pathlib Path, default=None
            Which folder to check for downloaded files, if check_if_done=True. If None, looks
            in data/folder_name
        yield_data_path: str, default='data/yield_data.csv'
            A path to the yield data
        """
        print('MODISExporter TASK : ')
        # Serializing json  
        json_object = json.dumps(TASK, indent = 4) 
        print(json_object)
        # yield_data_path = Path(yield_data_path)
        # exporter = MODISExporter(locations_filepath=yield_data_path, TASK=TASK)
        exporter = MODISExporter(TASK=TASK)
        exporter.export_all(
            export_limit, major_states_only, check_if_done, download_folder
        )

    
    @staticmethod
    def process(
        # mask_path=TASK['MASK_PATH'],
        # temperature_path=TASK['TEMPERATURE_PATH'],
        # image_path=TASK['IMAGE_PATH'],
        # yield_data_path=TASK['YIELD_DATA_PATH'],
        # cleaned_data_path=TASK['CLEANED_DATA_PATH'],
        multiprocessing=False,
        processes=4,
        parallelism=6,
        delete_when_done=False,
        # num_years=TASK['NUM_PROC_YEARS'],
        checkpoint=True,
        TASK=TASK,
    ):
        """
        Preprocess the data

        Parameters
        ----------
        mask_path: str, default='data/crop_yield-data_mask'
            Path to which the mask tif files have been saved
        temperature_path: str, default='data/crop_yield-data_temperature'
            Path to which the temperature tif files have been saved
        image_path: str, default='data/crop_yield-data_image'
            Path to which the image tif files have been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to
        multiprocessing: boolean, default=False
            Whether to use multiprocessing
        processes: int, default=4
            Number of processes to use if multiprocessing=True
        parallelism: int, default=6
            Parallelism if multiprocesisng=True
        delete_when_done: boolean, default=False
            Whether or not to delete the original .tif files once the .npy array
            has been generated.
        num_years: int, default=14
            How many years of data to create.
        checkpoint: boolean, default=True
            Whether or not to skip tif files which have already had their .npy arrays
            written
        """
        mask_path = Path( TASK['MASK_PATH'] )  # Path(mask_path)
        temperature_path = Path( TASK['TEMPERATURE_PATH'] )
        image_path = Path( TASK['IMAGE_PATH'] )  # Path(image_path)
        yield_data_path = Path( TASK['YIELD_DATA_PATH'] )  # Path(yield_data_path)
        cleaned_data_path = Path( TASK['CLEANED_DATA_PATH'] )   # Path(cleaned_data_path)
        # print( TASK['NUM_PROC_YEARS'] )
        # num_years = TASK['NUM_PROC_YEARS']

        print('MODISProcessing TASK : ')
        # Serializing json  
        json_object = json.dumps(TASK, indent = 4) 
        print(json_object)

        cleaner = DataCleaner(
            mask_path,
            temperature_path,
            image_path,
            yield_data_path,
            savedir=cleaned_data_path,
            multiprocessing=multiprocessing,
            processes=processes,
            parallelism=parallelism,
            TASK=TASK,
        )
        cleaner.process(
            delete_when_done=delete_when_done,
            checkpoint=checkpoint,
        )

    @staticmethod
    def engineer(
        # cleaned_data_path=TASK['CLEANED_DATA_PATH'],
        # yield_data_path=TASK['YIELD_DATA_PATH'],
        # county_data_path=TASK['COUNTY_DATA_PATH'],
        num_bins=512,
        max_bin_val=4999,
        TASK=TASK,
    ):
        """
        Take the preprocessed data and generate the input to the models

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to, and path to which processed data has been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        county_data_path: str, default='data/county_data.csv'
            Path to the county data csv file
        num_bins: int, default=32
            If generate=='histogram', the number of bins to generate in the histogram.
        max_bin_val: int, default=4999
            The maximum value of the bins. The default is taken from the original paper;
            note that the maximum pixel values from the MODIS datsets range from 16000 to
            18000 depending on the band
        """
        cleaned_data_path = Path(TASK['CLEANED_DATA_PATH'])   # Path(cleaned_data_path)
        yield_data_path = Path(TASK['YIELD_DATA_PATH'])       # Path(yield_data_path)
        county_data_path = Path(TASK['COUNTY_DATA_PATH'])     # Path(county_data_path)

        engineer = Engineer(cleaned_data_path, yield_data_path, county_data_path)
        engineer.process(
            num_bands=9,
            generate="histogram",
            num_bins=num_bins,
            max_bin_val=max_bin_val,
            channels_first=True,
        )

    @staticmethod
    def engineer2(
        # cleaned_data_path=TASK['CLEANED_DATA_PATH'],
        # yield_data_path=TASK['YIELD_DATA_PATH'],
        # county_data_path=TASK['COUNTY_DATA_PATH'],
        num_bins=512,
        max_bin_val=4999,
        TASK=TASK,
    ):
        """
        Take the preprocessed data and generate the input to the models

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to save the data to, and path to which processed data has been saved
        yield_data_path: str, default='data/yield_data.csv'
            Path to the yield data csv file
        county_data_path: str, default='data/county_data.csv'
            Path to the county data csv file
        num_bins: int, default=32
            If generate=='histogram', the number of bins to generate in the histogram.
        max_bin_val: int, default=4999
            The maximum value of the bins. The default is taken from the original paper;
            note that the maximum pixel values from the MODIS datsets range from 16000 to
            18000 depending on the band
        """
        cleaned_data_path = Path(TASK['CLEANED_DATA_PATH'])   # Path(cleaned_data_path)
        yield_data_path = Path(TASK['CLEAR_YIELD_DATA_PATH'])       # Path(yield_data_path)
        county_data_path = Path(TASK['COUNTY_DATA_PATH'])     # Path(county_data_path)

        engineer2 = Engineer2(cleaned_data_path, yield_data_path, county_data_path)
        engineer2.process(
            num_bands=9,
            generate="histogram",
            num_bins=num_bins,
            max_bin_val=max_bin_val,
            channels_first=True,
        )

    @staticmethod
    def train_cnn(
        # cleaned_data_path=Path(TASK['CLEANED_DATA_PATH']),
        dataset_path = Path(TASK['CLEANED_DATA_PATH']),
        dropout=0.5,
        dense_features=None,
        savedir=Path(TASK['MODELS_PATH']),
        times="all",
        pred_years=None,
        num_runs=2,
        train_steps=25000,
        batch_size=8,
        starter_learning_rate=1e-3,
        weight_decay=1,
        l1_weight=0,
        patience=10,
        use_gp=False,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Train a CNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        dropout: float, default=0.5
            Default taken from the original paper
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which year to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). The default is 0, but a value of 1.5 is used
            when training the model in batch
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """
        # histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"

        histogram_path = dataset_path

        model = ConvModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def train_rnn(
        train_dataset=Path( TASK['CLEANED_DATA_PATH'] ),
        valid_dataset=Path( TASK['CLEANED_DATA_PATH'] ),
        num_bins=512,
        hidden_size=96,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path( TASK['MODELS_PATH'] ),
        times="all",
        pred_years=2018,
        num_runs=2,
        train_steps=25000,
        batch_size=16,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0.3,
        patience=5000,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        # device = xm.xla_device(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        TB_prefix='TB'
    ):
        """
        Train an RNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        num_bins: int, default=32
            Number of bins in the generated histogram
        hidden_size: int, default=128
            The size of the hidden state. Default taken from the original paper
        rnn_dropout: float, default=0.75
            Default taken from the original paper. Note that this dropout is applied to the
            hidden state after each timestep, not after each layer (since there is only one layer)
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=10000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            L1 loss is not used for the RNN. Setting it to 0 avoids it being computed.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """
        # histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"
        # histogram_path = dataset_path

        model = RNNModel(
            in_channels=9,
            num_bins=num_bins,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
            TB_prefix=TB_prefix,
        )
        model.run(
            train_dataset,
            valid_dataset,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )


if __name__ == "__main__":
    fire.Fire(RunTask)
