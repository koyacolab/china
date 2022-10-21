import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime

from .gp import GaussianProcess
from .loss import l1_l2_loss

from pprint import pprint

import torch
from torch.utils.tensorboard import SummaryWriter

import os

from pathlib import Path

class ModelBase:
    """
    Base class for all models
    """

    def __init__(
        self,
        model,
        model_weight,
        model_bias,
        model_type,
        savedir,
        use_gp=False,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        TB_prefix='TB_',
    ):
        self.savedir = Path( savedir ) / model_type 
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f"Using {device.type}")
        model = model.to(device)
        # if device.type != "cpu":
        #     model = model.cuda()
        # elif device.type == "xla":
        #     model = model.to(device)
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        self.TB_prefix = TB_prefix

        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)

    def run(
        self,
        train_histogram=Path("data/img_output/histogram_all_full.npz"),
        valid_histogram=Path("data/img_output/histogram_all_full.npz"), 
        times="all",
        pred_years=2018,
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
    ):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.

        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int, list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
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
            In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.
        """
        print('run dataset : ', train_histogram, valid_histogram)
        print('run : pred_years: {}, batch_size: {}, num_runs: {}'.format(pred_years, batch_size, num_runs))

        # with np.load(path_to_histogram) as hist:
        #     images = hist["output_image"]
        #     locations = hist["output_locations"]
        #     yields = hist["output_yield"]
        #     years = hist["output_year"]
        #     indices = hist["output_index"]
        #     areas = hist["output_areas"]

        # # print('years : ', years)
        # print(images.shape, locations.shape, yields.shape, years.shape, indices.shape, areas.shape)
        # # fn
        # hists = []
        # img = np.empty( shape=[0,images.shape[1],images.shape[2],images.shape[3]] )
        # loc = np.empty( shape=[0, locations.shape[1]] )
        # yie = np.empty( shape=[0,] )
        # yea = np.empty( shape=[0,] )
        # ind = np.empty( shape=[0, indices.shape[1]] )
        # are = np.empty( shape=[0,] )
        # # img = np.repeat(np.expand_dims(images[0,...], axis=0), 4, axis=0)
        # print('img:', img.shape, areas[0])
        # print('output_image : ', type(images), images.shape, indices.shape)
        # for ii in range(0,indices.shape[0],1):
        #   # print(':', indices[ii,1], years[ii], areas[ii], yields[ii])
        #   if areas[ii] > 1000:
        #     areas[ii] = areas[ii] // 1000
        #   print(':', indices[ii,1], years[ii], areas[ii], yields[ii].round(4))
        #   # img = np.repeat(np.expand_dims(images[ii,...], axis=0), areas[ii], axis=0)
        #   # print('imgg:', imgg.shape)
        #   img = np.concatenate((img, np.repeat(np.expand_dims(images[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   loc = np.concatenate((loc, np.repeat(np.expand_dims(locations[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   yie = np.concatenate((yie, np.repeat(np.expand_dims(yields[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   yea = np.concatenate((yea, np.repeat(np.expand_dims(years[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   ind = np.concatenate((ind, np.repeat(np.expand_dims(indices[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   # are = np.concatenate((img, np.repeat(np.expand_dims(images[ii,...], axis=0), areas[ii], axis=0)), axis=0)
        #   # img = np.asarray(img)
        #   print(img.shape)
        #   # hists.append( img )

        # print('huh img: ', img.shape, loc.shape, yie.shape, yea.shape, ind.shape)

        # images = img
        # locations = loc
        # yields = yie
        # years = yea
        # indices = ind

        # np.savez(
        #     f"histogram_all_augmented_{pred_years}.npz",
        #     output_image = images,
        #     output_yield = yields,
        #     output_year = years,
        #     output_locations = locations,
        #     output_indices = indices,
        # )
        # print(f"Finished generating image augmentation for {pred_years}!")

        hist_train = dict()
        # with np.load(f"histogram_all_augmented.npz") as hist:
        with np.load(train_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_indices"]
            hist_train["output_image"] = hist["output_image"]
            hist_train["output_locations"] = hist["output_locations"]
            hist_train["output_yield"] = hist["output_yield"]
            hist_train["output_year"] = hist["output_year"]
            hist_train["output_indices"] = hist["output_indices"] 
            # hist_train = hist
            # areas = hist["output_areas"]

        hist_valid = dict()
        with np.load(valid_histogram) as hist:
            # print(hist.keys())
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]
            hist_valid["output_image"] = hist["output_image"]
            hist_valid["output_locations"] = hist["output_locations"]
            hist_valid["output_yield"] = hist["output_yield"]
            hist_valid["output_year"] = hist["output_year"]
            hist_valid["output_indices"] = hist["output_index"]
            # hist_valid = hist

        print(images.shape)

        # to collect results
        years_list, run_numbers, rmse_list, me_list, mape_list, times_list = [], [], [], [], [], []
        if self.gp is not None:
            rmse_gp_list, me_gp_list, mape_gp_list = [], [], []

        # print('pred_years 0:', pred_years)

        if pred_years is None:
            pred_years = range(2004, 2018)
            # print('pred_years 1:', pred_years)
            # fn
        elif type(pred_years) is int:
            pred_years = [pred_years]
            # print('pred_years 2:', pred_years)

        if times == "all":
            times = [32]
        else:
            times = range(10, 31, 4)

        for pred_year in pred_years:
            # print('pred_year : ', pred_year)
            for run_number in range(1, num_runs + 1):
                for time in times:
                    print(
                        f"Training to predict on {pred_year}, Run number {run_number}, Time {time}"
                    )

                    results = self._run_1_year(
                        hist_train,
                        hist_valid,
                        pred_year,
                        time,
                        run_number,
                        train_steps,
                        batch_size,
                        starter_learning_rate,
                        weight_decay,
                        l1_weight,
                        patience,
                    )

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(time)

                    if self.gp is not None:
                        rmse, me, mape, rmse_gp, me_gp, mape_gp = results
                        rmse_gp_list.append(rmse_gp)
                        me_gp_list.append(me_gp)
                        mape_gp_list.append(mape_gp)
                    else:
                        rmse, me, mape = results
                    rmse_list.append(rmse)
                    me_list.append(me)
                    mape_list.append(mape)
                print("--------------------------------------")

        # save results to a csv file
        data = {
            "year": years_list,
            "run_number": run_numbers,
            "time_idx": times_list,
            "RMSE": rmse_list,
            "ME": me_list,
            "MAPE": mape_list,
        }
        if self.gp is not None:
            data["RMSE_GP"] = rmse_gp_list
            data["ME_GP"] = me_gp_list
            data["MAPE_GP"] = mape_gp_list
        print('save : ', data.keys())
        results_df = pd.DataFrame(data=data)
        results_df.to_csv(self.savedir / f"{str(datetime.now())}.csv", index=False)

    def _run_1_year(
        self,
        hist_train,
        hist_valid,
        predict_year,
        time,
        run_number,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
    ):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """

        images = hist_train["output_image"]
        locations = hist_train["output_locations"]
        yields = hist_train["output_yield"]
        years = hist_train["output_year"]
        indices = hist_train["output_indices"]

        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )

        print('run_1_year : ', time, train_data.images.size())
        # fin

        # reinitialize the model, since self.model may be trained multiple
        # times in one call to run()
        self.reinitialize_model(time=time)

        TB_writer = SummaryWriter(log_dir=str(self.savedir) + f'/runs/' + f'{self.TB_prefix}_{predict_year}_{run_number}_{batch_size}', max_queue=2, flush_secs=10)

        train_scores, val_scores = self._train(
            train_data.images,
            train_data.yields,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
            TB_writer,
        )

        images = hist_valid["output_image"]
        locations = hist_valid["output_locations"]
        yields = hist_valid["output_yield"]
        years = hist_valid["output_year"]
        indices = hist_valid["output_indices"]

        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )

        results = self._predict(*train_data, *test_data, batch_size, TB_writer)

        model_information = {
            "state_dict": self.model.state_dict(),
            "val_loss": val_scores["loss"],
            "train_loss": train_scores["loss"],
        }
        for key in results:
            model_information[key] = results[key]

        # finally, get the relevant weights for the Gaussian Process
        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != "cpu":
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information["model_weight"] = model_weight.numpy()
        model_information["model_bias"] = model_bias.numpy()

        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(
                model_information["train_feat"],
                model_information["test_feat"],
                model_information["train_loc"],
                model_information["test_loc"],
                model_information["train_years"],
                model_information["test_years"],
                model_information["train_real"],
                model_information["model_weight"],
                model_information["model_bias"],
            )
            model_information["test_pred_gp"] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        print(type(model_information), len(model_information["train_loss"]), model_information["train_loss"])
        torch.save(model_information, self.savedir / filename)


        counties = []
        for ii in range(0,len(model_information['test_indices']), 1):
            counties.append(model_information['test_indices'][ii][1])
            print(type(np.array(model_information['test_indices'][ii][1])), type(model_information['test_indices'][ii][1]), model_information['test_indices'][ii][1])
            TB_writer.add_scalars( "Yield", {'Actual':np.array(model_information["test_real"][ii]), 
                                          'Predict':np.array(model_information["test_pred"][ii]),
                                          'GP':np.array(model_information["test_pred_gp"][ii])}, 
                                          model_information['test_indices'][ii][1] )
            # TB_writer.add_scalar("Yield/Actual", np.array(model_information["test_real"][ii]), model_information['test_indices'][ii][1])
            # TB_writer.add_scalar("Yield/Predict", np.array(model_information["test_pred"][ii]), model_information['test_indices'][ii][1])
            # TB_writer.add_scalar("Yield/GP", np.array(model_information["test_pred_gp"][ii]), model_information['test_indices'][ii][1])            
        # print(':', model_information["test_real"].shape, model_information["test_pred_gp"].shape, model_information["test_loc"].shape)
        # print(':', model_information.keys())
        # print(':', model_information["test_indices"])
        # print(':', counties)

        # fn

        TB_writer.close()

        return self.analyze_results(
            model_information["test_real"],
            model_information["test_pred"],
            model_information["test_pred_gp"] if self.gp is not None else None,
        )

    def _train(
        self,
        train_images,
        train_yields,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
        TB_writer,
    ):
        """Defines the training loop for a model"""

        import gc
        gc.collect()
        
        torch.cuda.empty_cache()

        # TB_writer = SummaryWriter(max_queue=1, flush_secs=10)

        # split the training dataset into a training and validation set
        total_size = train_images.shape[0]
        # "Learning rates and stopping criteria are tuned on a held-out
        # validation set (10%)."
        val_size = total_size // 10
        train_size = total_size - val_size
        print(
            f"After split, training on {train_size} examples, "
            f"validating on {val_size} examples"
        )
        train_dataset, val_dataset = random_split(
            TensorDataset(train_images, train_yields), (train_size, val_size)
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250,350,450,550,650,750,850,950], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500,1750,1850,1950], gamma=0.1)

        num_epochs = 100 * int(train_steps / (train_images.shape[0] / batch_size))
        print(f"Training for {num_epochs} epochs")

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()

        if patience is not None:
            epochs_without_improvement = 0

        EarlyStopping_train = []
        EarlyStopping_val = []

        for epoch in range(num_epochs):
            self.model.train()

            if epoch > 1:
              # scheduler.step(np.array(running_train_scores['loss']).mean())
              scheduler.step()

            # print('step:{}, epoch:{}, LR:{}'.format(step_number, epoch, scheduler.get_last_lr()))

            # running train and val scores are only for printing out
            # information
            running_train_scores = defaultdict(list)

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                loss, running_train_scores = l1_l2_loss(
                    pred_y, train_y, l1_weight, running_train_scores
                )
                loss.backward()
                optimizer.step()

                train_scores["loss"].append(loss.item())

                step_number += 1

                # if step_number in [4000, 20000]:
                #     print('change lr / 10')
                #     for param_group in optimizer.param_groups:
                #         param_group["lr"] /= 10

            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for (
                    val_x,
                    val_y,
                ) in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)

                    val_loss, running_val_scores = l1_l2_loss(
                        val_pred_y, val_y, l1_weight, running_val_scores
                    )

                    val_scores["loss"].append(val_loss.item())

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            l1vl2_train = round( np.array(running_train_scores['l2']).mean() / np.array(running_train_scores['l1']).mean(), 5)
            l1vl2_val   = round( np.array(running_val_scores['l2']).mean() / np.array(running_val_scores['l1']).mean(), 5)

            pprint("{}:TRAINING: {}, {}, {}".format(epoch , ", ".join(train_output_strings), l1vl2_train, optimizer.param_groups[0]['lr']))
            pprint("{}:VALIDATION: {}, {}".format(epoch, ", ".join(val_output_strings), l1vl2_val))

            # print('qqq : ', train_output_strings)

            EarlyStopping_train.append(np.array(l1vl2_train))
            EarlyStopping_val.append(np.array(l1vl2_val))

            # epoch_val_loss = np.array(running_val_scores["loss"]).mean()
            epoch_val_loss = np.array(running_train_scores["loss"]).mean() 

            # print('qaqaqaqa : ', running_train_scores.keys(), np.array(running_train_scores['l2']).mean())
            # fn

            if len(EarlyStopping_train) > 2000: 
              ES_train = round( sum(EarlyStopping_train[epoch-20:-1])/20, 3 )
              ES_val = round( sum(EarlyStopping_val[epoch-20:-1])/20, 3)
              pprint('Step:{}, Epoch:{}, LR:{}, ES_train:{}, ES_val:{}'.format(step_number, epoch, optimizer.param_groups[0]['lr'], ES_train, ES_val))
              if ES_train < 0.4:
                print('Early Stopping in epoch: {}'.format(epoch)) 
                break

            #For TensorBoard neediness
            TB_writer.add_scalar("Loss_L1/train", np.array(running_train_scores['l1']).mean(), epoch)
            TB_writer.add_scalar("Loss_L2/train", np.array(running_train_scores['l2']).mean(), epoch)
            TB_writer.add_scalar("Loss_loss/train", np.array(running_train_scores['loss']).mean(), epoch)
            TB_writer.add_scalar("Loss_L1vsL2/train", l1vl2_train, epoch)

            TB_writer.add_scalar("Loss_L1/valid", np.array(running_val_scores['l1']).mean(), epoch)
            TB_writer.add_scalar("Loss_L2/valid", np.array(running_val_scores['l2']).mean(), epoch)
            TB_writer.add_scalar("Loss_loss/valid", np.array(running_val_scores['loss']).mean(), epoch)
            TB_writer.add_scalar("Loss_L1vsL2/valid", l1vl2_val, epoch)

            TB_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

            TB_writer.flush()

            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    # revert to the best state dict
                    self.model.load_state_dict(best_state)
                    print("Early stopping!", patience, epochs_without_improvement)
                    break

        # TB_writer.close()
        self.model.load_state_dict(best_state)
        return train_scores, val_scores

    def _predict(
        self,
        train_images,
        train_yields,
        train_locations,
        train_indices,
        train_years,
        test_images,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        batch_size,
        TB_writer,
    ):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())
                # print('bihb')
                # fn

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results

    def prepare_arrays(
        self, images, yields, locations, indices, years, predict_year, time
    ):
        """Prepares the inputs for the model, in the following way:
        - normalizes the images
        - splits into a train and val set
        - turns the numpy arrays into tensors
        - removes excess months, if monthly predictions are being made
        """
        train_idx = np.nonzero(years != predict_year)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        # print('prepare : ', years[train_idx], years[test_idx])

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])

        print(
            f"Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}"
        )

        Data = namedtuple("Data", ["images", "yields", "locations", "indices", "years"])

        train_data = Data(
            images=torch.as_tensor(
                train_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[train_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor( locations[train_idx] ),
            indices=torch.as_tensor( indices[train_idx] ),
            years=torch.as_tensor( years[train_idx] ),
        )

        test_data = Data(
            images=torch.as_tensor(
                test_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[test_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[test_idx]),
            indices=torch.as_tensor(indices[test_idx]),
            years=torch.as_tensor(years[test_idx]),
        )

        return train_data, test_data

    @staticmethod
    def _normalize(train_images, val_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.

        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images

    @staticmethod
    def analyze_results(true, pred, pred_gp):
        """Calculate ME and RMSE"""
        print("analyze_results() - Calculate ME and RMSE")
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)
        mape = np.mean( (true-pred) / true ) 

        print(f"Without GP: RMSE: {rmse}, ME: {me}, MAPE: {mape}")

        if pred_gp is not None:
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            mape_gp = np.mean( (true-pred_gp) / true ) 
            print(f"With GP: RMSE: {rmse_gp}, ME: {me_gp}, MAPE:{mape_gp}")
            return rmse, me, mape, rmse_gp, me_gp, mape_gp
        return rmse, me, mape

    def reinitialize_model(self, time=None):
        raise NotImplementedError
