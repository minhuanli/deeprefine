import torch
from torch.utils.data import DataLoader, RandomSampler

from typing import Union
from tqdm import tqdm

from deeprefine.nn.flow.losses import MLlossNormal, KLloss
from deeprefine.utils import assert_numpy, assert_list
from deeprefine.nn.flow.networks import save_bg


class MLTrainer(object):
    def __init__(self, bg, optim, iwae=False):
        self.bg = bg
        self.optim = optim
        self.criterion = MLlossNormal(iwae=iwae)

    def train(
        self,
        x_train,
        x_val=None,
        max_iter=1000,
        batch_size=1024,
        std_z=1.0,
        record=[],
        checkpoint_step=500,
        checkpoint_name="MLTrain_checkpoint",
    ):
        if x_val is not None:
            sampler = RandomSampler(x_val)
            valloader = DataLoader(x_val, batch_size=512, sampler=sampler)
        
        train_sampler = RandomSampler(x_train)
        trainloader = DataLoader(
            x_train, batch_size=batch_size, sampler=train_sampler
        )
        progress_bar = tqdm(range(max_iter), desc=f"Batchsize {batch_size}")
        for i in progress_bar:
            x_batch = next(iter(trainloader)).to(self.bg.device)
            self.optim.zero_grad()
            output_train = self.bg.TxzJ(x_batch)
            loss = self.criterion(output_train, std_z)
            if torch.isnan(loss):
                print(f"Got NAN loss at step {i}! Skip remaining training!")
                break
            loss.backward()
            self.optim.step()
            if x_val is not None:
                x_batch_test = next(iter(valloader)).to(self.bg.device)
                output_test = self.bg.TxzJ(x_batch_test)
                loss_test = self.criterion(output_test)
                loss_np, loss_test_np = loss.item(), loss_test.item()
                progress_bar.set_postfix(
                    MLloss=loss_np,
                    MLTestloss=loss_test_np,
                    memory=torch.cuda.memory_allocated() / 1e9,
                )
                record.append(
                    [i, loss_np, loss_test_np, torch.cuda.memory_allocated() / 1e9]
                )
            else:
                loss_np = loss.item()
                progress_bar.set_postfix(
                    MLloss=loss_np, memory=torch.cuda.memory_allocated() / 1e9
                )
                record.append([i, loss_np, torch.cuda.memory_allocated() / 1e9])
            if (i % checkpoint_step == (checkpoint_step-1)) or (i == max_iter-1):
                save_bg(self.bg, checkpoint_name + f"_{i}.pkl")
        return record


class FlexibleTrainer(object):
    def __init__(self, bg, optim, iwae_ML=False, iwae_KL=False):
        self.bg = bg
        self.optim = optim
        self.ml_criterion = MLlossNormal(iwae_ML)
        assert bg.energy_model is not None, "No energy model in bg!"
        self.kl_criterion = KLloss(bg.energy_model.energy, iwae_KL)

    def train(
        self,
        x_train,
        x_val=None,
        max_iter=100,
        batchsize_ML=1024,
        batchsize_KL=None,
        samplez_std=1.0,
        temperature=1.0,
        w_ML=1.0,
        w_KL=1.0,
        Ehigh=20000,
        Emax=1e10,
        record=[],
        checkpoint_step=500,
        checkpoint_name="KLTrain_checkpoint",
    ):
        if batchsize_KL is None:
            batchsize_KL = batchsize_ML

        # batchsize_ML = assert_list(batchsize_ML, epochs, int)
        # batchsize_KL = assert_list(batchsize_KL, epochs, int)
        # w_ML = assert_list(w_ML, epochs, float)
        # w_KL = assert_list(w_KL, epochs, float)
        # samplez_std = assert_list(samplez_std, epochs, float)
        # temperature = assert_list(temperature, epochs, float)
        # Ehigh = assert_list(Ehigh, epochs, Union[int, float])
        # Emax = assert_list(Emax, epochs, Union[int, float])

        if x_val is not None:
            sampler = RandomSampler(x_val)
            valloader = DataLoader(x_val, batch_size=1024, sampler=sampler)

        
        train_sampler = RandomSampler(x_train)
        trainloader = DataLoader(
            x_train, batch_size=batchsize_ML, sampler=train_sampler
        )
        
        progress_bar = tqdm(
            range(max_iter),
            desc=f"w_ML {w_ML}, w_KL {w_KL}",
        )
        for i in progress_bar:
            x_batch = next(iter(trainloader)).to(self.bg.device)
            self.optim.zero_grad()
            output_MLtrain = self.bg.TxzJ(x_batch)
            mlloss = self.ml_criterion(output_MLtrain, std_z=1.0)
            z_batch = self.bg.sample_z(
                std=samplez_std, nsample=batchsize_KL
            )
            output_KLtrain = self.bg.TzxJ(z_batch)
            klloss = self.kl_criterion(
                output_KLtrain, temperature, Ehigh, Emax
            )
            loss = w_ML * mlloss + w_KL * klloss
            if torch.isnan(loss):
                print(f"Got NAN loss at step {i}! Skip remianing training!")
                break
            loss.backward()
            self.optim.step()
            if x_val is not None:
                x_batch_test = next(iter(valloader)).to(self.bg.device)
                output_test = self.bg.TxzJ(x_batch_test)
                loss_test = self.ml_criterion(output_test)
                loss_np, mlloss_np, klloss_np, loss_test_np = (
                    loss.item(),
                    mlloss.item(),
                    klloss.item(),
                    loss_test.item(),
                )
                progress_bar.set_postfix(
                    loss=loss_np,
                    MLloss=mlloss_np,
                    KLloss=klloss_np,
                    MLTestloss=loss_test_np,
                    memory=torch.cuda.memory_allocated() / 1e9,
                )
                record.append(
                    [
                        i,
                        loss_np,
                        mlloss_np,
                        klloss_np,
                        loss_test_np,
                        torch.cuda.memory_allocated() / 1e9,
                    ]
                )
            else:
                loss_np, mlloss_np, klloss_np = (
                    loss.item(),
                    mlloss.item(),
                    klloss.item(),
                )
                progress_bar.set_postfix(
                    loss=loss_np,
                    MLloss=mlloss_np,
                    KLloss=klloss_np,
                    memory=torch.cuda.memory_allocated() / 1e9,
                )
                record.append(
                    [
                    i,
                    loss_np,
                    mlloss_np,
                    klloss_np,
                    torch.cuda.memory_allocated() / 1e9,
                    ]
                )
        if (i % checkpoint_step == (checkpoint_step - 1)) or (i == max_iter-1):
            save_bg(self.bg, checkpoint_name + f"_{i}.pkl")
        return record
