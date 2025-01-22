import torch
from torch.utils.data import DataLoader, RandomSampler

from typing import Union
from tqdm import tqdm

import numpy as np

from deeprefine.nn.flow.losses import MLlossNormal, KLloss, RClossV1, RClossV2, SSEloss, InflatedSSEloss, NLLloss, MixNLLloss
from deeprefine.utils import assert_numpy, assert_list
from deeprefine.nn.flow.networks import save_bg

def starloss_v2(z_star, z_std):
    return torch.mean(torch.square(torch.mean(z_star**2, dim=1, keepdim=True) - z_std**2))

class MLTrainer(object):
    def __init__(self, bg, optim, iwae=False):
        self.bg = bg
        self.optim = optim
        self.criterion = MLlossNormal(iwae=iwae)

    def train(
        self,
        x_train,
        x_star=None,
        x_val=None,
        max_iter=1000,
        batch_size=1024,
        std_z=1.0,
        w_star=1.0,
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
        progress_bar = tqdm(range(max_iter), desc=f"Batchsize {batch_size}, w_star {w_star}")
        if x_star is not None:
            x_star = x_star.to(self.bg.device)
        for i in progress_bar:
            
            x_batch = next(iter(trainloader)).to(self.bg.device)
            output_train = self.bg.TxzJ(x_batch)
            MLloss = self.criterion(output_train, std_z)
            
            if x_star is not None:
                output_star = self.bg.TxzJ(x_star)
                loss_star = starloss_v2(output_star[0], std_z)
                loss = MLloss + w_star * loss_star
            else:
                loss = MLloss
                loss_star = torch.tensor(0.0)
            if torch.isnan(loss):
                print(f"Got NAN loss at step {i}! Skip remaining training!")
                break
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            if x_val is not None:
                x_batch_test = next(iter(valloader)).to(self.bg.device)
                output_test = self.bg.TxzJ(x_batch_test)
                loss_test = self.criterion(output_test)
                MLloss_np, MLloss_test_np, starloss_np = MLloss.item(), loss_test.item(), loss_star.item()
                progress_bar.set_postfix(
                    MLloss=MLloss_np,
                    MLTestloss=MLloss_test_np,
                    Starloss=starloss_np,
                    memory=torch.cuda.memory_allocated() / 1e9,
                )
                record.append(
                    [i, MLloss_np, MLloss_test_np, starloss_np, torch.cuda.memory_allocated() / 1e9]
                )
            else:
                MLloss_np, starloss_np = MLloss.item(), loss_star.item()
                progress_bar.set_postfix(
                    MLloss=MLloss_np, Starloss=starloss_np, memory=torch.cuda.memory_allocated() / 1e9
                )
                record.append([i, MLloss_np, starloss_np, torch.cuda.memory_allocated() / 1e9])
            if (i % checkpoint_step == (checkpoint_step-1)) or (i == max_iter-1):
                save_bg(self.bg, checkpoint_name + f"_{i}.pkl")
        return record


class FlexibleTrainer(object):
    def __init__(self, bg, 
                 iwae_ML=False, iwae_KL=False, 
                 RC_indice=None, target_var=None, dcp=None, LL_loss="MixNLLloss", 
                 energy_transform=None, sse_transform=None, nll_transform=None):
        self.bg = bg
        self.ml_criterion = MLlossNormal(iwae_ML)
        
        self.length_scale = self.bg.energy_model.length_scale.get_name()
        if self.length_scale == "nanometer":
            unit_change = 10.0
        else:
            unit_change = 1.0

        assert bg.energy_model is not None, "No energy model in bg!"
        self.kl_criterion = KLloss(bg.energy_model.energy, energy_transform, iwae_KL)
        if RC_indice is not None:
            if target_var is None:
                self.rc_criterion = RClossV1(RC_indice)
            else:
                self.rc_criterion = RClossV2(RC_indice, target_var)
        else:
            self.rc_criterion = None

        self.LL_loss = LL_loss
        if dcp is not None:
            if LL_loss == "SSEloss":
                self.ll_criterion = SSEloss(dcp, unit_change)
            elif LL_loss == "NLLloss":
                self.ll_criterion = NLLloss(dcp, unit_change)
            elif LL_loss == "MixNLLloss":
                self.ll_criterion = MixNLLloss(dcp, unit_change, sse_transform=sse_transform, nll_transform=nll_transform)
        else:
            self.ll_criterion = None

    def train(
        self,
        x_train,
        optim,
        x_star=None,
        x_val=None,
        max_iter=100,
        batchsize_ML=1024,
        batchsize_KL=None,
        gradaccum_LL=4,
        batchsize_LL=16,
        samplez_std=1.0,
        temperature=1.0,
        w_ML=1.0,
        w_star=None,
        w_KL=1.0,
        w_RC=None,
        w_NLL=1.0,
        r_NLL=0.0,
        r_NLL_list=None,
        stage=1,
        Ehigh=1e4,
        Emax=2e4,
        NLLhigh=1e7,
        NLLmax=1e10,
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

        train_sampler = RandomSampler(x_train)
        trainloader = DataLoader(
            x_train, batch_size=batchsize_ML, sampler=train_sampler
        )
        if r_NLL_list is not None:
            assert len(r_NLL_list) == max_iter
        else:
            r_NLL_list = [r_NLL]*max_iter

        progress_bar = tqdm(
            range(max_iter),
            desc=f"s {stage}, w_KL {w_KL}, r_NLL {np.mean(r_NLL_list):.2f}",
        )
        loss_terms = ["ML", "KL"]
        loss_weights = [w_ML, w_KL]

        # if x_val is not None:
        #     sampler = RandomSampler(x_val)
        #     valloader = DataLoader(x_val, batch_size=1024, sampler=sampler)
        #     loss_terms.append("")

        if x_star is not None:
            x_star = x_star.to(self.bg.device)
            loss_terms.append("star")
            loss_weights.append(w_star)
        
        if self.rc_criterion is not None: 
            loss_terms.append("rc")
            loss_weights.append(w_RC)

        if self.LL_loss == "SSEloss":
            loss_terms.append("sse")
        
        if self.LL_loss == "NLLloss":
            loss_terms.append("nll")
        
        if self.LL_loss == "MixNLLloss":
            loss_terms.append("sse")
            loss_terms.append("nll")

        for i in progress_bar:
            loss_step = []
            loss_step_record = []
            x_batch = next(iter(trainloader)).to(self.bg.device)
            output_MLtrain = self.bg.TxzJ(x_batch)
            mlloss = self.ml_criterion(output_MLtrain, std_z=1.0)
            loss_step.append(mlloss)
            loss_step_record.append(mlloss.item())

            z_batch = self.bg.sample_z(
                std=samplez_std, nsample=batchsize_KL
            )
            output_KLtrain = self.bg.TzxJ(z_batch)
            klloss = self.kl_criterion(
                output_KLtrain, temperature, Ehigh, Emax
            )
            loss_step.append(klloss)
            loss_step_record.append(klloss.item())

            if x_star is not None:
                output_star = self.bg.TxzJ(x_star)
                #starloss = self.ml_criterion(output_star, std_z=1.0)
                starloss = starloss_v2(output_star[0], samplez_std)
                loss_step.append(starloss)
                loss_step_record.append(starloss.item())

            if self.rc_criterion is not None: 
                rcloss = self.rc_criterion(output_KLtrain)
                loss_step.append(rcloss)
                loss_step_record.append(rcloss.item())
            
            loss = sum([w*l for w, l in zip(loss_weights, loss_step)])
            loss.backward()

            # Gradient accumulation for LL loss
            if self.ll_criterion is not None:
                if self.LL_loss == "MixNLLloss":
                    loss_nll_record = 0.0
                    loss_sse_record = 0.0
                else:
                    loss_ll_record = 0.0
                for _ in range(gradaccum_LL):
                    z_batch_ll = self.bg.sample_z(
                        std=samplez_std, nsample=batchsize_LL
                    )
                    outputx_LLtrain, _ = self.bg.TzxJ(z_batch_ll)
                    if self.LL_loss == "MixNLLloss":
                        llreg, nll_, sse_ = self.ll_criterion(outputx_LLtrain, NLLhigh, NLLmax, w_NLL, r_NLL_list[i])
                        if llreg.isnan():
                            continue
                        loss_nll_record = loss_nll_record + nll_ / gradaccum_LL
                        loss_sse_record = loss_sse_record + sse_ / gradaccum_LL
                        loss_ll = llreg / gradaccum_LL
                    else:
                        llreg =  self.ll_criterion(outputx_LLtrain, NLLhigh, NLLmax, w_NLL)
                        if llreg.isnan():
                            continue
                        loss_ll = llreg / gradaccum_LL
                        loss_ll_record = loss_ll_record + loss_ll.item()
                    weighted_loss_ll = w_KL * loss_ll
                    weighted_loss_ll.backward()
                if self.LL_loss == "MixNLLloss":
                    loss_step_record.append(loss_sse_record)
                    loss_step_record.append(loss_nll_record)
                else:
                    loss_step_record.append(loss_ll_record)
            optim.step()
            optim.zero_grad()

            # if x_val is not None:
            #     x_batch_test = next(iter(valloader)).to(self.bg.device)
            #     output_test = self.bg.TxzJ(x_batch_test)
            #     loss_test = self.ml_criterion(output_test)
            #     loss_np, mlloss_np, klloss_np, rcloss_np, loss_test_np, starloss_np = (
            #         loss.item(),
            #         mlloss.item(),
            #         klloss.item(),
            #         rcloss.item(), 
            #         loss_test.item(),
            #         starloss.item()
            #     )
            #     progress_bar.set_postfix(
            #         loss=loss_np,
            #         MLloss=mlloss_np,
            #         Starloss=starloss_np,
            #         KLloss=klloss_np,
            #         RCloss=rcloss_np,
            #         AdaScale=ada_scale.item(),
            #         MLTestloss=loss_test_np,
            #         memory=torch.cuda.memory_allocated() / 1e9,
            #     )
            #     record.append(
            #         [
            #             i,
            #             loss_np,
            #             mlloss_np,
            #             starloss_np,
            #             klloss_np,
            #             rcloss_np,
            #             loss_test_np,
            #             torch.cuda.memory_allocated() / 1e9,
            #         ]
            #     )
            # else:
            postfix_dict = {
                name: value for name, value in zip(loss_terms, loss_step_record)
            }
            postfix_dict["mem"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.memory.reset_peak_memory_stats()
            progress_bar.set_postfix(
                postfix_dict
            )
            record.append(
                list(postfix_dict.values())
            )
        if (i % checkpoint_step == (checkpoint_step - 1)) or (i == max_iter-1):
            save_bg(self.bg, checkpoint_name + f"_{i}.pkl")
        return record
