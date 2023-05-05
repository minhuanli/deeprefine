import torch
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm

from deeprefine.nn.flow.losses import MLlossNormal
from deeprefine.utils import assert_numpy


class MLTrainer(object):

    def __init__(self, bg, optim, std_z=1.0, iwae=False):
        self.bg = bg
        self.std_z = std_z
        self.optim = optim
        self.criterion = MLlossNormal(std_z=std_z, iwae=iwae)
    
    def train(self, x_train, x_val=None, 
              epochs=10, batch_size=1024,
              record=[],
              checkpoint_epoch=1, 
              checkpoint_name="MLTrain_checkpoint.pkl"):
        
        if isinstance(batch_size, int):
            batch_size = [batch_size] * epochs
        elif isinstance(batch_size, list):
            assert len(batch_size) == epochs     

        if x_val is not None:
            sampler = RandomSampler(x_val)
            valloader = DataLoader(x_val, batch_size=256, sampler=sampler)

        for epoch in range(epochs):
            trainloader = DataLoader(x_train, batch_size=batch_size[epoch], shuffle=True)
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")
            for x_batch in progress_bar:
                x_batch = x_batch.to(self.bg.device)
                self.optim.zero_grad()
                output_train = self.bg.TxzJ(x_batch)
                loss = self.criterion(output_train)
                loss.backward()
                self.optim.step()
                if x_val is not None:
                    x_batch_test = next(iter(valloader)).to(self.bg.device)
                    output_test = self.bg.TxzJ(x_batch_test)
                    loss_test = self.criterion(output_test)
                    loss_np, loss_test_np = loss.item(), loss_test.item()
                    progress_bar.set_postfix(MLloss=loss_np, MLTestloss=loss_test_np, memory=torch.cuda.memory_allocated()/1e9)
                    record.append([loss_np, loss_test_np, torch.cuda.memory_allocated()/1e9])
                else:
                    loss_np = loss.item()
                    progress_bar.set_postfix(MLloss=loss_np, memory=torch.cuda.memory_allocated()/1e9)
                    record.append(loss_np, torch.cuda.memory_allocated()/1e9)
        return record

                
                



                




                
        
        