import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import sys
import logging
import numpy as np
from tqdm import tqdm

from losses.mi_loss import *
from parameters import parse_args
from dataset import ReadDataset, MyDataset, mixup_data, mixup_criterion
from utils import plt_tensorboard,remove_prefix,create_table
from models.MI_Net import MI_Net

import warnings
warnings.simplefilter("ignore", UserWarning)
class train_and_test_model():
    def __init__(self,args):
        self.test_loss=[]
        self.best_AUC =  0
        self.start_epoch = 1
        self.plt_tb = plt_tensorboard(args)
        self.args = args
        self.net = MI_Net(model=self.args.model,num_regions=self.args.num_LIBs)
        self.device_ids=list(map(int, args.gpu_num.split(',')))
        self.dataset = ReadDataset(args.dataset)
        self.train_dataset = MyDataset(self.dataset.data['train'],self.dataset.labels['train'],size=args.size )
        self.val_dataset = MyDataset(self.dataset.data['val'], self.dataset.labels['val'],size=args.size , test=True)
        self.test_dataset = MyDataset(self.dataset.data['test'], self.dataset.labels['test'],size=args.size , test=True)

        self.device = torch.device("cuda", self.device_ids[0])
        try:
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=args.bs,
                                           num_workers=args.num_workers)
        except:
            print("train_dataset is null")
        try:
            self.val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=args.test_bs,
                                      num_workers=args.num_workers)
        except:
            print("val_dataset is null")
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=args.test_bs,
                                      num_workers=args.num_workers)
        self.loss_function = loss_functions(method='mi',
                                            mi_calculator=self.args.mi_calculator, temperature=self.args.temperature,
                                            bml_method=self.args.balance_loss_method, scales=self.args.scales,
                                            lil_loss=self.args.lil_loss,
                                            gil_loss=self.args.gil_loss,
                                            device=self.device)
        if len(self.device_ids) > 1:  # 单机多卡
            self.net = nn.DataParallel(self.net, device_ids=self.device_ids)

        self.net = self.net.cuda(self.device)
        if self.args.resume_model:
            self.load_model(self.args.resume_model)
        self.update_lr()
    def load_model(self,path):
        logging.info(f'Resuming training from {path}')
        loaded_params = torch.load(f"{path}",
                                   map_location=torch.device(self.device))
        state_dict = loaded_params['state_dict']
        try:
            self.net.load_state_dict(state_dict)
        except:
            state_dict = remove_prefix(state_dict, 'module.')
            self.net.load_state_dict(state_dict)
        self.start_epoch = loaded_params['epoch'] + 1

        logging.warning(f"Loaded parameters from saved model: current epoch is"
                        f" {self.start_epoch}")
    def update_lr(self):
        if len(self.test_loss)>=5:
            test_loss=np.array(self.test_loss[-5:])
            loss_drop=test_loss[:4]-test_loss[1:5]
            if min(loss_drop)<0:
                self.args.lr=self.args.lr/2
                self.test_loss=[]
                logging.info(f"Update lr to {self.args.lr}")
        self.optimizer = torch.optim.Adam([
            {'params': self.net.parameters(), 'lr': self.args.lr, 'weight_decay': args.weight_decay,
             'betas': (0.9, 0.999)},
            {'params': self.loss_function.balance_loss.parameters(), 'weight_decay': args.weight_decay}
        ])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)
    def train(self):
        epbar = tqdm(total=self.args.epoch)
        logging.info(f"Starting Training...")
        # self.test(self.net, self.start_epoch-1)
        for epoch in range(self.start_epoch,self.args.epoch+self.start_epoch):
            self.update_lr()
            self.net.train()
            avg_loss = []
            avg_ce_loss=[]
            avg_global_mi_loss=[]
            avg_local_loss=[]
            self.plt_tb.reset_metrics()
            # loader_pbar = tqdm(loader, position=1)
            for i,(data,y) in enumerate(self.train_loader):
                data = data.cuda(self.device)
                y=y.cuda(self.device)

                if self.args.mixup:
                    data, y_a, y_b, lam = mixup_data(data, y, self.args.alpha)
                    out = self.net(data)
                    losses = mixup_criterion(self.loss_function.criterion, out, y_a, y_b, lam)
                else:
                    out = self.net(data)
                    losses = self.loss_function.criterion(out, y)
                loss = self.loss_function.balance_mult_loss(losses)

                if torch.isnan(loss).any():
                    logging.info("loss is NAN, so stop training...")
                    sys.exit()
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                avg_loss.append(loss.item())
                avg_ce_loss.append(losses[0].item())
                if self.args.gil_loss:
                    avg_global_mi_loss.append(losses[1].item())
                if self.args.lil_loss:
                    avg_local_loss.append(losses[-1].item())

                self.plt_tb.accumulate_metrics(out['p_y_given_z'], y, loss)

                if i % 10 == 0:
                    log_info=f"Training total loss: {np.mean(avg_loss)}, CE loss: {np.mean(avg_ce_loss)}, "
                    if self.args.gil_loss:
                        log_info+=f"global MI loss: {np.mean(avg_global_mi_loss)},"
                    if self.args.lil_loss:
                        log_info += f"local MI loss: {np.mean(avg_local_loss)}"
                    logging.info(log_info)

            epbar.update(1)
            metric = self.plt_tb.report_metrics(epoch,
                                                tb_writer=self.plt_tb.tb_writer,
                                                tb_prefix=f'Test_False')
            logging.info(f"Epoch: {epoch} Training Average loss: {np.mean(avg_loss)}")
            self.test(self.net, epoch,val=False)


    def test(self,net, epoch,val=False):
        net.eval()
        self.plt_tb.reset_metrics()
        avg_total_loss = []
        avg_ce_loss = []
        avg_global_mi_loss = []
        avg_local_loss = []
        with torch.no_grad():
            if val:
                loader = self.val_loader
            else:
                loader = self.test_loader

            for i,(data,y) in tqdm(enumerate(loader)):
                data = data.cuda(self.device)
                y = y.cuda(self.device)
                out = net(data)
                losses = self.loss_function.criterion(out, y)
                loss = self.loss_function.balance_mult_loss(losses)

                avg_total_loss.append(loss.item())
                avg_ce_loss.append(losses[0].item())
                if self.args.gil_loss:
                    avg_global_mi_loss.append(losses[1].item())
                if self.args.lil_loss:
                    avg_local_loss.append(losses[-1].item())


                self.plt_tb.accumulate_metrics(out['p_y_given_z'], y,loss)
        self.test_loss.append(np.mean(avg_total_loss))
        metric = self.plt_tb.report_metrics(epoch,
                                 tb_writer=self.plt_tb.tb_writer,
                                 tb_prefix=f'Test_True')

        log_info = f"Test AUC: {metric}, Training CE loss: {np.mean(avg_ce_loss)}, "
        if self.args.gil_loss:
            log_info += f"global MI loss: {np.mean(avg_global_mi_loss)},"
        if self.args.lil_loss:
            log_info += f"local MI loss: {np.mean(avg_local_loss)}"
        logging.info(log_info)

        if self.best_AUC<metric:
            self.save_model(epoch,best=True)
            self.best_AUC=metric
        if epoch % 10 == 0:
            self.save_model(epoch)
        return metric

    def save_model(self,epoch,best=False):
        if self.args.save_model:
            logging.info(f"Saving model to {self.args.save_path}/{self.args.name}.")
            saved_dict = {'state_dict': self.net.state_dict(),
                          'epoch': epoch}
            if best:
                model_name=f"{self.args.save_path}/{self.args.name}/model_best.pth"
                logging.info(f"Saving best model.")
            else:
                model_name = f"{self.args.save_path}/{self.args.name}/model_{epoch}.pth"
            torch.save(saved_dict, model_name)



def unNormalize(tensor,mean,std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor
if __name__ == "__main__":
    args = parse_args()
    if not args.test:
        if args.name:
            os.makedirs(f"output/{args.name}", exist_ok=True)
        logging.basicConfig(filename=f"./logs/{args.name}.log",
                            filemode="w",
                            format='[%(asctime)s]%(levelname)s:%(message)s',
                            datefmt='%Y.%m.%d %I:%M:%S %p',
                            level=logging.INFO, )
        logging.warning(create_table(args))

    print(args.name)

    train_model=train_and_test_model(args)
    if not args.test:
        train_model.train()
        train_model.load_model(os.path.join(args.save_path,args.name,'model_best.pth'))
        train_model.test(train_model.net,0,val=False)
    else:
        print(args.resume_model)
        print(args.dataset)
        train_model.test(train_model.net, 0, val=False)
