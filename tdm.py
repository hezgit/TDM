
import numpy as np
import torch


from utils import nanmean, MAE, RMSE

import logging

import ot


class TDM():
    
    def __init__(self,
                 projector,
                 im_lr=1e-2,
                 proj_lr=1e-2,
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 save_dir_training=None):

        self.im_lr = im_lr
        self.proj_lr = proj_lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.projector = projector

        self.save_dir_training = save_dir_training


    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
       

        X = X.clone()
        n, d = X.shape

        
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()

        torch.autograd.set_detect_anomaly(True)


        if torch.sum(mask) < 1.0:
            is_no_missing = True
        else:
            is_no_missing = False

        X_filled = X.detach().clone()   

        if not is_no_missing:
            imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
            imps.requires_grad = True
            im_optimizer = self.opt([imps], lr=self.im_lr)
            X_filled[mask.bool()] = imps


        proj_optimizer = self.opt([p for p in self.projector.parameters()], lr=self.proj_lr)

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):

            X_filled = X.detach().clone()

            if not is_no_missing:
                X_filled[mask.bool()] = imps


            proj_loss = 0
            im_loss = 0


            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]


                X1_p, _ = self.projector(X1)
                X2_p, _ = self.projector(X2)


                M_p = torch.cdist(X1_p, X2_p, p=2)

                a1_p = torch.ones(X1.shape[0]) / X1.shape[0]
                a2_p = torch.ones(X2.shape[0]) / X2.shape[0]
                a1_p.requires_grad = False
                a2_p.requires_grad = False
                ot_p = ot.emd2(a1_p, a2_p, M_p)


                im_loss = im_loss + ot_p
                proj_loss = proj_loss + ot_p 


            if torch.isnan(im_loss).any() or torch.isinf(im_loss).any():
                logging.info("im_loss Nan or inf loss")
                break

            if torch.isnan(proj_loss).any() or torch.isinf(proj_loss).any():
                logging.info("proj_loss Nan or inf loss")
                break

            if not is_no_missing:
                im_optimizer.zero_grad()
                im_loss.backward(retain_graph=True)
                im_optimizer.step()
            


            proj_optimizer.zero_grad()
            proj_loss.backward()
            proj_optimizer.step()



            if verbose and (i % report_interval == 0):

                if X_true is not None:
                    maes[i] = MAE(X_filled, X_true, mask).item() 
                    rmses[i] = RMSE(X_filled, X_true, mask).item()

                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')


                else:
                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t ')
              
            

        X_filled = X.detach().clone()
        if not is_no_missing:
            X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled