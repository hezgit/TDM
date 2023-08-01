import numpy as np
import os
import torch
import logging
from tdm import TDM
import ot
from utils import MAE, RMSE
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')
data_dir = './datasets'

def run_TDM(X_missing, args, X_true=None):
  
    save_dir = args['out_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, filename=os.path.join(save_dir, 'log.txt'))

    # For small datasets, smaller batchsize may prevent overfitting; 
    # For larger datasets, larger batchsize may give better performance.
    if 'batchsize' in args: 
        batchsize = args['batchsize']
    else:
        batchsize = 512

    X_missing = torch.Tensor(X_missing)
    if X_true is not None:
        X_true = torch.Tensor(X_true)
    n, d = X_missing.shape
    mask = torch.isnan(X_missing)

    k = args['network_width']
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, k * d), nn.SELU(),  nn.Linear(k * d, k * d), nn.SELU(),
                            nn.Linear(k * d,  dims_out))
    projector = Ff.SequenceINN(d)
    for _ in range(args['network_depth']):
        projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)

    imputer = TDM(projector,  batchsize=batchsize, im_lr=args['lr'], proj_lr=args['lr'], niter=args['niter'], save_dir_training=save_dir)
    imp, maes, rmses = imputer.fit_transform(X_missing.clone(), verbose=True, report_interval=args['report_interval'], X_true=X_true)
    imp = imp.detach()

    result = {}
    result["imp"] = imp[mask.bool()].detach().cpu().numpy()
    if X_true is not None:
        result['learning_MAEs'] = maes
        result['learning_RMSEs'] = rmses
        result['MAE'] = MAE(imp, X_true, mask).item()
        result['RMSE'] = RMSE(imp, X_true, mask).item()
        OTLIM = 5000
        M = mask.sum(1) > 0
        nimp = M.sum().item()
        if nimp < OTLIM:
            M = mask.sum(1) > 0
            nimp = M.sum().item()
            dists = ((imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            result['OT'] = ot.emd2(np.ones(nimp) / nimp,
                                        np.ones(nimp) / nimp, \
                                        dists.cpu().numpy())
            logging.info(
                    f"MAE: {result['MAE']:.4f}\t"
                    f"RMSE: {result['RMSE']:.4f}\t"
                    f"OT: {result['OT']:.4f}")
        else:
            logging.info(
                    f"MAE: {result['MAE']:.4f}\t"
                    f"RMSE: {result['RMSE']:.4f}\t")
            
    np.save(os.path.join(save_dir, 'result.npy'), result)
