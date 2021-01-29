import time
import logging
import os

## logging
logfile = 'sphere_reid-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = os.path.join('res', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
# ++++++os.mknod(logfile)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def lr_scheduler(epoch, optimizer):
    warmup_epoch = 20
    warmup_lr = 5e-5
    start_lr = 1e-3
    lr_steps = [80, 100]
    lr_factor = 0.1

    if epoch < warmup_epoch:  # warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
        lr = warmup_lr * (warmup_scale ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr
    else:
        for i, el in enumerate(lr_steps):
            if epoch == el:
                lr = start_lr * (lr_factor ** (i + 1))
                logger.info('LR is set to: {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.defaults['lr'] = lr
    return optimizer
