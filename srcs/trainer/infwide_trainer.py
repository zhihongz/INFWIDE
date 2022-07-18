import time
from tqdm import tqdm
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import platform
from omegaconf import OmegaConf
from ._base import BaseTrainer
from srcs.utils._util import collect, instantiate, get_logger
from srcs.logger import BatchMetrics
import torch.nn.functional as F
from srcs.utils.utils_image_kair import tensor2uint, imsave
from srcs.utils.utils_deblur_kair import circular_padding
# ======================================
# Trainer: modify '_train_epoch'
# ======================================


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, input_denoise_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        if self.final_test:
            self.test_data_loader = test_data_loader
            if test_data_loader is None:
                self.logger.warning(
                    "Warning: test dataloader for final test is None, final test is omitted")
                self.final_test = False
        self.input_denoise_epoch = input_denoise_epoch
        self.lr_scheduler = lr_scheduler
        self.limit_train_batches = config['trainer'].get(
            'limit_train_batches', len(self.data_loader))
        if not self.limit_train_batches or self.limit_train_batches > len(self.data_loader):
            self.limit_train_batches = len(self.data_loader)
        self.limit_val_batches = config['trainer'].get(
            'limit_val_batches', len(self.valid_data_loader))
        if not self.limit_val_batches or self.limit_val_batches > len(self.valid_data_loader):
            self.limit_val_batches = len(self.valid_data_loader)
        args = ['loss', *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(
            *args, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics(
            *args, postfix='/valid', writer=self.writer)
        self.n_levels = 2  # model scale levels
        self.scales = [0.5, 1]  # model scale
        self.grad_clip = 0.5  # optimizer gradient clip value
        self.losses = self.config['loss']


    def clip_gradient(self, optimizer, grad_clip=0.5):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data_noisy, kernel, target, data) in enumerate(self.data_loader):
            data_noisy, kernel, target, data = data_noisy.to(self.device), kernel.to(self.device), target.to(
                self.device), data.to(self.device)

            output, data_denoise = self.model(data_noisy, kernel)

            # loss calc
            loss = 0
            for level in range(self.n_levels):
                scale = self.scales[level]
                n, c, h, w = target.shape
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                sharp_level = F.interpolate(target, (hi, wi), mode='bilinear')
                loss = loss + \
                    self.criterion['main_loss'](output[level], sharp_level)

            # input denoise loss
            if 'input_denoise_loss' in self.losses:
                input_denoise_loss = self.criterion['input_denoise_loss'](
                    data_denoise, data)
                loss = loss + \
                    self.losses['input_denoise_loss']*input_denoise_loss

            # forward_conv_loss
            if 'forward_conv_loss' in self.losses:
                # sharp image circular padding
                kernel_flip = kernel.flip(-2, -1)
                kernel_sz = kernel_flip.shape[2:]
                output_ = output[1]
                N, C, H, W = output_.shape
                output_pad = circular_padding(output_, kernel_sz)
                forward_conv = torch.zeros_like(output_)
                for k in range(N*C):
                    forward_conv[k//3][k % 3] = F.conv2d(output_pad[k//3][k % 3].unsqueeze(
                        0).unsqueeze(0), kernel_flip[k//3].unsqueeze(0), padding='valid').squeeze()

                forward_conv_loss = self.criterion['forward_conv_loss'](
                    forward_conv, data)
                loss = loss + \
                    self.losses['forward_conv_loss']*forward_conv_loss

            self.optimizer.zero_grad()
            loss.backward()

            
            # clip gradient
            self.clip_gradient(self.optimizer, self.grad_clip)
            self.optimizer.step()

            loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
            self.writer.set_step(
                (epoch - 1) * self.limit_train_batches + batch_idx, speed_chk='train')
            self.train_metrics.update('loss', loss_v)

            if batch_idx % self.logging_step == 0 or batch_idx == self.limit_train_batches:
                # save image
                self.writer.add_image(
                    'train/_input', make_grid(data_noisy[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'train/_input_denoise', make_grid(data_denoise[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'train/_output', make_grid(output[1][0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'train/_target', make_grid(target[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'train/kernel', make_grid(kernel[0:8, ...].cpu(), nrow=2, normalize=True))

                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output[1], target))
                    else:
                        metric_v = met(output[1], target)
                    self.train_metrics.update(met.__name__, metric_v)
                self.logger.info(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]["lr"]:.3e}')

            if batch_idx == self.limit_train_batches:
                break

        log = self.train_metrics.result()

        if self.valid_data_loader:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        # stop input denoise optimization when reaching maximum epoch
        if self.input_denoise_epoch is not None and epoch == self.input_denoise_epoch:
            self.model.DenoiseUnet.requires_grad = False
            self.logger.info(
                'Info: reach  INPUT_DENOISE_EPOCH(%d), stop denoise module optimization' % self.input_denoise_epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # show ce code update only when optimized
            for batch_idx, (data_noisy, kernel, target, data) in enumerate(self.data_loader):
                data_noisy, kernel, target, data = data_noisy.to(self.device), kernel.to(self.device), target.to(
                    self.device), data.to(self.device)

                output, data_denoise = self.model(data_noisy, kernel)

                loss = self.criterion['main_loss'](output[1], target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, speed_chk='valid')

                # save images
                self.writer.add_image(
                    'valid/_input', make_grid(data_noisy[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'valid/_input_denoise', make_grid(data_denoise[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'valid/_output', make_grid(output[1][0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'valid/_target', make_grid(target[0:8, ...].cpu(), nrow=2, normalize=True))
                self.writer.add_image(
                    'valid/kernel', make_grid(kernel[0:8, ...].cpu(), nrow=2, normalize=True))

                loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
                self.valid_metrics.update('loss', loss_v)
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output[1], target))

                if batch_idx == self.limit_val_batches:
                    break

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _test_epoch(self):
        """
        Final test logic after the training (! test the latest checkpoint)

        :param epoch: Current epoch number
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))
        time_start = time.time()

        with torch.no_grad():
            for i, (data_noisy, kernel, target, data) in enumerate(tqdm(self.test_data_loader)):
                data_noisy, kernel, target, data = data_noisy.to(self.device), kernel.to(self.device), target.to(
                    self.device), data.to(self.device)

                output, data_denoise = self.model(data_noisy, kernel)

                # save some sample images
                output = output[1]  # zzh: deblured image
                for k, (in_img, out_img, gt_img) in enumerate(zip(data_noisy, output, target)):
                    in_img = tensor2uint(in_img)
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    imsave(
                        in_img, f'{self.final_test_dir}/test{i+1:03d}_{k+1:03d}_in_img.png')
                    imsave(
                        out_img, f'{self.final_test_dir}/test{i+1:03d}_{k+1:03d}_out_img.png')
                    imsave(
                        gt_img, f'{self.final_test_dir}/test{i+1:03d}_{k+1:03d}_gt_img.png')

                # computing loss, metrics on test set
                loss = self.criterion['main_loss'](output, target)
                batch_size = data_noisy.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size
        time_end = time.time()
        time_cost = time_end-time_start
        n_samples = len(self.test_data_loader.sampler)
        log = {'loss': total_loss / n_samples,
               'time/sample': time_cost/n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info('-'*70+'\n Final test result: '+str(log))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            # total = len(self.data_loader.dataset)
            total = self.data_loader.batch_size * self.limit_train_batches
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.limit_train_batches
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)


# ======================================
# Trainning: run Trainer for trainning
# ======================================


def trainning(gpus, config):
    # enable access to non-existing keys
    OmegaConf.set_struct(config, False)
    n_gpu = len(gpus)
    config.n_gpu = n_gpu

    if n_gpu > 1:
        torch.multiprocessing.spawn(
            multi_gpu_train_worker, nprocs=n_gpu, args=(gpus, config))
    else:
        train_worker(config)


def train_worker(config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    logger = get_logger('train')
    # setup data_loader instances
    train_data_loader, valid_data_loader = instantiate(
        config.data_loader)

    # conduct test ater training
    if config.trainer.final_test:
        test_data_loader = instantiate(config.test_data_loader)
    else:
        test_data_loader = None

    # use assigned validation during training
    if config.trainer.assigned_valid:
        logger.info('== using assigned validation set ==')
        valid_data_loader = instantiate(config.valid_data_loader)

    if not valid_data_loader:
        logger.warning('!= validation set  is empty =!')

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(
        f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = {}
    if 'main_loss' in config.loss:
        criterion['main_loss'] = instantiate(config.main_loss, is_func=True)
    if 'input_denoise_loss' in config.loss:
        criterion['input_denoise_loss'] = instantiate(
            config.input_denoise_loss, is_func=True)
    if 'forward_conv_loss' in config.loss:
        criterion['forward_conv_loss'] = instantiate(
            config.forward_conv_loss, is_func=True)

    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler, input_denoise_epoch=config.input_denoise_epoch)
    trainer.train()


def multi_gpu_train_worker(rank, gpus, config):
    """
    Training with multiple GPUs

    Args:
        rank ([type]): [description]
        gpus ([type]): [description]
        config ([type]): [description]

    Raises:
        RuntimeError: [description]
    """
    # initialize training config
    config.local_rank = rank
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=len(gpus),
        rank=rank)
    torch.cuda.set_device(gpus[rank])

    # start training processes
    train_worker(config)
