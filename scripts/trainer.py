import argparse, os, datetime
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf
from transformers import logging as transf_logging
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from unifolm_wma.utils.train import set_logger, init_workspace, load_checkpoints, get_num_parameters


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed",
                        "-s",
                        type=int,
                        default=20250912,
                        help="seed for seed_everything")
    parser.add_argument("--name",
                        "-n",
                        type=str,
                        default="",
                        help="experiment name, as saving folder")
    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right.",
        default=list())
    parser.add_argument("--train",
                        "-t",
                        action='store_true',
                        default=False,
                        help='train')
    parser.add_argument("--val",
                        "-v",
                        action='store_true',
                        default=False,
                        help='val')
    parser.add_argument("--test",
                        action='store_true',
                        default=False,
                        help='test')
    parser.add_argument("--logdir",
                        "-l",
                        type=str,
                        default="logs",
                        help="directory for logging dat shit")
    parser.add_argument("--auto_resume",
                        action='store_true',
                        default=False,
                        help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only",
                        action='store_true',
                        default=False,
                        help="resume from weight-only checkpoint")
    parser.add_argument("--debug",
                        "-d",
                        action='store_true',
                        default=False,
                        help="enable post-mortem debugging")
    return parser


def get_nondefault_trainer_args(args):
    # Get default Trainer arguments by inspecting Trainer.__init__ signature
    import inspect
    sig = inspect.signature(Trainer.__init__)
    default_trainer_args = {}
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        default_trainer_args[param_name] = param.default if param.default != inspect.Parameter.empty else None
    
    # Return only non-default arguments that were set by the user
    return sorted(k for k in default_trainer_args.keys()
                  if hasattr(args, k) and getattr(args, k) != default_trainer_args[k])


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    # Manually add Trainer arguments (replacement for deprecated add_argparse_args)
    parser.add_argument('--accelerator', type=str, default='gpu', help='Supports passing different accelerator types')
    parser.add_argument('--strategy', type=str, default='auto', help='Supports different training strategies')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices to train on')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training')
    parser.add_argument('--precision', type=str, default=None, help='Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16)')
    parser.add_argument('--fast_dev_run', type=int, default=0, help='Runs n batch(es) of train, val and test to find any bugs')
    parser.add_argument('--max_epochs', type=int, default=None, help='Stop training once this number of epochs is reached')
    parser.add_argument('--min_epochs', type=int, default=None, help='Force training for at least these many epochs')
    parser.add_argument('--max_steps', type=int, default=-1, help='Stop training after this number of steps')
    parser.add_argument('--min_steps', type=int, default=None, help='Force training for at least these number of steps')
    parser.add_argument('--max_time', type=str, default=None, help='Stop training after this amount of time has passed')
    parser.add_argument('--limit_train_batches', type=float, default=None, help='How much of training dataset to check')
    parser.add_argument('--limit_val_batches', type=float, default=None, help='How much of validation dataset to check')
    parser.add_argument('--limit_test_batches', type=float, default=None, help='How much of test dataset to check')
    parser.add_argument('--limit_predict_batches', type=float, default=None, help='How much of prediction dataset to check')
    parser.add_argument('--overfit_batches', type=float, default=0.0, help='Overfit a fraction of training data')
    parser.add_argument('--val_check_interval', type=float, default=None, help='How often to check the validation set')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Check val every n train epochs')
    parser.add_argument('--num_sanity_val_steps', type=int, default=None, help='Sanity check runs n validation batches before starting the training routine')
    parser.add_argument('--log_every_n_steps', type=int, default=None, help='How often to log within steps')
    parser.add_argument('--enable_checkpointing', default=None, help='If True, enable checkpointing')
    parser.add_argument('--enable_progress_bar', default=None, help='Whether to enable to progress bar by default')
    parser.add_argument('--enable_model_summary', default=None, help='Whether to enable model summarization by default')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--gradient_clip_algorithm', type=str, default=None, help='The gradient clipping algorithm to use')
    parser.add_argument('--deterministic', default=None, help='If True, sets whether PyTorch operations must use deterministic algorithms')
    parser.add_argument('--benchmark', default=None, help='If True, enables cudnn.benchmark')
    parser.add_argument('--inference_mode', default=True, help='Whether to use torch.inference_mode() or torch.no_grad() during evaluation')
    parser.add_argument('--use_distributed_sampler', default=True, help='Whether to wrap the DataLoader sampler with DistributedSampler')
    parser.add_argument('--profiler', type=str, default=None, help='To profile individual steps during training and assist in identifying bottlenecks')
    parser.add_argument('--detect_anomaly', default=False, help='Enable anomaly detection for the autograd engine')
    parser.add_argument('--barebones', default=False, help='Whether to use the Trainer in barebones mode')
    parser.add_argument('--plugins', type=str, default=None, help='Plugins allow modification of core behavior')
    parser.add_argument('--sync_batchnorm', default=False, help='Synchronize batch norm layers between process groups/whole world')
    parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=0, help='Set to a non-negative integer to reload dataloaders every n epochs')
    parser.add_argument('--default_root_dir', type=str, default=None, help='Default path for logs and weights')
    args, unknown = parser.parse_known_args()
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # Setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir,
                                                       config,
                                                       lightning_config,
                                                       global_rank)
    logger = set_logger(
        logfile=os.path.join(loginfo, 'log_%d:%s.txt' % (global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]" % (pl.__version__))
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)
    # Load checkpoints
    model = load_checkpoints(model, config.model)

    # Register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas,
                                beta_schedule=model.beta_schedule,
                                timesteps=model.timesteps,
                                linear_start=model.linear_start,
                                linear_end=model.linear_end,
                                cosine_s=model.cosine_s)

    # Update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    # Setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr

    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.train_datasets:
        logger.info(
            f"{k}, {data.train_datasets[k].__class__.__name__}, {len(data.train_datasets[k])}"
        )
    if hasattr(data, 'val_datasets'):
        for k in data.val_datasets:
            logger.info(
                f"{k}, {data.val_datasets[k].__class__.__name__}, {len(data.val_datasets[k])}"
            )

    for item in unknown:
        if item.startswith('--total_gpus'):
            num_gpus = int(item.split('=')[-1])
            break
    model.datasets_len = len(data)

    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    # Setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # Setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir,
                                          ckptdir, logger)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(
        strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False

    # Trainer config: others (replacement for deprecated from_argparse_args)
    # Merge trainer_config with trainer_kwargs, with trainer_kwargs taking precedence
    all_trainer_args = dict(trainer_config)
    all_trainer_args.update(trainer_kwargs)
    trainer = Trainer(**all_trainer_args)

    # Allow checkpointing via USR1
    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # List the key model sizes
    total_params = get_num_parameters(model)

    logger.info("***** Running the Loop *****")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config[
                    'strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                if trainer_kwargs['precision'] == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>")
                trainer.fit(model, data)
        except Exception:
            raise
