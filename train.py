import argparse
from run import EMOE_run

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--smoke', action='store_true',
                    help='pipeline check: first 160/32/32 clips, 1 epoch, separate log/result dirs')
cli = parser.parse_args()

if cli.smoke:
    # early_stop=0 makes do_train return after the first epoch
    config = {'early_stop': 0, 'subset_sizes': {'train': 160, 'valid': 32, 'test': 32}}
    log_dir, res_save_dir = "./log_smoke", "./result_smoke"
else:
    config, log_dir, res_save_dir = None, "./log", "./result"

EMOE_run(model_name='emoe', dataset_name='mosi', config=config, is_tune=False, seeds=cli.seeds, model_save_dir="./pt",
         res_save_dir=res_save_dir, log_dir=log_dir, num_workers=cli.num_workers, mode='train')
