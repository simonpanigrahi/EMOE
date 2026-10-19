import argparse
from run import EMOE_run

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--smoke', action='store_true',
                    help='pipeline check: first 160/32/32 clips, 1 epoch, separate log/result dirs')
parser.add_argument('--router', choices=['orig', 'light', 'equal', 'self'], default='orig',
                    help='how the three modality weights are chosen: orig = EMOE as published, '
                         'light = small per-modality router, equal = 1/3 each, self = self-assessing experts')
parser.add_argument('--balance', choices=['clip', 'batch'], default='clip',
                    help='clip = EMOE (each clip pushed toward 1/3 each), batch = balance only the batch average')
parser.add_argument('--target', choices=['batch', 'history'], default='batch',
                    help="router target from this batch's errors (EMOE) or from each clip's running-average error")
cli = parser.parse_args()

# Each router writes to its own folders, so runs never mix: orig keeps ./log and ./result
suffix = '' if cli.router == 'orig' else f'_{cli.router}'
if cli.balance == 'batch':
    suffix += '_batchbal'
if cli.target == 'history':
    suffix += '_history'

if cli.smoke:
    # early_stop=0 makes do_train return after the first epoch
    config = {'early_stop': 0, 'subset_sizes': {'train': 160, 'valid': 32, 'test': 32}, 'router': cli.router, 'balance': cli.balance, 'target': cli.target}
    log_dir, res_save_dir = f"./log_smoke{suffix}", f"./result_smoke{suffix}"
else:
    config, log_dir, res_save_dir = {'router': cli.router, 'balance': cli.balance, 'target': cli.target}, f"./log{suffix}", f"./result{suffix}"

EMOE_run(model_name='emoe', dataset_name='mosi', config=config, is_tune=False, seeds=cli.seeds, model_save_dir="./pt",
         res_save_dir=res_save_dir, log_dir=log_dir, num_workers=cli.num_workers, mode='train')
