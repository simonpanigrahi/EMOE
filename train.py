import argparse
from run import EMOE_run

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[1111])
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--smoke', action='store_true',
                    help='pipeline check: first 160/32/32 clips, 1 epoch, separate log/result dirs')
parser.add_argument('--router', choices=['orig', 'light', 'equal'], default='orig',
                    help='how the three modality weights are chosen: orig = EMOE as published, '
                         'light = small per-modality router, equal = 1/3 each')
cli = parser.parse_args()

# Each router writes to its own folders, so runs never mix: orig keeps ./log and ./result
suffix = '' if cli.router == 'orig' else f'_{cli.router}'

if cli.smoke:
    # early_stop=0 makes do_train return after the first epoch
    config = {'early_stop': 0, 'subset_sizes': {'train': 160, 'valid': 32, 'test': 32}, 'router': cli.router}
    log_dir, res_save_dir = f"./log_smoke{suffix}", f"./result_smoke{suffix}"
else:
    config, log_dir, res_save_dir = {'router': cli.router}, f"./log{suffix}", f"./result{suffix}"

EMOE_run(model_name='emoe', dataset_name='mosi', config=config, is_tune=False, seeds=cli.seeds, model_save_dir="./pt",
         res_save_dir=res_save_dir, log_dir=log_dir, num_workers=cli.num_workers, mode='train')
