import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str, eva_imp, uni_distill, entropy_balance, batch_balance

logger = logging.getLogger('EMOE')


def corrupt_inputs(audio, vision, how):
    """Damage the voice or face features on purpose, to test robustness.

    voice_zero: the voice features become all zeros (voice missing)
    face_zero:  the face features become all zeros (face missing)
    noise:      random noise as strong as the features themselves is added to both
    """
    if how == 'voice_zero':
        audio = torch.zeros_like(audio)
    elif how == 'face_zero':
        vision = torch.zeros_like(vision)
    elif how == 'noise':
        audio = audio + audio.std() * torch.randn_like(audio)
        vision = vision + vision.std() * torch.randn_like(vision)
    else:
        raise ValueError(f"Unknown corruption '{how}'")
    return audio, vision


def summarize_weights(df):
    """One log line about the weights the router gave on the test clips.

    Questions it answers: what is the average weight per modality, do the weights
    change from clip to clip (std), how often is text the most trusted, how close
    to equal weights are they (entropy; 1.10 means exactly 1/3 each), and do
    bigger weights go to the modality whose own prediction is better on that clip
    (rank correlation with its error; negative is what a good router shows).
    """
    w = df[['w_text', 'w_face', 'w_voice']].to_numpy()
    entropy = -(w * np.log(np.clip(w, 1e-9, None))).sum(axis=1)
    # text counts as most trusted only if its weight is strictly the largest (ties do not count)
    text_top = ((w[:, 0] > w[:, 1] + 1e-6) & (w[:, 0] > w[:, 2] + 1e-6)).mean() * 100
    parts = []
    for m in ['text', 'face', 'voice']:
        if df[f'w_{m}'].std() < 1e-9:
            corr = 'n/a (weights never change)'
        else:
            corr = f"{df[f'w_{m}'].rank().corr(df[f'err_{m}'].rank()):+.2f}"
        parts.append(f"{m}: mean {df[f'w_{m}'].mean():.3f} std {df[f'w_{m}'].std():.3f} corr_with_error {corr}")
    return (" | ".join(parts) + f" | text most trusted on {text_top:.0f}% of clips"
            f" | mean entropy {entropy.mean():.3f} of max 1.099")


class EMOE():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)
                    w = output['channel_weight']

                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())

                    loss_task_l = self.criterion(output['logits_l'], labels)
                    loss_task_v = self.criterion(output['logits_v'], labels)
                    loss_task_a = self.criterion(output['logits_a'], labels)
                    loss_task_m = self.criterion(output['logits_c'], labels)

                    l_dist = eva_imp(output['logits_l'], labels)
                    a_dist = eva_imp(output['logits_a'], labels)
                    v_dist = eva_imp(output['logits_v'], labels)
                    dist = torch.zeros(l_dist.shape[0], 3).to(self.args.device)
                    for i,_ in enumerate(l_dist):
                        s = 1/(l_dist[i]+0.1) + 1/(v_dist[i]+0.1) + 1/(a_dist[i]+0.1)
                        dist[i][0] = (1/(l_dist[i]+0.1)) / s
                        dist[i][1] = (1/(v_dist[i]+0.1)) / s
                        dist[i][2] = (1/(a_dist[i]+0.1)) / s
                    loss_sim = torch.mean(torch.mean((dist.detach() - w) ** 2, dim=-1))
                    if self.args.get('balance', 'clip') == 'batch':
                        loss_ety = batch_balance(w)    # balance the batch average only
                    else:
                        loss_ety = entropy_balance(w)  # EMOE: push every clip toward 1/3 each

                    if self.args.fusion_method == "sum":
                        loss_ud = uni_distill(output['c_proj'], (output['l_proj'] * w[:,0].view(-1, 1) + output['v_proj'] * w[:,1].view(-1, 1) + 
                        output['a_proj'] * w[:,2].view(-1, 1)).detach())
                    elif self.args.fusion_method == "concat":
                        loss_ud = uni_distill(output['c_proj'], torch.cat([output['l_proj'] * w[:,0].view(-1, 1),output['v_proj'] * w[:,1].view(-1, 1),
                        output['a_proj'] * w[:,2].view(-1, 1)], dim=1).detach())

                    if self.args.get('router', 'orig') == 'self':
                        # Self-assessing experts: no separate router, so no balancing terms.
                        # Instead, teach each error head to guess its expert's real absolute
                        # error on this batch (detached: it is only a target).
                        real_err = torch.cat([
                            (output['logits_l'] - labels).abs(),
                            (output['logits_v'] - labels).abs(),
                            (output['logits_a'] - labels).abs(),
                        ], dim=1).detach()
                        loss_err = torch.mean(torch.abs(output['err_guess'] - real_err))
                        loss = loss_task_m + (loss_task_l + loss_task_v + loss_task_a)/3 + 0.1*loss_ud + loss_err
                    else:
                        loss = loss_task_m + (loss_task_l + loss_task_v + loss_task_a)/3 + 0.1*(loss_ety + 0.1*loss_sim) + 0.1*loss_ud

                    loss.backward()
                    train_loss += loss.item()

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/emoe.pth'
                torch.save(model.state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, f=0, corrupt=None, weights_csv=None):
        """corrupt: None, 'voice_zero', 'face_zero' or 'noise' (see corrupt_inputs).
        weights_csv: if a path is given, save one row per clip with its label, the
        predictions and the router weights, and log a summary line."""
        model.eval()
        y_pred, y_true = [], []
        weight, ability = [], []
        c_fea = []
        clip_rows = []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    if corrupt is not None:
                        audio, vision = corrupt_inputs(audio, vision, corrupt)
                    output = model(text, audio, vision)

                    loss = self.criterion(output['logits_c'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())
                    if weights_csv is not None:
                        w = output['channel_weight'].cpu()
                        for i in range(labels.shape[0]):
                            label = labels[i, 0].item()
                            row = {
                                'id': batch_data['id'][i],
                                'label': label,
                                'pred': output['logits_c'][i, 0].item(),
                                'w_text': w[i, 0].item(),
                                'w_face': w[i, 1].item(),
                                'w_voice': w[i, 2].item(),
                                # each expert's own error on this clip
                                'err_text': abs(output['logits_l'][i, 0].item() - label),
                                'err_face': abs(output['logits_v'][i, 0].item() - label),
                                'err_voice': abs(output['logits_a'][i, 0].item() - label),
                            }
                            if output.get('err_guess') is not None:
                                guess = output['err_guess'][i].cpu()
                                row.update({'guess_text': guess[0].item(), 'guess_face': guess[1].item(), 'guess_voice': guess[2].item()})
                            clip_rows.append(row)
                    
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        if weights_csv is not None:
            df = pd.DataFrame(clip_rows)
            df.to_csv(weights_csv, index=False)
            logger.info(f"WEIGHTS-({self.args.model_name}) >> {summarize_weights(df)}")
            logger.info(f"Per-clip weights saved to {weights_csv}")
        
        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results