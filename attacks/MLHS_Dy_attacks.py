import numpy as np
import logging
import torch
import math
import cv2
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import lpips
from refine_utils import refine_adversarial_visual
from scipy.fftpack import dct, idct

class MLHS_Dy(object):

    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        y = kwargs['y']
        eps = kwargs['eps']
        pop_size = kwargs['pop_size']
        print('pop_size:', pop_size)
        generation = kwargs['generation']
        batch_size = kwargs['batch_size']
        method = kwargs['method']
        q_max = kwargs['q_max']
        tmp_folder_path = kwargs['tmp_folder_path']
        print('tmp_folder_path:', tmp_folder_path)
        box_size = kwargs['box_size']
        nchannels, img_rows, img_cols = x_list.shape[1:4]
        loss_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
        total = len(x_list)
        stats_target = {'l1': [], 'l2': [], 'linf': [], 'meanr': [], 'rmsd': [], 'lpips': []}
        stats_l2 = {'l1': [], 'l2': [], 'linf': [], 'meanr': [], 'rmsd': [], 'lpips': []}
        stats_full = {'l1': [], 'l2': [], 'linf': [], 'meanr': [], 'rmsd': [], 'lpips': []}
        ori_list = []
        adv_list = []
        ori_y_list = []
        adv_tar_list = []
        queries_list = []
        for i in range(total):
            target_label = np.argwhere(y_target[i] > 0)
            if y[i] is not None:
                y_bin = (y[i] > 0).astype(int)
                yt_bin = (y_target[i] > 0).astype(int)
                hide_labels = np.argwhere((y_bin == 1) & (yt_bin == 0)).flatten()
                with torch.no_grad():
                    x_t = torch.tensor(np.expand_dims(x_list[i], 0), dtype=torch.float32)
                    if torch.cuda.is_available():
                        x_t = x_t.cuda()
                    pred_ori = self.model(x_t).cpu().numpy()[0]
                pred_bin = (pred_ori >= 0.5).astype(int)
                print('sample pre:', np.argwhere(pred_bin > 0).flatten(), 'sample ground truth:', np.argwhere(y_bin > 0).flatten())
                match_ori = np.all(pred_bin == y_bin)
                print(f'[Sample {i}] Original pred match y_ground: {match_ori}')
                print(f'[Sample {i}] Labels to hide: {hide_labels}')
            print('Method Name:', method)
            if method == 'SA':
                print('=' * 70)
                print('q_max', q_max)
                eps = 0.05
                x_adv1, count_tem, fs1 = SquareAttack(pop_size, q_max, generation, img_rows * img_cols * nchannels, self.model, x_list[i], target_label, eps, batch_size, gradient=None)
                queries = count_tem
            elif method == 'MLHS-Dy':
                print('=' * 70)
                print('q_max', q_max)
                eps = 0.05
                pop_size = 10
                block_size = 8
                print('pop_size:', pop_size, 'block_size:', block_size)
                x_adv2, count_tem2, fs2 = mlae_latent_bicubic_de_full(pop_size, q_max, generation, self.model, x_list[i], target_label, eps, batch_size, block_size=block_size, color_coherence=0.3)
                x_adv1 = x_adv2
                queries = count_tem2
                if fs2 == 1:
                    x_adv1, q_refine, info = refine_adversarial_visual(x_list[i], x_adv1, self.model, target_label, eps, max_queries=q_max - queries, q_limit=q_max, verbose=True)
                    queries += q_refine
                    print(f"[Refine] L2: {info['l2_before']:.2f} -> {info['l2_after']:.2f}, queries={q_refine}")
            elif method == 'MLAEDE':
                print('=' * 70)
                print('q_max', q_max)
                eps = 0.05
                x_adv1, count_tem, fs1 = DE(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i], target_label, eps, batch_size, gradient=None)
                queries = int(count_tem)
            elif method == 'SBA':
                print('=' * 70)
                eps = 0.3
                print(f'eps={eps}, q_max={q_max}')
                x_adv1, count_tem1, fs1 = SimBA_attack(pop_size, q_max, generation, img_rows * img_cols * nchannels, self.model, x_list[i], target_label, eps, batch_size, gradient=None)
                queries = count_tem1
            with torch.no_grad():
                x_t = torch.tensor(np.expand_dims(x_adv1, axis=0), dtype=torch.float32)
                if torch.cuda.is_available():
                    x_t = x_t.cuda()
                pred = self.model(x_t).cpu().numpy()
            pred_bin = pred.copy()
            pred_bin[pred_bin >= 0.5] = 1
            pred_bin[pred_bin < 0.5] = -1
            print(pred_bin, y_target[i])
            match_target = np.all(pred_bin == y_target[i], axis=1)[0]
            diff = x_adv1 - x_list[i]
            diff_255 = (x_adv1 / 2 + 0.5) * 255 - (x_list[i] / 2 + 0.5) * 255
            l1 = np.linalg.norm(diff.flatten(), ord=1)
            l2 = np.linalg.norm(diff.flatten(), ord=2)
            linf = np.linalg.norm(diff.flatten(), ord=np.inf)
            meanr = np.mean(np.abs(diff))
            rmsd = np.sqrt(np.mean(np.square(diff_255)))
            orig_tensor = torch.from_numpy(x_list[i]).float().unsqueeze(0)
            adv_tensor = torch.from_numpy(x_adv1).float().unsqueeze(0)
            if torch.cuda.is_available():
                orig_tensor = orig_tensor.cuda()
                adv_tensor = adv_tensor.cuda()
            orig_tensor = orig_tensor * 2.0 - 1.0
            adv_tensor = adv_tensor * 2.0 - 1.0
            with torch.no_grad():
                lpips_val = loss_fn(orig_tensor, adv_tensor).item()
            print(f'\n[Sample {i}] {method} | TargetMatch={match_target} | L2={l2:.2f} | LPIPS={lpips_val:.4f} | Queries={queries}')
            ori_list.append(x_list[i].copy())
            adv_list.append(x_adv1.copy())
            ori_y_list.append(y[i].copy())
            adv_tar_list.append(y_target[i].copy())
            queries_list.append(queries)
            if match_target:
                stats_target['l1'].append(l1)
                stats_target['l2'].append(l2)
                stats_target['linf'].append(linf)
                stats_target['meanr'].append(meanr)
                stats_target['rmsd'].append(rmsd)
                stats_target['lpips'].append(lpips_val)
                if l2 < 77.6:
                    stats_l2['l1'].append(l1)
                    stats_l2['l2'].append(l2)
                    stats_l2['linf'].append(linf)
                    stats_l2['meanr'].append(meanr)
                    stats_l2['rmsd'].append(rmsd)
                    stats_l2['lpips'].append(lpips_val)
                    if lpips_val < 0.2:
                        stats_full['l1'].append(l1)
                        stats_full['l2'].append(l2)
                        stats_full['linf'].append(linf)
                        stats_full['meanr'].append(meanr)
                        stats_full['rmsd'].append(rmsd)
                        stats_full['lpips'].append(lpips_val)
        save_root = f'../{tmp_folder_path}/{method}/{q_max}/'
        os.makedirs(save_root, exist_ok=True)
        np.save(f'{save_root}{method}_{q_max}_ori.npy', np.array(ori_list))
        np.save(f'{save_root}{method}_{q_max}_adv.npy', np.array(adv_list))
        np.save(f'{save_root}{method}_{q_max}_ori_y.npy', np.array(ori_y_list))
        np.save(f'{save_root}{method}_{q_max}_adv_tar.npy', np.array(adv_tar_list))
        np.save(f'{save_root}{method}_{q_max}_queries.npy', np.array(queries_list))
        print(f'\n[SaveNPY] All samples saved to {save_root}')
        print(f'  ori:     {method}_{q_max}_ori.npy     shape={np.array(ori_list).shape}')
        print(f'  adv:     {method}_{q_max}_adv.npy     shape={np.array(adv_list).shape}')
        print(f'  ori_y:   {method}_{q_max}_ori_y.npy   shape={np.array(ori_y_list).shape}')
        print(f'  adv_tar: {method}_{q_max}_adv_tar.npy shape={np.array(adv_tar_list).shape}')
        print(f'  queries: {method}_{q_max}_queries.npy shape={np.array(queries_list).shape}')

        def _print_summary(title, count, data, total):
            asr = count / total * 100.0 if total > 0 else 0.0
            print(f"\n{'=' * 70}")
            print(f'{title}')
            print(f'ASR: {count}/{total} = {asr:.2f}%')
            if count > 0:
                print(f"L1 mean:        {np.mean(data['l1']):.4f}")
                print(f"L2 mean:        {np.mean(data['l2']):.4f}")
                print(f"Linf mean:      {np.mean(data['linf']):.4f}")
                print(f"Mean perturb:   {np.mean(data['meanr']):.4f}")
                print(f"RMSD mean:      {np.mean(data['rmsd']):.4f}")
                print(f"LPIPS mean:     {np.mean(data['lpips']):.4f}")
            else:
                print('No successful samples, cannot compute mean')
            print(f"{'=' * 70}")
        n_target = len(stats_target['l1'])
        n_l2 = len(stats_l2['l1'])
        n_full = len(stats_full['l1'])
        _print_summary('Target only', n_target, stats_target, total)
        _print_summary('Target + L2 < 77.6', n_l2, stats_l2, total)
        _print_summary('Target + L2 < 77.6 + LPIPS < 0.2', n_full, stats_full, total)
        return ([], 0, 0, 0, 0, 0, 0, 0)

class Problem_eps:

    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size

    def evaluate_adv(self, x_adv):
        """
        Evaluate a concrete adversarial image (not a latent perturbation).
        x_adv: ndarray of shape (c, h, w) in [0, 1].
        Returns fitness array of shape (1, 1).
        """
        x_adv = x_adv[np.newaxis, ...]
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(x_adv, 0.0, 1.0), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(x_adv, 0.0, 1.0), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape) + 0.5
        fit = p - q
        fit[:, self.target_label] = -fit[:, self.target_label]
        fit[np.where(fit < 0)] = 0
        fitness = np.sum(fit, axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness

def verify_adversarial(model, image, target_label):
    """
    Verify whether `image` satisfies the attack target `target_label` (fitness == 0).
    Logic is identical to Problem_eps.evaluate_adv.
    """
    with torch.no_grad():
        x_t = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32)
        if torch.cuda.is_available():
            x_t = x_t.cuda()
        pred = model(x_t).cpu().numpy()[0]
    p = np.copy(pred)
    q = np.zeros(p.shape) + 0.5
    fit = p - q
    fit[target_label] = -fit[target_label]
    fit[fit < 0] = 0
    fitness = np.sum(fit)
    return fitness == 0

class BicubicLatentMapper:

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, z):
        z_t = torch.from_numpy(z).float().unsqueeze(0)
        delta = torch.nn.functional.interpolate(z_t, size=(self.h, self.w), mode='bicubic', align_corners=False)
        return delta.squeeze(0).numpy()

def mlae_latent_bicubic_de_full(pop_size, q_max, generation, model, image, target_label, eps, batch_size, block_size=16, color_coherence=0.2):
    np.random.seed(123)
    problem = Problem_eps(model, image, target_label, eps, batch_size)
    c, h, w = image.shape
    n_h = (h + block_size - 1) // block_size
    n_w = (w + block_size - 1) // block_size
    print(f'[LatentBicubic] block_size={block_size}, latent_dim=({c},{n_h},{n_w}), total={c * n_h * n_w}, color_coh={color_coherence}')
    mapper = BicubicLatentMapper(h, w)
    v_delta = np.random.choice([-eps, eps], size=(c, 1, w))
    h_delta = np.random.choice([-eps, eps], size=(c, h, 1))
    x_adv_v = np.clip(image + v_delta, 0.0, 1.0)
    x_adv_h = np.clip(image + h_delta, 0.0, 1.0)
    f_v = problem.evaluate_adv(x_adv_v)
    f_h = problem.evaluate_adv(x_adv_h)
    eval_count = 2
    if f_v[0, 0] <= f_h[0, 0]:
        base_delta = v_delta
        base_stripe = 'vertical'
        base_fit = f_v[0, 0]
        x_ret_full = x_adv_v
    else:
        base_delta = h_delta
        base_stripe = 'horizontal'
        base_fit = f_h[0, 0]
        x_ret_full = x_adv_h
    print(f'[LatentBicubic] Stripe base: {base_stripe}, fitness={base_fit:.4f}')
    if base_fit == 0.0:
        return (x_ret_full, eval_count, 1)
    x_adv = x_ret_full.copy()
    f_init = problem.evaluate_adv(x_adv)
    eval_count += 1
    current_fitness = f_init[0, 0]
    print(f'[LatentBicubic] Initial fitness={current_fitness:.4f}')
    if current_fitness == 0.0:
        return (x_ret_full, eval_count, 1)
    Z_MAX = 1.0
    pop_shared = np.random.uniform(-Z_MAX, Z_MAX, (pop_size, 1, n_h, n_w))
    pop_diff = np.random.uniform(-3 * Z_MAX, 3 * Z_MAX, (pop_size, c, n_h, n_w))
    pop = np.repeat(pop_shared, c, axis=1) + pop_diff
    pop = np.clip(pop, -Z_MAX, Z_MAX)
    F, CR = (0.8, 0.9)
    replace_rate = 0.2
    fitness = np.zeros((pop_size, 1))
    for i in range(pop_size):
        z = pop[i].copy()
        if color_coherence < 1.0 and c == 3:
            z_mean = z.mean(axis=0, keepdims=True)
            z_diff = z - z_mean
            z = z_mean + color_coherence * z_diff
        delta_i = mapper(z) * eps
        x_adv_i = np.clip(x_adv + delta_i, 0.0, 1.0)
        f = problem.evaluate_adv(x_adv_i)
        fitness[i] = f[0, 0]
        eval_count += 1
    for gen in range(generation):
        if eval_count >= q_max:
            break
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros_like(fitness)
        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                continue
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            trial = pop[i].copy()
            cross_mask = np.random.rand(c, n_h, n_w) < CR
            trial[cross_mask] = mutant[cross_mask]
            z_trial = trial.copy()
            if color_coherence < 1.0 and c == 3:
                z_mean = z_trial.mean(axis=0, keepdims=True)
                z_diff = z_trial - z_mean
                z_trial = z_mean + color_coherence * z_diff
            delta_trial = mapper(z_trial) * eps
            x_adv_trial = np.clip(x_adv + delta_trial, 0.0, 1.0)
            f_trial = problem.evaluate_adv(x_adv_trial)
            eval_count += 1
            if f_trial[0, 0] <= fitness[i, 0]:
                new_pop[i] = trial
                new_fitness[i] = f_trial[0, 0]
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
        pop = new_pop
        fitness = new_fitness
        best_idx = np.argmin(fitness[:, 0])
        best_fitness = fitness[best_idx, 0]
        if best_fitness < current_fitness:
            z_best = pop[best_idx].copy()
            if color_coherence < 1.0 and c == 3:
                z_mean = z_best.mean(axis=0, keepdims=True)
                z_diff = z_best - z_mean
                z_best = z_mean + color_coherence * z_diff
            best_delta = mapper(z_best) * eps
            x_adv = np.clip(x_adv + best_delta, 0.0, 1.0)
            current_fitness = best_fitness
            if best_fitness == 0.0:
                l2_now = np.linalg.norm(x_adv - image)
                print(f'[LatentBicubic] Gen {gen + 1}: SUCCESS, L2={l2_now:.2f}, queries={eval_count}')
                break
            best_blocks = pop[best_idx].copy()
            pop = pop - best_blocks
            pop[best_idx] = 0.0
        elif (gen + 1) % 10 == 0:
            print(f'[LatentBicubic] Gen {gen + 1}: best={best_fitness:.4f}, current={current_fitness:.4f}, queries={eval_count} (no update)')
        sorted_idx = np.argsort(fitness[:, 0])
        n_replace = max(1, int(pop_size * replace_rate))
        worst_idx = sorted_idx[-n_replace:]
        for idx in worst_idx:
            pop[idx] = 0.0
            mask = np.random.rand(c, n_h, n_w) < 0.1
            shared_noise = np.random.uniform(-Z_MAX, Z_MAX, (1, n_h, n_w))
            diff_noise = np.random.uniform(-0.3 * Z_MAX, 0.3 * Z_MAX, (c, n_h, n_w))
            noise = np.repeat(shared_noise, c, axis=0) + diff_noise
            pop[idx][mask] = noise[mask]
    f_final = problem.evaluate_adv(x_adv)
    eval_count += 1
    gbest_fitness = f_final[0, 0]
    success = 1 if gbest_fitness == 0.0 else 0
    l2_final = np.linalg.norm(x_adv - image)
    print(f'[LatentBicubic] Done. fitness={gbest_fitness:.4f}, L2={l2_final:.2f}, queries={eval_count}, success={success}')
    return (x_adv, eval_count, success)

def SquareAttack(pop_size, q_max, generation, length, model, image, target_label, eps, batch_size, gradient=None):
    """
    Square Attack (Linf) 多标签版本。
    """
    np.random.seed(123)
    problem = Problem_eps(model, image, target_label, eps, batch_size)
    c, h, w = image.shape
    n_features = c * h * w
    min_val, max_val = (0, 1)
    init_delta = np.random.choice([-eps, eps], size=(c, 1, w))
    x_best = np.clip(image + init_delta, min_val, max_val)
    fitness = problem.evaluate_adv(x_best)
    eval_count = 1
    loss_min = fitness[0, 0]
    p_init = 0.05
    n_iters = q_max
    for i_iter in range(n_iters - 1):
        if loss_min == 0.0:
            break
        if eval_count > q_max:
            break
        delta = x_best - image
        p = p_selection(p_init, i_iter, n_iters)
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        x_window = image[:, center_h:center_h + s, center_w:center_w + s]
        x_best_window = x_best[:, center_h:center_h + s, center_w:center_w + s]
        while np.sum(np.abs(np.clip(x_window + delta[:, center_h:center_h + s, center_w:center_w + s], min_val, max_val) - x_best_window) < 1e-07) == c * s * s:
            delta[:, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps], size=(c, 1, 1))
        x_new = np.clip(image + delta, min_val, max_val)
        fitness_new = problem.evaluate_adv(x_new)
        eval_count += 1
        loss_new = fitness_new[0, 0]
        if loss_new <= loss_min:
            loss_min = loss_new
            x_best = x_new
        if loss_min == 0.0:
            print('Success!', loss_min)
            return (x_best, eval_count, 1)
    print('Fail!', loss_min)
    return (x_best, eval_count, 0)

def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)
    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init
    return p

def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = (x // 2 + 1, y // 2 + 1)
    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x), max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2
        counter2[0] -= 1
        counter2[1] -= 1
    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
    return delta

def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * -1
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5:
            delta = np.transpose(delta)
    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
    return delta

class SimBA:

    def __init__(self, model, image_size):
        self.model = model
        self.image_size = image_size
        self.model.eval()
        self.query_count = 0

    @staticmethod
    def diagonal_order(image_size, channels):
        x = torch.arange(0, image_size).cumsum(0)
        order = torch.zeros(image_size, image_size)
        for i in range(image_size):
            order[i, :image_size - i] = i + x[i:]
        for i in range(1, image_size):
            reverse = order[image_size - i - 1].index_select(0, torch.LongTensor([j for j in range(i - 1, -1, -1)]))
            order[i, image_size - i:] = image_size * image_size - 1 - reverse
        if channels > 1:
            order_2d = order
            order = torch.zeros(channels, image_size, image_size)
            for i in range(channels):
                order[i, :, :] = 3 * order_2d + i
        return order.view(1, -1).squeeze().long().sort()[1]

    @staticmethod
    def block_order(image_size, channels, initial_size=1, stride=1):
        order = torch.zeros(channels, image_size, image_size)
        total_elems = channels * initial_size * initial_size
        perm = torch.randperm(total_elems)
        order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
        for i in range(initial_size, image_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = torch.randperm(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, :i + stride, i:i + stride] = perm[:num_first].view(channels, -1, stride)
            order[:, i:i + stride, :i] = perm[num_first:].view(channels, stride, -1)
            total_elems += num_elems
        return order.view(1, -1).squeeze().long().sort()[1]

    @staticmethod
    def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
        z = torch.zeros(x.size())
        num_blocks = int(x.size(2) / block_size)
        mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
        if type(ratio) != float:
            for i in range(x.size(0)):
                mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
        else:
            mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size].numpy()
                if masked:
                    submat = submat * mask
                z[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
        if linf_bound > 0:
            return z.clamp(-linf_bound, linf_bound)
        else:
            return z

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z

    def get_fitness(self, x, target_labels):
        output = self.model(x).detach().cpu().numpy()
        p = np.copy(output)
        q = np.zeros(p.shape) + 0.5
        fit = p - q
        if isinstance(target_labels, (int, np.integer)):
            target_labels = [int(target_labels)]
        else:
            target_labels = np.asarray(target_labels).flatten().tolist()
        fit[:, target_labels] = -fit[:, target_labels]
        fit[fit < 0] = 0
        fitness = np.sum(fit, axis=1)
        self.query_count += x.size(0)
        return torch.tensor(fitness, dtype=torch.double)

    def is_adversarial(self, x, target_labels):
        output = self.model(x).detach().cpu().numpy()
        pred_bin = (output >= 0.5).astype(int)
        if output.ndim == 1:
            output = output.reshape(1, -1)
            pred_bin = pred_bin.reshape(1, -1)
        if isinstance(target_labels, (int, np.integer)):
            target_labels = [int(target_labels)]
        else:
            target_labels = np.asarray(target_labels).flatten().tolist()
        all_labels = set(range(output.shape[1]))
        other_labels = list(all_labels - set(target_labels))
        success = np.ones(output.shape[0], dtype=bool)
        for i in range(output.shape[0]):
            target_ok = np.all(pred_bin[i, target_labels] == 1)
            other_ok = np.all(pred_bin[i, other_labels] == 0) if len(other_labels) > 0 else True
            success[i] = target_ok and other_ok
        self.query_count += x.size(0)
        return torch.tensor(success)

    def simba_single(self, x, target_labels, num_iters=10000, epsilon=0.2):
        torch.manual_seed(123)
        np.random.seed(123)
        self.query_count = 0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_fitness = self.get_fitness(x, target_labels).item()
        for i in range(num_iters):
            if self.query_count >= num_iters:
                return (x.squeeze(), self.query_count, last_fitness)
            if last_fitness == 0.0:
                return (x.squeeze(), self.query_count, last_fitness)
            diff = torch.zeros(n_dims, device=x.device)
            diff[perm[i]] = epsilon
            left_x = (x - diff.view(x.size())).clamp(0, 1)
            left_fitness = self.get_fitness(left_x, target_labels).item()
            if left_fitness < last_fitness:
                x = left_x
                last_fitness = left_fitness
            else:
                right_x = (x + diff.view(x.size())).clamp(0, 1)
                right_fitness = self.get_fitness(right_x, target_labels).item()
                if right_fitness < last_fitness:
                    x = right_x
                    last_fitness = right_fitness
        return (x.squeeze(), self.query_count, last_fitness)

def SimBA_attack(pop_size, q_max, generation, length, model, image, target_label, eps, batch_size, gradient=None):
    if isinstance(image, np.ndarray):
        x = torch.from_numpy(image).float()
    else:
        x = image.float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()
    if isinstance(target_label, np.ndarray):
        target_labels = target_label.flatten().tolist()
    elif isinstance(target_label, (int, list)):
        target_labels = [target_label] if isinstance(target_label, int) else list(target_label)
    else:
        target_labels = [int(target_label)]
    image_size = x.size(2)
    simba = SimBA(model, image_size)
    x_adv, queries, final_fitness = simba.simba_single(x, target_labels, num_iters=q_max, epsilon=eps)
    success = 1 if final_fitness == 0.0 else 0
    x_adv_np = x_adv.cpu().numpy() if isinstance(x_adv, torch.Tensor) else x_adv
    if x_adv_np.ndim == 4:
        x_adv_np = x_adv_np[0]
    return (x_adv_np, queries, success)

class ProblemDE:

    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size

    def evaluate(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1)) + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.0, 1.0), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1)) + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.0, 1.0), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape) + 0.5
        fit = p - q
        fit[:, self.target_label] = -fit[:, self.target_label]
        fit[np.where(fit < 0)] = 0
        fitness = np.sum(fit, axis=1)
        fitness = fitness[:, np.newaxis]
        return (fitness, fit)

def _de_mating(pop, F):
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    return mutation

def _de_select(pop, fitness, fit, off, off_fitness, off_fit):
    new_pop = pop.copy()
    new_fitness = fitness.copy()
    new_fit = fit.copy()
    i = np.argwhere(fitness > off_fitness)
    new_pop[i] = off[i].copy()
    new_fitness[i] = off_fitness[i].copy()
    new_fit[i] = off_fit[i].copy()
    return (new_pop, new_fitness, new_fit)

def _de_complement(fit, pop, fitness, problem):
    popnew = pop.copy()
    sort = np.argsort(fitness.reshape(-1))
    for q in range(len(pop)):
        i = sort[q]
        fit_item = fit.copy()
        c = np.argwhere(fit[i] == 0)
        fit_item[:, c] = 0
        fitness_tem = np.sum(fit_item, axis=1)
        j = np.argmin(fitness_tem)
        popnew[i] = pop[i] + pop[j] * 0.5
    off_fitness_new, off_fit_new = problem.evaluate(popnew)
    pop1, fitness1, fit1 = _de_select(pop, fitness, fit, popnew, off_fitness_new, off_fit_new)
    return (pop1, fitness1, fit1)

def DE(pop_size, generation, length, model, image, target_label, eps, batch_size, gradient):
    problem = ProblemDE(model, image, target_label, eps, batch_size)
    print('pop_size:', pop_size)
    pop = np.random.uniform(-1, 1, size=(pop_size, length))
    if gradient is not None:
        pop[0] = np.reshape(np.sign(gradient), length)
    max_eval = pop_size * generation
    eval_count = 0
    fitness, fit = problem.evaluate(pop)
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    generation_save = np.zeros((10000,))
    generation_save[count] = fitmin
    F = 0.5
    if len(np.where(fitness == 0)[0]) == 0:
        while eval_count < max_eval:
            count += 1
            off = _de_mating(pop, F)
            off_fitness, off_fit = problem.evaluate(off)
            eval_count += pop_size
            pop, fitness, fit = _de_select(pop, fitness, fit, off, off_fitness, off_fit)
            pop, fitness, fit = _de_complement(fit, pop, fitness, problem)
            eval_count += pop_size
            fitmin = np.min(fitness)
            generation_save[count] = fitmin
            if len(np.where(fitness == 0)[0]) != 0:
                break
    if len(np.where(fitness == 0)[0]) != 0:
        r = pop[np.where(fitness == 0)[0][0]]
        x_adv = np.clip(image + np.reshape(r, image.shape) * eps, 0, 1)
        return (x_adv, eval_count, 1)
    else:
        r = pop[0]
        x_adv = np.clip(image + np.reshape(r, image.shape) * eps, 0, 1)
        return (x_adv, eval_count, 0)
