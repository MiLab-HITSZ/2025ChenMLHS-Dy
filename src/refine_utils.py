# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def _evaluate_single(model, image, target_label):

    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(np.clip(image[np.newaxis, ...], 0., 1.), dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        pred = model(tensor).cpu().numpy()[0]
    q = np.full_like(pred, 0.5)
    fit = pred - q
    fit[target_label] = -fit[target_label]
    fit[fit < 0] = 0
    fitness = np.sum(fit)
    return fitness, fitness == 0.0


def _create_boundary_mask(h, w, block_size, boundary_width):

    mask = np.zeros((h, w), dtype=np.float32)
    for by in range(block_size, h, block_size):
        y0 = max(0, by - boundary_width)
        y1 = min(h, by + boundary_width)
        mask[y0:y1, :] = 1.0
    for bx in range(block_size, w, block_size):
        x0 = max(0, bx - boundary_width)
        x1 = min(w, bx + boundary_width)
        mask[:, x0:x1] = 1.0
    return mask


def _boundary_smooth_delta(delta, block_size=32, boundary_width=6, sigma=4.0):

    c, h, w = delta.shape
    mask = _create_boundary_mask(h, w, block_size, boundary_width)
    mask = mask[np.newaxis, ...]
    delta_smooth = np.zeros_like(delta)
    for ch in range(c):
        delta_smooth[ch] = gaussian_filter(delta[ch], sigma=sigma)
    return mask * delta_smooth + (1.0 - mask) * delta


def _compress_chrominance(delta, color_ratio=0.2):

    luminance = delta.mean(axis=0, keepdims=True)
    chrominance = delta - luminance
    return luminance + color_ratio * chrominance


def _pixel_sparse(model, ori, adv, target_label, max_query=40):

    delta = adv - ori
    flat = delta.flatten()
    abs_flat = np.abs(flat)
    sorted_idx = np.argsort(abs_flat)

    best = adv.copy()
    queries = 0
    for ratio in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        if queries >= max_query:
            break
        n_zero = int(len(flat) * ratio)
        mask = np.ones(len(flat), dtype=bool)
        mask[sorted_idx[:n_zero]] = False
        test = flat.copy()
        test[~mask] = 0
        adv_test = np.clip(ori + test.reshape(ori.shape), 0, 1)
        _, ok = _evaluate_single(model, adv_test, target_label)
        queries += 1
        if ok:
            best = adv_test
        else:
            break
    return best, queries


def _binary_shrink(model, ori, adv, target_label, max_query=25):

    delta = adv - ori
    low, high, best_a, best = 0.0, 1.0, 1.0, adv.copy()
    queries = 0
    while queries < max_query and (high - low) > 0.01:
        mid = (low + high) / 2.0
        test = np.clip(ori + mid * delta, 0, 1)
        _, ok = _evaluate_single(model, test, target_label)
        queries += 1
        if ok:
            best_a, best, high = mid, test, mid
        else:
            low = mid
    return best, queries, best_a


def _detect_perturbation_regions(delta, base_block=16, energy_ratio=0.05):

    c, h, w = delta.shape
    n_blocks_h = int(np.ceil(h / base_block))
    n_blocks_w = int(np.ceil(w / base_block))

    energies = []
    coords = []
    for by in range(n_blocks_h):
        for bx in range(n_blocks_w):
            y0 = by * base_block
            y1 = min((by + 1) * base_block, h)
            x0 = bx * base_block
            x1 = min((bx + 1) * base_block, w)
            energy = np.linalg.norm(delta[:, y0:y1, x0:x1])
            energies.append(energy)
            coords.append((y0, y1, x0, x1))

    energies = np.array(energies)
    max_e = energies.max()
    if max_e < 1e-6:
        return True, coords, 5

    threshold = max_e * energy_ratio
    perturbed = [coords[i] for i, e in enumerate(energies) if e > threshold]


    perturbed_ratio = len(perturbed) / len(coords)
    is_full = perturbed_ratio > 0.80


    if is_full:
        grid_size = 5
    else:

        ys = [y0 for y0, _, _, _ in perturbed] + [y1 for _, y1, _, _ in perturbed]
        xs = [x0 for _, _, x0, _ in perturbed] + [x1 for _, _, _, x1 in perturbed]
        h_span = max(ys) - min(ys)
        w_span = max(xs) - min(xs)
        grid_size = max(5, min(8, int(np.ceil(max(h_span, w_span) / (base_block * 3)))))

        if len(perturbed) <= 4:
            grid_size = 3

    return is_full, perturbed, grid_size

def _adaptive_block_restore(model, ori, adv, target_label,
                            is_full, perturbed_blocks, grid_size,
                            max_query=60, verbose=False):

    delta = adv - ori
    c, h, w = delta.shape
    adv_out = adv.copy()
    queries = 0

    if is_full:

        gy, gx = grid_size, grid_size
        bh = h // gy
        bw = w // gx
        blocks = []
        for by in range(gy):
            y0 = by * bh
            y1 = h if by == gy - 1 else (by + 1) * bh
            for bx in range(gx):
                x0 = bx * bw
                x1 = w if bx == gx - 1 else (bx + 1) * bw
                energy = np.linalg.norm(delta[:, y0:y1, x0:x1])
                blocks.append((energy, y0, y1, x0, x1))
    else:

        blocks = []
        for py0, py1, px0, px1 in perturbed_blocks:
            ph = py1 - py0
            pw = px1 - px0
            sub_bh = max(1, ph // grid_size)
            sub_bw = max(1, pw // grid_size)
            for by in range(grid_size):
                y0 = py0 + by * sub_bh
                y1 = py1 if by == grid_size - 1 else min(py0 + (by + 1) * sub_bh, py1)
                for bx in range(grid_size):
                    x0 = px0 + bx * sub_bw
                    x1 = px1 if bx == grid_size - 1 else min(px0 + (bx + 1) * sub_bw, px1)
                    if y1 <= y0 or x1 <= x0:
                        continue
                    energy = np.linalg.norm(delta[:, y0:y1, x0:x1])
                    blocks.append((energy, y0, y1, x0, x1))

    if not blocks:
        return adv_out, queries


    blocks.sort(key=lambda b: b[0])


    for round_idx in range(3):
        if queries >= max_query:
            break
        improved = False
        for energy, y0, y1, x0, x1 in blocks:
            if queries >= max_query:
                break

            if np.allclose(adv_out[:, y0:y1, x0:x1], ori[:, y0:y1, x0:x1], atol=1e-5):
                continue

            test = adv_out.copy()
            test[:, y0:y1, x0:x1] = ori[:, y0:y1, x0:x1]
            _, ok = _evaluate_single(model, test, target_label)
            queries += 1

            if ok:
                adv_out = test
                improved = True

        if not improved:
            break

    return adv_out, queries


def _region_binary_shrink(model, ori, adv, target_label,
                          perturbed_blocks, max_query=40):

    delta = adv - ori
    c, h, w = delta.shape
    adv_out = adv.copy()
    queries = 0

    if not perturbed_blocks:
        return adv_out, queries, 1.0


    blocks = []
    for y0, y1, x0, x1 in perturbed_blocks:
        energy = np.linalg.norm(delta[:, y0:y1, x0:x1])
        blocks.append((energy, y0, y1, x0, x1))

    blocks.sort(key=lambda b: b[0], reverse=True)

    best_alpha = 1.0
    for energy, y0, y1, x0, x1 in blocks:
        if queries >= max_query:
            break

        low, high = 0.0, 1.0
        block_alpha = 1.0
        for _ in range(5):
            if queries >= max_query:
                break
            mid = (low + high) / 2.0
            test = adv_out.copy()
            test[:, y0:y1, x0:x1] = np.clip(
                ori[:, y0:y1, x0:x1] + mid * (adv_out[:, y0:y1, x0:x1] - ori[:, y0:y1, x0:x1]),
                0, 1
            )
            _, ok = _evaluate_single(model, test, target_label)
            queries += 1
            if ok:
                block_alpha = mid
                adv_out = test
                high = mid
            else:
                low = mid

        if block_alpha < best_alpha:
            best_alpha = block_alpha

    return adv_out, queries, best_alpha


# ============================================================

def refine_adversarial_visual(ori, adv, model, target_label, eps,
                              max_queries=200,
                              q_limit=500,
                              verbose=True):


    np.random.seed(123)
    delta = adv - ori
    l2_before = np.linalg.norm(delta.flatten(), ord=2)
    queries_used = 0

    best_adv = adv.copy()
    best_l2 = l2_before

    if verbose:
        print(f"[VisualRefine] Start. L2={l2_before:.2f}, max_queries={max_queries}")


    is_full, perturbed_blocks, base_grid = _detect_perturbation_regions(
        delta, base_block=16, energy_ratio=0.05
    )
    if verbose:
        mode_str = "FULL-IMAGE" if is_full else "LOCAL"
        print(f"[VisualRefine] Perturbation mode: {mode_str}, "
              f"blocks={len(perturbed_blocks)}, base_grid={base_grid}x{base_grid}")



    reserve_post = min(100, max(20, int(max_queries * 0.10) + 20))
    budget_block = max(20, max_queries - reserve_post)

    if is_full:

        target_g = int(np.sqrt(budget_block / 1.5))
        grid_size = min(28, max(4, target_g))
    else:
        n_p = max(1, len(perturbed_blocks))

        target_g = int(np.sqrt(budget_block / (1.5 * n_p)))
        grid_size = min(16, max(2, target_g))

    if verbose:
        print(f"[BudgetPlan] reserve_post={reserve_post}, block_budget={budget_block}, "
              f"grid={grid_size}x{grid_size}")

    if queries_used < max_queries:
        adv_br, q_br = _adaptive_block_restore(
            model, ori, best_adv, target_label,
            is_full=is_full,
            perturbed_blocks=perturbed_blocks,
            grid_size=grid_size,
            max_query=min(budget_block, max_queries - queries_used),
            verbose=verbose
        )
        queries_used += q_br
        l2_br = np.linalg.norm((adv_br - ori).flatten(), ord=2)
        if l2_br < best_l2:
            best_adv = adv_br
            best_l2 = l2_br
            delta = best_adv - ori
            if verbose:
                print(f"  -> Block restore OK, L2={best_l2:.2f} (q={q_br})")


    if is_full and queries_used < max_queries:
        _, h, w = delta.shape
        gy, gx = grid_size, grid_size
        bh = h // gy
        bw = w // gx

        if bh * gy < h:
            test = best_adv.copy()
            test[:, bh * gy:, :] = ori[:, bh * gy:, :]
            _, ok = _evaluate_single(model, test, target_label)
            queries_used += 1
            if ok:
                best_adv = test
                best_l2 = np.linalg.norm((best_adv - ori).flatten(), ord=2)
                delta = best_adv - ori
                if verbose:
                    print(f"  -> Border cleanup (bottom) OK, L2={best_l2:.2f}")

        if bw * gx < w and queries_used < max_queries:
            test = best_adv.copy()
            test[:, :, bw * gx:] = ori[:, :, bw * gx:]
            _, ok = _evaluate_single(model, test, target_label)
            queries_used += 1
            if ok:
                best_adv = test
                best_l2 = np.linalg.norm((best_adv - ori).flatten(), ord=2)
                delta = best_adv - ori
                if verbose:
                    print(f"  -> Border cleanup (right) OK, L2={best_l2:.2f}")

    remain = max_queries - queries_used


    if remain >= 60:
        color_configs = [
            (0.3, 6, 4.0), (0.3, 8, 5.0),
            (0.2, 6, 4.0), (0.2, 8, 5.0),
            (0.1, 6, 4.0), (0.1, 8, 5.0),
            (0.0, 6, 4.0), (0.0, 8, 5.0),
        ]
    elif remain >= 30:
        color_configs = [
            (0.3, 6, 4.0),
            (0.2, 6, 4.0),
            (0.1, 8, 5.0),
        ]
    elif remain >= 10:
        color_configs = [
            (0.2, 6, 4.0),
        ]
    else:
        color_configs = []

    for cr, bw, sigma in color_configs:
        if queries_used >= max_queries:
            break

        delta_c = _compress_chrominance(delta, color_ratio=cr)
        delta_b = _boundary_smooth_delta(delta_c, block_size=32,
                                         boundary_width=bw, sigma=sigma)
        adv_test = np.clip(ori + delta_b, 0.0, 1.0)

        f, ok = _evaluate_single(model, adv_test, target_label)
        queries_used += 1

        if verbose and ok:
            l2_t = np.linalg.norm((adv_test - ori).flatten(), ord=2)
            print(f"[VisualRefine] color={cr}, bw={bw}, sigma={sigma}: "
                  f"OK, L2={l2_t:.2f}")

        if ok:
            l2_t = np.linalg.norm((adv_test - ori).flatten(), ord=2)
            if l2_t < best_l2:
                best_adv = adv_test
                best_l2 = l2_t
                delta = best_adv - ori
                if verbose:
                    print(f"  -> New best! L2={best_l2:.2f}")
        else:

            if verbose:
                print(f"[VisualRefine] color={cr}, bw={bw}, sigma={sigma}: lost target, keep previous best.")


    remain = max_queries - queries_used
    if remain > 10 and best_l2 < l2_before * 0.99:
        if verbose:
            print(f"[VisualRefine] Pixel sparsification...")
        adv_s, q_s = _pixel_sparse(model, ori, best_adv, target_label,
                                   max_query=min(remain - 5, 35))
        queries_used += q_s
        l2_s = np.linalg.norm((adv_s - ori).flatten(), ord=2)
        if l2_s < best_l2:
            best_adv = adv_s
            best_l2 = l2_s
            if verbose:
                print(f"  -> Sparse OK, L2={l2_s:.2f}")


    remain = max_queries - queries_used
    if remain > 15 and best_l2 < l2_before * 0.99:
        if verbose:
            print(f"[VisualRefine] Region binary shrink...")

        delta_current = best_adv - ori
        _, perturbed_current, _ = _detect_perturbation_regions(
            delta_current, base_block=16, energy_ratio=0.05
        )
        adv_r, q_r, alpha = _region_binary_shrink(
            model, ori, best_adv, target_label,
            perturbed_blocks=perturbed_current,
            max_query=min(remain - 5, 40)
        )
        queries_used += q_r
        l2_r = np.linalg.norm((adv_r - ori).flatten(), ord=2)
        if l2_r < best_l2:
            best_adv = adv_r
            best_l2 = l2_r
            if verbose:
                print(f"  -> Region shrink alpha={alpha:.3f}, L2={l2_r:.2f}")


    if queries_used < max_queries:
        f_final, ok_final = _evaluate_single(model, best_adv, target_label)
        queries_used += 1
    else:
        ok_final = True

    if not ok_final:

        best_adv = adv.copy()
        best_l2 = l2_before
        ok_final = True
        if verbose:
            print("[VisualRefine] Final check failed, revert to input adversarial example.")

    if verbose:
        print(f"[VisualRefine] Done. L2: {l2_before:.2f} -> {best_l2:.2f} "
              f"({'improved' if best_l2 < l2_before else 'no improvement'}), "
              f"queries={queries_used}, adv={ok_final}")

    return best_adv, queries_used, {
        'l2_before': l2_before,
        'l2_after': best_l2,
        'success': ok_final,
        'queries': queries_used
    }
