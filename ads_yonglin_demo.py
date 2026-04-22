"""
ADS × Yonglin Limit: Entropy-Adaptive Step Size Demo
Demonstrates that ADS log-barrier = dynamic compression of Euler step size
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── helpers ──────────────────────────────────────────────────────────────────

def entropy(p):
    p = np.array(p)
    return -np.sum(p * np.log(p + 1e-10))

def kl(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# ── simulation ────────────────────────────────────────────────────────────────

def run(n_classes=4, steps=60, eta_base=0.08, seed=42):
    rng = np.random.default_rng(seed)

    # anchor A = empirical training distribution (not the true answer)
    A = softmax(rng.normal(0, 1, n_classes))
    # true answer A* = one-hot on class 0
    A_star = np.zeros(n_classes); A_star[0] = 1.0

    # energy: cross-entropy to A  →  E(p) = KL(p||A) + const
    def grad_E(p):
        return np.log(p + 1e-10) - np.log(A + 1e-10)

    def proj_simplex(v):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.where(u > (cssv - 1) / np.arange(1, len(u)+1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)

    # fixed-eta run (standard gradient descent on simplex)
    p_fixed = softmax(rng.normal(0, 2, n_classes))
    traj_fixed, kl_fixed, eta_fixed_log = [p_fixed.copy()], [], []

    # ADS-adaptive-eta run
    p_ads = p_fixed.copy()
    traj_ads, kl_ads, eta_ads_log, alpha_log = [p_ads.copy()], [], [], []

    H_max = np.log(n_classes)

    for _ in range(steps):
        # ── fixed eta ──
        g = grad_E(p_fixed)
        p_fixed = proj_simplex(p_fixed - eta_base * g)
        traj_fixed.append(p_fixed.copy())
        kl_fixed.append(kl(p_fixed, A))
        eta_fixed_log.append(eta_base)

        # ── ADS adaptive eta ──
        H = entropy(p_ads)
        B = H / (H_max + 1e-10)
        alpha = -np.log(1 - B + 1e-10)          # log-barrier
        eta_t = eta_base / (1 + alpha)            # admissible step
        g = grad_E(p_ads)
        p_ads = proj_simplex(p_ads - eta_t * g)
        traj_ads.append(p_ads.copy())
        kl_ads.append(kl(p_ads, A))
        eta_ads_log.append(eta_t)
        alpha_log.append(alpha)

    return dict(
        A=A, A_star=A_star,
        kl_fixed=kl_fixed, kl_ads=kl_ads,
        eta_fixed=eta_fixed_log, eta_ads=eta_ads_log,
        alpha=alpha_log,
        traj_fixed=traj_fixed, traj_ads=traj_ads,
        steps=steps,
    )

# ── plot ──────────────────────────────────────────────────────────────────────

def plot(d):
    t = np.arange(1, d['steps']+1)
    fig = plt.figure(figsize=(14, 9), facecolor='#0d1117')
    fig.suptitle('ADS × Yonglin Limit: Entropy-Adaptive Step Size',
                 color='white', fontsize=14, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

    style = dict(facecolor='#0d1117', color='white')
    for a in ax:
        a.set_facecolor('#161b22')
        a.tick_params(colors='#8b949e')
        a.xaxis.label.set_color('#8b949e')
        a.yaxis.label.set_color('#8b949e')
        for spine in a.spines.values():
            spine.set_edgecolor('#30363d')

    # 1. KL convergence
    ax[0].plot(t, d['kl_fixed'], color='#f78166', lw=1.8, label='Fixed η')
    ax[0].plot(t, d['kl_ads'],   color='#79c0ff', lw=1.8, label='ADS adaptive η')
    ax[0].set_title('KL Divergence to Anchor A', color='white', fontsize=11)
    ax[0].set_xlabel('Reasoning step t')
    ax[0].set_ylabel('KL(p_t ‖ A)')
    ax[0].legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    ax[0].set_yscale('log')

    # 2. Adaptive step size vs fixed
    ax[1].plot(t, d['eta_fixed'], color='#f78166', lw=1.8, label='Fixed η')
    ax[1].plot(t, d['eta_ads'],   color='#79c0ff', lw=1.8, label='ADS η_t')
    ax[1].set_title('Step Size Schedule', color='white', fontsize=11)
    ax[1].set_xlabel('Reasoning step t')
    ax[1].set_ylabel('η_t')
    ax[1].legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # 3. Log-barrier alpha over time
    ax[2].plot(t, d['alpha'], color='#d2a8ff', lw=1.8)
    ax[2].set_title('Log-Barrier α(B_s) = −log(1−B_s)', color='white', fontsize=11)
    ax[2].set_xlabel('Reasoning step t')
    ax[2].set_ylabel('α  (information barrier)')
    ax[2].axhline(1.0, color='#3fb950', lw=1, ls='--', alpha=0.6, label='α=1 threshold')
    ax[2].legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # 4. Belief trajectory for class 0 (true answer)
    traj_f = np.array(d['traj_fixed'])
    traj_a = np.array(d['traj_ads'])
    ax[3].plot(traj_f[:, 0], color='#f78166', lw=1.8, label='Fixed η — class 0')
    ax[3].plot(traj_a[:, 0], color='#79c0ff', lw=1.8, label='ADS η  — class 0')
    ax[3].axhline(d['A'][0],      color='#f78166', lw=1, ls=':', alpha=0.7, label=f'Anchor A[0]={d["A"][0]:.2f}')
    ax[3].axhline(d['A_star'][0], color='#3fb950', lw=1, ls='--', alpha=0.8, label='True A*[0]=1.0')
    ax[3].set_title('Belief on True Answer (class 0)', color='white', fontsize=11)
    ax[3].set_xlabel('Reasoning step t')
    ax[3].set_ylabel('p(class 0)')
    ax[3].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

    out = '/Users/lizixi/Desktop/reasoning-kingdom/ads_yonglin_demo.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'Saved → {out}')
    return out

if __name__ == '__main__':
    data = run()
    plot(data)
