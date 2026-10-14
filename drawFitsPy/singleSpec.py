"""
plot_spectrum_at_lb.py

在指定 (l, b) 处画一条谱线。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myPYTHON import *

doFITS = myFITS()


def plot_spectrum_at_lb(
    fits_path,
    l,
    b,
    out_png=None,
    xlim=(-20, 20),
    ylim=None,
    chan_ref=9001,
    half_win=100,
    label=None,
    color='C0',
    title=None,
):
    """
    读取 FITS cube，在 (l, b) 处取一条谱线并画图。

    Parameters
    ----------
    fits_path : str
        FITS 文件路径。
    l, b : float
        目标银道坐标 (deg)。
    out_png : str or None
        输出 PNG 路径；None 时自动根据 fits 文件名和 (l,b) 生成。
    xlim : tuple
        x 轴范围（速度轴）。
    ylim : tuple or None
        y 轴范围；None 时自动。
    chan_ref : int
        参考通道号（仅用于图上标记）。
    half_win : int
        参考通道附近的半窗宽（仅用于标记）。
    label : str or None
        图例标签；None 时用文件名。
    color : str
        谱线颜色。
    title : str or None
        图标题；None 时自动生成。

    Returns
    -------
    dict
        包含 x, y, rms, mean, max_abs, out_png 等信息。
    """
    # ------------------------------------------------------------------
    # 1) 读数据 + 取谱
    # ------------------------------------------------------------------
    data, head = doFITS.readFITS(fits_path)
    spY, spX = doFITS.getSpectraByLB(data, head, l, b)

    spY = np.asarray(spY, dtype=float)
    spX = np.asarray(spX, dtype=float)

    # ------------------------------------------------------------------
    # 2) 在 [minX, maxX] 窗口内做统计
    # ------------------------------------------------------------------
    minX, maxX = xlim
    mask_win = (spX >= minX) & (spX <= maxX)
    if np.any(mask_win):
        rms_win  = np.nanstd(spY[mask_win])
        mean_win = np.nanmean(spY[mask_win])
        max_win  = np.nanmax(np.abs(spY[mask_win]))
        n_win    = int(np.count_nonzero(mask_win))
    else:
        rms_win = mean_win = max_win = np.nan
        n_win   = 0

    # ------------------------------------------------------------------
    # 3) 输出路径
    # ------------------------------------------------------------------
    if out_png is None:
        base = os.path.splitext(os.path.basename(fits_path))[0]
        out_png = f'spectrum_{base}_l{l:.4f}_b{b:.4f}.png'

    # ------------------------------------------------------------------
    # 4) 画图
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.step(spX, spY, where='mid', lw=1.5, color=color,
            label=label if label is not None else os.path.basename(fits_path))
    ax.axhline(0, color='k', lw=0.8)
    ax.grid(True, alpha=0.3)

    # 标记参考通道
    if chan_ref is not None and 0 <= chan_ref < len(spX):
        ax.axvline(spX[chan_ref], color='r', ls='--', lw=0.8, alpha=0.6,
                   label=f'chan_ref={chan_ref}')

    ax.set_xlim(minX, maxX)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(doFITS.unit_Vlsr)
    ax.set_ylabel('Intensity')

    if title is None:
        title = f'Spectrum at (l, b) = ({l:.4f}, {b:.4f})'
    ax.set_title(title)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) 打印统计
    # ------------------------------------------------------------------
    print(f'Saved: {out_png}')
    print(f'  nchan = {len(spY)}')
    print(f'  Window [x in {minX} .. {maxX}] (n={n_win}): '
          f'rms={rms_win:.4f}, mean={mean_win:.4f}, max|y|={max_win:.4f}')

    return {
        'x': spX,
        'y': spY,
        'rms': rms_win,
        'mean': mean_win,
        'max_abs': max_win,
        'n_win': n_win,
        'out_png': out_png,
    }


# ----------------------------------------------------------------------
# 用法示例
# ----------------------------------------------------------------------
if __name__ == '__main__':
    FITS_PY = "/home/qzyan/WORK/projects/OTFpyV2/G1380+000_MergeU.fits"

    L_TARGET = 137.9132
    B_TARGET = 0.0751583

    plot_spectrum_at_lb(
        fits_path=FITS_PY,
        l=L_TARGET,
        b=B_TARGET,
        xlim=(-20, 20),
        ylim=(-1, 3),
        color='blue',
        label='OTFpy (this work)',
    )