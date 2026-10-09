import sys
import bpy
import numpy as np

for path in sys.argv[sys.argv.index("--") + 1:]:
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    # 8-bit PNGs are exposed as sRGB-encoded values already
    rgb = px[..., :3]
    a = px[..., 3]
    m = a > 0.95
    s = rgb[m]
    mx, mn = s.max(1), s.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    white = (mn > 0.88).mean()
    yellow = ((s[:, 0] > 0.75) & (s[:, 1] > 0.6) & (s[:, 2] < 0.55)).mean()
    ys, xs = np.nonzero(a > 0.02)
    print(f"{path.split('/')[-1]}: cover={m.mean():.3f} meanRGB={np.round(s.mean(0)*255).astype(int)} "
          f"sat={sat.mean():.2f} nearWhite={white:.3f} yellow={yellow:.3f} "
          f"bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] of {w}x{h}")
