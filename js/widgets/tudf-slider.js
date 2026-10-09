const BASE = 'media/widgets/tudf-slice';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const hex = (h) => {
  const s = h.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16));
};
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

function palette() {
  return {
    empty: hex(css('--bg-raise')),
    band0: hex(css('--blue')),
    band1: hex(css('--surface-2')),
    surface: hex(css('--yellow')),
  };
}

function slider(id, label, min, max, step, value, unit, ticks) {
  const wrap = document.createElement('label');
  wrap.className = 'ctl';
  wrap.htmlFor = id;
  wrap.innerHTML = `<span class="ctl-name">${label}</span>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"${ticks ? ` list="${id}-ticks"` : ''}>
    <output class="ctl-val" for="${id}"></output>${ticks ? `<datalist id="${id}-ticks">${ticks.map((t) => `<option value="${t}"></option>`).join('')}</datalist>` : ''}`;
  const input = wrap.querySelector('input');
  const out = wrap.querySelector('output');
  const show = () => { out.textContent = `${Number(input.value).toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0)}${unit}`; };
  input.addEventListener('input', show);
  show();
  return { wrap, input };
}

export async function initTudfSlider(root) {
  const body = root.querySelector('[data-widget-body]');
  const [meta, img] = await Promise.all([
    fetch(`${BASE}.json`).then((r) => r.json()),
    new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = `${BASE}.png`;
    }),
  ]);
  const { width: W, height: H, dmax } = meta;
  const off = document.createElement('canvas');
  off.width = W;
  off.height = H;
  const octx = off.getContext('2d', { willReadFrequently: true });
  octx.drawImage(img, 0, 0);
  const raw = octx.getImageData(0, 0, W, H).data;
  const dist = new Float32Array(W * H);
  for (let i = 0; i < dist.length; i++) dist[i] = (raw[i * 4] / 255) * dmax;

  body.innerHTML = `
    <div class="tudf-grid">
      <div class="tudf-view">
        <canvas width="${W}" height="${H}" aria-label="TUDF slice; colours show truncated distance to the nearest surface"></canvas>
        <div class="tudf-cam" aria-hidden="true"><span></span>camera</div>
        <p class="tudf-probe" aria-live="off"></p>
      </div>
      <div class="tudf-side">
        <div class="ctls"></div>
        <dl class="readout">
          <div><dt>Voxels in band</dt><dd data-band></dd></div>
          <div><dt>Surface voxels</dt><dd data-surf></dd></div>
        </dl>
        <ul class="key-list">
          <li><span class="sw sw-surf"></span>surface, $V(\\mathbf p) &lt; \\epsilon$</li>
          <li><span class="sw sw-band"></span>inside band, $V(\\mathbf p) &lt; \\tau$</li>
          <li><span class="sw sw-empty"></span>truncated, $V(\\mathbf p) = \\tau$</li>
        </ul>
        <button type="button" class="chip-btn" data-reset-ctl>VolFill setting: τ = 3</button>
      </div>
    </div>`;
  const canvas = body.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(W, H);
  const probe = body.querySelector('.tudf-probe');
  const bandEl = body.querySelector('[data-band]');
  const surfEl = body.querySelector('[data-surf]');

  const tau = slider('tudf-tau', 'τ', 1, 24, 0.5, 3, ' vox', [3]);
  const eps = slider('tudf-eps', 'ε', 0.25, 3, 0.05, 1, ' vox');
  body.querySelector('.ctls').append(tau.wrap, eps.wrap);
  window.renderMathInElement?.(body.querySelector('.key-list'), {
    delimiters: [{ left: '$', right: '$', display: false }], throwOnError: false,
  });

  let pal = palette();
  const render = () => {
    const t = Number(tau.input.value);
    const e = Math.min(Number(eps.input.value), t);
    let band = 0;
    let surf = 0;
    const px = out.data;
    for (let i = 0; i < dist.length; i++) {
      const d = dist[i];
      let c;
      if (d < e) { c = pal.surface; surf++; band++; }
      else if (d < t) { c = mix(pal.band0, pal.band1, Math.sqrt((d - e) / Math.max(t - e, 1e-6))); band++; }
      else c = pal.empty;
      px[i * 4] = c[0]; px[i * 4 + 1] = c[1]; px[i * 4 + 2] = c[2]; px[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    bandEl.textContent = `${((band / dist.length) * 100).toFixed(1)}%`;
    surfEl.textContent = `${((surf / dist.length) * 100).toFixed(1)}%`;
    root.dataset.sig = `${t}-${e}`;
  };

  tau.input.addEventListener('input', () => {
    if (Number(eps.input.value) > Number(tau.input.value)) {
      eps.input.value = tau.input.value;
      eps.input.dispatchEvent(new Event('input'));
    }
    render();
  });
  eps.input.addEventListener('input', render);
  body.querySelector('[data-reset-ctl]').addEventListener('click', () => {
    tau.input.value = 3;
    eps.input.value = 1;
    tau.input.dispatchEvent(new Event('input'));
    eps.input.dispatchEvent(new Event('input'));
  });
  document.addEventListener('themechange', () => { pal = palette(); render(); });

  canvas.addEventListener('pointermove', (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * W);
    const y = Math.floor(((ev.clientY - r.top) / r.height) * H);
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const d = dist[y * W + x];
    const t = Number(tau.input.value);
    probe.textContent = `dist = ${d >= dmax ? `≥ ${dmax}` : d.toFixed(1)}  →  V(p) = ${Math.min(d, t).toFixed(1)} voxels`;
  });
  canvas.addEventListener('pointerleave', () => { probe.textContent = ''; });
  render();
}
