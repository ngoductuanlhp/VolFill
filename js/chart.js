// Numbers from Tables 1-2 of the paper (x10^2). Occluded = APD, complete = FS, at the tight threshold.
const DATA = {
  scrream: {
    gamma: '0.02',
    rows: [
      { m: 'VGGT',           g: 'pix',    occ: 10.04, comp: 36.77 },
      { m: 'DepthAnything3', g: 'pix',    occ: 10.76, comp: 43.10 },
      { m: 'DepthPro',       g: 'pix',    occ: 9.71,  comp: 28.44 },
      { m: 'MoGe2',          g: 'pix',    occ: 10.30, comp: 45.84 },
      { m: 'LaRI',           g: 'amodal', occ: 29.31, comp: 30.85 },
      { m: 'NOVA3R',         g: 'amodal', occ: 41.21, comp: 45.12 },
      { m: 'VolFill',        g: 'ours',   occ: 55.53, comp: 54.83 },
    ],
  },
  nrgbd: {
    gamma: '0.05',
    rows: [
      { m: 'VGGT',           g: 'pix',    occ: 26.35, comp: 58.91 },
      { m: 'DepthAnything3', g: 'pix',    occ: 28.15, comp: 63.52 },
      { m: 'DepthPro',       g: 'pix',    occ: 24.64, comp: 51.60 },
      { m: 'MoGe2',          g: 'pix',    occ: 28.43, comp: 61.93 },
      { m: 'LaRI',           g: 'amodal', occ: 45.82, comp: 43.47 },
      { m: 'NOVA3R',         g: 'amodal', occ: 58.55, comp: 49.92 },
      { m: 'VolFill',        g: 'ours',   occ: 62.48, comp: 63.92 },
    ],
  },
};

const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};

function panel(rows, key, title) {
  const W = 560, LABEL = 128, VAL = 64, ROW = 30, TOP = 34;
  const plotW = W - LABEL - VAL;
  const H = TOP + rows.length * ROW + 6;
  const max = 70;
  const x = (v) => (v / max) * plotW;
  const best = Math.max(...rows.filter((r) => r.g !== 'ours').map((r) => r[key]));

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
  svg.append(el('text', { x: 0, y: 14, class: 'c-title' }, title));
  for (const t of [0, 20, 40, 60]) {
    svg.append(el('line', { x1: LABEL + x(t), x2: LABEL + x(t), y1: TOP - 8, y2: H - 4, class: 'c-grid' }));
    svg.append(el('text', { x: LABEL + x(t), y: TOP - 12, 'text-anchor': 'middle', class: 'c-val', 'font-size': 10 }, t));
  }
  rows.forEach((r, i) => {
    const y = TOP + i * ROW;
    const ours = r.g === 'ours';
    const cls = ours ? 'is-ours' : r.g === 'amodal' ? 'is-amodal' : '';
    svg.append(el('text', { x: LABEL - 12, y: y + 17, 'text-anchor': 'end', class: `c-method ${cls}` }, ours ? 'VolFill (ours)' : r.m));
    svg.append(el('rect', { x: LABEL, y: y + 5, width: plotW, height: 18, rx: 3, class: 'c-track', opacity: 0.6 }));
    svg.append(el('rect', { x: LABEL, y: y + 5, width: Math.max(1, x(r[key])), height: 18, rx: 3, class: `c-bar ${cls}` }));
    const label = ours ? `${r[key].toFixed(2)}  +${(r[key] - best).toFixed(1)}` : r[key].toFixed(2);
    svg.append(el('text', { x: LABEL + x(r[key]) + 8, y: y + 18, class: `c-val ${cls}` }, label));
  });
  return svg;
}

export function initChart(root) {
  const body = root.querySelector('[data-chart-body]');
  const tabsRoot = root.closest('[data-dataset-tabs]');
  const render = (ds) => {
    const d = DATA[ds];
    body.replaceChildren(
      panel(d.rows, 'occ', `Occluded · APD@${d.gamma}`),
      panel(d.rows, 'comp', `Complete · FS@${d.gamma}`),
    );
  };
  tabsRoot?.addEventListener('tabchange', (e) => {
    const ds = e.detail.tab.dataset.dataset;
    if (ds) render(ds);
  });
  const sel = tabsRoot?.querySelector('[role="tab"][aria-selected="true"]');
  render(sel?.dataset.dataset || 'scrream');
}
