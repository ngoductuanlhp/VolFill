import { loadModelViewer, fetchList, trackLoading, setupFullscreen, toggleFullscreen, buildCarousel } from './model-viewer.js';

// SCENES come from models/valid_samples.txt (regenerate with scripts/validate_samples.py).
// `variant` must equal the GLB stem in models/<sample>/<variant>.glb, and the list must
// match the --methods passed to validate_samples.py. `group` only controls the UI grouping.
const SCENES_TXT = 'models/valid_samples.txt';

const BASELINES = [
  { group: 'Pixel-aligned', variant: 'moge2',  label: 'MoGe2' },
  { group: 'Pixel-aligned', variant: 'da3',    label: 'DepthAnything3' },
  { group: 'Amodal',        variant: 'nova3r', label: 'NOVA3R' },
  { group: 'Amodal',        variant: 'lari',   label: 'LaRI' },
];

const MODELS_ROOT = 'models';
const DEFAULT_ORBIT = '0deg 75deg 150%';

function buildVariantControls(container, onSelect) {
  const groups = new Map();
  for (const b of BASELINES) {
    if (!groups.has(b.group)) groups.set(b.group, []);
    groups.get(b.group).push(b);
  }
  const buttons = [];
  container.replaceChildren(...[...groups].map(([name, entries]) => {
    const g = document.createElement('div');
    g.className = 'seg-group';
    const label = document.createElement('span');
    label.className = 'seg-label';
    label.id = `seg-${name.toLowerCase().replace(/\W+/g, '-')}`;
    label.textContent = name;
    const seg = document.createElement('div');
    seg.className = 'seg';
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-labelledby', label.id);
    for (const b of entries) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = b.label;
      btn.dataset.variant = b.variant;
      btn.setAttribute('aria-pressed', 'false');
      btn.addEventListener('click', () => onSelect(b));
      buttons.push(btn);
      seg.append(btn);
    }
    g.append(label, seg);
    return g;
  }));
  return buttons;
}

export async function initCompare(root) {
  const left = root.querySelector('#viewer-left');
  const right = root.querySelector('#viewer-right');
  const rightLabel = root.querySelector('#viewer-right-label');
  const inputImg = root.querySelector('#viewer-input-img');
  const syncBtn = root.querySelector('[data-sync]');

  const [scenes] = await Promise.all([
    fetchList(SCENES_TXT).catch((err) => { console.error('[compare] scene list:', err); return []; }),
    loadModelViewer(),
  ]);
  if (!scenes.length) return;

  trackLoading(left);
  trackLoading(right);

  const state = { index: 0, variant: BASELINES[0], sync: true };

  const refresh = () => {
    const id = scenes[state.index];
    left.setAttribute('src', `${MODELS_ROOT}/${id}/ours.glb`);
    right.setAttribute('src', `${MODELS_ROOT}/${id}/${state.variant.variant}.glb`);
    inputImg.src = `${MODELS_ROOT}/${id}/input.jpg`;
    resetViews();
  };

  const resetViews = () => {
    for (const mv of [left, right]) {
      mv.cameraOrbit = DEFAULT_ORBIT;
      mv.cameraTarget = 'auto auto auto';
      mv.fieldOfView = 'auto';
    }
  };

  const variantButtons = buildVariantControls(root.querySelector('[data-variants]'), (b) => setVariant(b));
  const carousel = buildCarousel(root.querySelector('[data-carousel]'), scenes, MODELS_ROOT, (i) => setScene(i));

  function setScene(i) {
    state.index = i;
    carousel.select(i);
    refresh();
  }
  function setVariant(b, load = true) {
    state.variant = b;
    variantButtons.forEach((btn) => btn.setAttribute('aria-pressed', String(btn.dataset.variant === b.variant)));
    rightLabel.textContent = b.label;
    if (load) refresh();
  }

  // Only user-driven camera changes propagate, so programmatic resets don't feed back.
  const copyCameraState = (src, dst) => {
    try {
      const o = src.getCameraOrbit();
      const t = src.getCameraTarget();
      dst.cameraOrbit = `${o.theta}rad ${o.phi}rad ${o.radius}m`;
      dst.cameraTarget = `${t.x}m ${t.y}m ${t.z}m`;
      const fov = src.getFieldOfView();
      if (fov) dst.fieldOfView = `${fov}deg`;
    } catch (_) { /* viewer not ready yet */ }
  };
  const onUserCamera = (src, dst) => (e) => {
    if (!state.sync) return;
    if (e && e.detail && e.detail.source !== 'user-interaction') return;
    copyCameraState(src, dst);
  };
  left.addEventListener('camera-change', onUserCamera(left, right));
  right.addEventListener('camera-change', onUserCamera(right, left));

  syncBtn.addEventListener('click', () => {
    state.sync = !state.sync;
    syncBtn.setAttribute('aria-pressed', String(state.sync));
    if (state.sync) copyCameraState(left, right);
  });
  root.querySelector('[data-reset]').addEventListener('click', resetViews);
  setupFullscreen(root.querySelector('[data-fullscreen]'), root);

  let inView = false;
  new IntersectionObserver(([e]) => { inView = e.isIntersecting; }, { threshold: 0.35 }).observe(root);
  document.addEventListener('keydown', (e) => {
    if (!inView && !root.contains(document.activeElement)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (document.activeElement?.closest('[data-track]')) return;
    const n = Number(e.key);
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      carousel.step(e.key === 'ArrowRight' ? 1 : -1);
    } else if (n >= 1 && n <= BASELINES.length) {
      setVariant(BASELINES[n - 1]);
    } else if (e.key === 'r' || e.key === 'R') {
      resetViews();
    } else if (e.key === 'f' || e.key === 'F') {
      toggleFullscreen(root);
    }
  });

  setVariant(BASELINES[0], false);
  setScene(0);
}
