import { loadModelViewer, fetchList, trackLoading, setupFullscreen, buildCarousel } from './model-viewer.js';

const TEASER_TXT = 'models/valid_samples_teaser.txt';
const MODELS_ROOT = 'models';
const DEFAULT_ORBIT = '0deg 75deg 112%';

const idle = () => new Promise((resolve) => {
  const go = () => (window.requestIdleCallback ? requestIdleCallback(resolve, { timeout: 1500 }) : setTimeout(resolve, 300));
  if (document.readyState === 'complete') go();
  else window.addEventListener('load', go, { once: true });
});

// `deferred`: don't fetch <model-viewer> until the user asks for 3D (small screens).
export async function initTeaser(root, { deferred = false } = {}) {
  const stage = root.querySelector('[data-stage]');
  const viewer = root.querySelector('#teaser-viewer');
  const inputImg = root.querySelector('#teaser-input-img');
  const counter = root.querySelector('[data-teaser-count]');
  const cta = root.querySelector('[data-load-3d]');

  const scenes = await fetchList(TEASER_TXT).catch((err) => { console.error('[teaser] scene list:', err); return []; });
  if (!scenes.length) return;

  let current = 0;
  let live = null;
  const activate = () => {
    live ??= loadModelViewer().then(() => {
      trackLoading(viewer);
      viewer.addEventListener('load', () => stage.classList.add('is-ready'));
      stage.classList.add('is-live');
    });
    return live;
  };

  const carousel = buildCarousel(root.querySelector('[data-carousel]'), scenes, MODELS_ROOT, (i) => show(i));

  function show(i) {
    current = i;
    const id = scenes[i];
    inputImg.src = `${MODELS_ROOT}/${id}/input.jpg`;
    if (counter) counter.textContent = `scene ${i + 1} / ${scenes.length}`;
    carousel.select(i);
    if (i !== 0) stage.classList.add('is-ready');
    activate().then(() => {
      if (current !== i) return;
      viewer.setAttribute('src', `${MODELS_ROOT}/${id}/ours.glb`);
      viewer.cameraOrbit = DEFAULT_ORBIT;
    });
  }

  root.querySelector('[data-reset]')?.addEventListener('click', () => {
    viewer.cameraOrbit = DEFAULT_ORBIT;
    viewer.cameraTarget = 'auto auto auto';
    viewer.fieldOfView = 'auto';
  });
  setupFullscreen(root.querySelector('[data-fullscreen]'), root);

  if (deferred) {
    carousel.select(0);
    if (counter) counter.textContent = `scene 1 / ${scenes.length}`;
    cta?.addEventListener('click', () => show(current));
  } else {
    await idle();
    show(0);
  }
}
