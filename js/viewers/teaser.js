import { loadModelViewer, fetchList, trackLoading, setupFullscreen, buildCarousel } from './model-viewer.js';

const TEASER_TXT = 'models/valid_samples_teaser.txt';
const MODELS_ROOT = 'models';
const DEFAULT_ORBIT = '0deg 75deg 112%';

export async function initTeaser(root) {
  const viewer = root.querySelector('#teaser-viewer');
  const inputImg = root.querySelector('#teaser-input-img');
  const counter = root.querySelector('[data-teaser-count]');
  const carouselEl = root.querySelector('[data-carousel]');

  const [scenes] = await Promise.all([
    fetchList(TEASER_TXT).catch((err) => { console.error('[teaser] scene list:', err); return []; }),
    loadModelViewer(),
  ]);
  if (!scenes.length) return;

  trackLoading(viewer);
  let current = -1;
  const carousel = buildCarousel(carouselEl, scenes, MODELS_ROOT, (i) => show(i));

  function show(i) {
    if (i === current) return;
    current = i;
    const id = scenes[i];
    viewer.setAttribute('src', `${MODELS_ROOT}/${id}/ours.glb`);
    inputImg.src = `${MODELS_ROOT}/${id}/input.jpg`;
    viewer.cameraOrbit = DEFAULT_ORBIT;
    if (counter) counter.textContent = `scene ${i + 1} / ${scenes.length}`;
    carousel.select(i);
  }

  root.querySelector('[data-reset]')?.addEventListener('click', () => {
    viewer.cameraOrbit = DEFAULT_ORBIT;
    viewer.cameraTarget = 'auto auto auto';
    viewer.fieldOfView = 'auto';
  });
  setupFullscreen(root.querySelector('[data-fullscreen]'), root);
  show(0);
}
