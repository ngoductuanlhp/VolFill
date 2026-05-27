// Teaser viewer — single <model-viewer> showing ours.glb for a curated set
// of scenes listed in models/valid_samples_teaser.txt.
//
// Loads the scene list at runtime, builds the thumb row, and updates the
// model-viewer + input image whenever the user picks a different scene.
//
// Wrapped in an IIFE so its top-level consts don't clash with viewer-swap.js.

(function () {
'use strict';

const TEASER_TXT   = 'models/valid_samples_teaser.txt';
const MODELS_ROOT  = 'models';
const DEFAULT_ORBIT = '0deg 75deg 150%';

let TEASER_SCENES = [];

async function loadTeaserScenes() {
  try {
    const resp = await fetch(TEASER_TXT, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${TEASER_TXT}`);
    const text = await resp.text();
    return text.split('\n').map(s => s.trim()).filter(Boolean);
  } catch (err) {
    console.error(`[teaser-viewer] failed to load ${TEASER_TXT}:`, err);
    return [];
  }
}

function buildTeaserThumbs(container) {
  container.innerHTML = TEASER_SCENES.map((id, i) => {
    const label = `Scene ${i + 1}`;
    return `
      <button class="viewer-thumb${i === 0 ? ' is-active' : ''}"
              data-stem="${id}" data-label="${label}">
        <span class="viewer-thumb-img-wrap">
          <img src="${MODELS_ROOT}/${id}/input.jpg" alt="${label}" loading="lazy"
               onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'placeholder',textContent:'${i + 1}'}))">
        </span>
        <span class="viewer-thumb-label">${label}</span>
      </button>
    `;
  }).join('');
}

async function initTeaserViewer() {
  const viewer    = document.getElementById('teaser-viewer');
  const inputImg  = document.getElementById('teaser-input-img'); // optional
  const thumbRow  = document.querySelector('.viewer-thumbs[data-row="teaser"]');
  if (!viewer || !thumbRow) {
    console.warn('[teaser-viewer] required DOM nodes not found');
    return;
  }

  TEASER_SCENES = await loadTeaserScenes();
  console.log(`[teaser-viewer] init: ${TEASER_SCENES.length} scenes`);
  if (TEASER_SCENES.length === 0) {
    thumbRow.innerHTML = '<p style="color:#888;font-size:13px;margin:0;">No teaser samples — populate models/valid_samples_teaser.txt.</p>';
    return;
  }

  buildTeaserThumbs(thumbRow);
  const thumbs = thumbRow.querySelectorAll('.viewer-thumb');

  const setScene = (btn) => {
    thumbs.forEach(t => t.classList.remove('is-active'));
    btn.classList.add('is-active');
    const stem = btn.dataset.stem;
    if (!stem) return;
    viewer.setAttribute('src', `${MODELS_ROOT}/${stem}/ours.glb`);
    if (inputImg) inputImg.src = `${MODELS_ROOT}/${stem}/input.jpg`;
    viewer.cameraOrbit = DEFAULT_ORBIT;
  };

  thumbs.forEach(btn => btn.addEventListener('click', () => setScene(btn)));

  // Activate the first thumb on init.
  if (thumbs.length) setScene(thumbs[0]);
}

const runInit = () => {
  initTeaserViewer().catch(err => console.error('[teaser-viewer] init failed:', err));
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runInit);
} else {
  runInit();
}

})();
