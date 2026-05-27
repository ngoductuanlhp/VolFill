// Wrapped in an IIFE so its top-level consts don't clash with teaser-viewer.js.

(function () {
'use strict';

// Qualitative-results viewer — data-driven.
//
//   * SCENES are loaded at runtime from `models/valid_samples.txt`. Regenerate
//     that file by running `scripts/validate_samples.py`, which keeps only
//     samples that contain every required `.glb` plus `input.jpg`.
//   * BASELINES is the only list still inline below. Edit it to add/remove
//     comparison methods; the `variant` value must match the `.glb` stem in
//     `models/<sample>/<variant>.glb`. After editing, re-run
//     `scripts/validate_samples.py --methods ours <variants...>` so the txt
//     stays in sync with what the page expects.
//
// The script:
//   1. Fetches the scene list, then builds the thumb buttons for both rows.
//   2. Loads the left viewer with ours.glb and the right viewer with the
//      currently selected baseline's .glb.
//   3. Syncs the two cameras (drag either, both orbit).

// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------

const SCENES_TXT = 'models/valid_samples.txt';
let   SCENES = [];

// Order is the visual order. The first entry overall is the default-active
// baseline on page load. `group` controls how thumbs are stacked in Row B.
const BASELINES = [
  { group: 'Pixel-aligned', variant: 'moge2',  label: 'MoGe2' },
  { group: 'Pixel-aligned', variant: 'da3',    label: 'DepthAnything3' },
  { group: 'Amodal',        variant: 'nova3r', label: 'NOVA3R' },
  { group: 'Amodal',        variant: 'lari',   label: 'LaRI' },
];

const MODELS_ROOT = 'models';

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function buildSceneThumbs(container) {
  container.innerHTML = SCENES.map((id, i) => {
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

function buildVariantThumbs(container) {
  // Group entries by `group` while preserving the order they first appear
  // in BASELINES.
  const groups = [];
  const byName = new Map();
  BASELINES.forEach((b) => {
    const name = b.group || '';
    if (!byName.has(name)) {
      const g = { name, entries: [] };
      byName.set(name, g);
      groups.push(g);
    }
    byName.get(name).entries.push(b);
  });

  let globalIndex = 0;
  container.innerHTML = groups.map((g) => `
    <div class="viewer-thumb-group">
      ${g.name ? `<div class="viewer-thumb-group-label">${g.name}</div>` : ''}
      <div class="viewer-thumb-group-thumbs">
        ${g.entries.map((b) => {
          const isActive = globalIndex === 0;
          globalIndex += 1;
          return `<button class="viewer-thumb viewer-thumb-text${isActive ? ' is-active' : ''}"
                          data-variant="${b.variant}" data-label="${b.label}">${b.label}</button>`;
        }).join('')}
      </div>
    </div>
  `).join('');
}

// -----------------------------------------------------------------------------
// Boot
// -----------------------------------------------------------------------------

async function loadScenes() {
  try {
    const resp = await fetch(SCENES_TXT, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${SCENES_TXT}`);
    const text = await resp.text();
    return text.split('\n').map(s => s.trim()).filter(Boolean);
  } catch (err) {
    console.error(`[viewer-swap] failed to load ${SCENES_TXT}:`, err);
    return [];
  }
}

async function initViewerSwap() {
  const left  = document.getElementById('viewer-left');
  const right = document.getElementById('viewer-right');
  if (!left || !right) {
    console.warn('[viewer-swap] viewer-left / viewer-right not found in DOM');
    return;
  }

  const sceneRow   = document.querySelector('.viewer-thumbs[data-row="scene"]');
  const variantRow = document.querySelector('.viewer-thumbs[data-row="variant"]');
  if (!sceneRow || !variantRow) {
    console.warn('[viewer-swap] .viewer-thumbs containers not found in DOM');
    return;
  }

  SCENES = await loadScenes();
  console.log(`[viewer-swap] init: ${SCENES.length} scenes, ${BASELINES.length} baselines`);
  if (SCENES.length === 0) {
    sceneRow.innerHTML = '<p style="color:#888;font-size:13px;margin:0;">No valid samples — run scripts/validate_samples.py.</p>';
    return;
  }

  // Inject thumb buttons from the data arrays.
  buildSceneThumbs(sceneRow);
  buildVariantThumbs(variantRow);

  const sceneThumbs   = sceneRow.querySelectorAll('.viewer-thumb');
  const variantThumbs = variantRow.querySelectorAll('.viewer-thumb');
  const rightLabel    = document.getElementById('viewer-right-label');
  const caption       = document.getElementById('viewer-caption');

  const state = {
    stem:    SCENES[0]   || null,
    variant: (BASELINES[0] && BASELINES[0].variant) || 'nova3r',
  };

  const DEFAULT_ORBIT = '0deg 75deg 150%';
  const inputImg = document.getElementById('viewer-input-img');
  const refresh = () => {
    if (!state.stem) return;
    left.setAttribute('src',  `${MODELS_ROOT}/${state.stem}/ours.glb`);
    right.setAttribute('src', `${MODELS_ROOT}/${state.stem}/${state.variant}.glb`);
    if (inputImg) inputImg.src = `${MODELS_ROOT}/${state.stem}/input.jpg`;
    // Reset camera to the default zoom on every scene/method swap so each
    // freshly loaded scene starts from the same vantage point.
    left.cameraOrbit  = DEFAULT_ORBIT;
    right.cameraOrbit = DEFAULT_ORBIT;
  };

  const setScene = (btn) => {
    sceneThumbs.forEach(t => t.classList.remove('is-active'));
    btn.classList.add('is-active');
    state.stem = btn.dataset.stem;
    if (caption) caption.textContent = btn.dataset.label || '';
    refresh();
  };

  const setVariant = (btn) => {
    variantThumbs.forEach(t => t.classList.remove('is-active'));
    btn.classList.add('is-active');
    state.variant = btn.dataset.variant;
    if (rightLabel) rightLabel.textContent = btn.dataset.label || btn.dataset.variant;
    refresh();
  };

  sceneThumbs.forEach(btn => btn.addEventListener('click', () => setScene(btn)));
  variantThumbs.forEach(btn => btn.addEventListener('click', () => setVariant(btn)));

  // Bidirectional camera sync — only propagate user-driven changes so that
  // programmatic cameraOrbit sets (from refresh()) don't feed back into the
  // viewer the user is actively dragging.
  const copyCameraState = (src, dst) => {
    try {
      const o = src.getCameraOrbit();
      const t = src.getCameraTarget();
      dst.cameraOrbit  = `${o.theta}rad ${o.phi}rad ${o.radius}m`;
      dst.cameraTarget = `${t.x}m ${t.y}m ${t.z}m`;
      const fov = src.getFieldOfView();
      if (fov) dst.fieldOfView = `${fov}deg`;
    } catch (_) { /* viewer not ready yet */ }
  };
  const onUserCamera = (src, dst) => (e) => {
    if (e && e.detail && e.detail.source !== 'user-interaction') return;
    copyCameraState(src, dst);
  };
  left.addEventListener('camera-change',  onUserCamera(left,  right));
  right.addEventListener('camera-change', onUserCamera(right, left));

  // Initialise.
  if (sceneThumbs.length)   setScene(sceneThumbs[0]);
  if (variantThumbs.length) setVariant(variantThumbs[0]);
}

// Run immediately if the DOM is already parsed; otherwise wait for it.
const runInit = () => {
  initViewerSwap().catch(err => console.error('[viewer-swap] init failed:', err));
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runInit);
} else {
  runInit();
}

})();
