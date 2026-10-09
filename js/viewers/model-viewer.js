const MV_SRC = 'https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js';
const MV_SRI = 'sha384-sr9b4Ux0WhAUGclJ0ym0FSY2zSOMmNSn0bP/SA0e6bNCrpn/5W3QL8mm+LdlQMKw';

let mvPromise = null;

export function loadModelViewer() {
  if (customElements.get('model-viewer')) return Promise.resolve();
  if (!mvPromise) {
    const s = document.createElement('script');
    s.type = 'module';
    s.src = MV_SRC;
    s.integrity = MV_SRI;
    s.crossOrigin = 'anonymous';
    document.head.appendChild(s);
    mvPromise = customElements.whenDefined('model-viewer');
  }
  return mvPromise;
}

export async function fetchList(url) {
  const resp = await fetch(url, { cache: 'no-cache' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  return (await resp.text()).split('\n').map((s) => s.trim()).filter(Boolean);
}

export function trackLoading(mv) {
  const stage = mv.parentElement;
  const bar = stage.querySelector('.stage-loader span');
  const done = () => { stage.classList.remove('is-loading'); if (bar) bar.style.width = '0'; };
  mv.addEventListener('progress', (e) => {
    const p = e.detail.totalProgress;
    stage.classList.toggle('is-loading', p < 1);
    if (bar) bar.style.width = `${Math.round(p * 100)}%`;
    if (p >= 1) setTimeout(done, 250);
  });
  mv.addEventListener('load', done);
  mv.addEventListener('error', done);
}

export function setupFullscreen(btn, target) {
  if (!btn) return;
  if (!document.fullscreenEnabled) { btn.hidden = true; return; }
  btn.addEventListener('click', () => toggleFullscreen(target));
  document.addEventListener('fullscreenchange', () => {
    const on = document.fullscreenElement === target;
    target.classList.toggle('is-fullscreen', on);
    btn.setAttribute('aria-label', on ? 'Exit fullscreen' : 'Fullscreen');
  });
}

export function toggleFullscreen(target) {
  if (!document.fullscreenEnabled) return;
  if (document.fullscreenElement) document.exitFullscreen();
  else target.requestFullscreen().catch(() => {});
}

// Scene carousel: thumbnails from models/<id>/input.jpg; arrow keys move selection.
export function buildCarousel(root, ids, modelsRoot, onSelect) {
  const track = root.querySelector('[data-track]');
  track.replaceChildren(...ids.map((id, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'thumb';
    b.setAttribute('role', 'option');
    b.setAttribute('aria-selected', 'false');
    b.setAttribute('aria-label', `Scene ${i + 1}`);
    b.tabIndex = -1;
    b.dataset.index = String(i);
    const img = document.createElement('img');
    img.src = `${modelsRoot}/${id}/input.jpg`;
    img.alt = '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.width = 118;
    img.height = 88;
    const label = document.createElement('span');
    label.textContent = String(i + 1).padStart(2, '0');
    b.append(img, label);
    b.addEventListener('click', () => onSelect(i));
    return b;
  }));
  const thumbs = [...track.children];

  const select = (i) => {
    thumbs.forEach((t, j) => {
      const on = j === i;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
    });
    const t = thumbs[i];
    track.scrollTo({ left: t.offsetLeft - (track.clientWidth - t.offsetWidth) / 2, behavior: 'smooth' });
  };

  const step = (d) => {
    const cur = thumbs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
    onSelect((cur + d + ids.length) % ids.length);
  };
  root.querySelector('[data-prev]')?.addEventListener('click', () => step(-1));
  root.querySelector('[data-next]')?.addEventListener('click', () => step(1));
  track.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      step(e.key === 'ArrowRight' ? 1 : -1);
      track.querySelector('[aria-selected="true"]')?.focus();
    }
  });
  return { select, step };
}
