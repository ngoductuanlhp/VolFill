import { getTheme } from './theme.js';

const MANIFEST_URL = 'media/anim/manifest.json';
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function setSources(video, name, theme) {
  const base = `media/anim/${name}-${theme}`;
  const t = video.currentTime;
  const playing = !video.paused;
  video.poster = `${base}.jpg`;
  video.replaceChildren(
    Object.assign(document.createElement('source'), { src: `${base}.webm`, type: 'video/webm' }),
    Object.assign(document.createElement('source'), { src: `${base}.mp4`, type: 'video/mp4' }),
  );
  video.load();
  if (t > 0) {
    video.addEventListener('loadedmetadata', () => {
      video.currentTime = t;
      if (playing) video.play().catch(() => {});
    }, { once: true });
  }
}

function initFigure(fig, meta) {
  const name = fig.dataset.animFig;
  const video = fig.querySelector('video');
  const chips = [...fig.querySelectorAll('.beats button')];
  const marks = meta.marks;
  const ends = [...marks.slice(1), meta.duration];
  let loaded = false;
  let visible = false;

  const load = () => {
    if (loaded) return;
    loaded = true;
    video.preload = 'auto';
    setSources(video, name, getTheme());
  };
  const play = () => { if (!reduceMotion.matches) video.play().catch(() => {}); };

  const applyMotion = () => {
    video.controls = reduceMotion.matches;
    if (reduceMotion.matches) video.pause();
    else if (visible) play();
  };
  applyMotion();
  reduceMotion.addEventListener('change', applyMotion);
  document.addEventListener('themechange', (e) => { if (loaded) setSources(video, name, e.detail.theme); });

  // Beat chips: highlight the current beat and fill a progress bar through it.
  let active = -1;
  const sync = () => {
    const t = video.currentTime;
    let i = marks.length - 1;
    while (i > 0 && t < marks[i]) i--;
    if (i !== active) {
      active = i;
      chips.forEach((c, j) => c.setAttribute('aria-pressed', String(j === i)));
    }
    const p = Math.min(1, Math.max(0, (t - marks[i]) / (ends[i] - marks[i])));
    chips[i]?.style.setProperty('--p', p.toFixed(3));
  };
  let raf = 0;
  const loop = () => { sync(); raf = requestAnimationFrame(loop); };
  video.addEventListener('play', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); });
  video.addEventListener('pause', () => { cancelAnimationFrame(raf); sync(); });
  video.addEventListener('seeked', sync);

  chips.forEach((c, i) => {
    c.setAttribute('aria-pressed', 'false');
    c.addEventListener('click', () => {
      load();
      const go = () => { video.currentTime = marks[i] + 0.01; sync(); play(); };
      if (video.readyState >= 1) go();
      else video.addEventListener('loadedmetadata', go, { once: true });
    });
  });

  new IntersectionObserver(([e]) => { if (e.isIntersecting) load(); }, { rootMargin: '50% 0px' }).observe(fig);
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) play();
    else video.pause();
  }, { threshold: 0.5 }).observe(video);
}

export async function initAnimations() {
  let manifest;
  try {
    const r = await fetch(MANIFEST_URL, { cache: 'no-cache' });
    if (!r.ok) return;
    manifest = await r.json();
  } catch (_) {
    return;
  }
  document.querySelectorAll('[data-anim-fig]').forEach((fig) => {
    const meta = manifest[fig.dataset.animFig];
    if (meta) initFigure(fig, meta);
  });
}
