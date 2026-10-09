import { getTheme } from './theme.js';

const MANIFEST_URL = 'media/anim/manifest.json';
const desktop = window.matchMedia('(min-width: 961px)');
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function setSources(video, name, theme) {
  const base = `media/anim/${name}-${theme}`;
  const t = video.currentTime;
  const wasPlaying = !video.paused;
  video.poster = `${base}.jpg`;
  video.replaceChildren(
    Object.assign(document.createElement('source'), { src: `${base}.webm`, type: 'video/webm' }),
    Object.assign(document.createElement('source'), { src: `${base}.mp4`, type: 'video/mp4' }),
  );
  video.load();
  if (t > 0) video.addEventListener('loadedmetadata', () => { video.currentTime = t; if (wasPlaying) video.play().catch(() => {}); }, { once: true });
}

// Plays [from, to) and pauses at `to`; seeking is only needed when jumping backwards or ahead.
function makePlayer(video) {
  let stopAt = Infinity;
  let raf = 0;
  const tick = () => {
    if (video.currentTime >= stopAt - 0.03) {
      video.pause();
      video.currentTime = stopAt - 0.03;
      return;
    }
    raf = requestAnimationFrame(tick);
  };
  return {
    segment(from, to) {
      cancelAnimationFrame(raf);
      stopAt = to;
      if (video.currentTime < from - 0.05 || video.currentTime > to - 0.05) video.currentTime = from;
      video.play().then(() => { raf = requestAnimationFrame(tick); }).catch(() => {});
    },
    stop() { cancelAnimationFrame(raf); video.pause(); },
  };
}

function initScrolly(root, manifest) {
  const video = root.querySelector('video.anim');
  const name = root.dataset.scrolly;
  const meta = manifest[name];
  if (!video || !meta) return;
  const steps = [...root.querySelectorAll('.step')];
  const marks = [...meta.marks, meta.duration];
  const player = makePlayer(video);
  let loaded = false;
  let active = -1;

  const load = () => {
    if (loaded) return;
    loaded = true;
    video.preload = 'auto';
    setSources(video, name, getTheme());
  };
  document.addEventListener('themechange', (e) => { if (loaded) setSources(video, name, e.detail.theme); });

  const mode = () => (reduceMotion.matches ? 'static' : desktop.matches ? 'scroll' : 'loop');
  const applyMode = () => {
    const m = mode();
    video.loop = m === 'loop';
    video.controls = m === 'static';
    steps.forEach((s) => s.classList.toggle('is-active', m !== 'scroll'));
    if (m !== 'scroll') active = -1;
  };
  applyMode();
  desktop.addEventListener('change', applyMode);
  reduceMotion.addEventListener('change', applyMode);

  new IntersectionObserver(([e]) => {
    if (e.isIntersecting) load();
  }, { rootMargin: '100% 0px' }).observe(root);

  new IntersectionObserver(([e]) => {
    const m = mode();
    if (!e.isIntersecting) { player.stop(); return; }
    if (m === 'loop') video.play().catch(() => {});
  }, { threshold: 0.25 }).observe(root.querySelector('.scrolly-stage'));

  const io = new IntersectionObserver((entries) => {
    if (mode() !== 'scroll') return;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const i = steps.indexOf(e.target);
      if (i === active) continue;
      active = i;
      steps.forEach((s, j) => s.classList.toggle('is-active', j === i));
      const idx = Math.min(Number(e.target.dataset.t ?? i), marks.length - 2);
      player.segment(marks[idx], marks[idx + 1]);
    }
  }, { rootMargin: '-48% 0px -48% 0px' });
  steps.forEach((s) => io.observe(s));
}

export async function initAllScrolly() {
  let manifest;
  try {
    const r = await fetch(MANIFEST_URL, { cache: 'no-cache' });
    if (!r.ok) return;
    manifest = await r.json();
  } catch (_) {
    return;
  }
  document.querySelectorAll('[data-scrolly]').forEach((root) => initScrolly(root, manifest));
}
