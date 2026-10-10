// Hero logo loop (paper_figs/logo_anim/animate_logo.py): a transparent video, HEVC+alpha .mov for WebKit
// (Safari, every iOS browser) and VP9+alpha .webm elsewhere. Blink can decode HEVC but drops its alpha,
// so the choice is by engine, not canPlayType. Nothing downloads until the page is idle and the hero art
// is on screen (it is display:none on narrow screens); under reduced motion the poster (the static logo
// pose, = frame 0) stays.
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const webkit = 'webkitSupportsPresentationMode' in HTMLVideoElement.prototype;
const whenIdle = (fn) => (window.requestIdleCallback ? requestIdleCallback(fn, { timeout: 3000 }) : setTimeout(fn, 1500));

export function initHeroAnim(video) {
  let attached = false;
  let onScreen = false;
  let idle = false;

  const update = () => {
    if (reduceMotion.matches || !onScreen || !idle) { video.pause(); return; }
    if (!attached) {
      attached = true;
      video.src = webkit ? video.dataset.hevc : video.dataset.vp9;
    }
    video.play().catch(() => {});
  };

  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; update(); }, { threshold: 0.2 }).observe(video);
  reduceMotion.addEventListener('change', update);
  const start = () => whenIdle(() => { idle = true; update(); });
  if (document.readyState === 'complete') start();
  else window.addEventListener('load', start, { once: true });
}
