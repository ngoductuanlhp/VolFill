import { initTheme } from './theme.js';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function renderMath() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    throwOnError: false,
  });
}

function initTopbar() {
  const bar = document.querySelector('[data-topbar]');
  const progress = document.querySelector('[data-progress]');
  const title = document.querySelector('#title');
  new IntersectionObserver(([e]) => bar.classList.toggle('is-scrolled', !e.isIntersecting), {
    rootMargin: '-56px 0px 0px 0px',
  }).observe(title);

  let ticking = false;
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
    ticking = false;
  };
  window.addEventListener('scroll', () => {
    if (!ticking) { ticking = true; requestAnimationFrame(update); }
  }, { passive: true });
  requestAnimationFrame(update);

  const links = new Map([...document.querySelectorAll('.topnav a')].map((a) => [a.hash.slice(1), a]));
  const sections = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  const visible = new Set();
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id));
    const active = sections.find((s) => visible.has(s.id));
    for (const [id, a] of links) {
      if (active && id === active.id) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  sections.forEach((s) => spy.observe(s));
}

function initReveal() {
  if (reduceMotion.matches || !('IntersectionObserver' in window)) return;
  const targets = document.querySelectorAll('.chapter-head, .stats, .widget, .compare, .gallery, .chart, .table-wrap, .paper-fig, .prose, .limits, .code-block');
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
    }
  }, { rootMargin: '0px 0px -8% 0px' });
  const below = [...targets].filter((el) => el.getBoundingClientRect().top >= window.innerHeight);
  for (const el of below) {
    el.classList.add('reveal');
    io.observe(el);
  }
}

function initTabs() {
  for (const root of document.querySelectorAll('[data-tabs]')) {
    const tabs = [...root.querySelectorAll(':scope > .tabs [role="tab"], :scope > [role="tablist"] [role="tab"]')];
    const select = (tab, focus) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
      }
      if (focus) tab.focus();
      root.dispatchEvent(new CustomEvent('tabchange', { detail: { tab } }));
    };
    tabs.forEach((t, i) => {
      t.type = 'button';
      t.addEventListener('click', () => select(t));
      t.addEventListener('keydown', (e) => {
        const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length], true); }
        if (e.key === 'Home') { e.preventDefault(); select(tabs[0], true); }
        if (e.key === 'End') { e.preventDefault(); select(tabs[tabs.length - 1], true); }
      });
    });
  }
}

function initLightbox() {
  const dlg = document.getElementById('lightbox');
  const img = dlg.querySelector('[data-lightbox-img]');
  const cap = dlg.querySelector('[data-lightbox-cap]');
  let opener = null;
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-lightbox]');
    if (!btn) return;
    opener = btn;
    img.src = btn.dataset.lightbox;
    img.alt = btn.querySelector('img')?.alt || btn.dataset.caption || '';
    cap.textContent = btn.dataset.caption || '';
    dlg.showModal();
  });
  dlg.querySelector('[data-close]').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => { img.removeAttribute('src'); opener?.focus(); });
}

function initCopy() {
  for (const btn of document.querySelectorAll('[data-copy]')) {
    const label = btn.querySelector('[data-copy-label]');
    btn.addEventListener('click', async () => {
      const text = document.querySelector(btn.dataset.copy).textContent;
      try {
        await navigator.clipboard.writeText(text);
        label.textContent = 'Copied';
      } catch (_) {
        label.textContent = 'Press ⌘C';
        const range = document.createRange();
        range.selectNodeContents(document.querySelector(btn.dataset.copy));
        getSelection().removeAllRanges();
        getSelection().addRange(range);
      }
      btn.classList.add('is-done');
      setTimeout(() => { label.textContent = 'Copy'; btn.classList.remove('is-done'); }, 1800);
    });
  }
}

// Import a module only when its element approaches the viewport.
function lazyInit(selector, load, margin = '600px') {
  const el = document.querySelector(selector);
  if (!el) return;
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    load().then((init) => init(el)).catch((err) => console.error(`[init ${selector}]`, err));
  }, { rootMargin: margin });
  io.observe(el);
}

function boot() {
  initTheme();
  renderMath();
  initTopbar();
  initTabs();
  initLightbox();
  initCopy();
  initReveal();

  const small = window.matchMedia('(max-width: 720px)').matches;
  lazyInit('[data-teaser]', () => import('./viewers/teaser.js').then((m) => (el) => m.initTeaser(el, { deferred: small })));
  lazyInit('[data-compare]', () => import('./viewers/compare.js').then((m) => m.initCompare));
  lazyInit('[data-chart]', () => import('./chart.js').then((m) => m.initChart), '400px');
  lazyInit('[data-widget="tudf"]', () => import('./widgets/tudf-slider.js').then((m) => m.initTudfSlider), '400px');
  lazyInit('[data-widget="flow"]', () => import('./widgets/flow-scrubber.js').then((m) => m.initFlowScrubber), '400px');
  lazyInit('#problem', () => import('./anim-player.js').then((m) => () => m.initAnimations()), '120%');
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
