// Light by default; dark only after the visitor picks it with the toggle. Only explicit choices are stored
// (under a new key, so the old 'volfill-theme' values that every visit used to save are ignored). The inline
// <head> script applies a stored choice before first paint.
const KEY = 'volfill-theme-choice';
const BAR = { dark: '#0E1015', light: '#FAF9F6' };      // browser UI colour (meta theme-color) = --bg

export const getTheme = () => (document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');

function apply(theme, persist) {
  document.documentElement.setAttribute('data-theme', theme);
  if (persist) {
    try { localStorage.setItem(KEY, theme); } catch (_) { /* private mode */ }
  }
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', BAR[theme]);
  for (const btn of document.querySelectorAll('[data-theme-toggle]')) {
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

export function initTheme() {
  apply(getTheme(), false);
  for (const btn of document.querySelectorAll('[data-theme-toggle]')) {
    btn.addEventListener('click', () => apply(getTheme() === 'dark' ? 'light' : 'dark', true));
  }
}
