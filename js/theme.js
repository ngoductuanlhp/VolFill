const KEY = 'volfill-theme';

export const getTheme = () => document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';

function apply(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem(KEY, theme); } catch (_) { /* private mode */ }
  for (const btn of document.querySelectorAll('[data-theme-toggle]')) {
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

export function initTheme() {
  apply(getTheme());
  for (const btn of document.querySelectorAll('[data-theme-toggle]')) {
    btn.addEventListener('click', () => apply(getTheme() === 'dark' ? 'light' : 'dark'));
  }
}
