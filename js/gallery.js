// Click-to-play gallery for VolFill demos.
// Each thumbnail has data-src (mp4) and data-label. Clicking swaps the main player.

document.addEventListener('DOMContentLoaded', () => {
  const player = document.getElementById('main-player');
  if (!player) return;
  const video = player.querySelector('video');
  const overlay = player.querySelector('.overlay-info');
  const thumbs = document.querySelectorAll('.gallery .thumb');

  const setLoading = (on) => {
    player.classList.toggle('loading', on);
  };

  const switchTo = (btn, autoplay = true) => {
    const src = btn.dataset.src;
    const label = btn.dataset.label || '';
    if (!src) return;

    thumbs.forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    setLoading(true);
    video.oncanplay = () => setLoading(false);
    video.onerror   = () => setLoading(false);

    setTimeout(() => {
      video.poster = btn.dataset.poster || '';
      video.src = src;
      overlay.textContent = label;
      video.load();
      if (autoplay) {
        const p = video.play();
        if (p && p.catch) p.catch(() => { /* autoplay may be blocked */ });
      }
    }, 80);
  };

  thumbs.forEach(btn => btn.addEventListener('click', () => switchTo(btn)));

  if (thumbs.length) {
    const first = thumbs[0];
    first.classList.add('active');
    overlay.textContent = first.dataset.label || '';
    video.poster = first.dataset.poster || '';
    video.src = first.dataset.src;
    video.load();
    const p = video.play();
    if (p && p.catch) p.catch(() => { /* swallow */ });
  }
});
