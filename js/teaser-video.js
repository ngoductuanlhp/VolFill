// Teaser video (video_teaser/VolFill_teaser_share.mp4: 2:26, 1080p30 with the score, ~93 MB, faststart). The
// source is attached when the figure nears the viewport (preload="none", so bytes flow only once it plays).
// Wide screens without reduced motion: autoplay muted while at least half visible, pause when scrolled away,
// never auto-resume after the visitor paused it. Phones and reduced motion: no autoplay; a tap plays with
// sound. The chip mirrors and toggles the sound (the native controls can too).
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const narrow = window.matchMedia('(max-width: 720px)');

export function initTeaserVideo(fig) {
  const video = fig.querySelector('video');
  const chip = fig.querySelector('[data-sound]');
  const autoplay = () => !reduceMotion.matches && !narrow.matches;
  let userPaused = false;   // the visitor paused it: don't resume on scroll
  let ours = false;         // the pending play/pause event comes from this module

  video.src = video.dataset.src;
  if (autoplay()) {
    // Safari dims a never-played video under its native controls; show them once playback starts.
    video.controls = false;
    video.addEventListener('playing', () => { video.controls = true; }, { once: true });
  } else {
    video.muted = false;
  }

  const syncChip = () => {
    const on = !video.muted && video.volume > 0;
    chip.setAttribute('aria-pressed', String(on));
    chip.querySelector('use').setAttribute('href', on ? '#i-sound' : '#i-mute');
    chip.querySelector('span').textContent = on ? 'Sound on' : 'Sound off';
  };
  chip.addEventListener('click', () => {
    video.muted = !video.muted;
    if (!video.muted && video.paused) video.play().catch(() => {});
  });
  video.addEventListener('volumechange', syncChip);
  video.addEventListener('play', () => { if (!ours) userPaused = false; ours = false; });
  video.addEventListener('pause', () => { if (!ours) userPaused = true; ours = false; });
  syncChip();

  new IntersectionObserver(([e]) => {
    const inView = e.intersectionRatio >= 0.5;
    if (inView && autoplay() && !userPaused && video.paused) {
      ours = true;
      video.play().catch(() => { ours = false; });
    } else if (!inView && !video.paused) {
      ours = true;
      video.pause();
    }
  }, { threshold: [0, 0.5] }).observe(fig);
}
