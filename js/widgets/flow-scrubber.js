const ROOT = 'media/widgets/flow';
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function loadImg(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.decoding = 'async';
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

export async function initFlowScrubber(root) {
  const body = root.querySelector('[data-widget-body]');
  const index = await fetch(`${ROOT}/index.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!index?.scenes?.length) return;

  body.innerHTML = `
    <div class="flow-grid">
      <div class="flow-view">
        <img class="flow-frame" alt="Decoded clean-latent estimate at the selected denoising step">
        <figure class="flow-input"><img alt="Input image"><figcaption>Input</figcaption></figure>
        <p class="flow-step" aria-live="polite"></p>
      </div>
      <div class="flow-side">
        <div class="seg flow-scenes" role="group" aria-label="Scene"></div>
        <label class="ctl" for="flow-t">
          <span class="ctl-name">t</span>
          <input type="range" id="flow-t" min="0" max="1" step="1" value="0">
          <output class="ctl-val" for="flow-t"></output>
        </label>
        <div class="flow-buttons">
          <button type="button" class="chip-btn" data-play aria-pressed="false">Play</button>
        </div>
        <p class="flow-eq" data-eq></p>
        <p class="flow-note">Early steps decode to a blurry average layout; detail and thin structures appear as $t \\to 1$. Behind the visible surfaces, the model is <em>generating</em>.</p>
      </div>
    </div>`;
  const frameImg = body.querySelector('.flow-frame');
  const inputImg = body.querySelector('.flow-input img');
  const stepEl = body.querySelector('.flow-step');
  const range = body.querySelector('#flow-t');
  const outT = body.querySelector('output');
  const eq = body.querySelector('[data-eq]');
  const playBtn = body.querySelector('[data-play]');
  const sceneSeg = body.querySelector('.flow-scenes');
  window.renderMathInElement?.(body.querySelector('.flow-note'), {
    delimiters: [{ left: '$', right: '$', display: false }], throwOnError: false,
  });

  let scene = null;
  let images = [];
  let timer = 0;

  const show = (i) => {
    const f = scene.frames[i];
    if (images[i]?.complete) frameImg.src = images[i].src;
    else frameImg.src = `${ROOT}/${scene.name}/${f.src}`;
    outT.textContent = f.t.toFixed(2);
    stepEl.textContent = `Euler step ${f.step} / ${scene.total_steps}`;
    const k = (s) => window.katex?.renderToString(s, { throwOnError: false }) ?? s;
    eq.innerHTML = `${k(`\\mathbf{z}_t = ${(1 - f.t).toFixed(2)}\\,\\boldsymbol{\\epsilon} + ${f.t.toFixed(2)}\\,\\hat{\\mathbf{z}}_1`)}`;
    root.dataset.sig = `${scene.name}-${i}`;
  };

  const stop = () => {
    clearInterval(timer);
    timer = 0;
    playBtn.setAttribute('aria-pressed', 'false');
    playBtn.textContent = 'Play';
  };
  const play = () => {
    if (Number(range.value) >= Number(range.max)) range.value = 0;
    playBtn.setAttribute('aria-pressed', 'true');
    playBtn.textContent = 'Pause';
    timer = setInterval(() => {
      const v = Number(range.value) + 1;
      if (v > Number(range.max)) { stop(); return; }
      range.value = v;
      show(v);
    }, 110);
  };
  playBtn.addEventListener('click', () => (timer ? stop() : play()));
  range.addEventListener('input', () => { stop(); show(Number(range.value)); });

  const setScene = async (name) => {
    stop();
    const meta = await fetch(`${ROOT}/${name}/flow.json`).then((r) => r.json());
    scene = { name, ...meta };
    sceneSeg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.name === name)));
    inputImg.src = `${ROOT}/${name}/input.jpg`;
    range.max = String(scene.frames.length - 1);
    range.value = String(scene.frames.length - 1);
    show(scene.frames.length - 1);
    images = scene.frames.map((f) => { const im = new Image(); im.src = `${ROOT}/${name}/${f.src}`; return im; });
    await Promise.allSettled(images.slice(0, 3).map((im) => im.decode?.()));
  };

  sceneSeg.replaceChildren(...index.scenes.map((s, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.name = s.name;
    b.textContent = s.label || `Scene ${i + 1}`;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => setScene(s.name));
    return b;
  }));
  await setScene(index.scenes[0].name);

  if (!reduceMotion.matches) {
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      range.value = 0;
      show(0);
      setTimeout(play, 400);
    }, { threshold: 0.6 });
    io.observe(frameImg);
  }
  await loadImg(frameImg.src).catch(() => {});
}
