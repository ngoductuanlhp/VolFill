"""Screenshot + smoke-test harness for the project page.

  python scripts/screenshot.py --out /tmp/volfill_shots/phase1 [--sizes desktop,tablet,mobile]
                               [--themes dark,light] [--gl swiftshader] [--sections] [--no-full]

Serves the site from the repo root, then for each viewport x theme:
  * captures a full-page shot and (with --sections) one crop per <section id>
  * checks: no console errors, every image/GLB/video/font request returns 2xx/3xx,
    both viewers load, scene/baseline clicks swap `src`, camera sync propagates,
    theme toggle persists across reload, scrolly steps advance video time,
    widgets respond to slider input.
Exits non-zero if any check fails. Needs `pip install playwright && playwright install chromium`.
"""
import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve import make_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIZES = {"desktop": (1440, 900, 1), "tablet": (1024, 768, 1), "mobile": (390, 844, 2)}
SECTIONS = ["top", "problem", "tudf", "vae", "generation", "results",
            "quantitative", "ablations", "limitations", "bibtex"]
ASSET_TYPES = {"image", "media", "font", "stylesheet", "script"}


def serve(port):
    httpd = make_server(port, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)

    def skip(self, name, why):
        self.rows.append((name, None, why))
        print(f"  [SKIP] {name}  {why}", flush=True)

    @property
    def failed(self):
        return [r for r in self.rows if r[1] is False]


async def scroll_through(page, step=700, pause=120):
    h = await page.evaluate("document.documentElement.scrollHeight")
    y = 0
    while y < h:
        await page.evaluate(f"window.scrollTo(0, {y})")
        await page.wait_for_timeout(pause)
        y += step
        h = await page.evaluate("document.documentElement.scrollHeight")
    await page.evaluate("window.scrollTo(0, 0)")


async def mv_src(page, sel):
    return await page.evaluate(f"document.querySelector('{sel}')?.getAttribute('src')")


async def wait_mv_loaded(page, sel, timeout=45000):
    try:
        await page.wait_for_function(
            f"(() => {{ const m = document.querySelector('{sel}'); return m && m.loaded; }})()",
            timeout=timeout)
        return True
    except Exception:
        return False


async def functional_checks(page, rep, size):
    # Teaser viewer
    await page.locator("[data-teaser]").scroll_into_view_if_needed()
    cta = page.locator("[data-load-3d]")
    if await cta.is_visible():
        await cta.click()
    rep.check("teaser loads GLB", await wait_mv_loaded(page, "#teaser-viewer"),
              await mv_src(page, "#teaser-viewer") or "")
    thumbs = page.locator("[data-teaser] .thumb")
    if await thumbs.count() > 1:
        before = await mv_src(page, "#teaser-viewer")
        await thumbs.nth(1).click()
        await page.wait_for_timeout(300)
        rep.check("teaser scene swap", before != await mv_src(page, "#teaser-viewer"))

    # Comparison viewer
    await page.locator("[data-compare]").scroll_into_view_if_needed()
    ok = await wait_mv_loaded(page, "#viewer-left") and await wait_mv_loaded(page, "#viewer-right")
    rep.check("compare loads both GLBs", ok, f"{await mv_src(page, '#viewer-left')} | {await mv_src(page, '#viewer-right')}")
    first = (ROOT / "models/valid_samples.txt").read_text().split()[0]
    rep.check("compare default = first valid sample", first in (await mv_src(page, "#viewer-left") or ""))
    before = await mv_src(page, "#viewer-right")
    await page.locator("[data-variants] button").nth(2).click()
    await page.wait_for_timeout(300)
    rep.check("baseline swap changes src", before != await mv_src(page, "#viewer-right"),
              await mv_src(page, "#viewer-right") or "")
    before = await mv_src(page, "#viewer-left")
    await page.locator("[data-compare] .thumb").nth(1).click()
    await page.wait_for_timeout(300)
    rep.check("scene swap changes src", before != await mv_src(page, "#viewer-left"))
    await wait_mv_loaded(page, "#viewer-left")
    await wait_mv_loaded(page, "#viewer-right")

    if size != "mobile":
        # Lazy widgets above the viewer can shift layout; re-measure right before dragging.
        await page.evaluate("document.querySelector('#viewer-left').scrollIntoView({block: 'center'})")
        await page.wait_for_timeout(800)
        box = await page.locator("#viewer-left").bounding_box()
        hit = await page.evaluate(f"document.elementFromPoint({box['x'] + box['width'] / 2}, {box['y'] + box['height'] / 2})?.id")
        if hit != "viewer-left":
            print(f"    (drag target covered by #{hit})")
        orbit0 = await page.evaluate("document.querySelector('#viewer-right').getCameraOrbit().theta")
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        await page.mouse.move(cx, cy)
        await page.mouse.down()
        for i in range(12):
            await page.mouse.move(cx + 12 * (i + 1), cy, steps=2)
        await page.mouse.up()
        await page.wait_for_timeout(400)
        orbit1 = await page.evaluate("document.querySelector('#viewer-right').getCameraOrbit().theta")
        left1 = await page.evaluate("document.querySelector('#viewer-left').getCameraOrbit().theta")
        rep.check("camera sync propagates", abs(orbit1 - orbit0) > 1e-3,
                  f"right theta {orbit0:.3f} -> {orbit1:.3f} (left {left1:.3f}, hit #{hit})")

    # Inline animations
    fig = page.locator("[data-anim-fig='s1']")
    if await fig.count():
        await fig.scroll_into_view_if_needed()
        await page.wait_for_timeout(3000)
        t = await page.evaluate("document.querySelector('[data-anim-fig=s1] video').currentTime")
        rep.check("animation plays inline", t > 0.3, f"currentTime={t:.2f}")
        await fig.locator(".beats button").nth(3).click()
        await page.wait_for_timeout(600)
        mark = json.loads((ROOT / "media/anim/manifest.json").read_text())["s1"]["marks"][3]
        t = await page.evaluate("document.querySelector('[data-anim-fig=s1] video').currentTime")
        pressed = await fig.locator(".beats button").nth(3).get_attribute("aria-pressed")
        rep.check("beat chip seeks", t >= mark - 0.05 and pressed == "true", f"t={t:.2f} mark={mark} pressed={pressed}")
    else:
        rep.skip("animation plays inline", "no animation figures")

    # Widgets
    for wid in ("tudf", "flow"):
        sel = f"[data-widget='{wid}']"
        await page.locator(sel).scroll_into_view_if_needed()
        await page.wait_for_timeout(800)
        slider = page.locator(f"{sel} input[type=range]").first
        if not await slider.count():
            rep.skip(f"widget {wid} responds", "not built yet")
            continue
        sig0 = await page.evaluate(f"document.querySelector(\"{sel}\").dataset.sig || ''")
        await slider.focus()
        await page.keyboard.press("Home")
        for _ in range(3):
            await page.keyboard.press("ArrowRight")
        await page.wait_for_timeout(500)
        sig1 = await page.evaluate(f"document.querySelector(\"{sel}\").dataset.sig || ''")
        rep.check(f"widget {wid} responds", sig0 != sig1, f"{sig0[:24]} -> {sig1[:24]}")


async def run(args):
    httpd = serve(args.port)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}/"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    launch_args = []
    if args.gl == "swiftshader":
        launch_args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=launch_args)
        for size in args.sizes.split(","):
            vw, vh, scale = SIZES[size]
            for theme in args.themes.split(","):
                print(f"== {size} {theme}", flush=True)
                ctx = await browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=scale,
                                                color_scheme=theme, reduced_motion=args.reduced_motion)
                # The page ignores the OS scheme (light unless the visitor chose dark), so seed the stored choice;
                # only if unset, so the toggle-persists check below still sees the toggled value after reload.
                await ctx.add_init_script("if (!localStorage.getItem('volfill-theme-choice')) "
                                          f"localStorage.setItem('volfill-theme-choice', '{theme}')")
                page = await ctx.new_page()
                errors, bad = [], []
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: errors.append(str(e)))

                def on_response(r):
                    if r.request.resource_type in ASSET_TYPES or r.url.endswith(".glb"):
                        if r.status >= 400:
                            bad.append(f"{r.status} {r.url}")
                page.on("response", on_response)
                # Media range requests are routinely aborted by the browser while seeking.
                page.on("requestfailed", lambda r: bad.append(f"FAILED {r.url} {r.failure}")
                        if not (r.resource_type == "media" and "ERR_ABORTED" in (r.failure or "")) else None)

                await page.goto(base, wait_until="networkidle")
                await page.wait_for_timeout(1500)
                tag = f"{size}_{theme}"
                await page.screenshot(path=str(out / f"{tag}_00_fold.png"))
                await scroll_through(page)
                await page.wait_for_timeout(args.settle)

                if args.full:
                    await page.screenshot(path=str(out / f"{tag}_full.png"), full_page=True)
                if args.sections:
                    for sid in SECTIONS:
                        loc = page.locator(f"#{sid}")
                        if await loc.count():
                            await loc.scroll_into_view_if_needed()
                            await page.wait_for_timeout(700)
                            await loc.screenshot(path=str(out / f"{tag}_{sid}.png"))

                if args.checks and theme == args.themes.split(",")[0]:
                    await functional_checks(page, rep, size)
                    if args.sections:
                        for sid in ("top", "results"):
                            await page.locator(f"#{sid}").scroll_into_view_if_needed()
                            await page.wait_for_timeout(1500)
                            await page.locator(f"#{sid}").screenshot(path=str(out / f"{tag}_{sid}_loaded.png"))

                if args.checks and size == args.sizes.split(",")[0] and theme == args.themes.split(",")[0]:
                    await page.evaluate("window.scrollTo(0,0)")
                    t0 = await page.evaluate("document.documentElement.dataset.theme")
                    await page.click("[data-theme-toggle]")
                    t1 = await page.evaluate("document.documentElement.dataset.theme")
                    stored = await page.evaluate("localStorage.getItem('volfill-theme-choice')")
                    page2 = await ctx.new_page()
                    await page2.goto(base, wait_until="domcontentloaded")
                    t2 = await page2.evaluate("document.documentElement.dataset.theme")
                    rep.check("theme toggle flips", t0 != t1, f"{t0} -> {t1}")
                    rep.check("theme persists across reload", stored == t1 and t2 == t1, f"stored={stored} reloaded={t2}")
                    await page2.close()

                rep.check(f"no console errors [{tag}]", not errors, "; ".join(errors)[:600])
                rep.check(f"all asset requests ok [{tag}]", not bad, "; ".join(bad)[:600])
                await ctx.close()
        await browser.close()
    httpd.shutdown()

    (out / "report.json").write_text(json.dumps(rep.rows, indent=1))
    print(f"\n{len(rep.failed)} failed / {len(rep.rows)} checks. Shots in {out}")
    return 1 if rep.failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="desktop,tablet,mobile")
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--gl", default="default", choices=["default", "swiftshader"])
    ap.add_argument("--settle", type=int, default=2500)
    ap.add_argument("--sections", action="store_true")
    ap.add_argument("--no-full", dest="full", action="store_false")
    ap.add_argument("--no-checks", dest="checks", action="store_false")
    ap.add_argument("--reduced-motion", default="no-preference", choices=["no-preference", "reduce"])
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
