import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import {
  spiral2D, LESSON_SPIRAL, ddpmSchedule, qSample, timeFeatures, denoiserInit, denoiserForward,
  denoiserFlatten, denoiserUnflatten, ddpmSample, nearestDist, gmmInit, gmmEM,
} from '../assets/gen.js';
import * as S from '../diffusion/stage.js';

const html = readFileSync(new URL('../diffusion/index.html', import.meta.url), 'utf8');
// The page's words, without tags; book titles in <cite> are left out of the wording checks.
const text = html.replace(/<cite>[^<]*<\/cite>/g, '').replace(/<[^>]*>/g, ' ');

// The spiral written out again: two arms of 200, radius 0.3 + 2.7 i/200, one turn,
// Gaussian noise SD 0.1 (Box–Muller), then each axis standardised (divide by n).
function spiralDirect(seed) {
  const rng = mulberry32(seed);
  const g = () => {
    let u = 0;
    while (u === 0) u = rng();
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const X = [];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < 200; i++) {
      const r = 0.3 + 2.7 * (i / 200);
      const th = 2 * Math.PI * (i / 200) + Math.PI * c;
      const x = r * Math.cos(th) + 0.1 * g();
      const y = r * Math.sin(th) + 0.1 * g();
      X.push([x, y]);
    }
  }
  const st = [0, 1].map((k) => {
    const m = X.reduce((s, p) => s + p[k], 0) / X.length;
    const v = X.reduce((s, p) => s + (p[k] - m) ** 2, 0) / X.length;
    return [m, Math.sqrt(v)];
  });
  return X.map((p) => [(p[0] - st[0][0]) / st[0][1], (p[1] - st[1][0]) / st[1][1]]);
}

// Every starting weight times (1 ± 1e-12), the sign alternating.
const nudged = (eps) => {
  const n = S.startNet();
  return denoiserUnflatten(n, denoiserFlatten(n).map((v, i) => v * (1 + (i % 2 ? -eps : eps))));
};

let RUN = null;
const pageRun = () => (RUN ??= S.storyRun());

// The first PRESSES presses after full training: noise, diffusion and mixture quality.
function pressTable(net) {
  return Array.from({ length: S.PRESSES }, (_, i) => {
    const { X: out, path } = ddpmSample(net, 400, S.schedule(), 313 + i, { keepPath: true });
    return { noise: S.quality(path[0]), diff: S.quality(out), mix: S.quality(S.mixSamples(i + 1)) };
  });
}
const better = (a, b) => a.precision < b.precision && a.coverage < b.coverage;

test('data reproducible', () => {
  assert.deepEqual(LESSON_SPIRAL, { n: 400, seed: 305, turns: 1, noise: 0.1 });
  const X = S.spiral();
  assert.equal(X.length, 400);
  const D = spiralDirect(305);
  X.forEach((p, i) => { assert.ok(Math.abs(p[0] - D[i][0]) < 1e-12 && Math.abs(p[1] - D[i][1]) < 1e-12, String(i)); });
  assert.deepEqual(X, spiral2D(LESSON_SPIRAL));
  assert.deepEqual(S.freshSpiral(), spiral2D({ ...LESSON_SPIRAL, seed: 307 }));
  assert.equal(X[0][0].toFixed(6), D[0][0].toFixed(6));
  assert.equal(X[0][0].toFixed(4), '0.1591');
  assert.equal(S.T, 50);
  assert.deepEqual(S.HIDDEN, [64, 64]);
  assert.deepEqual([S.SEED_INIT, S.SEED_TRAIN, S.SEED_SAMPLE, S.SEED_FORWARD, S.SEED_FRESH], [311, 312, 313, 300, 307]);
  assert.deepEqual([S.ETA, S.BATCH, S.STEPS], [3e-3, 128, 6000]);
  assert.deepEqual([S.MIX_K, S.MIX_SEED, S.MIX_TOL, S.MIX_MAX, S.MIX_SAMPLE], [5, 304, 1e-6, 500, 310]);
  assert.deepEqual(denoiserFlatten(S.startNet()), denoiserFlatten(denoiserInit({ hidden: [64, 64], seed: 311 })));
  assert.equal(S.paramCount(), 21 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2);
  const rng = mulberry32(300);
  const E = Array.from({ length: 400 }, () => [randn(rng), randn(rng)]);
  assert.deepEqual(S.forwardNoise(), E);
});

test('forward slider points equal qSample', () => {
  const sc = ddpmSchedule(50);
  for (const t of [0, 1, 7, 25, 50]) {
    const pts = S.forwardPoints(t);
    pts.forEach((p, i) => {
      const x0 = S.spiral()[i];
      const e = S.forwardNoise()[i];
      const a = Math.sqrt(sc.abar[t]);
      const b = Math.sqrt(1 - sc.abar[t]);
      assert.deepEqual(p, qSample(x0, t, e, sc));
      assert.ok(Math.abs(p[0] - (a * x0[0] + b * e[0])) < 1e-12);
    });
  }
  assert.deepEqual(S.forwardPoints(0), S.spiral());
  // The schedule written out: β linear from 0.001 to 0.2, ᾱ the running product of 1 − β.
  let ab = 1;
  for (let t = 1; t <= 50; t++) {
    const beta = 0.001 + (0.2 - 0.001) * (t - 1) / 49;
    ab *= 1 - beta;
    assert.ok(Math.abs(S.schedule().beta[t] - beta) < 1e-15);
    assert.ok(Math.abs(S.schedule().abar[t] - ab) < 1e-15);
  }
});

test('the one-step jump has the same mean and spread as 50 small steps', () => {
  // Monte Carlo: 20,000 runs of the chain x_t = √α_t x_{t−1} + √β_t ε from one x₀ to t = 25.
  // Tolerances: 5 standard errors of the mean (SD ≤ 1, n = 20,000) and of the variance.
  const sc = S.schedule();
  const x0 = S.spiral()[S.EXAMPLE];
  const rng = mulberry32(9);
  const n = 20000;
  const t = 25;
  let m0 = 0, m1 = 0, v0 = 0, v1 = 0;
  const xs = [];
  for (let r = 0; r < n; r++) {
    let x = x0.slice();
    for (let s = 1; s <= t; s++) {
      const a = Math.sqrt(sc.alpha[s]);
      const b = Math.sqrt(sc.beta[s]);
      x = [a * x[0] + b * randn(rng), a * x[1] + b * randn(rng)];
    }
    xs.push(x);
    m0 += x[0] / n; m1 += x[1] / n;
  }
  for (const x of xs) { v0 += (x[0] - m0) ** 2 / (n - 1); v1 += (x[1] - m1) ** 2 / (n - 1); }
  const a = Math.sqrt(sc.abar[t]);
  const v = 1 - sc.abar[t];
  assert.ok(Math.abs(m0 - a * x0[0]) < 5 * Math.sqrt(v / n) && Math.abs(m1 - a * x0[1]) < 5 * Math.sqrt(v / n));
  const seV = v * Math.sqrt(2 / (n - 1));
  assert.ok(Math.abs(v0 - v) < 5 * seV && Math.abs(v1 - v) < 5 * seV, `${v0} ${v1} ${v}`);
});

test('abar_T shown equals the schedule', () => {
  const sc = ddpmSchedule(50);
  const pv = S.pageValues();
  assert.equal(pv['abar-T'], Number(sc.abar[50].toPrecision(2)).toFixed(4));
  assert.equal(pv['abar-T'], '0.0045');
  assert.equal(pv['sqrt-abar-T'], Math.sqrt(sc.abar[50]).toFixed(3));
  assert.equal(pv['beta-1'], '0.001');
  assert.equal(pv['beta-T'], '0.2');
  assert.ok(sc.abar[50] < 0.01);
  assert.equal(S.readout(0, { t: 50 })[0], `t = 50 of 50 · ᾱₜ = ${pv['abar-T']}`);
  assert.equal(S.abarText(sc.abar[10]), sc.abar[10].toFixed(3));
});

test('guessing 0 scores the mean squared noise of the fixed batch', () => {
  const z = S.zeroGuessLoss();
  assert.ok(z > 1.8 && z < 2.2, String(z));
  assert.equal(S.pageValues()['zero-loss'], z.toFixed(2));
});

test('training and sampling deterministic', () => {
  const a = S.makeLive();
  const b = S.makeLive();
  a.advance(150);
  b.advance(100);
  b.advance(50);
  assert.equal(a.trainer.state.iter, 150);
  assert.deepEqual(Array.from(a.trainer.state.net.params), Array.from(b.trainer.state.net.params));
  assert.deepEqual(a.trainer.state.evalLoss, b.trainer.state.evalLoss);
  const full = pageRun().live;
  assert.equal(full.trainer.state.iter, 6000);
  assert.ok(full.done());
  full.advance(10);
  assert.equal(full.trainer.state.iter, 6000);
  // The page's sampler and its frames give exactly ddpmSample's path.
  const net = a.trainer.state.net;
  const ref = ddpmSample(net, 400, S.schedule(), 313 + 2, { keepPath: true });
  const g = S.makeGen(net, 3, 150);
  assert.equal(g.seed, 315);
  let frames = 0;
  while (!g.done()) { g.frame(); frames++; }
  assert.equal(frames, 50 * S.GEN_FRAMES);
  assert.deepEqual(g.sampler.path, ref.path);
  assert.deepEqual(g.points(), ref.X);
  assert.equal(g.tNow(), 0);
  const q = S.makeGen(net, 3, 150);
  while (!q.done()) q.quiet();
  assert.deepEqual(q.sampler.path, ref.path);
  // Half-way through a step the points lie half-way between path[k] and path[k + 1].
  const h = S.makeGen(net, 1, 150);
  for (let i = 0; i < S.GEN_FRAMES / 2; i++) h.frame();
  const p = h.points();
  const r = h.sampler.path;
  assert.ok(Math.abs(p[5][0] - (r[0][5][0] + r[1][5][0]) / 2) < 1e-12);
  assert.equal(h.tNow(), 50);
  // The sampler keeps its own copy of the weights.
  const live = S.makeLive();
  const g2 = S.makeGen(live.trainer.state.net, 1, 0);
  live.advance(5);
  g2.frame();
  const g3 = S.makeGen(S.startNet(), 1, 0);
  g3.frame();
  assert.deepEqual(g2.sampler.path[1], g3.sampler.path[1]);
});

test('the mixture equals the generative-models fit and has no collapsed component', async () => {
  const X = S.spiral();
  let p = gmmInit(X, 5, 304);
  let last = -Infinity;
  let it = 0;
  for (;;) {
    const r = gmmEM(X, p, 1);
    p = r.params;
    it++;
    const L = r.logLik[1];
    const rise = L - (it === 1 ? r.logLik[0] : last);
    last = L;
    if (rise < 1e-6 || it >= 500) break;
  }
  const m = S.mixFit();
  assert.equal(m.iter, it);
  assert.deepEqual(m.params, p);
  for (let i = 1; i < m.logLik.length; i++) assert.ok(m.logLik[i] >= m.logLik[i - 1] - 1e-12);
  // No component holds fewer than 20 points' worth of weight or has a near-flat ellipse.
  for (let k = 0; k < 5; k++) {
    assert.ok(m.params.pi[k] * 400 >= 20, `pi ${m.params.pi[k]}`);
    const Sg = m.params.S[k];
    assert.ok(Sg[0][0] * Sg[1][1] - Sg[0][1] ** 2 > 1e-3);
  }
  const F = await import('../diffusion/stage.js?sliced');
  let slices = 0;
  while (!F.mixWork(3)) slices++;
  assert.ok(slices > 10);
  assert.deepEqual(F.mixFit().params, m.params);
  const gm = new URL('../generative-models/stage.js', import.meta.url);
  if (existsSync(gm)) {
    const G = await import(gm.href);
    assert.deepEqual(G.fitResult('spiral').params, m.params);
    assert.deepEqual(G.spiralSamples(310), S.mixSamples(1));
  }
});

test('story: precision and coverage of noise, the mixture and the diffusion samples', () => {
  const run = pageRun();
  const s = run.state;
  assert.equal(s.iter, 6000);
  assert.equal(s.diverged, false);
  const sv = S.storyValues();
  const ev = s.evalLoss;
  assert.equal(ev.length, 61);
  // The starting network does worse than guessing 0; after 100 steps it is far below.
  assert.ok(ev[0] > S.zeroGuessLoss());
  assert.ok(ev[1] < 0.5 * S.zeroGuessLoss());
  // It levels off well above 0: the last 20 checks all lie within 0.03 of the end value.
  for (const v of ev.slice(-20)) assert.ok(Math.abs(v - ev.at(-1)) < 0.03 && v > 0.5, String(v));
  const fx = (v) => v.toFixed(2);
  assert.equal(sv['loss-start'], fx(ev[0]));
  assert.equal(sv['loss-100'], fx(ev[1]));
  assert.equal(sv['loss-end'], fx(ev.at(-1)));
  // Press 1 against noise, mixture and fresh spiral points.
  const out = run.gen.sampler.path[50];
  const noise = run.gen.sampler.path[0];
  const ref = ddpmSample(s.net, 400, S.schedule(), 313, { keepPath: true });
  assert.deepEqual(out, ref.X);
  const X = S.spiral();
  const q = (A) => ({ p: fx(nearestDist(A, X)), c: fx(nearestDist(X, A)) });
  assert.deepEqual([sv['diff-prec'], sv['diff-cov']], [q(out).p, q(out).c]);
  assert.deepEqual([sv['noise-prec'], sv['noise-cov']], [q(noise).p, q(noise).c]);
  assert.deepEqual([sv['mix-prec'], sv['mix-cov']], [q(S.mixSamples(1)).p, q(S.mixSamples(1)).c]);
  assert.deepEqual([sv['fresh-prec'], sv['fresh-cov']], [q(S.freshSpiral()).p, q(S.freshSpiral()).c]);
  const num = (k) => Number(sv[k]);
  // As rounded on the page: diffusion beats the mixture, which beats noise, both ways;
  // the samples are not copies: farther from the data than fresh spiral points are.
  assert.ok(num('diff-prec') < num('mix-prec') && num('mix-prec') < num('noise-prec'));
  assert.ok(num('diff-cov') < num('mix-cov') && num('mix-cov') < num('noise-cov'));
  assert.ok(num('diff-prec') > num('fresh-prec'));
  assert.ok(num('diff-cov') - num('fresh-cov') <= 0.01);
  assert.ok(num('noise-prec') >= 2 * num('noise-cov'));
  // Every one of the first 20 presses: diffusion closer than the mixture and the noise, both ways.
  for (const row of pressTable(s.net)) {
    assert.ok(better(row.diff, row.mix) && better(row.diff, row.noise));
  }
});

test('quoted numbers survive a 1e-12 nudge', () => {
  const sv = S.storyValues();
  for (const eps of [1e-12, -1e-12]) {
    const run = S.storyRun(nudged(eps));
    assert.equal(run.state.iter, 6000);
    assert.deepEqual(S.storyFrom(run), sv);
    for (const row of pressTable(run.state.net)) assert.ok(better(row.diff, row.mix) && better(row.diff, row.noise));
  }
});

test('read-out rows', () => {
  const live = S.makeLive();
  const infos = [[0, { t: 0 }], [0, { t: 50 }], [1, { t: 25 }], [2, { ex: S.example(S.startNet(), 25), iter: 0 }],
    [3, S.liveInfo(live, false)], [3, S.liveInfo(live, 'computing')], [4, { gen: null, netIter: 0 }], [4, { gen: null, netIter: 6000 }], [5, {}]];
  const run = pageRun();
  infos.push([2, { ex: S.example(run.state.net, 50), iter: 6000 }], [3, S.liveInfo(run.live, false)]);
  const g = S.makeGen(run.state.net, 12, 6000);
  g.frame();
  infos.push([4, S.genInfo(g, null)]);
  infos.push([4, S.genInfo(run.gen, S.quality(S.mixSamples(1)))]);
  for (const [step, s] of infos) {
    const rows = S.readout(step, s);
    assert.ok(rows.length >= 1 && rows.length <= 3, String(step));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 48, r);
    }
  }
  const sv = S.storyValues();
  const end = S.readout(4, S.genInfo(run.gen, null));
  assert.ok(end.some((r) => r.includes(`samples → data ${sv['diff-prec']} · data → samples ${sv['diff-cov']}`)), end.join('|'));
  assert.ok(end.some((r) => r.includes('seed 313')));
  assert.ok(S.readout(3, S.liveInfo(run.live, false)).some((r) => r.includes('6,000 of 6,000 · finished')));
  assert.ok(S.readout(3, S.liveInfo(S.makeLive(), false)).some((r) => r.includes('press Train')));
  // The example's numbers are the network's: ε̂ from denoiserForward at xₜ.
  const ex = S.example(run.state.net, 25);
  const sc = S.schedule();
  const xt = qSample(S.spiral()[S.EXAMPLE], 25, S.forwardNoise()[S.EXAMPLE], sc);
  assert.deepEqual(ex.guess, denoiserForward(run.state.net, xt, timeFeatures(25, 50)));
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Diffusion: from noise to data · ML Explorer<\/title>/);
  assert.match(html, /<h1>Diffusion: from noise to data<\/h1>/);
  assert.match(html, /Level 5 · Modern AI/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/diffusion\/">/);
  for (const s of [300, 304, 305, 307, 310, 311, 312, 313]) assert.match(html, new RegExp(`seed ${s}\\b`));
  for (const href of ['/ml/generative-models/', '/ml/measuring-error/', '/ml/optimisers/']) assert.match(html, new RegExp(`href="${href}"`));
  assert.match(html, /<a class="next-lesson" href="\/ml\/">Back to the path →<\/a>/);
  assert.match(text, /Level 6 is coming next/);
  assert.match(html, /far larger scale/);
  assert.match(html, /2006\.11239/);
  assert.match(html, /1503\.03585/);
  assert.match(html, /Chapter 18, "Diffusion models"/);
  assert.match(html, /Algorithm 2/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(text, /understand|thinks|\bknows\b|reasons|creative|intelligen|conscious|hallucinat|imagin|dreams/i);
  assert.doesNotMatch(text, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(text, /GPT|DALL|Stable Diffusion|Midjourney|Imagen|Sora/);
  // The maths the spec asks for.
  for (const tex of [
    'q(\\mathbf{x}_t \\mid \\mathbf{x}_{t-1}) = \\mathcal{N}\\big(\\sqrt{\\alpha_t}\\,\\mathbf{x}_{t-1},\\ \\beta_t \\mathbf{I}\\big)',
    'q(\\mathbf{x}_t \\mid \\mathbf{x}_0) = \\mathcal{N}\\big(\\sqrt{\\bar\\alpha_t}\\,\\mathbf{x}_0,\\ (1 - \\bar\\alpha_t)\\mathbf{I}\\big)',
    '\\mathbb{E}\\,\\lVert \\boldsymbol{\\varepsilon} - \\hat{\\boldsymbol{\\varepsilon}}_\\theta(\\mathbf{x}_t, t) \\rVert^2',
    '\\sigma_t^2 = \\beta_t',
  ]) assert.ok(html.includes(tex.replace(/&/g, '&amp;')), tex);
  assert.match(html, /σ<sub>t<\/sub> here is the standard deviation|here is the standard deviation of the fresh noise/);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What is the network trained to predict\?/);
  assert.match(html, /Why start generation from pure noise\?/);
  const pv = S.pageValues();
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  for (const [, key, val] of spans) {
    assert.ok(key in pv, key);
    assert.equal(val, pv[key], key);
  }
  const usedPv = new Set(spans.map((m) => m[1]));
  for (const key of Object.keys(pv)) assert.ok(usedPv.has(key), `unused page value ${key}`);
  const sv = S.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, val] of story) {
    assert.ok(key in sv, key);
    assert.equal(val, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of [...Object.values(pv), ...Object.values(sv)]) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
  // 64 × 64 × 3, quoted in step 6.
  assert.match(text, /12,288/);
  assert.equal(64 * 64 * 3, 12288);
});
