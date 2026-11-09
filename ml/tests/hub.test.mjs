// Tests for the two hub pages: the landing page /ml/ and the notation page.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { PATH, TOTAL_LIVE, lessonHref, firstNotDone, startLink, nextLesson } from '../assets/path.js';
import { mean, variance, sd, olsFit, mse, mae } from '../assets/maths.js';
import { sigmoid, logLoss, gini, accuracy, confusion } from '../assets/classify.js';
import { inertia, pca2 } from '../assets/unsupervised.js';

const ML = fileURLToPath(new URL('..', import.meta.url));
const read = (p) => readFileSync(ML + p, 'utf8');
const landing = () => read('index.html');
const notation = () => read('notation/index.html');

const decode = (s) => s.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
const stripTags = (s) => decode(s.replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim();
const LIVE = PATH.flatMap((l) => l.lessons);

// The path block on the landing page, up to its closing marker.
function pathBlock(html) {
  const m = /<ol class="hub-path"[\s\S]*?<!-- \/hub-path -->/.exec(html);
  assert.ok(m, 'landing page has ol.hub-path closed by <!-- /hub-path -->');
  return m[0];
}

// Splits the path block into one chunk per level, keyed by data-level.
function levelChunks(block) {
  const parts = block.split(/(?=<li class="hub-level[^"]*" data-level=")/).slice(1);
  return parts.map((chunk) => ({ level: /data-level="([^"]+)"/.exec(chunk)[1], chunk }));
}

test('firstNotDone walks the path in order', () => {
  const none = new Set();
  assert.equal(firstNotDone((s) => none.has(s)).slug, LIVE[0].slug);
  const two = new Set([LIVE[0].slug, LIVE[1].slug]);
  assert.equal(firstNotDone((s) => two.has(s)).slug, LIVE[2].slug);
  const gap = new Set([LIVE[0].slug, LIVE[2].slug]);
  assert.equal(firstNotDone((s) => gap.has(s)).slug, LIVE[1].slug);
  const all = new Set(LIVE.map((l) => l.slug));
  assert.equal(firstNotDone((s) => all.has(s)), null);
});

test('landing path lists every live lesson, in order, matching path.js', () => {
  const block = pathBlock(landing());
  const links = [...block.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g)].map((m) => ({
    href: /href="([^"]*)"/.exec(m[1])?.[1],
    slug: /data-slug="([^"]*)"/.exec(m[1])?.[1],
    title: stripTags(/<span class="hub-title">([\s\S]*?)<\/span>/.exec(m[2])?.[1] ?? ''),
  }));
  assert.equal(links.length, TOTAL_LIVE);
  assert.deepEqual(
    links,
    LIVE.map((l) => ({ href: lessonHref(l.slug), slug: l.slug, title: l.title })),
  );
});

test('landing path shows every level; coming levels have topics and no links', () => {
  const chunks = levelChunks(pathBlock(landing()));
  assert.deepEqual(chunks.map((c) => c.level), PATH.map((l) => l.level));
  chunks.forEach(({ chunk }, i) => {
    const level = PATH[i];
    assert.ok(stripTags(chunk).includes(level.name), `${level.level} shows its name`);
    if (level.soon) {
      assert.ok(!/<a\b/.test(chunk), `${level.level} has no links`);
      assert.ok(stripTags(chunk).includes('Coming next'), `${level.level} says Coming next`);
      assert.ok(stripTags(chunk).includes(level.topics), `${level.level} lists its topics`);
    }
  });
});

test('startLink picks the one start button from progress', () => {
  const first = LIVE[0];
  assert.deepEqual(startLink(() => false), { label: `Start at L0: ${first.title}`, href: lessonHref(first.slug) });
  const gap = new Set([LIVE[0].slug, LIVE[2].slug]);
  assert.deepEqual(startLink((s) => gap.has(s)), { label: `Continue: ${LIVE[1].title}`, href: lessonHref(LIVE[1].slug) });
  assert.deepEqual(startLink(() => true), { label: `Start again at L0: ${first.title}`, href: lessonHref(first.slug) });
  // Slugs not on the path never count.
  assert.deepEqual(startLink((s) => s === 'retired-lesson'), startLink(() => false));
});

test('landing static start button matches startLink with no progress', () => {
  const m = /<a class="hub-start[^"]*" href="([^"]+)">([\s\S]*?)<\/a>/.exec(landing());
  assert.deepEqual({ label: stripTags(m[2]), href: m[1] }, startLink(() => false));
  assert.equal((landing().match(/class="hub-start/g) || []).length, 1, 'one start button');
});

test('landing start link goes to the first lesson', () => {
  const m = /<a class="hub-start[^"]*" href="([^"]+)">([\s\S]*?)<\/a>/.exec(landing());
  assert.ok(m, 'a.hub-start exists');
  assert.equal(m[1], lessonHref(LIVE[0].slug));
  assert.ok(stripTags(m[2]).includes('L0'));
});

test('landing JSON-LD describes the course and every live lesson', () => {
  const scripts = [...landing().matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
  const inline = scripts.filter((s) => !/\bsrc=/.test(s[1]));
  assert.equal(inline.length, 1, 'one inline script only');
  assert.match(inline[0][1], /type="application\/ld\+json"/);
  const ld = JSON.parse(inline[0][2]);
  assert.equal(ld['@context'], 'https://schema.org');
  assert.equal(ld['@type'], 'Course');
  assert.equal(ld.name, 'ML Explorer');
  assert.equal(ld.inLanguage, 'en-NZ');
  assert.equal(ld.isAccessibleForFree, true);
  for (const k of ['provider', 'author']) {
    assert.equal(ld[k]['@type'], 'Person');
    assert.equal(ld[k].name, 'Reju Sam John');
  }
  assert.deepEqual(
    ld.hasPart.map((p) => [p['@type'], p.name, p.url]),
    LIVE.map((l) => ['LearningResource', l.title, `https://rejusamjohn.pages.dev/ml/${l.slug}/`]),
  );
});

for (const [name, get, canon] of [
  ['landing', landing, 'https://rejusamjohn.pages.dev/ml/'],
  ['notation', notation, 'https://rejusamjohn.pages.dev/ml/notation/'],
]) {
  test(`${name} page has the shell: meta, scripts, beacon, no coffee line`, () => {
    const html = get();
    assert.match(html, /<html lang="en-NZ">/);
    assert.match(html, /<title>[^<]+<\/title>/);
    assert.match(html, /<meta name="description" content="[^"]{50,}">/);
    assert.ok(html.includes(`<link rel="canonical" href="${canon}">`));
    assert.ok(html.includes(`<meta property="og:url" content="${canon}">`));
    assert.ok(html.includes('<meta property="og:image" content="https://rejusamjohn.pages.dev/ml/og.png">'));
    assert.ok(html.includes(`<script type="module" src="https://static.cloudflareinsights.com/beacon.min.js" data-cf-beacon='{"token": "e46c62f910fe4def882feaed064efced"}'></script>`));
    const flag = html.indexOf('<script src="/ml/assets/js-flag.js"></script>');
    const ml = html.indexOf('<script type="module" src="/ml/assets/ml.js"></script>');
    assert.ok(flag > 0 && ml > flag, 'js-flag.js loads before ml.js');
    assert.ok(html.includes('<link rel="stylesheet" href="/ml/assets/ml.css">'));
    assert.ok(!/class="coffee-line"/.test(html), 'no coffee line on hub pages');
    assert.ok(!/\son[a-z]+=/i.test(html), 'no inline event handlers');
    assert.ok(!/<main class="lesson"/.test(html), 'hub pages have no lesson main');
  });
}

test('notation page links only to lessons that exist', () => {
  const html = notation();
  const hrefs = [...html.matchAll(/<a\b[^>]*href="(\/ml\/[^"#]*)(#[^"]*)?"/g)].map((m) => m[1]);
  const allowed = new Set(['/ml/', '/ml/notation/', ...LIVE.map((l) => lessonHref(l.slug))]);
  assert.ok(hrefs.length > 20);
  for (const h of hrefs) assert.ok(allowed.has(h), `unknown link ${h}`);
  for (const l of LIVE) assert.ok(existsSync(`${ML}${l.slug}/index.html`));
});

test('notation page covers the symbols the lessons use', () => {
  const tex = new Set([...notation().matchAll(/data-tex="([^"]*)"/g)].map((m) => decode(m[1])));
  const has = (t) => [...tex].some((s) => s.includes(t));
  for (const t of [
    'x', 'y', '\\hat{y}', 'x_i', '\\mathbf{x}_i', 'x_{ij}', 'n', 'e_i = y_i - \\hat{y}_i', '\\varepsilon',
    '\\mathbf{X}', '\\bar{x}', '\\bar{y}', '\\mathbf{y}', '\\mathbb{R}^2', '\\tilde{y}', '\\bar{f}(x_0)', 'df/dx', '\\lim', '\\boldsymbol{\\delta}', '\\cdot', 'I', 'w_j', 'x_1, \\dots, x_d', '\\mu', '\\sigma', 's^2', 'z', 'x_{(k)}', '\\mathbb{E}', '\\operatorname{Var}',
    '\\hat{a}', '\\hat{b}', '\\boldsymbol{\\theta}', '\\theta_j', 'w', '\\tilde{x}', 'a_z', 'w_z',
    '\\eta', 'w^*', '\\gamma_{\\max}', '\\kappa', '\\lambda', 'J(\\boldsymbol{\\theta})', 'H',
    '\\mathrm{MSE}', '\\mathrm{MAE}', '\\mathrm{RMSE}', '\\mathrm{SSE}', '\\mathrm{SST}', 'R^2', 'r',
    'S_{xy}', '\\mathrm{CV}_k', '|T|', "f'(x)", '\\partial', '\\nabla', '\\sum', '\\lVert', '\\arg\\min', '^{\\top}',
  ]) assert.ok(has(t), `notation page shows ${t}`);
  assert.match(notation(), /<h2[^>]*>Sample versus population variance<\/h2>/);
});

// The worked examples all use three points: (1, 1), (2, 3), (3, 2).
test('notation worked examples match the computed values', () => {
  const xs = [1, 2, 3], ys = [1, 3, 2];
  const f2 = (v) => {
    const s = v.toFixed(2).replace(/\.?0+$/, '');
    return s;
  };
  const fit = olsFit(xs, ys);
  assert.ok(fit.ok);
  const yhat = xs.map((x) => fit.intercept + fit.slope * x);
  const e = ys.map((y, i) => y - yhat[i]);
  const sse = e.reduce((s, v) => s + v * v, 0);
  const sst = ys.reduce((s, y) => s + (y - mean(ys)) ** 2, 0);
  const sigma = sd(xs, { sample: false });
  const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);
  close(fit.intercept, 1);
  close(fit.slope, 0.5);
  close(1 - sse / sst, fit.r2);

  const expected = [
    `\\bar{x} = (1 + 2 + 3)/3 = ${f2(mean(xs))}`,
    `\\sigma^2 = 2/3 \\approx ${f2(variance(xs, { sample: false }))}`,
    `\\sigma \\approx ${f2(sigma)}`,
    `s^2 = 2/2 = ${f2(variance(xs, { sample: true }))}`,
    `s = ${f2(sd(xs, { sample: true }))}`,
    `z = (3 - 2)/0.82 \\approx ${f2((3 - mean(xs)) / sigma)}`,
    `\\hat{a} = ${f2(fit.intercept)},\\ \\hat{b} = ${f2(fit.slope)}`,
    `\\hat{y}_2 = 1 + 0.5 \\times 2 = ${f2(yhat[1])}`,
    `e_2 = 3 - 2 = ${f2(e[1])}`,
    `\\mathrm{MSE} = (0.25 + 1 + 0.25)/3 = ${f2(mse(ys, yhat))}`,
    `\\mathrm{MAE} = (0.5 + 1 + 0.5)/3 \\approx ${f2(mae(ys, yhat))}`,
    `\\mathrm{RMSE} = \\sqrt{0.5} \\approx ${f2(Math.sqrt(mse(ys, yhat)))}`,
    `\\mathrm{SSE} = 0.25 + 1 + 0.25 = ${f2(sse)}`,
    `\\mathrm{SST} = 1 + 1 + 0 = ${f2(sst)}`,
    `R^2 = 1 - 1.5/2 = ${f2(1 - sse / sst)}`,
    `a_z = ${f2(mean(ys))},\\ w_z = 0.5 \\times 0.82 \\approx ${f2(fit.slope * sigma)}`,
  ];
  const tex = [...notation().matchAll(/data-tex="([^"]*)"/g)].map((m) => decode(m[1]));
  for (const t of expected) assert.ok(tex.includes(t), `notation page shows ${t}`);
  // The variance box quotes both versions in words.
  const box = stripTags(/<section class="callout"[\s\S]*?<\/section>/.exec(notation())[0]);
  assert.ok(box.includes(`the population variance is about ${f2(variance(xs, { sample: false }))} and the sample variance is ${f2(variance(xs))}`));
});

const ROOT = fileURLToPath(new URL('../../', import.meta.url));

test('TOTAL_LIVE equals the count of lessons in PATH', () => {
  assert.equal(TOTAL_LIVE, PATH.flatMap((l) => l.lessons).length);
  assert.equal(TOTAL_LIVE, 19);
});

test('L3 is live: not marked soon, nine lessons, each with a page whose h1 is its title', () => {
  const l3 = PATH.find((l) => l.level === 'L3');
  assert.ok(!l3.soon);
  assert.equal(l3.topics, undefined);
  assert.deepEqual(l3.lessons.map((l) => l.slug), [
    'k-nearest-neighbours', 'logistic-regression', 'naive-bayes', 'support-vector-machines',
    'decision-trees', 'random-forests', 'gradient-boosting', 'k-means', 'pca',
  ]);
  for (const l of l3.lessons) {
    const html = read(`${l.slug}/index.html`);
    assert.equal(stripTags(/<h1>([\s\S]*?)<\/h1>/.exec(html)[1]), l.title, l.slug);
  }
});

test('landing intro says which levels are ready', () => {
  const sub = stripTags(/<p class="hub-sub">([\s\S]*?)<\/p>/.exec(landing())[1]);
  assert.ok(sub.includes('L0 to L3 are ready; L4 to L6 are coming next.'), sub);
});

test('landing numbers the lessons in order and quotes each page\'s promise and time', () => {
  const block = pathBlock(landing());
  const items = [...block.matchAll(/<li class="hub-lesson">([\s\S]*?)<\/li>/g)].map((m) => m[1]);
  assert.equal(items.length, TOTAL_LIVE);
  items.forEach((item, i) => {
    const slug = /data-slug="([^"]+)"/.exec(item)[1];
    const page = read(`${slug}/index.html`);
    const mins = /<p class="meta">About (\d+) minutes/.exec(page)[1];
    assert.equal(stripTags(/<p class="hub-n">([\s\S]*?)<\/p>/.exec(item)[1]), `Lesson ${i + 1} · About ${mins} min`);
    assert.equal(
      stripTags(/<p class="hub-promise">([\s\S]*?)<\/p>/.exec(item)[1]),
      stripTags(/<p class="promise">([\s\S]*?)<\/p>/.exec(page)[1]),
      slug,
    );
  });
});

test("overfitting's next link points to /ml/k-nearest-neighbours/", () => {
  const html = read('overfitting/index.html');
  assert.ok(html.includes('<a class="next-lesson" href="/ml/k-nearest-neighbours/">Next lesson: k-nearest neighbours →</a>'));
  assert.ok(!html.includes('Back to the path'));
});

test('every lesson\'s next link follows the path and lands on a page that exists', () => {
  for (const l of LIVE) {
    const html = read(`${l.slug}/index.html`);
    const links = [...html.matchAll(/<a class="next-lesson" href="([^"]+)">/g)].map((m) => m[1]);
    assert.equal(links.length, 1, l.slug);
    const next = nextLesson(l.slug);
    assert.equal(links[0], next ? lessonHref(next.slug) : '/ml/', l.slug);
    assert.ok(existsSync(`${ROOT}${links[0].slice(1)}index.html`), links[0]);
  }
});

test('sitemap lists every lesson URL in PATH', () => {
  const xml = readFileSync(`${ROOT}sitemap.xml`, 'utf8');
  for (const l of LIVE) {
    assert.ok(
      new RegExp(`<url><loc>https://rejusamjohn\\.pages\\.dev/ml/${l.slug}/</loc><lastmod>\\d{4}-\\d{2}-\\d{2}</lastmod></url>`).test(xml),
      `sitemap has ${l.slug}`,
    );
  }
});

test('home page card names levels 0 to 3', () => {
  const html = readFileSync(`${ROOT}index.html`, 'utf8');
  assert.ok(html.includes('<div class="project-label">Interactive guide · Levels 0–3 live</div>'));
  assert.ok(html.includes('classic models such as decision trees, random forests, k-means and PCA'));
});

// Every HTML page under ml/, vendor files aside.
function pages(dir = ML) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name === 'vendor' || name === 'node_modules') continue;
    const p = `${dir}${name}`;
    if (statSync(p).isDirectory()) out.push(...pages(`${p}/`));
    else if (name.endsWith('.html')) out.push(p);
  }
  return out;
}

test('ISLR is cited with its four authors', () => {
  const all = pages();
  assert.ok(all.length > TOTAL_LIVE);
  let cites = 0;
  for (const p of all) {
    const html = readFileSync(p, 'utf8');
    assert.ok(!/Taylor/.test(html), `${p} names Taylor`);
    assert.ok(!/Hastie, Hastie/.test(html), p);
    for (const m of html.matchAll(/([^>]*)<cite>An Introduction to Statistical Learning<\/cite>/g)) {
      cites++;
      assert.match(m[1], /James, Witten, Hastie &amp; Tibshirani, $/, p);
    }
  }
  assert.ok(cites > 0);
});

test('notation page lists the Level 3 symbols', () => {
  const html = notation();
  const tex = new Set([...html.matchAll(/data-tex="([^"]*)"/g)].map((m) => decode(m[1])));
  const has = (t) => [...tex].some((s) => s.includes(t));
  for (const t of [
    'y \\in \\{0, 1\\}', 'y_i = \\pm 1', '\\hat{p}', '\\sigma(z) = \\frac{1}{1 + e^{-z}}', '\\mathbf{w}\\cdot\\mathbf{x} + b',
    'N_k(\\mathbf{x})', '\\hat{k}', 'K', 'C', '\\alpha_i', '\\xi_i', 'K(\\mathbf{x}_i, \\mathbf{x}_j)',
    'G = 1 - \\sum_k p_k^2', 'G_{\\text{split}}', 'B', '\\hat{f}_b', 'S_b', '\\rho', 'M', 'F_m', 'h_m', 'r_{mi}',
    'c(i)', '\\boldsymbol{\\mu}_k', 'S', '\\gamma_1 \\ge \\gamma_2', '\\mathbf{v}', 'e_k', '\\operatorname{tr}',
    '\\mathbf{1}[\\cdot]', '\\propto', '\\prod', '\\ln', 'P(c \\mid \\mathbf{x})', 'p(x_j \\mid c)', '\\in', '\\notin', '\\arg\\max',
  ]) assert.ok(has(t), `notation page shows ${t}`);
  for (const id of ['label', 'phat', 'sigmoid', 'wb', 'K', 'C', 'gini', 'B', 'Fm', 'rmi', 'S', 'accuracy', 'inertia', 'indicator', 'clashes']) {
    assert.ok(html.includes(`id="${id}"`), `notation page has #${id}`);
  }
  for (const l of PATH.find((x) => x.level === 'L3').lessons) {
    assert.ok(html.includes(`href="${lessonHref(l.slug)}"`), `notation page links ${l.slug}`);
  }
});

test('notation page lists the Level 3 clashes', () => {
  const sec = /<section class="sym-group" id="clashes"[\s\S]*?<\/section>/.exec(notation());
  assert.ok(sec, 'a clash section exists');
  const items = [...sec[0].matchAll(/<li>([\s\S]*?)<\/li>/g)].map((m) => m[1]);
  const tex = (item) => [...item.matchAll(/data-tex="([^"]*)"/g)].map((m) => decode(m[1]));
  const find = (...ts) => items.some((it) => ts.every((t) => tex(it).some((s) => s.includes(t))));
  assert.ok(find('\\sigma(z)', '\\sigma'), 'sigma: sigmoid vs SD');
  assert.ok(find('\\mathbf{w}\\cdot\\mathbf{x} + b', 'a + b\\,x'), 'b: intercept vs slope');
  assert.ok(find('K', 'k'), 'k vs K');
  assert.ok(find('S', '\\Sigma'), 'S vs Sigma');
  assert.ok(find('r_{mi}', 'r'), 'r vs r_mi');
  assert.ok(find('M'), 'M');
  assert.ok(find('d'), 'd');
  assert.ok(find('y_i = \\pm 1'), 'SVM labels');
});

// The Level 3 worked examples use the same three points (1, 1), (2, 3), (3, 2).
test('notation Level 3 worked examples match the computed values', () => {
  const f2 = (v) => v.toFixed(2).replace(/\.?0+$/, '');
  const X = [[1, 1], [2, 3], [3, 2]];
  const ys = [1, 3, 2];
  const { S, gammas, vecs } = pca2(X);
  const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);
  close(S[0][0], 1); close(S[1][1], 1); close(S[0][1], 0.5);
  close(gammas[0], 1.5); close(gammas[1], 0.5);
  close(vecs[0][0], Math.SQRT1_2); close(vecs[0][1], Math.SQRT1_2);
  const cm = confusion([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
  const expected = [
    `\\sigma(0) = ${f2(sigmoid(0))}`,
    `\\sigma(2) \\approx ${f2(sigmoid(2))}`,
    `\\sigma(-2) \\approx ${f2(sigmoid(-2))}`,
    `\\hat{p} = \\sigma(-3 + 1 \\times 5) = \\sigma(2) \\approx ${f2(sigmoid(-3 + 5))}`,
    `-\\ln ${f2(sigmoid(2))} \\approx ${f2(logLoss([1], [sigmoid(2)]))}`,
    `G = 1 - (3/4)^2 - (1/4)^2 = ${gini([3, 1])}`,
    `\\text{accuracy} = 45/50 = ${accuracy([...Array(45).fill(1), ...Array(5).fill(0)], Array(50).fill(1))}`,
    `\\text{precision} = 8/(8 + 2) = ${f2(cm.tp / (cm.tp + cm.fp))}`,
    `\\text{recall} = 8/(8 + 4) \\approx ${f2(cm.tp / (cm.tp + cm.fn))}`,
    `\\text{inertia} = 2 + 1 + 1 = ${inertia(X, [[2, 2]], [0, 0, 0])}`,
    `(1 - 1/3)^3 = 8/27 \\approx ${f2((2 / 3) ** 3)}`,
    `r_{1i} = y_i - F_0(x_i) = ${ys.map((y) => y - mean(ys)).join(',\\ ')}`,
    `S = \\begin{pmatrix} ${f2(S[0][0])} & ${f2(S[0][1])} \\\\ ${f2(S[1][0])} & ${f2(S[1][1])} \\end{pmatrix}`,
    `\\gamma_1 = ${f2(gammas[0])},\\ \\gamma_2 = ${f2(gammas[1])}`,
    `e_1 = ${f2(gammas[0])}/(${f2(gammas[0])} + ${f2(gammas[1])}) = ${f2(gammas[0] / (gammas[0] + gammas[1]))}`,
  ];
  const tex = [...notation().matchAll(/data-tex="([^"]*)"/g)].map((m) => decode(m[1]));
  for (const t of expected) assert.ok(tex.includes(t), `notation page shows ${t}`);
});
