// Tests for the two hub pages: the landing page /ml/ and the notation page.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { PATH, TOTAL_LIVE, lessonHref, firstNotDone } from '../assets/path.js';
import { mean, variance, sd, olsFit, mse, mae } from '../assets/maths.js';

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

test('landing start link goes to the first lesson', () => {
  const m = /<a class="hub-start[^"]*" href="([^"]+)">([\s\S]*?)<\/a>/.exec(landing());
  assert.ok(m, 'a.hub-start exists');
  assert.equal(m[1], lessonHref(LIVE[0].slug));
  assert.ok(stripTags(m[2]).includes('L0'));
});

test('landing JSON-LD describes the course and its ten lessons', () => {
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
    '\\mathbf{X}', '\\bar{x}', '\\mu', '\\sigma', 's^2', 'z', 'x_{(k)}', '\\mathbb{E}', '\\operatorname{Var}',
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
  const box = stripTags(/<aside class="callout"[\s\S]*?<\/aside>/.exec(notation())[0]);
  assert.ok(box.includes(`the population variance is about ${f2(variance(xs, { sample: false }))} and the sample variance is ${f2(variance(xs))}`));
});
