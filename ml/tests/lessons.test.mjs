// Tests for the data and the numbers each lesson's stage shows.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import * as lesson1 from '../what-is-a-model/stage.js';

test('lesson1 data is reproducible', () => {
  const a = lesson1.lessonData();
  const b = lesson1.lessonData();
  assert.equal(a.xs.length, 20);
  assert.equal(a.ys.length, 20);
  assert.deepEqual(a.xs, b.xs);
  assert.deepEqual(a.ys, b.ys);
  for (const x of a.xs) assert.ok(x >= 1 && x <= 10, `x out of range: ${x}`);
  for (const y of a.ys) assert.ok(y >= 0 && y <= 100, `y out of range: ${y}`);
});

test('lesson1 fitted slope is near the generating slope', () => {
  const { xs, ys } = lesson1.lessonData();
  const fit = lesson1.fitRule(xs, ys);
  assert.ok(fit.ok);
  assert.ok(Math.abs(fit.b - 5) < 1.5, `slope ${fit.b}`);
});

test('lesson1 stage numbers agree with the fitted rule', () => {
  const { xs, ys } = lesson1.lessonData();
  const fit = lesson1.fitRule(xs, ys);
  const s = lesson1.missSummary(xs, ys, fit);
  // With an intercept, least-squares residuals sum to zero.
  assert.ok(Math.abs(s.residuals.reduce((t, e) => t + e, 0)) < 1e-9);
  // Moving a or b either way from the fit raises the average squared miss.
  for (const d of [-0.5, 0.5]) {
    assert.ok(lesson1.missSummary(xs, ys, { a: fit.a + d, b: fit.b }).mse > s.mse);
    assert.ok(lesson1.missSummary(xs, ys, { a: fit.a, b: fit.b + d }).mse > s.mse);
  }
  // The biggest miss is the largest absolute residual, at the reported index.
  assert.equal(s.maxAbs, Math.max(...s.residuals.map(Math.abs)));
  assert.equal(Math.abs(s.residuals[s.worst]), s.maxAbs);
  assert.equal(lesson1.predict(fit, 0), fit.a);
  assert.equal(lesson1.fmt(-2, 1), '\u22122.0');
  assert.equal(lesson1.fmt(34.264, 1), '34.3');
});

// ---------------------------------------------------------------- lesson 2

import { readFileSync } from 'node:fs';
import { mean, sd } from '../assets/maths.js';
import * as lesson2 from '../data-as-numbers/stage.js';

const lesson2Html = readFileSync(new URL('../data-as-numbers/index.html', import.meta.url), 'utf8');

test('lesson2 fruit table is physically plausible', () => {
  assert.equal(lesson2.FRUIT.length, 6);
  for (const f of lesson2.FRUIT) {
    assert.ok(f.mass >= 20 && f.mass <= 300, `${f.kind} mass ${f.mass}`);
    assert.ok(f.diameter >= 3 && f.diameter <= 10, `${f.kind} diameter ${f.diameter}`);
    // Treated as a sphere, the fruit's density should be near that of water
    // (apples are a little lighter and float; stone fruit are close to 1).
    const volume = (Math.PI * f.diameter ** 3) / 6;
    const density = f.mass / volume;
    assert.ok(density > 0.7 && density < 1.1, `${f.kind} density ${density.toFixed(2)} g/cm³`);
  }
});

test('lesson2 standardised features have mean 0 and population SD 1', () => {
  const { z, mu, sigma } = lesson2.standardiseRows(lesson2.vectors());
  for (const k of [0, 1]) {
    const col = z.map((v) => v[k]);
    assert.ok(Math.abs(mean(col)) < 1e-12);
    assert.ok(Math.abs(sd(col, { sample: false }) - 1) < 1e-12);
    const raw = lesson2.vectors().map((v) => v[k]);
    assert.ok(Math.abs(mu[k] - mean(raw)) < 1e-12);
    assert.ok(Math.abs(sigma[k] - sd(raw, { sample: false })) < 1e-12);
  }
});

test('lesson2 distance is the square root of the summed squared gaps', () => {
  const pts = lesson2.vectors();
  const pairs = lesson2.pairDistances(pts);
  assert.equal(pairs.length, 15);
  for (const p of pairs) {
    const dm = pts[p.i][0] - pts[p.j][0];
    const dd = pts[p.i][1] - pts[p.j][1];
    assert.ok(Math.abs(p.d - Math.sqrt(dm * dm + dd * dd)) < 1e-12);
  }
  assert.equal(lesson2.distance([0, 0], [3, 4]), 5);
});

test('lesson2 nearest pair changes after standardising', () => {
  const raw = lesson2.vectors();
  const { z } = lesson2.standardiseRows(raw);
  const before = lesson2.nearestPair(raw);
  const after = lesson2.nearestPair(z);
  assert.notEqual(lesson2.pairKey(before), lesson2.pairKey(after));
  // Both winners are clear, not near-ties.
  const second = (pts, best) => lesson2.pairDistances(pts).filter((p) => lesson2.pairKey(p) !== lesson2.pairKey(best))
    .reduce((m, p) => Math.min(m, p.d), Infinity);
  assert.ok(second(raw, before) / before.d > 1.5);
  assert.ok(second(z, after) / after.d > 1.5);
});

test('lesson2 page agrees with the data', () => {
  const raw = lesson2.vectors();
  const { z } = lesson2.standardiseRows(raw);
  const before = lesson2.nearestPair(raw);
  const after = lesson2.nearestPair(z);
  // The quiz answers are the computed nearest pairs.
  const answer = (id) => lesson2Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), lesson2.pairKey(before));
  assert.equal(answer('q2'), lesson2.pairKey(after));
  // The data table on the page lists exactly the stage's fruit.
  const body = lesson2Html.match(/<table class="data-table">[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/)[1];
  const rows = [...body.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map((m) => [...m[1].matchAll(/<t[dh][^>]*>([^<]*)<\/t[dh]>/g)].map((c) => c[1]));
  assert.deepEqual(rows, lesson2.FRUIT.map((f, i) => [String(i + 1), f.kind, String(f.mass), f.diameter.toFixed(1)]));
  // Claims in the quiz explanations.
  const gap = (i, j, k) => Math.abs(raw[i][k] - raw[j][k]);
  const [a, b] = [before.i, before.j];
  const [c, d] = [after.i, after.j];
  assert.ok(gap(c, d, 0) > gap(a, b, 0), 'the standardised winners differ by more grams');
  assert.ok(gap(c, d, 1) < gap(a, b, 1), 'and by fewer centimetres');
  assert.equal(lesson2.FRUIT[c].kind, lesson2.FRUIT[d].kind, 'the standardised winners are the same kind');
});

test('lesson2 claims in the text hold for the data', () => {
  const raw = lesson2.vectors();
  const { z } = lesson2.standardiseRows(raw);
  const pairs = lesson2.pairDistances(raw);
  // Densities, treated as spheres, lie between about 0.8 and 1 g/cm³ (About the data).
  for (const f of lesson2.FRUIT) {
    const density = f.mass / ((Math.PI * f.diameter ** 3) / 6);
    assert.ok(density >= 0.75 && density <= 1.0, `${f.kind} ${density}`);
  }
  // In grams, mass gives almost all of every squared distance;
  // in kilograms, diameter does.
  const share = (pts, k) => pairs.map(({ i, j }) => {
    const g = pts[i].map((v, c) => (v - pts[j][c]) ** 2);
    return g[k] / (g[0] + g[1]);
  });
  const kg = raw.map(([m, d]) => [m / 1000, d]);
  for (const s of share(raw, 0)) assert.ok(s > 0.95, `mass share ${s}`);
  for (const s of share(kg, 1)) assert.ok(s > 0.95, `diameter share in kg ${s}`);
  // Standardising gives the same result whatever the units.
  const zkg = lesson2.standardiseRows(kg).z;
  z.forEach((row, i) => row.forEach((v, c) => assert.ok(Math.abs(v - zkg[i][c]) < 1e-12)));
  // Quiz explanations: the raw winner has the smallest mass gap of all pairs.
  const before = lesson2.nearestPair(raw);
  const gap = (pts, i, j, k) => Math.abs(pts[i][k] - pts[j][k]);
  for (const p of pairs) {
    if (lesson2.pairKey(p) !== lesson2.pairKey(before)) assert.ok(gap(raw, p.i, p.j, 0) > gap(raw, before.i, before.j, 0));
  }
  // After standardising, fruit 1 and 3 are pushed apart mostly by diameter.
  assert.ok(gap(z, 0, 2, 1) > gap(z, 0, 2, 0));
  // The two apples are closer than the plum and apricot in both features.
  for (const k of [0, 1]) assert.ok(gap(raw, 0, 4, k) < gap(raw, 3, 5, k));
  // Heavier fruit are also wider here: sorting by mass sorts by diameter too.
  const byMass = raw.map((_, i) => i).sort((i, j) => raw[i][0] - raw[j][0]);
  for (let k = 1; k < byMass.length; k++) assert.ok(raw[byMass[k]][1] > raw[byMass[k - 1]][1]);
  // The nectarine is narrower than both apples.
  assert.ok(raw[2][1] < raw[0][1] && raw[2][1] < raw[4][1]);
  assert.deepEqual([lesson2.FRUIT[0].kind, lesson2.FRUIT[2].kind, lesson2.FRUIT[3].kind, lesson2.FRUIT[4].kind, lesson2.FRUIT[5].kind],
    ['apple', 'nectarine', 'plum', 'apple', 'apricot']);
});

// ---------------------------------------------------------------- lesson 3

import { median as medianOf } from '../assets/maths.js';
import * as lesson3 from '../centre-and-spread/stage.js';

const lesson3Html = readFileSync(new URL('../centre-and-spread/index.html', import.meta.url), 'utf8');

// The nine values from mulberry32(21), each round(50 + 8·randn), worked out
// once and written down here so the tests below can check by hand.
const L3_DATA = [40, 41, 58, 48, 52, 42, 46, 65, 37];

test('lesson3 data is reproducible and in range', () => {
  const v = lesson3.lessonData();
  assert.deepEqual(v, L3_DATA);
  assert.deepEqual(lesson3.lessonData(), v);
  for (const x of v) assert.ok(Number.isInteger(x) && x >= lesson3.LO && x <= lesson3.HI, `value ${x}`);
  // Dragged values are clamped to 0–120 and rounded to whole minutes.
  assert.equal(lesson3.clampValue(-7), 0);
  assert.equal(lesson3.clampValue(130.2), 120);
  assert.equal(lesson3.clampValue(64.6), 65);
});

test('lesson3 median unchanged by moving the maximum further out', () => {
  const v = lesson3.lessonData();
  const start = lesson3.stats(v);
  assert.equal(start.median, 46);
  for (const to of [66, 80, 100, 120]) {
    const moved = lesson3.moveLargest(v, to);
    assert.equal(Math.max(...moved), to);
    assert.deepEqual(moved.filter((x) => x !== to), v.filter((x) => x !== 65), 'only the largest value moved');
    const st = lesson3.stats(moved);
    assert.equal(st.median, start.median);
    // The mean moves by (distance moved) / n.
    assert.ok(Math.abs(st.mean - (start.mean + (to - 65) / 9)) < 1e-12);
  }
  // The input array is left alone.
  assert.deepEqual(v, L3_DATA);
});

test('lesson3 displayed variance uses n-1', () => {
  // By hand: the sum is 429, so the mean is 429/9 = 143/3.
  // 3x − 143 for each value: −23, −20, 31, 1, 13, −17, −5, 52, −32.
  // Their squares add to 529 + 400 + 961 + 1 + 169 + 289 + 25 + 2704 + 1024 = 6102,
  // so the sum of squared distances from the mean is 6102 / 9 = 678.
  const st = lesson3.stats(lesson3.lessonData());
  assert.equal(st.sum, 429);
  assert.ok(Math.abs(st.mean - 143 / 3) < 1e-12);
  assert.ok(Math.abs(st.ss - 678) < 1e-9);
  assert.ok(Math.abs(st.sampleVar - 678 / 8) < 1e-9);  // 84.75
  assert.ok(Math.abs(st.popVar - 678 / 9) < 1e-9);     // 75.33…
  assert.ok(Math.abs(st.sampleSd - Math.sqrt(84.75)) < 1e-12);
  assert.ok(Math.abs(st.popSd - Math.sqrt(678 / 9)) < 1e-12);
  // What the stage shows on the n − 1 step and the SD step.
  const rows5 = lesson3.readout(4, lesson3.lessonData(), 7).join('\n');
  assert.match(rows5, /sample variance \(÷ 8\):\s+84\.8 min²/);
  assert.match(rows5, /population variance \(÷ 9\):\s+75\.3 min²/);
  assert.match(rows5, /÷ 8 gives 9\/8 = 1\.125 × the area/);
  assert.ok(Math.abs(st.sampleVar / st.popVar - 9 / 8) < 1e-12);
  const rows6 = lesson3.readout(5, lesson3.lessonData(), 7).join('\n');
  assert.match(rows6, /sample SD \(÷ 8\):\s+9\.2 min/);
  assert.match(rows6, /population SD \(÷ 9\):\s+8\.7 min/);
});

test('lesson3 claims in the text hold for any single drag', () => {
  const v = lesson3.lessonData();
  const sorted = [...v].sort((a, b) => a - b);
  for (let i = 0; i < v.length; i++) {
    for (let to = lesson3.LO; to <= lesson3.HI; to++) {
      const w = v.slice();
      w[i] = to;
      const st = lesson3.stats(w);
      // The median stays between the old 4th and 6th values.
      assert.ok(st.median >= sorted[3] && st.median <= sorted[5], `drag ${i} to ${to}`);
      assert.equal(st.median, medianOf(w));
      // The balance point: the distances below and above the mean are equal.
      assert.ok(Math.abs(st.below - st.above) < 1e-9);
      // Dividing by 8 gives a bigger variance than dividing by 9, unless all are equal.
      assert.ok(st.sampleVar >= st.popVar);
    }
  }
  // z-scores use the population SD, and z = −2 sits two SDs below the mean.
  const st = lesson3.stats(v);
  assert.ok(Math.abs(lesson3.zscore(st.mean - 2 * st.popSd, st) + 2) < 1e-12);
  assert.ok(Math.abs(lesson3.zscore(65, st) - (65 - st.mean) / st.popSd) < 1e-12);
  // No spread: the z-score is undefined rather than NaN or 0.
  assert.equal(lesson3.zscore(50, lesson3.stats(new Array(9).fill(50))), null);
  // Stacking keeps dots that would overlap on separate levels.
  const lv = lesson3.levels([100, 104, 108, 140, 140], 13);
  assert.deepEqual(lv, [0, 1, 2, 0, 1]);
});

test('lesson3 page agrees with the stage', () => {
  // Every data-value span on the page is filled by the stage.
  const keys = [...lesson3Html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length > 0);
  const values = lesson3.pageValues(lesson3.lessonData());
  for (const k of keys) assert.ok(k in values, `no value for ${k}`);
  assert.equal(values['mean-start'], '47.7');
  assert.equal(values['median-start'], '46');
  assert.equal(values['mean-moved'], '53.8');
  // The quiz answers.
  const answer = (id) => lesson3Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), 'median');
  assert.equal(answer('q2'), 'two-sd-below');
  // The seed and the formula are stated in About the data.
  const note = lesson3Html.match(/<p class="data-note">([\s\S]*?)<\/p>/)[1];
  assert.match(note, /seed 21/);
  assert.match(note, /50 \+ 8/);
});

// ---------------------------------------------------------------- lesson 4

import { mean as meanOf, mse as mseOf, mae as maeOf, olsFit } from '../assets/maths.js';
import * as lesson4 from '../measuring-error/stage.js';

const lesson4Html = readFileSync(new URL('../measuring-error/index.html', import.meta.url), 'utf8');

// Grid search over constants c in [lo, hi] with the given step; returns the best c.
function gridArgmin(f, lo, hi, step) {
  let best = lo;
  let fBest = f(lo);
  const steps = Math.round((hi - lo) / step);
  for (let k = 1; k <= steps; k++) {
    const c = lo + k * step;
    const v = f(c);
    if (v < fBest) { fBest = v; best = c; }
  }
  return best;
}

test('lesson4 data is reproducible and the line is the least-squares fit', () => {
  const { xs, ys } = lesson4.lessonData();
  assert.equal(xs.length, 11);
  assert.equal(ys.length, 11);
  assert.deepEqual(lesson4.lessonData(), { xs, ys });
  for (const x of xs) assert.ok(x >= 0 && x <= 10);
  // n is odd, so the median of y is unique.
  assert.equal(xs.length % 2, 1);
  const fit = olsFit(xs, ys);
  const line = lesson4.fitLine(xs, ys);
  assert.ok(Math.abs(line.a - fit.intercept) < 1e-12);
  assert.ok(Math.abs(line.b - fit.slope) < 1e-12);
  // The residuals of the least-squares line (with an intercept) add to zero.
  const es = lesson4.residualsOf(xs, ys, line);
  assert.ok(Math.abs(es.reduce((s, e) => s + e, 0)) < 1e-9);
});

test('lesson4 argmin of MSE over constants is the mean', () => {
  const { ys } = lesson4.lessonData();
  const c = gridArgmin((c) => lesson4.mseAt(ys, c), lesson4.C_LO, lesson4.C_HI, 0.001);
  assert.ok(Math.abs(c - meanOf(ys)) <= 0.001, `grid ${c} vs mean ${meanOf(ys)}`);
  // The stage's own helper agrees, and the lowest MSE is the population variance of y.
  assert.ok(Math.abs(lesson4.mseAt(ys, meanOf(ys)) - ys.reduce((s, y) => s + (y - meanOf(ys)) ** 2, 0) / ys.length) < 1e-12);
  assert.ok(Math.abs(lesson4.mseAt(ys, 3) - mseOf(ys, ys.map(() => 3))) < 1e-12);
});

test('lesson4 argmin of MAE over constants is the median', () => {
  const { ys } = lesson4.lessonData();
  const c = gridArgmin((c) => lesson4.maeAt(ys, c), lesson4.C_LO, lesson4.C_HI, 0.001);
  assert.ok(Math.abs(c - medianOf(ys)) <= 0.001, `grid ${c} vs median ${medianOf(ys)}`);
  assert.ok(Math.abs(lesson4.maeAt(ys, 3) - maeOf(ys, ys.map(() => 3))) < 1e-12);
  // The mean and the median differ here, so the two losses pick different constants.
  assert.ok(Math.abs(meanOf(ys) - medianOf(ys)) > 0.1);
  // With an even number of values, every c between the two middle values ties.
  const even = [1, 2, 4, 9];
  assert.ok(Math.abs(lesson4.maeAt(even, 2.2) - lesson4.maeAt(even, 3.9)) < 1e-12);
  assert.ok(lesson4.maeAt(even, 1.9) > lesson4.maeAt(even, 2));
});

test('lesson4 loss summary and the read-out agree with maths.js', () => {
  const { xs, ys } = lesson4.lessonData();
  const line = lesson4.fitLine(xs, ys);
  const yhat = xs.map((x) => lesson4.predict(line, x));
  const L = lesson4.lossSummary(lesson4.residualsOf(xs, ys, line));
  assert.ok(Math.abs(L.mae - maeOf(ys, yhat)) < 1e-12);
  assert.ok(Math.abs(L.mse - mseOf(ys, yhat)) < 1e-12);
  assert.ok(Math.abs(L.rmse - Math.sqrt(L.mse)) < 1e-12);
  // Positive and negative residuals cancel.
  assert.ok(Math.abs(L.above + L.below) < 1e-9);
  assert.ok(L.above > 0 && L.nAbove + L.nBelow === 11);
  const rows = (i) => lesson4.readout(i, { xs, ys, line, start: ys, c: 3 }).join('\n');
  assert.match(rows(1), /sum of e: 0\.00/);
  assert.match(rows(2), new RegExp(`MAE = ${L.sumAbs.toFixed(2)}/11 = ${L.mae.toFixed(2)}`));
  assert.match(rows(3), new RegExp(`MSE = ${L.sumSq.toFixed(2)}/11 = ${L.mse.toFixed(2)}`));
  assert.match(rows(4), /miss: ×1\.00/);
  assert.match(rows(5), /c = 3\.00/);
  for (let i = 0; i < 6; i++) for (const r of lesson4.readout(i, { xs, ys, line, start: ys, c: 3 })) {
    assert.ok(!/NaN|undefined|Infinity/.test(r), r);
    assert.ok(r.length <= 22, `row too long for a phone: "${r}"`);
  }
});

test('lesson4 one miss k times as big: its square is k² times as big', () => {
  const { xs, ys } = lesson4.lessonData();
  const line = lesson4.fitLine(xs, ys);
  const i = lesson4.CHOSEN;
  const r = lesson4.dragRange(xs, ys, line);
  // The drag range keeps the point on the stage and the factor k within 0–KMAX.
  assert.ok(r.lo >= lesson4.Y_LO && r.hi <= lesson4.Y_HI);
  assert.ok(Math.abs(r.e0) > 0.5);
  for (const k of [0, 0.5, 1, 2, 3, lesson4.KMAX]) {
    const y = r.yhat + k * r.e0;
    assert.ok(y >= r.lo - 1e-9 && y <= r.hi + 1e-9, `k = ${k}`);
    const moved = ys.slice();
    moved[i] = y;
    const es = lesson4.residualsOf(xs, moved, line);
    assert.ok(Math.abs(lesson4.kFactor(es[i], r.e0) - k) < 1e-9);
    const L0 = lesson4.lossSummary(lesson4.residualsOf(xs, ys, line));
    const L1 = lesson4.lossSummary(es);
    // Its part of MAE grows k times, its part of MSE k² times; the other ten parts stay put.
    assert.ok(Math.abs((L1.mae - L0.mae) * 11 - (k - 1) * Math.abs(r.e0)) < 1e-9);
    assert.ok(Math.abs((L1.mse - L0.mse) * 11 - (k * k - 1) * r.e0 ** 2) < 1e-9);
  }
  const rows = lesson4.readout(4, { xs, ys: (() => { const m = ys.slice(); m[i] = r.yhat + 3 * r.e0; return m; })(), line, start: ys, c: 3 });
  assert.match(rows.join('\n'), /miss: ×3\.00/);
  assert.match(rows.join('\n'), /miss²: ×9\.00/);
});

test('lesson4 page agrees with the stage', () => {
  const keys = [...lesson4Html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length > 0);
  const { xs, ys } = lesson4.lessonData();
  const values = lesson4.pageValues(xs, ys);
  for (const k of keys) assert.ok(k in values, `no value for ${k}`);
  for (const v of Object.values(values)) assert.ok(!/NaN|undefined|Infinity/.test(v));
  assert.equal(values.mean, meanOf(ys).toFixed(2));
  assert.equal(values.median, medianOf(ys).toFixed(2));
  // The quiz answers.
  const answer = (id) => lesson4Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), 'mse');
  assert.equal(answer('q2'), 'median');
  // The seed and the formula are stated in About the data.
  const note = lesson4Html.match(/<p class="data-note">([\s\S]*?)<\/p>/)[1];
  assert.match(note, /seed 5/);
  assert.match(note, /2 \+ 0\.8/);
});

// ---------------------------------------------------------------- lesson 5
import { centralDiff } from '../assets/maths.js';
import * as lesson5 from '../slopes/stage.js';

const lesson5Html = readFileSync(new URL('../slopes/index.html', import.meta.url), 'utf8');

test('lesson5 analytic derivative matches centralDiff on 20 points', () => {
  for (let k = 0; k < 20; k++) {
    const x = lesson5.X_LO + (k / 19) * (lesson5.X_HI - lesson5.X_LO);
    const numeric = centralDiff(lesson5.f, x);
    assert.ok(Math.abs(lesson5.fPrime(x) - numeric) < 1e-6, `x = ${x}: ${lesson5.fPrime(x)} vs ${numeric}`);
  }
  // The facts the page states: f′(x) = x − 2, slope 2 at x = 4, −2 at x = 0.
  assert.equal(lesson5.fPrime(4), 2);
  assert.equal(lesson5.fPrime(0), -2);
  assert.equal(lesson5.f(0), 3);
});

test('lesson5 secant slope approaches derivative as h shrinks', () => {
  for (const x of [-1, 0, lesson5.X_SECANT, 4]) {
    const errs = [1, 0.5, 0.1, 0.01].map((h) => Math.abs(lesson5.secantSlope(x, h) - lesson5.fPrime(x)));
    for (let i = 1; i < errs.length; i++) assert.ok(errs[i] <= errs[i - 1], `x = ${x}: ${errs}`);
    // For this quadratic the gap is exactly h/2.
    for (const h of [1, 0.5, 0.1, 0.01]) {
      assert.ok(Math.abs(lesson5.secantSlope(x, h) - (lesson5.fPrime(x) + 0.5 * h)) < 1e-9);
    }
  }
  // The slider never reaches h = 0, and the secant stays on the plot.
  assert.ok(lesson5.H_MIN > 0 && lesson5.H_MIN < lesson5.H_MAX);
  assert.ok(lesson5.X_SECANT + lesson5.H_MAX <= lesson5.X_HI);
  assert.ok(Number.isFinite(lesson5.secantSlope(lesson5.X_SECANT, lesson5.H_MIN)));
});

test('lesson5 bottom of the bowl, the downhill rule and the straight line', () => {
  const v = lesson5.vertex();
  assert.equal(v.x, 2);
  assert.equal(v.y, 1);
  assert.equal(lesson5.fPrime(v.x), 0);
  for (let x = lesson5.X_LO; x <= lesson5.X_HI; x += 0.25) assert.ok(lesson5.f(x) >= v.y);
  assert.equal(lesson5.downhill(lesson5.fPrime(0)), 'right');
  assert.equal(lesson5.downhill(lesson5.fPrime(4)), 'left');
  assert.equal(lesson5.downhill(lesson5.fPrime(2)), 'flat');
  // Moving a small step against the slope's sign lowers f.
  for (const x of [-1, 0, 1, 3, 5, 6]) {
    const dir = lesson5.downhill(lesson5.fPrime(x)) === 'right' ? 1 : -1;
    assert.ok(lesson5.f(x + dir * 0.01) < lesson5.f(x), `x = ${x}`);
  }
  // Step 1: both triangles on the straight line give the same slope.
  const big = lesson5.slopeBetween(...lesson5.BIG);
  const small = lesson5.slopeBetween(...lesson5.SMALL);
  assert.equal(big, lesson5.LINE.b);
  assert.equal(small, lesson5.LINE.b);
  for (const [px, py] of [...lesson5.BIG, ...lesson5.SMALL]) assert.ok(Math.abs(lesson5.LINE.a + lesson5.LINE.b * px - py) < 1e-12);
});

test('lesson5 read-out rows and page', () => {
  for (let i = 0; i < 5; i++) {
    for (const x of [lesson5.X_LO, 0, 1.99, 2, 4, lesson5.X_HI]) {
      for (const h of [lesson5.H_MIN, 1, lesson5.H_MAX]) {
        for (const r of lesson5.readout(i, { x, h })) {
          assert.ok(!/NaN|undefined|Infinity/.test(r), r);
          assert.ok(r.length <= 22, `row too long for a phone: "${r}"`);
        }
      }
    }
  }
  assert.match(lesson5.readout(0, { x: 0, h: 1 }).join('\n'), /slope = 6\/4 = 1\.5/);
  assert.match(lesson5.readout(1, { x: 4, h: 1 }).join('\n'), /slope = 2\.00/);
  assert.match(lesson5.readout(2, { x: 0, h: 2 }).join('\n'), /secant: 2\.000/);
  assert.match(lesson5.readout(3, { x: 0, h: 1 }).join('\n'), /go right/);
  assert.match(lesson5.readout(4, { x: 2, h: 1 }).join('\n'), /slope = 0\.00/);
  const answer = (id) => lesson5Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), String(lesson5.fPrime(4)));
  assert.equal(answer('q2'), lesson5.downhill(lesson5.fPrime(0)));
  const note = lesson5Html.match(/<p class="data-note">([\s\S]*?)<\/p>/)[1];
  assert.match(note, /0\.5x² − 2x \+ 3/);
  assert.equal(lesson5Html.match(/class="step"/g).length, 5);
});

test('lesson5 downhill arrow never ends higher than it starts', () => {
  const v = lesson5.vertex();
  for (let k = 0; k <= 700; k++) {
    const x = lesson5.X_LO + k / 100;
    const d = lesson5.downhill(lesson5.fPrime(x));
    const arc = lesson5.downhillArrow(x);
    if (d === 'flat') { assert.equal(arc, null); continue; }
    if (arc === null) { assert.ok(Math.abs(v.x - x) < 0.12 + 0.15 + 1e-9, `hidden far from the bottom at x = ${x}`); continue; }
    assert.equal(arc.dir, d === 'right' ? 1 : -1);
    const [x0, y0] = arc.pts[0];
    const [x1, y1] = arc.pts[arc.pts.length - 1];
    assert.ok(y1 <= y0, `x = ${x}: ends at ${y1}, starts at ${y0}`);
    // It never passes the bottom, so it runs downhill all the way.
    assert.ok(arc.dir > 0 ? x1 <= v.x + 1e-12 : x1 >= v.x - 1e-12, `x = ${x}: passes the bottom`);
    for (let j = 1; j < arc.pts.length; j++) assert.ok(arc.pts[j][1] <= arc.pts[j - 1][1] + 1e-12);
    assert.ok(Math.sign(x1 - x0) === arc.dir);
  }
  assert.notEqual(lesson5.downhillArrow(0.5), null);
  assert.equal(lesson5.downhillArrow(1.99), null);
});

// ---------------------------------------------------------------- lesson 6

import { mulberry32 as rng6, uniform as uniform6, randn as randn6 } from '../assets/maths.js';
import * as lesson6 from '../linear-regression/stage.js';

const lesson6Html = readFileSync(new URL('../linear-regression/index.html', import.meta.url), 'utf8');

test('lesson6 regressionData is seed 11, n = 25, x drawn before y', () => {
  const { xs, ys } = lesson6.regressionData();
  assert.equal(xs.length, 25);
  assert.equal(ys.length, 25);
  const r = rng6(11);
  for (let i = 0; i < 25; i++) {
    const x = uniform6(r, 0, 10);
    assert.equal(xs[i], x);
    assert.equal(ys[i], 4 + 1.5 * x + 2 * randn6(r));
  }
  assert.deepEqual(lesson6.regressionData(), { xs, ys });
});

test('lesson6 user SSE is never below OLS SSE', () => {
  const { xs, ys } = lesson6.regressionData();
  const ols = lesson6.fitLine(xs, ys);
  const best = lesson6.sse(xs, ys, ols);
  const r = rng6(2024);
  // 200 random lines anywhere.
  for (let k = 0; k < 200; k++) {
    const line = { a: uniform6(r, -20, 40), b: uniform6(r, -6, 6) };
    assert.ok(lesson6.sse(xs, ys, line) >= best, `a ${line.a}, b ${line.b}`);
  }
  // 200 lines the reader can make with the two handles.
  for (let k = 0; k < 200; k++) {
    const line = lesson6.lineFromHandles(uniform6(r, lesson6.Y_LO, lesson6.Y_HI), uniform6(r, lesson6.Y_LO, lesson6.Y_HI));
    assert.ok(lesson6.sse(xs, ys, line) >= best);
  }
  // 200 lines very close to the least-squares line.
  for (let k = 0; k < 200; k++) {
    const line = { a: ols.a + uniform6(r, -1e-4, 1e-4), b: ols.b + uniform6(r, -1e-5, 1e-5) };
    assert.ok(lesson6.sse(xs, ys, line) >= best - 1e-9);
  }
  // The flat line at the mean of y is one of the candidates, so SSE <= SST.
  assert.ok(best <= lesson6.sst(ys));
  // "Solve" lands exactly on the least-squares line.
  const from = lesson6.lineFromHandles(15, 9);
  assert.deepEqual(lesson6.solveFrame(from, ols, 1), ols);
  const f = olsFit(xs, ys);
  assert.equal(lesson6.solveFrame(from, ols, 1).a, f.intercept);
  assert.equal(lesson6.solveFrame(from, ols, 1).b, f.slope);
  assert.deepEqual(lesson6.solveFrame(from, ols, 0), from);
});

test('lesson6 R2 equals squared correlation for simple regression', () => {
  const { xs, ys } = lesson6.regressionData();
  const ols = lesson6.fitLine(xs, ys);
  const r2 = lesson6.rSquared(xs, ys, ols);
  const r = lesson6.correlation(xs, ys);
  assert.ok(Math.abs(r2 - r * r) < 1e-12, `${r2} vs ${r * r}`);
  assert.ok(Math.abs(r2 - olsFit(xs, ys).r2) < 1e-12);
  assert.ok(r2 > 0 && r2 < 1);
  // For any other line, 1 − SSE/SST is lower than r², so the identity needs the least-squares line.
  assert.ok(lesson6.rSquared(xs, ys, { a: ols.a + 1, b: ols.b }) < r * r);
  // Also on other seeds.
  for (const seed of [1, 2, 3]) {
    const d = lesson6.regressionData(seed);
    const l = lesson6.fitLine(d.xs, d.ys);
    const c = lesson6.correlation(d.xs, d.ys);
    assert.ok(Math.abs(lesson6.rSquared(d.xs, d.ys, l) - c * c) < 1e-12);
  }
});

test('lesson6 closed form matches the partial derivatives being zero', () => {
  const { xs, ys } = lesson6.regressionData();
  const ols = lesson6.fitLine(xs, ys);
  const h = 1e-5;
  const dA = (lesson6.sse(xs, ys, { a: ols.a + h, b: ols.b }) - lesson6.sse(xs, ys, { a: ols.a - h, b: ols.b })) / (2 * h);
  const dB = (lesson6.sse(xs, ys, { a: ols.a, b: ols.b + h }) - lesson6.sse(xs, ys, { a: ols.a, b: ols.b - h })) / (2 * h);
  assert.ok(Math.abs(dA) < 1e-6 && Math.abs(dB) < 1e-6, `${dA} ${dB}`);
  const p = lesson6.pageValues(xs, ys);
  assert.equal(p['a-hat'], ols.a.toFixed(2));
  assert.equal(p['b-hat'], ols.b.toFixed(2));
});

test('lesson6 read-out rows and page', () => {
  const { xs, ys } = lesson6.regressionData();
  const ols = lesson6.fitLine(xs, ys);
  const lines = [lesson6.lineFromHandles(lesson6.Y_LO, lesson6.Y_HI), lesson6.lineFromHandles(lesson6.Y_HI, lesson6.Y_LO), lesson6.START, ols];
  for (let i = 0; i < 6; i++) {
    for (const user of lines) {
      for (const x0 of [lesson6.X0_LO(xs), lesson6.X0_HI(xs)]) {
        for (const from of ['line', 'mean']) {
          for (const row of lesson6.readout(i, { xs, ys, user, line: ols, x0, from })) {
            assert.ok(!/NaN|undefined|Infinity/.test(row), row);
            assert.ok(row.length <= 22, `row too long for a phone: "${row}"`);
          }
        }
      }
    }
  }
  // Step 5's slider stays inside the data, so x and x + 1 are both within it.
  assert.ok(lesson6.X0_LO(xs) >= Math.min(...xs));
  assert.ok(lesson6.X0_HI(xs) + 1 <= Math.max(...xs));
  assert.equal(lesson6Html.match(/class="step"/g).length, 6);
  const answer = (id) => lesson6Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), 'squares');
  assert.equal(answer('q2'), 'per-unit');
  const note = lesson6Html.match(/<p class="data-note">([\s\S]*?)<\/p>/)[1];
  assert.match(note, /seed 11/);
  assert.match(note, /4 \+ 1\.5x/);
  // Every data-value span in the page has a value.
  const p = lesson6.pageValues(xs, ys);
  for (const [, key] of lesson6Html.matchAll(/data-value="([^"]+)"/g)) assert.ok(key in p, key);
});

test('lesson6 step 5 dashed extrapolation segments are clearly visible', () => {
  const { xs, ys } = lesson6.regressionData();
  const [left, right] = lesson6.extrapolationPx(xs);
  assert.ok(left >= 40 && right >= 40, `dashed ends ${left} px and ${right} px`);
  // The extended least-squares line still fits inside the y range.
  const ols = lesson6.fitLine(xs, ys);
  for (const x of lesson6.X_DOMAIN) {
    const y = lesson6.predict(ols, x);
    assert.ok(y >= lesson6.Y_LO && y <= lesson6.Y_HI, `ŷ(${x}) = ${y}`);
  }
  // x = 0 (the intercept marker) and both handles sit inside the axis.
  for (const x of [0, ...lesson6.HX]) assert.ok(x > lesson6.X_DOMAIN[0] && x < lesson6.X_DOMAIN[1]);
});

// ---------------------------------------------------------------- lesson 7

import { gdRun as gdRun7, mse as mse7, mean as mean7 } from '../assets/maths.js';
import * as lesson7 from '../gradient-descent/stage.js';

const lesson7Html = readFileSync(new URL('../gradient-descent/index.html', import.meta.url), 'utf8');

test('lesson7 loss on centred data is a(w − w*)² + c, with w* the OLS slope', () => {
  const d = lesson7.centredData();
  const { xs, ys } = lesson6.regressionData();
  assert.ok(Math.abs(mean7(d.cx)) < 1e-12 && Math.abs(mean7(d.cy)) < 1e-12);
  assert.equal(d.xbar, mean7(xs));
  assert.equal(d.ybar, mean7(ys));
  const p = lesson7.lossParams(d.cx, d.cy);
  assert.ok(Math.abs(p.a - mean7(d.cx.map((x) => x * x))) < 1e-12);
  assert.ok(Math.abs(p.wStar - lesson6.fitLine(xs, ys).b) < 1e-12);
  for (let w = -1; w <= 4; w += 0.25) {
    const direct = mse7(d.cy, d.cx.map((x) => w * x));
    assert.ok(Math.abs(lesson7.loss(p, w) - direct) < 1e-9, `w = ${w}`);
    const g = lesson7.gradOf(d)([w])[0];
    assert.ok(Math.abs(lesson7.lossSlope(p, w) - g) < 1e-9, `w = ${w}`);
  }
  // The minimum loss is Lesson 6's MSE of the least-squares line.
  assert.ok(Math.abs(p.c - lesson6.sse(xs, ys, lesson6.fitLine(xs, ys)) / xs.length) < 1e-9);
});

test('lesson7 regime boundaries', () => {
  const d = lesson7.centredData();
  const { a, wStar } = lesson7.lossParams(d.cx, d.cy);
  const grad = lesson7.gradOf(d);
  const run = (eta) => gdRun7({ grad, init: [lesson7.W0], eta, steps: 20000 });
  assert.equal(run(0.98 / a).status, 'converged');
  assert.equal(run(1.02 / a).status, 'diverged');
  // Between 1/(2a) and 1/a the iterate lands on alternate sides of w*.
  const zz = gdRun7({ grad, init: [lesson7.W0], eta: 0.75 / a, steps: 30, tol: 0 });
  for (let k = 1; k < zz.path.length; k++) {
    const e0 = zz.path[k - 1][0] - wStar;
    const e1 = zz.path[k][0] - wStar;
    assert.ok(e0 * e1 < 0, `step ${k}: ${e0}, ${e1}`);
  }
  // Below 1/(2a) it stays on the starting side.
  const st = gdRun7({ grad, init: [lesson7.W0], eta: 0.25 / a, steps: 30, tol: 0 });
  for (const [w] of st.path) assert.ok(w < wStar);
  // The labels follow r = 1 − 2aη.
  assert.equal(lesson7.regime(0.4 / a, a), 'steady');
  assert.equal(lesson7.regime(0.5 / a, a), 'onestep');
  assert.equal(lesson7.regime(0.75 / a, a), 'zigzag');
  assert.equal(lesson7.regime(1 / a, a), 'never');
  assert.equal(lesson7.regime(1.01 / a, a), 'diverges');
  assert.equal(lesson7.regime(2.5 / a, a), 'diverges');
  assert.equal(lesson7.REGIME_LABEL.never, 'never settles');
  assert.equal(lesson7.REGIME_LABEL.diverges, 'diverges');
  // Exactly at 1/a the distance to w* never changes.
  const nv = lesson7.runPath(1 / a, d, 20);
  for (const [w] of nv.path) assert.ok(Math.abs(Math.abs(w - wStar) - Math.abs(lesson7.W0 - wStar)) < 1e-9);
  // Every slider position gets a label that matches the maths.
  for (let t = 0; t <= lesson7.SLIDER_MAX; t++) {
    const eta = lesson7.etaFromSlider(t, a);
    const r = 1 - 2 * a * eta;
    const lab = lesson7.regime(eta, a);
    if (lab === 'steady') assert.ok(r > 0 && r < 1);
    else if (lab === 'onestep') assert.ok(Math.abs(r) < 1e-12);
    else if (lab === 'zigzag') assert.ok(r < 0 && r > -1);
    else if (lab === 'diverges') assert.ok(r < -1);
    else assert.ok(Math.abs(r + 1) < 1e-9);
  }
});

test('lesson7 converges to OLS slope', () => {
  const d = lesson7.centredData();
  const { xs, ys } = lesson6.regressionData();
  const ols = lesson6.fitLine(xs, ys);
  const c = lesson7.converge(d, lesson7.ETA_DEFAULT);
  assert.equal(c.status, 'converged');
  assert.ok(Math.abs(c.w - ols.b) < 1e-6, `${c.w} vs ${ols.b}`);
  // The intercept recovered from the centring gives Lesson 6's line.
  assert.ok(Math.abs(lesson7.intercept(d, c.w) - ols.a) < 1e-6);
  // Any η in the stable range reaches the same slope.
  const { a } = lesson7.lossParams(d.cx, d.cy);
  for (const eta of [0.1 / a, 0.5 / a, 0.9 / a]) assert.ok(Math.abs(lesson7.converge(d, eta).w - ols.b) < 1e-6);
});

test('lesson7 divergence at η = 2.5/a: label, clamped ball, no NaN', () => {
  const d = lesson7.centredData();
  const { a } = lesson7.lossParams(d.cx, d.cy);
  const etaMax = lesson7.etaFromSlider(lesson7.SLIDER_MAX, a);
  assert.equal(etaMax, 2.5 / a);
  assert.equal(lesson7.etaFromSlider(0, a), lesson7.ETA_MIN);
  assert.equal(lesson7.regime(etaMax, a), 'diverges');
  const run = lesson7.runPath(etaMax, d, lesson7.K);
  assert.equal(run.status, 'diverged');
  for (const [w] of run.path) assert.ok(Number.isFinite(w));
  const ball = lesson7.ballView(run.path[run.path.length - 1][0]);
  assert.equal(ball.clamped, true);
  assert.ok(ball.w === lesson7.W_DOMAIN[0] || ball.w === lesson7.W_DOMAIN[1]);
  const rows = lesson7.readout(4, { d, eta: etaMax, w: lesson7.W0, trail: run.path.map(([w]) => w) });
  assert.ok(rows.join(' ').includes('diverges'));
  for (const r of rows) assert.ok(!/NaN|undefined|Infinity/.test(r), r);
  // In the stable range the ball is not clamped.
  const ok = lesson7.runPath(lesson7.ETA_DEFAULT, d, lesson7.K);
  assert.equal(lesson7.ballView(ok.path[ok.path.length - 1][0]).clamped, false);
});

test('lesson7 slider, downhill cue, bumpy inset, read-out rows and page', () => {
  const d = lesson7.centredData();
  const p = lesson7.lossParams(d.cx, d.cy);
  // The log slider round-trips and starts at the default.
  const t0 = lesson7.sliderFromEta(lesson7.ETA_DEFAULT, p.a);
  assert.ok(Math.abs(lesson7.etaFromSlider(t0, p.a) / lesson7.ETA_DEFAULT - 1) < 0.01);
  assert.equal(lesson7.regime(lesson7.ETA_DEFAULT, p.a), 'steady');
  // The downhill arrow points against the slope's sign everywhere on the axis.
  for (let w = lesson7.W_DOMAIN[0]; w <= lesson7.W_DOMAIN[1]; w += 0.01) {
    const dir = lesson7.downhillDir(lesson7.lossSlope(p, w));
    if (dir === 0) {
      assert.ok(Math.abs(w - p.wStar) < 0.01);
      continue;
    }
    assert.ok(lesson7.loss(p, w + dir * 1e-3) < lesson7.loss(p, w), `w = ${w}`);
  }
  // The data-driven ranges fit the plot.
  assert.ok(p.wStar > lesson7.W_DOMAIN[0] && p.wStar < lesson7.W_DOMAIN[1]);
  for (const w of lesson7.W_DOMAIN) assert.ok(lesson7.loss(p, w) < lesson7.L_HI);
  // The bumpy inset: a run stops in the higher dip, not the lowest one.
  const b = lesson7.bumpyRun();
  assert.equal(b.status, 'converged');
  const g = lesson7.bumpyLowest();
  assert.ok(Math.abs(lesson7.bumpyPrime(b.u)) < 1e-6);
  assert.ok(lesson7.bumpy(b.u) > lesson7.bumpy(g) + 0.3, `${b.u} vs ${g}`);
  assert.ok(Math.abs(b.u - g) > 1);
  // Read-out rows are finite and short for every step and a range of η.
  for (let i = 0; i < 6; i++) {
    for (const t of [0, 300, t0, 700, 800, lesson7.SLIDER_MAX]) {
      const eta = lesson7.etaFromSlider(t, p.a);
      const run = lesson7.runPath(eta, d, lesson7.K);
      for (const w of [lesson7.W_DOMAIN[0], lesson7.W0, p.wStar, lesson7.W_DOMAIN[1]]) {
        for (const r of lesson7.readout(i, { d, eta, w, trail: run.path.map(([v]) => v) })) {
          assert.ok(!/NaN|undefined|Infinity/.test(r), r);
          assert.ok(r.length <= 22, `row too long for a phone: "${r}"`);
        }
      }
    }
  }
  assert.equal(lesson7Html.match(/class="step"/g).length, 6);
  const answer = (id) => lesson7Html.match(new RegExp(`<form class="q" id="${id}" data-answer="([^"]+)"`))[1];
  assert.equal(answer('q1'), 'overshoot');
  assert.equal(answer('q2'), 'slope');
  const note = lesson7Html.match(/<p class="data-note">([\s\S]*?)<\/p>/)[1];
  assert.match(note, /seed 11/);
  const pv = lesson7.pageValues();
  for (const [, key] of lesson7Html.matchAll(/data-value="([^"]+)"/g)) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('lesson7 slider stops on both boundaries, and the shown η agrees with its label', () => {
  const d = lesson7.centredData();
  const { a, wStar } = lesson7.lossParams(d.cx, d.cy);
  const snap = lesson7.snapPositions(a);
  assert.ok(snap.half > 0 && snap.one > snap.half && snap.one < lesson7.SLIDER_MAX);
  // The stop at 1/(2a) lands on w* in one step; the stop at 1/a never settles.
  assert.equal(lesson7.etaFromSlider(snap.half, a), 1 / (2 * a));
  assert.equal(lesson7.regime(lesson7.etaFromSlider(snap.half, a), a), 'onestep');
  assert.equal(lesson7.REGIME_LABEL.onestep, 'lands in one step');
  const one = lesson7.runPath(lesson7.etaFromSlider(snap.half, a), d, 3);
  assert.ok(Math.abs(one.path[1][0] - wStar) < 1e-9);
  assert.equal(lesson7.etaFromSlider(snap.one, a), 1 / a);
  assert.equal(lesson7.regime(lesson7.etaFromSlider(snap.one, a), a), 'never');
  const pv = lesson7.pageValues();
  const half = Number(pv['eta-half']);
  const inv = Number(pv['eta-one']);
  for (let t = 0; t <= lesson7.SLIDER_MAX; t++) {
    const eta = lesson7.etaFromSlider(t, a);
    const q = 1 - 2 * a * eta;
    const lab = lesson7.regime(eta, a);
    const shown = Number(lesson7.etaText(eta, a));
    // The label follows q from the actual η ...
    const want = Math.abs(q + 1) < 1e-12 ? 'never' : q < -1 ? 'diverges' : Math.abs(q) < 1e-12 ? 'onestep' : q < 0 ? 'zigzag' : 'steady';
    assert.equal(lab, want, `t = ${t}`);
    // ... and the shown η reads the same way against the boundaries the page shows.
    if (lab === 'steady') assert.ok(shown < half, `t = ${t}: ${shown}`);
    if (lab === 'onestep') assert.equal(shown, half);
    if (lab === 'zigzag') assert.ok(shown > half && shown < inv, `t = ${t}: ${shown}`);
    if (lab === 'never') assert.equal(shown, inv);
    if (lab === 'diverges') assert.ok(shown > inv, `t = ${t}: ${shown}`);
    for (const r of lesson7.readout(4, { d, eta, w: lesson7.W0, trail: lesson7.runPath(eta, d).path.map(([w]) => w) })) {
      assert.ok(r.length <= 22, r);
    }
  }
});

// ---------------------------------------------------------------- lesson 8
import { mse as mse8, olsFit as olsFit8, gdRun as gdRun8, mseGrad2D as grad8, standardise as standardise8 } from '../assets/maths.js';
import { linearScale as scale8 } from '../assets/plot.js';
import * as lesson8 from '../loss-surface/stage.js';

const lesson8Html = readFileSync(new URL('../loss-surface/index.html', import.meta.url), 'utf8');

test('lesson8 standardised condition number is 1', () => {
  const { xs } = lesson8.lossData();
  const { z } = standardise8(xs);
  const c = lesson8.curvatures(z);
  assert.ok(Math.abs(c.kappa - 1) < 1e-9, `kappa ${c.kappa}`);
  assert.ok(Math.abs(c.gmax - 2) < 1e-9 && Math.abs(c.gmin - 2) < 1e-9);
  // The raw Hessian is (2/n)·[[n, Σx], [Σx, Σx²]].
  const n = xs.length;
  const sx = xs.reduce((s, x) => s + x, 0);
  const sxx = xs.reduce((s, x) => s + x * x, 0);
  const H = lesson8.hessian(xs);
  assert.ok(Math.abs(H[0][0] - 2) < 1e-12 && Math.abs(H[0][1] - (2 / n) * sx) < 1e-9 && Math.abs(H[1][1] - (2 / n) * sxx) < 1e-6);
  const r = lesson8.curvatures(xs);
  assert.ok(r.kappa > 1e5, `raw kappa ${r.kappa}`);
});

test('lesson8 standardised run, mapped back to original units, equals the OLS line within 1e-4', () => {
  const { xs, ys } = lesson8.lossData();
  const ols = olsFit8(xs, ys);
  const fit = lesson8.standardisedFit();
  assert.ok(Math.abs(fit.a - ols.intercept) < 1e-4, `a ${fit.a} v ${ols.intercept}`);
  assert.ok(Math.abs(fit.w - ols.slope) < 1e-4, `w ${fit.w} v ${ols.slope}`);
  // The mapping itself: a = a′ − w′μ/σ, w = w′/σ gives the same predictions.
  const { z, mu, sigma } = standardise8(xs);
  const back = lesson8.mapBack([3, 7], mu, sigma);
  xs.forEach((x, i) => assert.ok(Math.abs(back.a + back.w * x - (3 + 7 * z[i])) < 1e-9));
});

test('lesson8 raw run\'s closed-form step count exceeds the standardised run\'s', () => {
  const s = lesson8.runFacts();
  assert.ok(Number.isInteger(s.raw.steps) && Number.isInteger(s.std.steps));
  assert.ok(s.raw.steps > s.std.steps, `${s.raw.steps} v ${s.std.steps}`);
  assert.ok(s.raw.steps > 1000 * s.std.steps);
  // Both use η = 1.8 / γmax of their own Hessian, which is stable (below 2 / γmax).
  assert.ok(Math.abs(s.raw.eta * s.raw.gmax - 1.8) < 1e-12 && Math.abs(s.std.eta * s.std.gmax - 1.8) < 1e-12);
});

test('lesson8 closed-form step count matches gdRun for the standardised run', () => {
  const { xs, ys } = lesson8.lossData();
  const { z } = standardise8(xs);
  const s = lesson8.runFacts();
  const ols = olsFit8(z, ys);
  const Ls = mse8(ys, z.map((v) => ols.intercept + ols.slope * v));
  const run = gdRun8({ grad: grad8(z, ys), init: lesson8.START, eta: s.std.eta, steps: 200, tol: 0 });
  const ex = run.path.map(([a, w]) => mse8(ys, z.map((v) => a + w * v)) - Ls);
  const first = ex.findIndex((e) => e <= lesson8.TOL * ex[0]);
  assert.ok(first > 0);
  assert.equal(first, s.std.steps, `gdRun ${first} v closed form ${s.std.steps}`);
  // The closed-form extra loss follows gdRun step by step, for both runs.
  for (const [data, eta] of [[z, s.std.eta], [xs, s.raw.eta]]) {
    const o = olsFit8(data, ys);
    const L0 = mse8(ys, data.map((v) => o.intercept + o.slope * v));
    const p = gdRun8({ grad: grad8(data, ys), init: lesson8.START, eta, steps: 60, tol: 0 }).path;
    p.forEach(([a, w], k) => {
      const direct = mse8(ys, data.map((v) => a + w * v)) - L0;
      const closed = lesson8.closedExcess(data, ys, eta, k);
      assert.ok(Math.abs(direct - closed) <= 1e-6 * ex[0] + 1e-6 * Math.abs(direct), `k ${k}: ${direct} v ${closed}`);
    });
  }
});

test('lesson8 contours come from the real loss, and the downhill arrow points downhill everywhere', () => {
  const { xs, ys } = lesson8.lossData();
  const L = (a, w) => mse8(ys, xs.map((x) => a + w * x));
  const g = lesson8.lossGrid(xs, ys, lesson8.RAW_A, lesson8.RAW_W, 40, 60);
  assert.equal(g.vals[7][11], L(g.as[11], g.ws[7]));
  const Ls = lesson8.minLoss(xs, ys);
  // A segment end sits on a cell edge, so its loss is the level up to linear-interpolation error.
  const ha = g.as[1] - g.as[0];
  const hw = g.ws[1] - g.ws[0];
  const H = lesson8.hessian(xs);
  const bound = Math.max(H[0][0] * ha * ha, H[1][1] * hw * hw) / 8 + 1e-9;
  for (const c of lesson8.LEVELS) {
    const segs = lesson8.contourSegments(g, Ls + c);
    for (const seg of segs) for (const [a, w] of seg) assert.ok(Math.abs(L(a, w) - (Ls + c)) <= bound, `level ${c}`);
  }
  assert.ok(lesson8.contourSegments(g, Ls + lesson8.LEVELS[2]).length > 0);
  // The drawn arrow, mapped back from pixels to (a, w), lowers the loss at every slider position.
  const X = scale8(lesson8.RAW_A, [56, 372]);
  const Y = scale8(lesson8.RAW_W, [290, 20]);
  for (let a = lesson8.RAW_A[0]; a <= lesson8.RAW_A[1]; a += 2.5) {
    for (let w = lesson8.RAW_W[0]; w <= lesson8.RAW_W[1] + 1e-9; w += 0.025) {
      const v = lesson8.arrowVec(xs, ys, a, w, X, Y);
      if (!v) continue;
      const da = (X.invert(X(a) + v[0]) - a) * 1e-4;
      const dw = (Y.invert(Y(w) + v[1]) - w) * 1e-4;
      assert.ok(L(a + da, w + dw) < L(a, w), `a ${a}, w ${w}`);
    }
  }
});

test('lesson8 read-out rows and page', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { a: -10, w: 2.3 }, { a: 125, w: -0.1 }, { k: 0 }, { k: 40 }, { view: 'raw' }]) {
      for (const r of lesson8.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((lesson8Html.match(/<li class="step"/g) || []).length, 6);
  assert.match(lesson8Html, /data-answer="across"/);
  assert.match(lesson8Html, /data-answer="shape"/);
  assert.match(lesson8Html, /seed 13/);
  assert.match(lesson8Html, /<title>Many features and the loss surface · ML Explorer<\/title>/);
  const pv = lesson8.pageValues();
  for (const [, key] of lesson8Html.matchAll(/data-value="([^"]+)"/g)) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
  assert.equal(pv['std-steps'], '31');
});

test('lesson8 closed-form count equals gdRun on moderately conditioned data', () => {
  const counts = [];
  for (const [lo, hi] of [[1, 6], [5, 15], [10, 30]]) {
    const rng = rng6(5);
    const xs = [];
    const ys = [];
    for (let i = 0; i < 30; i++) {
      const x = uniform6(rng, lo, hi);
      xs.push(x);
      ys.push(2 + 0.5 * x + randn6(rng));
    }
    const eta = lesson8.etaFor(xs);
    const closed = lesson8.stepsToTol(xs, ys, eta);
    const o = olsFit8(xs, ys);
    const Ls = mse8(ys, xs.map((x) => o.intercept + o.slope * x));
    const run = gdRun8({ grad: grad8(xs, ys), init: lesson8.START, eta, steps: closed + 50, tol: 0 });
    const ex = run.path.map(([a, w]) => mse8(ys, xs.map((x) => a + w * x)) - Ls);
    const first = ex.findIndex((e) => e <= lesson8.TOL * ex[0]);
    assert.equal(first, closed, `x in [${lo}, ${hi}]: gdRun ${first} v closed form ${closed}`);
    counts.push(closed);
  }
  // Wider, further-from-zero x is worse conditioned, so it needs more steps.
  assert.ok(counts[0] < counts[1] && counts[1] < counts[2], String(counts));
});

// ---------------------------------------------------------------- shared sine data (lessons 9 and 10)
import { mulberry32 as rngS, uniform as uniformS, randn as randnS, mse as mseS, polyFit as polyFitS, polyEval as polyEvalS, trainTestSplit as splitS } from '../assets/maths.js';
import { sineData } from '../assets/sine-data.js';

test('sineData: same seed gives the same data, x in [−1, 1], n respected, x drawn before y', () => {
  const a = sineData({ n: 40, seed: 3 });
  const b = sineData({ n: 40, seed: 3 });
  assert.deepEqual(a, b);
  assert.equal(a.xs.length, 40);
  assert.equal(a.ys.length, 40);
  assert.equal(sineData({ n: 7, seed: 9 }).xs.length, 7);
  for (const x of a.xs) assert.ok(x >= -1 && x <= 1, `x out of range: ${x}`);
  assert.notDeepEqual(sineData({ n: 40, seed: 4 }).xs, a.xs);
  // For each point in turn: x ~ Uniform(−1, 1), then y = sin(πx) + sd·randn.
  const rng = rngS(3);
  for (let i = 0; i < 40; i++) {
    const x = uniformS(rng, -1, 1);
    const y = Math.sin(Math.PI * x) + 0.25 * randnS(rng);
    assert.equal(a.xs[i], x);
    assert.equal(a.ys[i], y);
  }
  // sd is honoured, and sd = 0 puts every point on the curve.
  const c = sineData({ n: 10, sd: 0, seed: 5 });
  c.xs.forEach((x, i) => assert.equal(c.ys[i], Math.sin(Math.PI * x)));
});

// ---------------------------------------------------------------- lesson 9
import * as lesson9 from '../train-test/stage.js';

test('lesson9 train MSE <= test MSE on the default seed', () => {
  const d = lesson9.lessonData();
  assert.deepEqual(d, sineData({ n: 40, seed: 3 }));
  assert.equal(lesson9.SPLIT_SEED, 3);
  const sp = lesson9.splitFor(lesson9.SPLIT_SEED);
  assert.deepEqual(sp, splitS(40, 0.3, rngS(3)));
  assert.equal(sp.train.length, 28);
  assert.equal(sp.test.length, 12);
  const s = lesson9.score(d, sp, 3);
  const pick = (arr, idx) => idx.map((i) => arr[i]);
  const fit = polyFitS(pick(d.xs, sp.train), pick(d.ys, sp.train), 3);
  assert.ok(fit.ok);
  const yhat = (idx) => pick(d.xs, idx).map((x) => polyEvalS(fit.coef, x));
  assert.ok(Math.abs(s.train - mseS(pick(d.ys, sp.train), yhat(sp.train))) < 1e-12);
  assert.ok(Math.abs(s.test - mseS(pick(d.ys, sp.test), yhat(sp.test))) < 1e-12);
  assert.ok(s.train <= s.test, `${s.train} vs ${s.test}`);
});

test('lesson9 join-the-dots scores zero on its own points; the true curve does not', () => {
  const d = lesson9.lessonData();
  const f = lesson9.joinDots(d.xs, d.ys);
  assert.ok(mseS(d.ys, d.xs.map(f)) < 1e-24);
  const t = lesson9.stepOneScores(d);
  assert.ok(t.dots < 1e-24 && t.truth > 0.01);
  assert.ok(Math.abs(t.truth - mseS(d.ys, d.xs.map((x) => Math.sin(Math.PI * x)))) < 1e-12);
  // Between two neighbouring points it is the straight line joining them.
  const o = d.xs.map((x, i) => [x, d.ys[i]]).sort((p, q) => p[0] - q[0]);
  const m = (o[3][0] + o[4][0]) / 2;
  assert.ok(Math.abs(f(m) - (o[3][1] + o[4][1]) / 2) < 1e-12);
});

test('lesson9 some splits put the training MSE above the test MSE, and the page can say so', () => {
  const d = lesson9.lessonData();
  const rows = [];
  for (let seed = lesson9.SPLIT_SEED; seed < lesson9.SPLIT_SEED + lesson9.STRIP_N; seed++) rows.push(lesson9.score(d, lesson9.splitFor(seed), 3));
  const below = rows.filter((r) => r.train < r.test).length;
  assert.ok(below > 0 && below < rows.length, String(below));
  const st = lesson9.splitStats(d, 200);
  assert.ok(st.below > 100 && st.below < 200, String(st.below));
  assert.ok(st.meanTrain < st.meanTest);
});

test('lesson9 k-fold: every training point validates once, the test points never', () => {
  const d = lesson9.lessonData();
  const sp = lesson9.splitFor(lesson9.SPLIT_SEED);
  const folds = lesson9.folds(sp.train, lesson9.K);
  assert.equal(folds.length, 5);
  const all = folds.flat().sort((a, b) => a - b);
  assert.deepEqual(all, [...sp.train].sort((a, b) => a - b));
  for (const f of folds) assert.ok(f.length === 5 || f.length === 6);
  for (const t of sp.test) assert.ok(!all.includes(t));
  const cv = lesson9.cvTable(d, sp);
  assert.deepEqual(cv.scores.map((r) => r.degree), lesson9.CANDIDATES);
  const min = Math.min(...cv.scores.map((r) => r.cv));
  assert.equal(cv.scores.find((r) => r.degree === cv.best).cv, min);
  // Each score is the mean over the k rounds of the validation MSE.
  const pick = (arr, idx) => idx.map((i) => arr[i]);
  const deg = 2;
  let tot = 0;
  for (const v of folds) {
    const tr = sp.train.filter((i) => !v.includes(i));
    const fit = polyFitS(pick(d.xs, tr), pick(d.ys, tr), deg);
    tot += mseS(pick(d.ys, v), pick(d.xs, v).map((x) => polyEvalS(fit.coef, x)));
  }
  assert.ok(Math.abs(cv.scores.find((r) => r.degree === deg).cv - tot / 5) < 1e-12);
  // The test score is the chosen degree refitted on all the training points.
  assert.ok(Math.abs(cv.test - lesson9.score(d, sp, cv.best).test) < 1e-12);
});

test('lesson9 read-out rows and page', () => {
  const html = readFileSync(new URL('../train-test/index.html', import.meta.url), 'utf8');
  for (let step = 0; step < 5; step++) {
    for (const seed of [3, 4, 11, 40, 150]) {
      for (const r of lesson9.readout(step, { seed })) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 5);
  assert.match(html, /<title>Train, validate, test · ML Explorer<\/title>/);
  assert.match(html, /seed 3/);
  assert.match(html, /href="\/ml\/overfitting\/"/);
  const pv = lesson9.pageValues(lesson9.SPLIT_SEED);
  for (const [, key] of html.matchAll(/data-value="([^"]+)"/g)) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
  assert.equal(lesson9.stepOneScores(lesson9.lessonData()).dots < 1e-24, true);
});

// ---------------------------------------------------------------- lesson 10
import { lstsq as lstsq10, polyDesign as polyDesign10, mean as mean10, variance as variance10 } from '../assets/maths.js';
import * as lesson10 from '../overfitting/stage.js';

test('lesson10 data: 15 training points (seed 3) and 200 test points (seed 4) from the shared generator', () => {
  assert.deepEqual(lesson10.trainData(), sineData({ n: 15, seed: 3 }));
  assert.deepEqual(lesson10.testData(), sineData({ n: 200, seed: 4 }));
});

test('lesson10 train MSE non-increasing in degree', () => {
  const rows = lesson10.degreeTable();
  assert.deepEqual(rows.map((r) => r.degree), Array.from({ length: 13 }, (_, p) => p));
  for (let p = 1; p < rows.length; p++) assert.ok(rows[p].train <= rows[p - 1].train + 1e-12, `degree ${p}: ${rows[p].train} > ${rows[p - 1].train}`);
  // 13 numbers from 15 points: degree 12 does not pass through every point.
  assert.ok(rows[12].train > 1e-3, String(rows[12].train));
  const tr = lesson10.trainData();
  const te = lesson10.testData();
  const fit = polyFitS(tr.xs, tr.ys, 3);
  assert.ok(Math.abs(rows[3].train - mseS(tr.ys, tr.xs.map((x) => polyEvalS(fit.coef, x)))) < 1e-12);
  assert.ok(Math.abs(rows[3].test - mseS(te.ys, te.xs.map((x) => polyEvalS(fit.coef, x)))) < 1e-12);
});

test('lesson10 test MSE at degree 12 exceeds its minimum over degrees', () => {
  const rows = lesson10.degreeTable();
  const min = Math.min(...rows.map((r) => r.test));
  assert.ok(rows[12].test > min * 10, `${rows[12].test} vs ${min}`);
  const best = lesson10.bestDegree();
  assert.equal(rows[best].test, min);
  assert.ok(best > 1 && best < 12, String(best));
  // Too simple is worse than the best as well.
  assert.ok(rows[0].test > min && rows[1].test > min);
  // On this data the test curve is not a smooth U: past the best degree it
  // goes up and down again before it climbs.
  assert.ok(rows.some((r, p) => p > best && p < 12 && rows[p + 1].test < r.test));
});

test('lesson10 ridge at degree 12 lowers test MSE versus lambda 1e-6', () => {
  const rows = lesson10.ridgeTable();
  assert.equal(rows.length, lesson10.SLIDER_MAX + 1);
  assert.equal(rows[0].lambda, 1e-6);
  assert.equal(rows[rows.length - 1].lambda, 10);
  const base = rows[0].test;
  assert.ok(rows.some((r) => r.test < base / 10), 'no λ on the slider lowers the test MSE');
  const best = rows.reduce((a, b) => (b.test < a.test ? b : a));
  assert.ok(best.lambda > 1e-6 && best.lambda < 10);
  // Large λ squeezes the slopes towards 0: the curve flattens and the training MSE rises.
  assert.ok(rows[rows.length - 1].train > rows[0].train);
  assert.ok(rows[rows.length - 1].maxAbs < rows[0].maxAbs);
  // The training MSE never falls as λ grows.
  for (let k = 1; k < rows.length; k++) assert.ok(rows[k].train >= rows[k - 1].train - 1e-12, `t = ${k}`);
});

test('lesson10 ridge objective matches lstsq', () => {
  const { xs, ys } = lesson10.trainData();
  const n = xs.length;
  const X = polyDesign10(xs, 12);
  for (let t = 0; t <= lesson10.SLIDER_MAX; t += 5) {
    const lam = lesson10.lambdaFromSlider(t);
    const coef = lesson10.ridgeFit(xs, ys, 12, lam);
    // Gradient of (1/n)Σeᵢ² + λ·Σ_{j≥1}θⱼ²; the intercept θ₀ is not penalised.
    const e = X.map((row, i) => ys[i] - row.reduce((s, v, j) => s + v * coef[j], 0));
    const g = coef.map((c, j) => (-2 / n) * X.reduce((s, row, i) => s + row[j] * e[i], 0) + (j >= 1 ? 2 * lam * c : 0));
    for (const gj of g) assert.ok(Math.abs(gj) < 1e-6, `λ = ${lam}: gradient ${gj}`);
    assert.ok(Math.abs(lesson10.ridgeObjective(coef, xs, ys, lam) - (mean10(e.map((v) => v * v)) + lam * coef.slice(1).reduce((s, c) => s + c * c, 0))) < 1e-12);
    // Nudging any coefficient raises the stated objective.
    for (let j = 0; j < coef.length; j++) {
      for (const h of [-1e-3, 1e-3]) {
        const c2 = coef.slice();
        c2[j] += h;
        assert.ok(lesson10.ridgeObjective(c2, xs, ys, lam) > lesson10.ridgeObjective(coef, xs, ys, lam));
      }
    }
  }
  // Passing λ itself to lstsq (a sum, not a mean) would give a different fit.
  const lam = 1e-2;
  const wrong = lstsq10(X, ys, { lambda: lam }).coef;
  assert.ok(Math.abs(lesson10.ridgeFit(xs, ys, 12, lam)[1] - wrong[1]) > 1e-3);
});

test('lesson10 λ slider: log scale from 1e-6 to 10, and the shown value agrees with λ', () => {
  assert.equal(lesson10.lambdaFromSlider(0), 1e-6);
  assert.equal(lesson10.lambdaFromSlider(lesson10.SLIDER_MAX), 10);
  let prev = 0;
  for (let t = 0; t <= lesson10.SLIDER_MAX; t++) {
    const lam = lesson10.lambdaFromSlider(t);
    assert.ok(lam > prev);
    prev = lam;
    if (t > 0) assert.ok(Math.abs(Math.log10(lam / lesson10.lambdaFromSlider(t - 1)) - 0.1) < 1e-9);
    const shown = lesson10.lambdaValue(lesson10.lambdaText(lam));
    // Two significant figures are within 5% of the value.
    assert.ok(Math.abs(shown / lam - 1) < 0.05, `t = ${t}: ${lesson10.lambdaText(lam)} vs ${lam}`);
    assert.ok(!/NaN|undefined|Infinity|e/.test(lesson10.lambdaText(lam)), lesson10.lambdaText(lam));
  }
  assert.equal(lesson10.lambdaText(1e-6), '1.0 × 10⁻⁶');
  assert.equal(lesson10.lambdaText(10), '10');
  // The log-axis labels name the right powers of ten.
  assert.deepEqual([-2, -1, 0, 1, 2, 4].map(lesson10.logTick), ['0.01', '0.1', '1', '10', '10²', '10⁴']);
});

test('lesson10 bias² and variance: corrected for the noisy average, shown without false precision', () => {
  const M = lesson10.BV_SETS;
  for (const p of lesson10.BV_DEGREES) {
    const bv = lesson10.biasVariance(p);
    const f0 = Math.sin(Math.PI * lesson10.X0);
    const preds = bv.preds;
    assert.equal(preds.length, M);
    const m = mean10(preds);
    assert.ok(Math.abs(bv.gap2 - (f0 - m) ** 2) < 1e-12);
    assert.ok(Math.abs(bv.variance - variance10(preds, { sample: false })) < 1e-12);
    // The mean squared deviation from f splits exactly into gap² + variance.
    const direct = mean10(preds.map((v) => (f0 - v) ** 2));
    assert.ok(Math.abs(direct - (bv.gap2 + bv.variance)) <= 1e-9 * Math.max(1, direct));
    // E[(f − mean)²] = bias² + Var/M, so the estimate subtracts Var/M, floored at 0.
    assert.ok(Math.abs(bv.bias2 - Math.max(0, bv.gap2 - bv.variance / M)) < 1e-15);
    assert.equal(bv.measurable, bv.gap2 > 2 * bv.variance / M);
    assert.equal(bv.sigma2, 0.0625);
  }
  const lo = lesson10.biasVariance(1);
  const mid = lesson10.biasVariance(3);
  const hi = lesson10.biasVariance(7);
  assert.ok(lo.measurable && lo.bias2 > 0.1, 'degree 1 should clearly miss the bend');
  assert.equal(lesson10.fmtBias(lo), '0.26');
  // Degrees 3 and 7: the squared gap is within the noise of the average.
  assert.equal(mid.measurable, false);
  assert.equal(hi.measurable, false);
  assert.equal(lesson10.fmtBias(mid, true), 'too small to measure with 200 sets');
  // Variance: two significant figures below 1, one ("about") from 1 up.
  assert.equal(lesson10.fmtVar(hi.variance), 'about 2');
  assert.equal(lesson10.fmtVar(mid.variance), '0.017');
  // Degree 7's variance is far above degree 3's in every block of seeds,
  // and its spread across blocks is what the page quotes.
  for (const b of lesson10.BV_BLOCKS) {
    const v7 = lesson10.biasVariance(7, b).variance;
    const v3 = lesson10.biasVariance(3, b).variance;
    assert.ok(v7 > 10 * v3, `block ${b}: ${v7} vs ${v3}`);
  }
  assert.ok([301, 501].every((b) => lesson10.BV_BLOCKS.includes(b)));
  const sp = lesson10.varianceSpread(7);
  const pv = lesson10.pageValues();
  assert.equal(pv['bv7-var-lo'], '0.5');
  assert.equal(pv['bv7-var-hi'], '9');
  assert.ok(Math.abs(sp.lo - 0.5) < 0.05 && Math.abs(sp.hi - 9) < 0.5);
  // Degree 12's fits spread further still (the page says so).
  assert.ok(lesson10.biasVariance(12).variance > hi.variance * 100);
  // Step 3's framing: the widest gap between training x values sits near −0.2,
  // and nearly all of degree 12's test error comes from past the last training point.
  assert.equal(pv['gap-mid'], '−0.2');
  assert.equal(pv['d12-maxabs'], '35,000');
  const c12 = lesson10.degreeTable()[12].coef;
  const te = lesson10.testData();
  const xmax = Math.max(...lesson10.trainData().xs);
  const sq = te.xs.map((x, i) => [x, (te.ys[i] - polyEvalS(c12, x)) ** 2]);
  const total = sq.reduce((t, [, e]) => t + e, 0);
  const right = sq.filter(([x]) => x > xmax).reduce((t, [, e]) => t + e, 0);
  assert.ok(right / total > 0.9, String(right / total));
});

test('lesson10 read-out rows and page', () => {
  const html = readFileSync(new URL('../overfitting/index.html', import.meta.url), 'utf8');
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { degree: 0 }, { degree: 7 }, { degree: 12 }, { t: 0 }, { t: 35 }, { t: lesson10.SLIDER_MAX }, { bv: 1 }, { bv: 7 }]) {
      for (const r of lesson10.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Overfitting and regularisation · ML Explorer<\/title>/);
  assert.match(html, /seed 3/);
  assert.match(html, /seed 4/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/">/);
  assert.doesNotMatch(html, /through every point/);
  const pv = lesson10.pageValues();
  for (const [, key] of html.matchAll(/data-value="([^"]+)"/g)) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});
