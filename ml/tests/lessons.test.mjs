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
