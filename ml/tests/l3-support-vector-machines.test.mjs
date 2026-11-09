import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dot } from '../assets/maths.js';
import { blobs2 } from '../assets/classify.js';
import { svmFit, kktViolation } from '../assets/svm.js';
import * as L from '../support-vector-machines/stage.js';

const html = readFileSync(new URL('../support-vector-machines/index.html', import.meta.url), 'utf8');
const yf = (f, X, y) => X.map((x, i) => y[i] * (dot(f.w, x) + f.b));
const close = (u, v, tol = 1e-9) => assert.ok(Math.abs(u - v) <= tol, `${u} vs ${v}`);

// No direction splits the classes: the gap min₊ n·x − max₋ n·x is below zero at
// every angle. The gap changes by at most 2R per radian, so a grid of step h
// proves it when every grid gap is below −R·h.
function provablyNotSeparable(X, y, steps = 36000) {
  const R = Math.max(...X.map((x) => Math.hypot(x[0], x[1])));
  const h = Math.PI / steps;
  for (let k = 0; k < 2 * steps; k++) {
    const n = [Math.cos(k * h), Math.sin(k * h)];
    let lo = Infinity;
    let hi = -Infinity;
    X.forEach((x, i) => {
      const p = dot(n, x);
      if (y[i] === 1) lo = Math.min(lo, p); else hi = Math.max(hi, p);
    });
    if (lo - hi >= -R * h) return false;
  }
  return true;
}

test('svm data reproducible, labels −1 and +1', () => {
  const s = L.sepData();
  assert.deepEqual(s.X, blobs2({ n0: 12, n1: 12, mu0: [-2, -1.5], mu1: [2, 1.5], sd: 0.8, seed: 71 }).X);
  const o = L.overData();
  assert.deepEqual(o.X, blobs2({ n0: 12, n1: 12, mu0: [-2, -1.5], mu1: [2, 1.5], sd: 1.4, seed: L.SEED_OVER }).X);
  assert.equal(L.SEED_SEP, 71);
  assert.equal(L.SEED_OVER, 77);
  for (const d of [s, o]) {
    assert.equal(d.X.length, 24);
    assert.deepEqual(d.y, d.cls.map((c) => (c === 1 ? 1 : -1)));
    assert.equal(d.y.filter((v) => v === 1).length, 12);
  }
  const c = L.circleData();
  assert.equal(c.X.length, 40);
  c.X.forEach((x, i) => {
    const r = Math.hypot(x[0], x[1]);
    if (i < 20) { assert.equal(c.y[i], -1); assert.ok(r < 1); } else { assert.equal(c.y[i], 1); assert.ok(r > 1.6 && r < 2.4); }
  });
  assert.deepEqual(L.circleData(), c);
  assert.deepEqual(L.lift([1, 2]), [1, 2, 5]);
});

test('separable data has a hard margin', () => {
  const { X, y } = L.sepData();
  const f = L.fitSep();
  assert.ok(f.ok);
  assert.ok(f.kkt < 1e-6);
  // Every point on or outside the street, and no α at its cap: the hard margin.
  for (const v of yf(f, X, y)) assert.ok(v >= 1 - 1e-6, String(v));
  for (const a of f.alpha) assert.ok(a < L.C_HARD - 1e-6);
  // Any larger C gives the same answer.
  const big = svmFit(X, y, { C: 1e6 });
  close(big.w[0], f.w[0], 1e-7);
  close(big.w[1], f.w[1], 1e-7);
  close(big.b, f.b, 1e-7);
  close(f.margin, 2 / Math.hypot(...f.w), 1e-12);
  assert.equal(f.margin.toFixed(2), '3.61');
  assert.deepEqual(f.sv, [6, 11, 15]);
  // The widest street over every tilt (0.01° grid) matches the solver. The
  // width changes by at most 2R per radian of tilt, so the grid's best is within 2R·h.
  const scan = L.bestTilt(0.01);
  const R = Math.max(...X.map((x) => Math.hypot(x[0], x[1])));
  assert.ok(f.margin - scan.width < 2 * R * (0.01 * Math.PI / 180), `${scan.width} vs ${f.margin}`);
  assert.ok(scan.width <= f.margin + 1e-9);
});

test('support vectors lie on the margin lines', () => {
  for (const [X, y, f, C] of [
    [L.sepData().X, L.sepData().y, L.fitSep(), L.C_HARD],
    ...L.C_GRID.map((c) => [L.overData().X, L.overData().y, L.fitOver(c), c]),
  ]) {
    const v = yf(f, X, y);
    f.alpha.forEach((a, i) => {
      if (a > 1e-8 && a < C - 1e-8) assert.ok(Math.abs(v[i] - 1) < 1e-6, `C ${C} point ${i}: ${v[i]}`);
      if (a <= 0) assert.ok(v[i] >= 1 - 1e-6);
      if (a >= C) assert.ok(v[i] <= 1 + 1e-6);
    });
  }
});

test('margin shrinks as C grows', () => {
  assert.deepEqual(L.C_GRID, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]);
  const m = L.C_GRID.map((c) => L.fitOver(c).margin);
  for (let k = 1; k < m.length; k++) assert.ok(m[k] <= m[k - 1] + 1e-9, `${L.C_GRID[k]}: ${m[k]} > ${m[k - 1]}`);
  // Both ends of the slider meet KKT within 1e-6.
  const { X, y } = L.overData();
  for (const c of [L.C_GRID[0], L.C_GRID[L.C_GRID.length - 1]]) {
    const f = L.fitOver(c);
    assert.ok(f.ok, String(c));
    assert.ok(f.kkt < 1e-6, String(c));
    assert.ok(kktViolation(X, y, c, f.alpha, f.b) < 1e-6);
  }
  for (const c of L.C_GRID) assert.ok(L.fitOver(c).kkt < 1e-6, String(c));
  // The overlapping data really overlap: no straight line splits them.
  assert.ok(provablyNotSeparable(X, y));
  // Violations never rise with C here; the story numbers.
  const st = L.C_GRID.map((c) => L.overStats(c));
  for (let k = 1; k < st.length; k++) assert.ok(st[k].violations <= st[k - 1].violations);
  assert.equal(st[0].margin.toFixed(2), '6.66');
  assert.equal(st[0].violations, 22);
  assert.equal(st.at(-1).margin.toFixed(2), '1.59');
  assert.equal(st.at(-1).violations, 4);
  assert.equal(st.at(-1).wrong, 1);
  for (const s of st) assert.ok(s.wrong >= 1);
  // Hinge total equals Σ max(0, 1 − y f).
  const f = L.fitOver(1);
  close(st[6].hinge, yf(f, X, y).reduce((a, v) => a + Math.max(0, 1 - v), 0), 1e-12);
});

test('lifted circle data is linearly separable', () => {
  const { X, y } = L.circleData();
  const Z = X.map(L.lift);
  const f = L.fitLifted();
  assert.ok(f.ok);
  assert.ok(f.kkt < 1e-6);
  assert.equal(f.w.length, 3);
  for (const v of yf(f, Z, y)) assert.ok(v >= 1 - 1e-6);
  for (const a of f.alpha) assert.ok(a < L.C_HARD - 1e-6);
  const big = svmFit(Z, y, { C: 1e6 });
  close(big.margin, f.margin, 1e-7);
  // The slider's ends on the lifted data too.
  for (const c of [0.01, 100]) {
    const g = svmFit(Z, y, { C: c });
    assert.ok(g.ok && g.kkt < 1e-6, String(c));
  }
  // In two dimensions no straight line splits them.
  assert.ok(provablyNotSeparable(X, y));
  const lin = L.fitCircleLine();
  assert.ok(lin.ok && lin.kkt < 1e-6);
  assert.equal(L.pct(L.circleLineAcc()), '75.0%');
  // The flat plane, seen back in the plane of x₁, x₂, is a circle: f < 0 inside, > 0 outside.
  const c = L.circleBoundary();
  assert.ok(c.r > 1 && c.r < 1.6);
  for (const t of [0, 1, 2, 3, 4, 5]) {
    const p = (rr) => [c.cx + rr * Math.cos(t), c.cy + rr * Math.sin(t)];
    close(dot(f.w, L.lift(p(c.r))) + f.b, 0, 1e-9);
    close(dot(f.w, L.lift(p(c.rIn))) + f.b, -1, 1e-9);
    close(dot(f.w, L.lift(p(c.rOut))) + f.b, 1, 1e-9);
  }
  // The side view sees the plane edge-on: f depends only on (u, z).
  Z.forEach((z, i) => {
    const [u, h] = L.sideView(X[i]);
    close(L.sideF(u, h), dot(f.w, z) + f.b, 1e-9);
  });
});

test('moving or removing a point that is not a support vector changes nothing', () => {
  const { X, y } = L.sepData();
  const f = L.fitSep();
  assert.ok(!f.sv.includes(L.POINT_A));
  assert.ok(f.sv.includes(L.POINT_B));
  // Move A further from the line, and somewhere else outside the street.
  for (const to of [[X[L.POINT_A][0] + 1, X[L.POINT_A][1] + 1], [3.5, 0.5]]) {
    const g = L.dragFit({ [L.POINT_A]: to });
    assert.ok(y[L.POINT_A] * (dot(f.w, to) + f.b) > 1);
    close(g.w[0], f.w[0], 1e-6);
    close(g.w[1], f.w[1], 1e-6);
    close(g.b, f.b, 1e-6);
    assert.equal(L.lineMoved(g), false);
  }
  // Removing any non-support point: the same answer.
  for (let i = 0; i < X.length; i++) {
    if (f.sv.includes(i)) continue;
    const g = svmFit(X.filter((_, k) => k !== i), y.filter((_, k) => k !== i), { C: L.C_HARD });
    close(g.w[0], f.w[0], 1e-6);
    close(g.w[1], f.w[1], 1e-6);
    close(g.b, f.b, 1e-6);
  }
  // Moving the support vector B away from the line moves the line.
  const xb = X[L.POINT_B];
  const g = L.dragFit({ [L.POINT_B]: [xb[0] + 0.5, xb[1] + 0.5] });
  assert.equal(L.lineMoved(g), true);
  assert.ok(g.margin > f.margin);
});

test('dragged fits converge everywhere on the plot', () => {
  // A and B dragged over a grid covering the plot: each fit meets KKT, never NaN.
  for (const idx of [L.POINT_A, L.POINT_B]) {
    for (let a = L.X_DOM[0]; a <= L.X_DOM[1]; a += 0.5) {
      for (let b = L.Y_DOM[0]; b <= L.Y_DOM[1]; b += 0.5) {
        const g = L.dragFit({ [idx]: [a, b] });
        assert.ok(g.ok && g.kkt < 1e-6, `${idx} at ${a}, ${b}`);
        for (const r of L.readout(3, { fit: g })) assert.ok(!/NaN|undefined|Infinity|null/.test(r), r);
      }
    }
  }
  // Both on top of each other.
  const g = L.dragFit({ [L.POINT_A]: [0, 0], [L.POINT_B]: [0, 0] });
  assert.ok(Number.isFinite(g.b));
});

test('candidate line: distances, sides and the street', () => {
  const { X, y } = L.sepData();
  const c = L.candidate(L.START.tilt, L.START.shift);
  assert.equal(c.wrong, 0);
  // Distances are the perpendicular distances to the line.
  const n = [Math.sin(L.START.tilt * Math.PI / 180), -Math.cos(L.START.tilt * Math.PI / 180)];
  const d = X.map((x) => Math.abs(dot(n, x) - L.START.shift));
  close(c.d0, Math.min(...d.filter((_, i) => y[i] === -1)), 1e-12);
  close(c.d1, Math.min(...d.filter((_, i) => y[i] === 1)), 1e-12);
  close(c.width, 2 * Math.min(c.d0, c.d1), 1e-12);
  assert.ok(c.width < L.fitSep().margin);
  // Centred, this tilt gives d0 + d1, still narrower than the widest street.
  close(c.centred, c.d0 + c.d1, 1e-12);
  assert.ok(c.centred < L.fitSep().margin);
  // A line through the middle of the circles does not separate.
  const bad = L.candidate(90, 2);
  assert.ok(bad.wrong > 0);
  assert.equal(bad.width, null);
  for (let t = 0; t < 180; t += 15) {
    for (let s = -3; s <= 3; s += 1.5) {
      for (const st of [0, 1]) for (const r of L.readout(st, { tilt: t, shift: s })) assert.ok(!/NaN|undefined|Infinity|null/.test(r), r);
    }
  }
});

test('svm read-out rows and page', () => {
  for (let step = 0; step < 6; step++) {
    for (const st of [{}, { tilt: 0, shift: -3 }, { tilt: 179, shift: 3 }, { cIdx: 0 }, { cIdx: 12 }, { lifted: true }]) {
      for (const r of L.readout(step, st)) {
        assert.ok(!/NaN|undefined|Infinity|null/.test(r), r);
        assert.ok(r.length <= 24, r);
      }
    }
  }
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Support vector machines · ML Explorer<\/title>/);
  assert.match(html, /L3 · Classic ML/);
  assert.match(html, /seed 71/);
  assert.match(html, /seed 77/);
  assert.match(html, /seed 73/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/decision-trees\/">/);
  assert.match(html, /href="\/ml\/naive-bayes\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /static\.cloudflareinsights\.com\/beacon\.min\.js/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  const pv = L.pageValues();
  const keys = [...html.matchAll(/data-value="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(keys.length >= 10);
  for (const key of keys) assert.ok(key in pv, key);
  for (const v of Object.values(pv)) assert.ok(!/NaN|undefined|Infinity|null/.test(v), v);
  // Every value has its pre-filled copy in the page, equal to the computed one.
  for (const m of html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)) assert.equal(m[2], pv[m[1]], m[1]);
});

test('svm page wording holds', () => {
  const pv = L.pageValues();
  assert.equal(pv['sep-width'], '3.61');
  assert.equal(pv['sep-sv'], '3');
  assert.equal(pv['c-lo-width'], '6.66');
  assert.equal(pv['c-lo-viol'], '22');
  assert.equal(pv['c-hi-width'], '1.59');
  assert.equal(pv['c-hi-viol'], '4');
  assert.equal(pv['circ-line-acc'], '75.0%');
  assert.equal(pv['circ-line-right'], '30');
  assert.equal(pv['sep-sv0'], '2');
  assert.equal(pv['sep-sv1'], '1');
  assert.equal(pv['sep-nonsv'], '21');
  assert.equal(pv['sep-tilt'], '144.4');
  // "nothing changes above C = 10": the answer at 10 equals every larger C, and 5 differs.
  assert.equal(pv['c-flat'], '10');
  for (const c of [20, 50, 100]) {
    for (const j of [0, 1]) close(L.fitOver(c).w[j], L.fitOver(10).w[j], 1e-6);
    close(L.fitOver(c).b, L.fitOver(10).b, 1e-6);
  }
  assert.ok(Math.abs(L.fitOver(5).margin - L.fitOver(10).margin) > 0.1);
  // "The αᵢ still grow with C": the total α rises from 10 to 20, 50 and 100.
  const sumA = (c) => L.fitOver(c).alpha.reduce((t, v) => t + v, 0);
  for (const [lo, hi] of [[10, 20], [20, 50], [50, 100]]) assert.ok(sumA(hi) > sumA(lo) + 1);
  // "circles below 1, triangles above 2.56"; "1.65 + 0.99" add to the centred width.
  const c = L.circleData();
  c.X.forEach((x, i) => { const z = L.lift(x)[2]; assert.ok(c.y[i] === -1 ? z < 1 : z > 2.56); });
  assert.equal((Number(pv['start-d0']) + Number(pv['start-d1'])).toFixed(2), pv['start-centred']);
  // y ∈ {−1, +1} is stated, the kernel trick and RBF are named in The maths.
  assert.match(html, /y = −1/);
  assert.match(html, /y = \+1/);
  assert.match(html, /kernel trick/);
  assert.match(html, /radial basis function/);
  // ISLR's C is a budget; the page warns that it runs the other way.
  assert.match(html, /James, Witten, Hastie &amp; Tibshirani,/);
  assert.doesNotMatch(html, /Taylor/);
  assert.match(html, /budget/);
  // The C used for steps 3 and 4.
  assert.equal(L.C_HARD, 1);
  assert.equal(Math.max(...L.fitSep().alpha) < 0.2, true);
});
