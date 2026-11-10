import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, randn } from '../assets/maths.js';
import { conv2d, convOutSize } from '../assets/conv.js';
import * as C from '../convolutions/stage.js';

const html = readFileSync(new URL('../convolutions/index.html', import.meta.url), 'utf8');

// The image written out again: 12 × 12, background 0; a 4 × 4 square of 1s in
// rows 2–5, columns 2–5; a diagonal line of 0.5s at r + c = 15 for rows 6–10.
function image() {
  const img = [];
  for (let r = 0; r < 12; r++) {
    const row = [];
    for (let c = 0; c < 12; c++) {
      if (r >= 2 && r <= 5 && c >= 2 && c <= 5) row.push(1);
      else if (r >= 6 && r <= 10 && r + c === 15) row.push(0.5);
      else row.push(0);
    }
    img.push(row);
  }
  return img;
}
// The noisy copy: one randn per pixel, row-major, from mulberry32(191), sd 0.1, clipped to [0, 1].
function noisy() {
  const rng = mulberry32(191);
  return image().map((row) => row.map((v) => Math.min(1, Math.max(0, v + 0.1 * randn(rng)))));
}
const LITERAL_NOISY_0_4 = 0.020762272518980194;

const VERT = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
const cellsOf = (map, pick) => {
  const vals = map.flat();
  const best = pick === 'max' ? Math.max(...vals) : Math.min(...vals);
  const cells = [];
  map.forEach((row, i) => row.forEach((v, j) => { if (v === best) cells.push([i, j]); }));
  return { best, cells };
};

test('data reproducible', () => {
  assert.equal(C.SEED_NOISE, 191);
  assert.equal(C.NOISE_SD, 0.1);
  assert.equal(C.N, 12);
  assert.deepEqual(C.cleanImage(), image());
  assert.deepEqual(C.noisyImage(), noisy());
  assert.equal(C.noisyImage()[0][4], LITERAL_NOISY_0_4);
  assert.equal(C.noisyImage()[0][4], 0.020762272518980194);
  for (const row of C.noisyImage()) for (const v of row) assert.ok(v >= 0 && v <= 1);
  assert.deepEqual(C.FILTERS.vertical, VERT);
  assert.deepEqual(C.FILTERS.horizontal, [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]);
  for (const row of C.FILTERS.blur) for (const v of row) assert.equal(v, 1 / 9);
});

test('output sizes match the formula for every control setting', () => {
  const want = { '1,0': 10, '2,0': 5, '1,1': 12, '2,1': 6 };
  for (const s of C.STRIDES) {
    for (const p of C.PADS) {
      const m = Math.floor((12 + 2 * p - 3) / s) + 1;
      assert.equal(m, want[`${s},${p}`]);
      assert.equal(convOutSize(12, 3, s, p), m);
      for (const img of [C.cleanImage(), C.noisyImage()]) {
        for (const W of Object.values(C.FILTERS)) {
          const out = C.mapFor(img, W, { stride: s, pad: p });
          assert.equal(out.length, m);
          for (const row of out) assert.equal(row.length, m);
        }
      }
      const info = C.sizeInfo(s, p);
      assert.equal(info.formula, m);
      assert.equal(info.actual, m);
    }
  }
  assert.deepEqual(C.STRIDES, [1, 2]);
  assert.deepEqual(C.PADS, [0, 1]);
});

test('vertical-edge map is largest on the square\'s vertical edges', () => {
  const map = C.mapFor(C.cleanImage(), C.FILTERS.vertical);
  const hi = cellsOf(map, 'max');
  const lo = cellsOf(map, 'min');
  assert.equal(hi.best, 3);
  assert.deepEqual(hi.cells, [[2, 0], [2, 1], [3, 0], [3, 1]]);
  assert.equal(lo.best, -3);
  assert.deepEqual(lo.cells, [[2, 4], [2, 5], [3, 4], [3, 5]]);
  // Each of those windows has all three rows in the square and straddles an edge:
  // left edge between columns 1 and 2, right edge between 5 and 6.
  for (const [i, j] of hi.cells) {
    assert.ok(i >= 2 && i + 2 <= 5);
    assert.ok(j <= 1 && j + 2 >= 2);
  }
  for (const [i, j] of lo.cells) {
    assert.ok(i >= 2 && i + 2 <= 5);
    assert.ok(j <= 5 && j + 2 >= 6);
  }
  // Windows away from the square (no square pixel inside) only see the line: |value| ≤ 0.5.
  assert.equal(C.awayFromSquare(map), 0.5);
  // The same on the noisy copy: the largest value is still on the left edge.
  const nmap = C.mapFor(C.noisyImage(), C.FILTERS.vertical);
  const nhi = cellsOf(nmap, 'max');
  assert.deepEqual(nhi.cells, [[3, 0]]);
  assert.ok(Math.abs(nhi.best - 2.9494806687401063) < 1e-12);
  const nlo = cellsOf(nmap, 'min');
  assert.deepEqual(nlo.cells, [[2, 4]]);
});

test('shifting the image shifts the map', () => {
  // The page's image: no pixel falls off the image for shifts of up to 2. Every map value
  // that stays on the map moves exactly d columns right; columns j < d of the shifted map
  // are 0; the original map's last d columns slide out of view (some of them are not 0).
  for (const d of C.SHIFTS) {
    const a = C.mapFor(C.cleanImage(), VERT);
    const b = C.mapFor(C.shiftRight(C.cleanImage(), d), VERT);
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        if (j >= d) assert.equal(b[i][j], a[i][j - d], `${d} ${i} ${j}`);
        else assert.equal(b[i][j], 0);
      }
    }
    const kept = C.cleanImage().flat().reduce((t, v) => t + v, 0);
    assert.equal(C.shiftRight(C.cleanImage(), d).flat().reduce((t, v) => t + v, 0), kept, 'no pixel pushed off');
    if (d === 2) {
      const lost = [];
      a.forEach((row, i) => row.forEach((v, j) => { if (j >= 10 - d && v !== 0) lost.push([i, j]); }));
      assert.deepEqual(lost, [[4, 9], [5, 8], [5, 9], [6, 8], [6, 9], [7, 8]]);
    }
    const hb = cellsOf(b, 'max');
    assert.deepEqual(hb.cells, [[2, d], [2, 1 + d], [3, d], [3, 1 + d]]);
  }
  assert.deepEqual(C.SHIFTS, [0, 1, 2]);
  // Any image, away from the borders: map columns j ≥ d (stride 1, padding 0 or 1)
  // equal the unshifted map d columns to the left. At the right edge it can fail.
  const rng = mulberry32(7);
  const rand = Array.from({ length: 12 }, () => Array.from({ length: 12 }, () => rng()));
  const W = [[0.5, -1, 2], [0, 1.5, -0.5], [1, 0, -2]];
  for (const pad of [0, 1]) {
    for (const d of [1, 2]) {
      const a = C.mapFor(rand, W, { pad });
      const b = C.mapFor(C.shiftRight(rand, d), W, { pad });
      const m = a.length;
      for (let i = 0; i < m; i++) {
        for (let j = d; j < m; j++) {
          // Interior: the shifted window holds only pixels that moved, none lost off the right.
          if (j + 2 - pad <= 11) assert.ok(Math.abs(b[i][j] - a[i][j - d]) < 1e-12, `${pad} ${d} ${i} ${j}`);
        }
      }
      // With padding 1 the last column differs: its window lost a column off the right.
      if (pad === 1) assert.ok(Math.abs(b[0][m - 1] - a[0][m - 1 - d]) > 1e-6);
    }
  }
});

test('editable filter output equals conv2d', () => {
  assert.equal(C.parseWeight('1'), 1);
  assert.equal(C.parseWeight('-0.5'), -0.5);
  assert.equal(C.parseWeight('−1.5'), -1.5);
  assert.equal(C.parseWeight('3'), 2);
  assert.equal(C.parseWeight('-7'), -2);
  assert.equal(C.parseWeight('0.7'), 0.5);
  assert.equal(C.parseWeight('0.8'), 1);
  assert.equal(C.parseWeight('-0.2'), 0);
  assert.equal(C.parseWeight(''), 0);
  assert.equal(C.parseWeight('abc'), 0);
  assert.equal(C.parseWeight('NaN'), 0);
  assert.equal(C.parseWeight('Infinity'), 2);
  assert.equal(C.parseWeight(undefined), 0);
  assert.ok(Object.is(C.parseWeight('-0.1'), 0));
  // Ties are snapped away from 0 on both sides.
  assert.equal(C.parseWeight('0.25'), 0.5);
  assert.equal(C.parseWeight('-0.25'), -0.5);
  assert.equal(C.parseWeight('0.75'), 1);
  assert.equal(C.parseWeight('-0.75'), -1);
  assert.equal(C.parseWeight('1.75'), 2);
  assert.equal(C.parseWeight('-1.75'), -2);
  for (let k = -40; k <= 40; k++) {
    const v = k / 10;
    assert.equal(C.parseWeight(String(v)), -C.parseWeight(String(-v)) || 0, String(v));
  }
  const rng = mulberry32(11);
  for (let t = 0; t < 20; t++) {
    const raw = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => String((rng() * 6 - 3).toFixed(2))));
    const W = C.parseFilter(raw);
    for (const row of W) for (const w of row) assert.ok(w >= -2 && w <= 2 && Number.isInteger(w * 2));
    for (const img of [C.cleanImage(), C.noisyImage()]) {
      for (const s of C.STRIDES) for (const p of C.PADS) {
        assert.deepEqual(C.mapFor(img, W, { stride: s, pad: p }), conv2d(img, W, { stride: s, pad: p }));
      }
    }
  }
  // The suggested line filter: 1 from top right to bottom left, −0.5 elsewhere.
  assert.deepEqual(C.TRY_LINE, [[-0.5, -0.5, 1], [-0.5, 1, -0.5], [1, -0.5, -0.5]]);
  for (const row of C.TRY_LINE) for (const w of row) assert.equal(C.parseWeight(String(w)), w);
  const lm = C.mapFor(C.cleanImage(), C.TRY_LINE);
  assert.equal(Math.max(...lm.flat()), 1.5);
  assert.equal(C.lineVsSquare(lm).line, 1.5);
  assert.equal(C.lineVsSquare(lm).square, 1);
});

test('window products sum to the map value at every position', () => {
  for (const img of [C.cleanImage(), C.noisyImage()]) {
    for (const s of C.STRIDES) for (const p of C.PADS) {
      const map = C.mapFor(img, VERT, { stride: s, pad: p });
      for (let i = 0; i < map.length; i++) for (let j = 0; j < map.length; j++) {
        const w = C.windowCells(img, VERT, i, j, { stride: s, pad: p });
        assert.equal(w.cells.length, 9);
        let sum = 0;
        for (const k of w.cells) {
          const inside = k.r >= 0 && k.r < 12 && k.c >= 0 && k.c < 12;
          assert.equal(k.pad, !inside);
          assert.equal(k.x, inside ? img[k.r][k.c] : 0);
          assert.equal(k.prod, k.x * k.w);
          sum += k.prod;
        }
        assert.ok(Math.abs(w.sum - map[i][j]) < 1e-12);
        assert.ok(Math.abs(sum - map[i][j]) < 1e-12);
        assert.equal(w.cells[0].r, i * s - p);
        assert.equal(w.cells[0].c, j * s - p);
      }
    }
  }
});

test('every negative map value that reads as negative carries a minus sign', () => {
  for (const img of [C.cleanImage(), C.noisyImage()]) {
    for (const W of [...Object.values(C.FILTERS), C.TRY_LINE]) {
      for (const v of C.mapFor(img, W).flat()) {
        assert.equal(C.minusShown(v), C.fmtV(v).startsWith('\u2212'), String(v));
        if (v <= -0.005) assert.ok(C.minusShown(v), String(v));
      }
    }
  }
  assert.ok(!C.minusShown(-0.001));
  assert.ok(!C.minusShown(0.4));
});

test('weight sharing counts', () => {
  assert.equal(C.sharedCount(), 9);
  assert.equal(C.denseCount(), 14400);
  assert.equal(C.denseCount(), 144 * 100);
});

test('formatting and read-out rows', () => {
  assert.equal(C.fmtV(-3), '−3');
  assert.equal(C.fmtV(0.5), '0.5');
  assert.equal(C.fmtV(-0), '0');
  assert.equal(C.fmtV(-0.0001), '0');
  assert.equal(C.fmtV(2.9494806687401063), '2.95');
  assert.equal(C.fmtV(1 / 9), '0.11');
  const states = [
    { step: 0, cursor: [3, 4] },
    { step: 1, cursor: [2, 0] },
    { step: 2, cursor: [2, 0], filled: 100 },
    { step: 2, cursor: [0, 0], filled: 7 },
    { step: 3, cursor: [1, 1], filter: 'blur', image: 'noisy' },
    { step: 3, cursor: [1, 1], filter: 'custom', weights: C.TRY_LINE, image: 'clean' },
    { step: 4, cursor: [0, 0], stride: 2, pad: 1 },
    { step: 4, cursor: [4, 4], stride: 1, pad: 0 },
    { step: 5, cursor: [2, 2], shift: 2 },
  ];
  for (const st of states) {
    const rows = C.readout(st);
    assert.ok(rows.length >= 1 && rows.length <= 3, JSON.stringify(st));
    for (const r of rows) {
      assert.ok(!/NaN|undefined|Infinity/.test(r), r);
      assert.ok(r.length <= 62, r);
    }
  }
  assert.ok(C.readout({ step: 4, cursor: [0, 0], stride: 2, pad: 1 }).some((r) => r.includes('⌊(12 + 2·1 − 3)/2⌋ + 1 = 6')));
  assert.ok(C.readout({ step: 4, cursor: [0, 0], stride: 2, pad: 1 }).some((r) => r.includes('6 × 6')));
  assert.ok(C.readout({ step: 1, cursor: [2, 0] }).some((r) => r.includes('sum = 3')));
  assert.ok(C.readout({ step: 2, cursor: [0, 0], filled: 7 }).some((r) => /sliding/.test(r)));
});

test('page', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>Convolutions · ML Explorer<\/title>/);
  assert.match(html, /<h1>Convolutions<\/h1>/);
  assert.match(html, /Level 4 · Deep learning/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/convolutions\/">/);
  assert.match(html, /seed 191\b/);
  assert.match(html, /cross-correlation/);
  assert.match(html, /no flip/);
  assert.match(html, /away from the edges/);
  assert.match(html, /clips each pixel to \[0, 1\]/);
  assert.match(html, /href="\/ml\/deep-networks\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/cnns\/">Next lesson: Convolutional networks →<\/a>/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.match(html, /Chapter 9, "Convolutional Networks"/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.doesNotMatch(html, /Taylor/);
  assert.doesNotMatch(html, /understands|learns like|always|can learn anything/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  const sv = C.storyValues();
  const story = [...html.matchAll(/<span data-story="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(story.length >= 15);
  const used = new Set(story.map((m) => m[1]));
  for (const [, key, text] of story) {
    assert.ok(key in sv, key);
    assert.equal(text, sv[key], key);
  }
  for (const key of Object.keys(sv)) assert.ok(used.has(key), `unused story value ${key}`);
  for (const v of Object.values(sv)) assert.ok(!/NaN|undefined|Infinity/.test(v), v);
});

test('story values', () => {
  const sv = C.storyValues();
  assert.equal(sv['out-10'], '10');
  assert.equal(sv['v-max'], '3');
  assert.equal(sv['v-min'], '−3');
  assert.equal(sv['v-line'], '0.5');
  assert.equal(sv['h-max'], '3');
  assert.equal(sv['n-max'], '2.95');
  assert.equal(sv['try-line'], '1.5');
  assert.equal(sv['try-square'], '1');
  assert.equal(sv['size-s1p0'], '10');
  assert.equal(sv['size-s2p0'], '5');
  assert.equal(sv['size-s1p1'], '12');
  assert.equal(sv['size-s2p1'], '6');
  assert.equal(sv['dense'], '14,400');
  assert.equal(sv['shared'], '9');
  assert.equal(sv['v-max-at'], 'rows 2–3, columns 0–1');
  assert.equal(sv['v-min-at'], 'rows 2–3, columns 4–5');
  assert.equal(sv['h-max-at'], 'rows 0–1, columns 2–3');
  assert.equal(sv['n-max-at'], 'row 3, column 0');
  assert.equal(sv['shift-at'], 'rows 2–3, columns 2–3');
  // Noisy map: where the clean map is 0, the noisy one stays small.
  const clean = C.mapFor(C.cleanImage(), VERT).flat();
  const nm = C.mapFor(C.noisyImage(), VERT).flat();
  const off = Math.max(...nm.filter((_, k) => clean[k] === 0).map(Math.abs));
  assert.equal(sv['n-off'], C.fmtV(off));
  assert.ok(off < 1);
  // Blur: a window fully inside the square gives 1; the line's best is 1.5/9.
  const bm = C.mapFor(C.cleanImage(), C.FILTERS.blur);
  assert.ok(Math.abs(bm[2][2] - 1) < 1e-12);
  assert.equal(sv['blur-in'], '1');
  assert.equal(sv['blur-line'], C.fmtV(C.awayFromSquare(bm)));
  assert.equal(sv['blur-line'], '0.17');
});
