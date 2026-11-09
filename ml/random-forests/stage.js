// Lesson L3.6, "Bagging and random forests": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-random-forests.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { growTree, predictTree, forest, accuracy } from '../assets/classify.js';
import { treeData } from '../assets/tree-data.js';
import { linearScale, clamp, el, axes, tween, classMarker, regionMap } from '../assets/plot.js';

// The decision-tree lesson's data, so the two lessons can be compared.
export const SEED_TRAIN = 99;
export const SEED_TEST = 100;
export const SEED_FOREST = 10;
export const N_TRAIN = 80;
export const N_TEST = 400;
export const BS = [1, 3, 5, 11, 25, 51, 101];
const B_MAX = 101;
// Region map cells: square, 5 user units a side on the main plot.
export const GRID_N = 42;
export const PANEL_N = 32;
export const ACC_DOM = [0.7, 0.9];
const DOMAIN = [0, 10];

export const trainData = () => treeData({ n: N_TRAIN, seed: SEED_TRAIN });
export const testData = () => treeData({ n: N_TEST, seed: SEED_TEST });

const TR = trainData();
const TE = testData();

// 'bag' looks at both features at every split; 'rf' at one, chosen at random.
const forests = new Map();
export function forestOf(kind) {
  if (!forests.has(kind)) {
    forests.set(kind, forest(TR.X, TR.y, { B: B_MAX, mtry: kind === 'rf' ? 1 : null, seed: SEED_FOREST }));
  }
  return forests.get(kind);
}

let fullMemo = null;
export function fullTreeTest() {
  if (fullMemo === null) {
    const t = growTree(TR.X, TR.y);
    fullMemo = accuracy(TE.y, TE.X.map((x) => predictTree(t, x)));
  }
  return fullMemo;
}

// Each tree's 0/1 predictions on the test cases, computed once per forest.
const testPreds = new Map();
function treeTestPreds(kind) {
  if (!testPreds.has(kind)) testPreds.set(kind, forestOf(kind).trees.map((t) => TE.X.map((x) => predictTree(t, x))));
  return testPreds.get(kind);
}

// Majority vote of the first B trees (B is odd, so no ties) on each test case.
const tables = new Map();
export function testTable(kind) {
  if (!tables.has(kind)) {
    const P = treeTestPreds(kind);
    tables.set(kind, BS.map((B) => {
      const pred = TE.X.map((_, i) => {
        let ones = 0;
        for (let t = 0; t < B; t++) ones += P[t][i];
        return ones / B > 0.5 ? 1 : 0;
      });
      return { B, test: accuracy(TE.y, pred) };
    }));
  }
  return tables.get(kind);
}

const oobs = new Map();
export function oobTable(kind) {
  if (!oobs.has(kind)) {
    const f = forestOf(kind);
    oobs.set(kind, BS.map((B) => {
      const o = f.oob(B);
      return { B, oob: o.accuracy, covered: o.covered, excluded: N_TRAIN - o.covered };
    }));
  }
  return oobs.get(kind);
}

export const firstFullB = (kind) => oobTable(kind).find((r) => r.covered === N_TRAIN).B;

// The mean test accuracy of one tree, and the mean share of test cases on
// which two trees disagree (over every pair of the 101).
const divs = new Map();
export function diversity(kind) {
  if (!divs.has(kind)) {
    const P = treeTestPreds(kind);
    const meanTree = P.reduce((s, p) => s + accuracy(TE.y, p), 0) / P.length;
    let d = 0;
    let pairs = 0;
    for (let a = 0; a < P.length; a++) {
      for (let b = a + 1; b < P.length; b++) {
        let k = 0;
        for (let i = 0; i < N_TEST; i++) if (P[a][i] !== P[b][i]) k++;
        d += k / N_TEST;
        pairs++;
      }
    }
    divs.set(kind, { meanTree, disagree: d / pairs });
  }
  return divs.get(kind);
}

// The first bootstrap sample (tree 1's), against the formula for n = 80.
let bootMemo = null;
export function bootstrapStats() {
  if (!bootMemo) {
    const n = N_TRAIN;
    const samples = forestOf('bag').samples;
    const times = new Array(n).fill(0);
    for (const i of samples[0]) times[i]++;
    const drawCounts = [0, 0, 0, 0, 0];
    for (const t of times) drawCounts[Math.min(4, t)]++;
    const distinct = times.filter((t) => t > 0).length;
    bootMemo = {
      times,
      drawCounts,
      distinct,
      share: distinct / n,
      expected: 1 - (1 - 1 / n) ** n,
      pOut: (1 - 1 / n) ** n,
      limit: 1 - Math.exp(-1),
      meanShare: samples.reduce((s, smp) => s + new Set(smp).size, 0) / samples.length / n,
    };
  }
  return bootMemo;
}

// Cell centres in regionCells order: row 0 at the top (x₂ = 10), left to right.
export function gridCentres(n) {
  const out = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) out.push([((i + 0.5) / n) * 10, 10 - ((j + 0.5) / n) * 10]);
  }
  return out;
}
export const cellIndex = (x, y, n) => clamp(Math.floor((10 - y) / 10 * n), 0, n - 1) * n + clamp(Math.floor(x / 10 * n), 0, n - 1);

// The vote map for every B, built in one pass over the trees and kept, so
// switching B only repaints the cells.
const grids = new Map();
export function voteGrid(kind, B) {
  if (!grids.has(kind)) {
    const trees = forestOf(kind).trees;
    const byB = new Map(BS.map((b) => [b, []]));
    for (const p of gridCentres(GRID_N)) {
      let ones = 0;
      for (let t = 0; t < B_MAX; t++) {
        ones += predictTree(trees[t], p);
        if (byB.has(t + 1)) byB.get(t + 1).push(ones / (t + 1) > 0.5 ? 1 : 0);
      }
    }
    grids.set(kind, byB);
  }
  return grids.get(kind).get(B);
}

// Step 1's small maps: trees 1 to 3 of the bagged forest.
const panels = new Map();
export function panelGrid(t) {
  if (!panels.has(t)) panels.set(t, gridCentres(PANEL_N).map((p) => predictTree(forestOf('bag').trees[t], p)));
  return panels.get(t);
}
export const panelTests = () => [0, 1, 2].map((t) => accuracy(TE.y, treeTestPreds('bag')[t]));

// How many trees ask about each feature first.
export function rootCounts(kind) {
  const c = [0, 0];
  for (const t of forestOf(kind).trees) if (!t.leaf) c[t.feature]++;
  return c;
}

// Cells of the main map whose class changes from one B on the slider to the next.
export const mapChanges = (kind) => BS.slice(1).map((B, k) => {
  const a = voteGrid(kind, BS[k]);
  const b = voteGrid(kind, B);
  return a.filter((v, j) => v !== b[j]).length;
});

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);

export function readout(step, { bi = 3 } = {}) {
  const B = BS[clamp(bi, 0, BS.length - 1)];
  if (step === 0) return ['one full tree, all rows', `test accuracy: ${pct(fullTreeTest())}`];
  if (step === 1) {
    const s = bootstrapStats();
    return [`distinct rows: ${s.distinct} of ${N_TRAIN}`, `this sample: ${pct(s.share)}`, `expected: ${pct(s.expected)}`, `mean of ${B_MAX}: ${pct(s.meanShare)}`];
  }
  const bag = testTable('bag').find((r) => r.B === B);
  const rf = testTable('rf').find((r) => r.B === B);
  if (step === 2) return [`B = ${B}`, `bagging test: ${pct(bag.test)}`, `one full tree: ${pct(fullTreeTest())}`];
  if (step === 3) return [`B = ${B}`, `bagging test: ${pct(bag.test)}`, `forest test: ${pct(rf.test)}`];
  const o = oobTable('rf').find((r) => r.B === B);
  return [`B = ${B}`, `OOB: ${pct(o.oob)}`, `rows with a vote: ${o.covered}`, `forest test: ${pct(rf.test)}`];
}

export function pageValues() {
  const s = bootstrapStats();
  const bag = testTable('bag');
  const rf = testTable('rf');
  const o = oobTable('rf');
  const ob = oobTable('bag');
  const at = (rows, B) => rows.find((r) => r.B === B);
  const dB = diversity('bag');
  const dR = diversity('rf');
  const t = panelTests();
  return {
    'n-train': String(N_TRAIN),
    'n-test': String(N_TEST),
    'full-test': pct(fullTreeTest()),
    't1-test': pct(t[0]),
    't2-test': pct(t[1]),
    't3-test': pct(t[2]),
    's-distinct': String(s.distinct),
    's-out': String(N_TRAIN - s.distinct),
    's-share': pct(s.share),
    'mean-share': pct(s.meanShare),
    expected: s.expected.toFixed(3),
    'expected-pct': pct(s.expected),
    'p-out': s.pOut.toFixed(3),
    'p-out-pct': pct(s.pOut),
    limit: s.limit.toFixed(3),
    'bag-1': pct(at(bag, 1).test),
    'bag-11': pct(at(bag, 11).test),
    'bag-101': pct(at(bag, 101).test),
    'rf-101': pct(at(rf, 101).test),
    'bag-tree': pct(dB.meanTree),
    'rf-tree': pct(dR.meanTree),
    'bag-dis': pct(dB.disagree),
    'rf-dis': pct(dR.disagree),
    'oob-1': pct(o[0].oob),
    'oob-1-rows': String(o[0].covered),
    'oob-1-excl': String(o[0].excluded),
    'oob-full-b': String(firstFullB('rf')),
    'oob-101': pct(at(o, 101).oob),
    'bag-oob-101': pct(at(ob, 101).oob),
    'bag-root-x2': String(rootCounts('bag')[1]),
    'rf-root-x1': String(rootCounts('rf')[0]),
    'rf-root-x2': String(rootCounts('rf')[1]),
    'bag-chg-first': String(mapChanges('bag')[0]),
    'bag-chg-last': String(mapChanges('bag').at(-1)),
    cells: (GRID_N * GRID_N).toLocaleString('en-NZ'),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const SURFACE = '#1e293b';
const LINE = '#334155';

// The scatter: 21 user units per unit on both axes.
const PX0 = 48;
const PX1 = 258;
const PY0 = 268;
const PY1 = 58;
// Right-hand charts.
const CX0 = 340;
const CX1 = 588;
const CY0 = 262;
const CY1 = 150;
const BOX_Y = 8;
const ROW = 20;
// Step 1's three panels.
const PAN_W = 172;
const PAN_X = [22, 216, 410];
const PAN_Y0 = 272;
const PAN_Y1 = PAN_Y0 - PAN_W;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const boot = bootstrapStats();
  let bi = 3;

  const X = linearScale(DOMAIN, [PX0, PX1]);
  const Y = linearScale(DOMAIN, [PY0, PY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Eighty training points on two features, x₁ across and x₂ up: class 0 as cyan circles, class 1 as violet triangles. Depending on the step: three small maps from three single trees; one bootstrap sample with a bar chart of how often each row was drawn; or the map a vote of many trees predicts, shaded by class with a line where the prediction changes, beside a chart of accuracy against the number of trees.');

  // --- Step 1: three single trees.
  const panelsG = el('g', {}, svg);
  const forestTrees = forestOf('bag');
  const ptests = panelTests();
  PAN_X.forEach((x0, t) => {
    const g = el('g', {}, panelsG);
    const PXs = linearScale(DOMAIN, [x0, x0 + PAN_W]);
    const PYs = linearScale(DOMAIN, [PAN_Y0, PAN_Y1]);
    const grid = panelGrid(t);
    regionMap(g, PXs, PYs, (x, y) => grid[cellIndex(x, y, PANEL_N)], { nx: PANEL_N, ny: PANEL_N });
    el('rect', { x: x0, y: PAN_Y1, width: PAN_W, height: PAN_W, fill: 'none', stroke: LINE }, g);
    const seen = new Set(forestTrees.samples[t]);
    TR.X.forEach((p, i) => { if (seen.has(i)) classMarker(g, PXs(p[0]), PYs(p[1]), TR.y[i], { r: 3.2 }); });
    el('text', { class: 'stage-text', x: x0, y: PAN_Y1 - 8, text: `tree ${t + 1}: ${seen.size} rows` }, g);
    const lab = el('text', { class: 'stage-text', x: x0, y: PAN_Y0 + 20, text: `test: ${pct(ptests[t])}` }, g);
    lab.style.fill = STRONG;
  });

  // --- The main scatter.
  const mainG = el('g', {}, svg);
  const regionG = el('g', {}, mainG);
  const region = regionMap(regionG, X, Y, () => 0, { nx: GRID_N, ny: GRID_N });
  axes(mainG, X, Y, { xLabel: 'x₁', yLabel: 'x₂' });
  const ringsG = el('g', {}, mainG);
  const countsG = el('g', {});
  TR.X.forEach((p, i) => {
    if (boot.times[i] === 0) {
      el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 8, fill: 'none', stroke: MUTED, 'stroke-width': 1.3, 'stroke-dasharray': '3 2' }, ringsG);
    } else if (boot.times[i] > 1) {
      el('text', { class: 'rf-count', x: X(p[0]) + 6, y: Y(p[1]) - 5, text: `×${boot.times[i]}` }, countsG);
    }
  });
  const pointsG = el('g', {}, mainG);
  TR.X.forEach((p, i) => classMarker(pointsG, X(p[0]), Y(p[1]), TR.y[i], { r: 4.5 }));
  // Counts sit above the markers so they stay readable.
  mainG.append(countsG);

  // Key, above the scatter (or the panels).
  const keyG = el('g', {}, svg);
  classMarker(keyG, PX0 + 6, 24, 0, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 16, y: 28, text: 'class 0' }, keyG);
  classMarker(keyG, PX0 + 112, 24, 1, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 122, y: 28, text: 'class 1' }, keyG);
  const keyOut = el('g', {}, keyG);
  // Beside the bar chart, under the read-out box.
  el('circle', { cx: CX0 + 6, cy: 124, r: 6, fill: 'none', stroke: MUTED, 'stroke-width': 1.3, 'stroke-dasharray': '3 2' }, keyOut);
  el('text', { class: 'stage-text', x: CX0 + 16, y: 128, text: 'left out' }, keyOut);
  el('text', { class: 'rf-count', x: CX0 + 104, y: 128, text: '×2' }, keyOut);
  el('text', { class: 'stage-text', x: CX0 + 124, y: 128, text: 'drawn twice' }, keyOut);

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // --- Step 2: how often each row was drawn.
  const barsG = el('g', {}, svg);
  const HX = linearScale([-0.5, 4.5], [CX0, CX1]);
  const HY = linearScale([0, 40], [CY0, CY1]);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, barsG);
  boot.drawCounts.forEach((c, k) => {
    const x = HX(k);
    el('rect', { x: x - 16, y: HY(c), width: 32, height: CY0 - HY(c), fill: LINE, stroke: INK, 'stroke-width': 1.2 }, barsG);
    el('text', { class: 'stage-text', x, y: HY(c) - 6, 'text-anchor': 'middle', text: String(c) }, barsG).style.fill = STRONG;
    el('text', { class: 'stage-text', x, y: CY0 + 16, 'text-anchor': 'middle', text: k < 4 ? String(k) : '4+' }, barsG);
  });
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'times drawn (rows)' }, barsG);

  // --- Steps 3 to 5: accuracy against B.
  const accG = el('g', {}, svg);
  const BX = (k) => CX0 + 14 + (k / (BS.length - 1)) * (CX1 - CX0 - 24);
  const AY = linearScale(ACC_DOM, [CY0, CY1]);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, accG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, accG);
  for (const a of [0.7, 0.8, 0.9]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: AY(a), y2: AY(a), stroke: LINE }, accG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: AY(a) + 4, 'text-anchor': 'end', text: `${Math.round(a * 100)}%` }, accG);
  }
  BS.forEach((B, k) => {
    el('line', { x1: BX(k), x2: BX(k), y1: CY0, y2: CY0 + 4, stroke: LINE }, accG);
    el('text', { class: 'stage-text', x: BX(k), y: CY0 + 16, 'text-anchor': 'middle', text: String(B) }, accG);
  });
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'B, the number of trees' }, accG);
  const bMark = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, accG);
  // One full tree: a muted dotted reference line.
  const fullG = el('g', {}, accG);
  el('line', { x1: CX0, x2: CX1, y1: AY(fullTreeTest()), y2: AY(fullTreeTest()), stroke: MUTED, 'stroke-width': 1.4, 'stroke-dasharray': '1 3' }, fullG);
  const path = (vals) => vals.map((v, k) => `${k ? 'L' : 'M'}${BX(k).toFixed(2)},${AY(v).toFixed(2)}`).join('');
  // Bagging: ink, solid, dots. Forest: cyan, dashed, squares. OOB: strong, dotted, diamonds.
  const bagG = el('g', {}, accG);
  const bagVals = testTable('bag').map((r) => r.test);
  el('path', { d: path(bagVals), fill: 'none', stroke: INK, 'stroke-width': 2 }, bagG);
  bagVals.forEach((v, k) => el('circle', { cx: BX(k), cy: AY(v), r: 3, fill: INK }, bagG));
  const rfG = el('g', {}, accG);
  const rfVals = testTable('rf').map((r) => r.test);
  el('path', { d: path(rfVals), fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, rfG);
  rfVals.forEach((v, k) => el('rect', { x: BX(k) - 3, y: AY(v) - 3, width: 6, height: 6, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, rfG));
  const oobG = el('g', {}, accG);
  const oobVals = oobTable('rf').map((r) => r.oob);
  el('path', { d: path(oobVals), fill: 'none', stroke: STRONG, 'stroke-width': 1.6, 'stroke-dasharray': '2 3' }, oobG);
  oobVals.forEach((v, k) => el('path', { d: diamondD(BX(k), AY(v), 4.5), fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, oobG));

  // Chart key: up to two lines above the chart.
  const keyRows = [0, 1].map((r) => {
    const g = el('g', {}, accG);
    const y = CY1 - 34 + r * 18;
    return { g, y, line: el('line', { x1: CX0 + 10, x2: CX0 + 30, y1: y, y2: y, 'stroke-width': 2 }, g), mark: el('g', {}, g), text: el('text', { class: 'stage-text', x: CX0 + 38, y: y + 4 }, g) };
  });
  const KEYS = {
    bag: { stroke: INK, dash: null, text: 'bagging, test', mark: (g, x, y) => el('circle', { cx: x, cy: y, r: 3, fill: INK }, g) },
    rf: { stroke: CYAN, dash: '5 3', text: 'forest, test', mark: (g, x, y) => el('rect', { x: x - 3, y: y - 3, width: 6, height: 6, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, g) },
    oob: { stroke: STRONG, dash: '2 3', text: 'forest, out-of-bag', mark: (g, x, y) => el('path', { d: diamondD(x, y, 4.5), fill: BG, stroke: STRONG, 'stroke-width': 1.5 }, g) },
    full: { stroke: MUTED, dash: '1 3', text: 'one full tree, test', mark: () => {} },
  };
  const setKey = (r, name) => {
    const k = KEYS[name];
    r.line.setAttribute('stroke', k.stroke);
    if (k.dash) r.line.setAttribute('stroke-dasharray', k.dash);
    else r.line.removeAttribute('stroke-dasharray');
    r.mark.replaceChildren();
    k.mark(r.mark, CX0 + 20, r.y);
    r.text.textContent = k.text;
    r.text.style.fill = k.stroke;
  };

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const bGroup = make('div');
  const bLabel = make('label');
  const bInput = make('input', { type: 'range', min: 0, max: BS.length - 1, step: 1, value: bi });
  const bOut = make('output');
  bOut.setAttribute('aria-hidden', 'true');
  bLabel.append('Trees, B', ' ', bInput, ' ', bOut);
  bGroup.append(bLabel);
  controls.append(bGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  let s = { panels: 1, main: 0, sample: 0, region: 0, bars: 0, acc: 0 };
  const target = (i) => ({
    panels: i === 0 ? 1 : 0,
    main: i >= 1 ? 1 : 0,
    sample: i === 1 ? 1 : 0,
    region: i >= 2 ? 1 : 0,
    bars: i === 1 ? 1 : 0,
    acc: i >= 2 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const kindAt = (i) => (i === 2 ? 'bag' : 'rf');

  // The map is repainted only when its forest or B changes, at most once a frame.
  let regionKey = '';
  let pending = false;
  function drawRegion() {
    if (step < 2) return;
    const kind = kindAt(step);
    const key = `${kind}:${BS[bi]}`;
    if (key === regionKey) return;
    regionKey = key;
    const grid = voteGrid(kind, BS[bi]);
    region.update((x, y) => grid[cellIndex(x, y, GRID_N)]);
  }
  function queueRegion() {
    if (pending) return;
    pending = true;
    const run = () => { pending = false; drawRegion(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  }

  function render() {
    queueRegion();
    op(panelsG, s.panels);
    op(mainG, s.main);
    op(ringsG, s.sample);
    op(countsG, s.sample);
    op(keyOut, s.sample);
    op(regionG, s.region);
    op(barsG, s.bars);
    op(accG, s.acc);

    // Chart series per step.
    fullG.style.display = step === 2 ? '' : 'none';
    bagG.style.display = step === 2 || step === 3 ? '' : 'none';
    rfG.style.display = step >= 3 ? '' : 'none';
    oobG.style.display = step === 4 ? '' : 'none';
    const names = step === 2 ? ['bag', 'full'] : step === 3 ? ['bag', 'rf'] : ['rf', 'oob'];
    keyRows.forEach((r, k) => setKey(r, names[k]));
    const bx = BX(bi).toFixed(2);
    bMark.setAttribute('x1', bx);
    bMark.setAttribute('x2', bx);

    const texts = readout(step, { bi });
    rowsT.forEach((r, i) => {
      r.textContent = texts[i] || '';
      r.style.display = i < texts.length ? '' : 'none';
    });
    let w = 210;
    try {
      const lens = rowsT.slice(0, texts.length).map((r) => r.getComputedTextLength());
      if (lens.every((len) => len > 0)) w = Math.max(...lens) + 24;
    } catch {
      // Keep the fixed width.
    }
    const x0 = 598 - w;
    boxRect.setAttribute('x', x0.toFixed(2));
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', w.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', (x0 + 12).toFixed(2));
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    bOut.textContent = String(BS[bi]);
    bInput.setAttribute('aria-valuetext', `B = ${BS[bi]} trees`);
  }

  function animateTo(to, ms) {
    const my = ++token;
    const start = { ...s };
    return tween(ms, (kt) => {
      if (my !== token) return;
      const next = {};
      for (const k of Object.keys(to)) next[k] = start[k] + (to[k] - start[k]) * kt;
      s = next;
      render();
    }, { reduced });
  }

  const spoken = (i) => {
    const B = BS[bi];
    if (i === 0) {
      const t = ptests;
      return `Three small maps, one per tree. Each tree was grown in full on its own bootstrap sample, and the maps differ. Test accuracy: tree 1 ${pct(t[0])}, tree 2 ${pct(t[1])}, tree 3 ${pct(t[2])}. One full tree on all the rows: ${pct(fullTreeTest())}.`;
    }
    if (i === 1) {
      const c = boot.drawCounts;
      return `Tree 1's bootstrap sample: 80 draws with replacement. ${boot.distinct} distinct rows, ${pct(boot.share)}; the expected share is ${pct(boot.expected)}. ${c[0]} rows were drawn 0 times, ${c[1]} once, ${c[2]} twice, ${c[3]} three times and ${c[4]} four or more. Left-out rows have a dashed ring.`;
    }
    const bag = testTable('bag').find((r) => r.B === B);
    const rf = testTable('rf').find((r) => r.B === B);
    if (i === 2) return `Bagging, B = ${B} trees. Test accuracy of the vote ${pct(bag.test)}; one full tree ${pct(fullTreeTest())}.`;
    if (i === 3) return `B = ${B} trees. Bagging test accuracy ${pct(bag.test)}; random forest test accuracy ${pct(rf.test)}. The map shows the forest's vote.`;
    const o = oobTable('rf').find((r) => r.B === B);
    return `Random forest, B = ${B} trees. Out-of-bag accuracy ${pct(o.oob)} on the ${o.covered} rows with at least one out-of-bag tree; ${o.excluded} rows have none. Test accuracy ${pct(rf.test)}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  bInput.addEventListener('input', () => {
    bi = clamp(Math.round(Number(bInput.value)), 0, BS.length - 1);
    render();
    say(spoken(step));
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in filled)) p.hidden = false;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 5,
    show(i, animate) {
      step = i;
      showGroup(bGroup, i >= 2);
      announce(spoken(i));
      const to = target(i);
      if (!animate) {
        token++;
        s = { ...to };
        render();
        return Promise.resolve();
      }
      return animateTo(to, 500);
    },
  };
}

function diamondD(cx, cy, r) {
  const P = (v) => Number(v.toFixed(2));
  return `M${P(cx)},${P(cy - r)}L${P(cx + r)},${P(cy)}L${P(cx)},${P(cy + r)}L${P(cx - r)},${P(cy)}Z`;
}
