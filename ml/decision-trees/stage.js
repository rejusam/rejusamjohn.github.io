// Lesson L3.5, "Decision trees": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l3-decision-trees.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { gini, bestSplit, growTree, predictTree, treeStats, accuracy } from '../assets/classify.js';
import { treeData, treeRule } from '../assets/tree-data.js';
import { linearScale, clamp, el, axes, tween, classMarker, draggable, treeLayout } from '../assets/plot.js';

export const SEED_TRAIN = 99;
export const SEED_TEST = 100;
export const N_TRAIN = 80;
export const N_TEST = 400;
export const DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8];
// Step 1's question, "is x₁ less than 5?".
export const T_GUESS = 5;
// The point followed down the tree in step 6.
export const Q0 = [2, 8];
// The diagram draws three levels below the root; deeper subtrees become stubs.
export const DISPLAY_DEPTH = 3;
// Where the diagram sits, and the widest mono character it may hold
// (13-unit text on a phone, set in lesson.css).
export const TREE_AREA = { x: 276, y: 128, w: 322, h: 206 };
export const CHAR_W = 7.8;
const BOX_H = 36;
const DOMAIN = [0, 10];

export const trainData = () => treeData({ n: N_TRAIN, seed: SEED_TRAIN });
export const testData = () => treeData({ n: N_TEST, seed: SEED_TEST });

const TR = trainData();
const TE = testData();

export const classCounts = (y) => [y.filter((v) => v === 0).length, y.filter((v) => v === 1).length];
export const parentGini = () => gini(classCounts(TR.y));

// Split the training points by "x_f < t?": counts each side, each side's Gini,
// and the weighted Gini (each side weighted by its share of the points).
export function splitScore(f, t) {
  const left = [0, 0];
  const right = [0, 0];
  TR.X.forEach((x, i) => (x[f] < t ? left : right)[TR.y[i]]++);
  const nl = left[0] + left[1];
  const nr = right[0] + right[1];
  const gLeft = gini(left);
  const gRight = gini(right);
  return { left, right, gLeft, gRight, weighted: (nl * gLeft + nr * gRight) / (nl + nr) };
}

// The weighted Gini as t runs over [0, 10]: constant between neighbouring
// data values, so a list of flat steps.
const stepCache = new Map();
export function giniSteps(f) {
  if (stepCache.has(f)) return stepCache.get(f);
  const v = [...new Set(TR.X.map((x) => x[f]))].sort((a, b) => a - b);
  const cuts = [DOMAIN[0], ...v, DOMAIN[1]];
  const out = [];
  for (let k = 0; k + 1 < cuts.length; k++) {
    out.push({ t0: cuts[k], t1: cuts[k + 1], g: splitScore(f, (cuts[k] + cuts[k + 1]) / 2).weighted });
  }
  stepCache.set(f, out);
  return out;
}

// The lowest step (the first, if two tie); its midpoint is the tree's threshold.
export function minInterval(f) {
  const s = giniSteps(f).reduce((a, b) => (b.g < a.g - 1e-12 ? b : a));
  return { a: s.t0, b: s.t1, mid: (s.t0 + s.t1) / 2, g: s.g };
}

const ALL = TR.X.map((_, i) => i);
let rootMemo = null;
export function rootChoice() {
  if (!rootMemo) {
    rootMemo = {
      byFeature: [0, 1].map((f) => bestSplit(TR.X, TR.y, ALL, { features: [f] })),
      best: bestSplit(TR.X, TR.y, ALL),
    };
  }
  return rootMemo;
}

const leafNode = (counts, depth) => ({ leaf: true, counts, cls: counts[1] > counts[0] ? 1 : 0, depth, n: counts[0] + counts[1] });

// A one-question tree "x_f < t?", in growTree's shape.
export function stump(f, t) {
  const s = splitScore(f, t);
  const all = classCounts(TR.y);
  return { ...leafNode(all, 0), leaf: false, feature: f, threshold: t, left: leafNode(s.left, 1), right: leafNode(s.right, 1) };
}

const treeCache = new Map();
export function treeAt(d) {
  if (!treeCache.has(d)) treeCache.set(d, growTree(TR.X, TR.y, { maxDepth: d }));
  return treeCache.get(d);
}
export const fullTree = () => treeAt(Infinity);

let depthRows = null;
export function depthTable() {
  if (!depthRows) {
    depthRows = DEPTHS.map((d) => {
      const t = treeAt(d);
      const st = treeStats(t);
      return {
        d,
        train: accuracy(TR.y, TR.X.map((x) => predictTree(t, x))),
        test: accuracy(TE.y, TE.X.map((x) => predictTree(t, x))),
        leaves: st.leaves,
        depth: st.depth,
      };
    });
  }
  return depthRows;
}

// Highest test accuracy; a tie would go to the shallower tree.
export function bestDepth() {
  return depthTable().reduce((a, b) => (b.test > a.test ? b : a)).d;
}

export function ruleAccuracy() {
  return { train: accuracy(TR.y, TR.X.map(treeRule)), test: accuracy(TE.y, TE.X.map(treeRule)) };
}

// Each leaf's box in the square [0, 10]²: yes (x_f < t) is the lower side.
export function leafBoxes(tree) {
  const out = [];
  const walk = (n, x0, x1, y0, y1) => {
    if (n.leaf) {
      out.push({ x0, x1, y0, y1, cls: n.cls, node: n });
      return;
    }
    const t = clamp(n.threshold, DOMAIN[0], DOMAIN[1]);
    if (n.feature === 0) {
      walk(n.left, x0, Math.max(x0, Math.min(x1, t)), y0, y1);
      walk(n.right, Math.min(x1, Math.max(x0, t)), x1, y0, y1);
    } else {
      walk(n.left, x0, x1, y0, Math.max(y0, Math.min(y1, t)));
      walk(n.right, x0, x1, Math.min(y1, Math.max(y0, t)), y1);
    }
  };
  walk(tree, DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]);
  return out;
}

export function leafOf(tree, p) {
  let n = tree;
  while (!n.leaf) n = p[n.feature] < n.threshold ? n.left : n.right;
  return n;
}

// Pieces of box sides where the predicted class changes: [x0, y0, x1, y1].
export function classEdges(boxes) {
  const segs = [];
  for (const a of boxes) {
    for (const b of boxes) {
      if (a.cls === b.cls) continue;
      if (a.cls !== 0) continue; // each pair once
      if (a.x1 === b.x0 || a.x0 === b.x1) {
        const x = a.x1 === b.x0 ? a.x1 : a.x0;
        const lo = Math.max(a.y0, b.y0);
        const hi = Math.min(a.y1, b.y1);
        if (hi > lo) segs.push([x, lo, x, hi]);
      }
      if (a.y1 === b.y0 || a.y0 === b.y1) {
        const y = a.y1 === b.y0 ? a.y1 : a.y0;
        const lo = Math.max(a.x0, b.x0);
        const hi = Math.min(a.x1, b.x1);
        if (hi > lo) segs.push([lo, y, hi, y]);
      }
    }
  }
  return segs;
}

export const fmtT = (t) => t.toFixed(2);
export const fmtG = (g) => g.toFixed(3);
const NAMES = ['x₁', 'x₂'];
const question = (n) => `${NAMES[n.feature]} < ${fmtT(n.threshold)}`;
const countSplits = (n) => (n.leaf ? 0 : 1 + countSplits(n.left) + countSplits(n.right));

// The diagram's boxes: leaves (class marker, then counts class 0 · class 1),
// splits (the question) and stubs (a split below the drawn levels: the number of
// splits it hides, then its counts). Each bottom box has its own column, and a
// split is no wider than the columns beneath it, so boxes never overlap.
export function treeBoxes(tree, area = TREE_AREA) {
  const cut = (n, level) => {
    if (n.leaf) return { leaf: true, orig: n, kind: 'leaf' };
    if (level >= DISPLAY_DEPTH) return { leaf: true, orig: n, kind: 'stub' };
    return { leaf: false, orig: n, kind: 'split', left: cut(n.left, level + 1), right: cut(n.right, level + 1) };
  };
  const lay = treeLayout(cut(tree, 0));
  const colW = area.w / lay.leaves;
  const rowH = lay.depth ? Math.min(64, (area.h - BOX_H) / lay.depth) : 0;
  const span = (n) => (n.leaf ? 1 : span(n.left) + span(n.right));
  const counts = (n, w) => {
    const [c0, c1] = n.counts;
    const wide = `${c0} · ${c1}`;
    return wide.length * CHAR_W + 4 <= w ? wide : `${c0}·${c1}`;
  };
  const nodes = lay.nodes.map((p) => {
    const d = p.node;
    const n = d.orig;
    const cx = area.x + (p.x + 0.5) * colW;
    const w = d.kind === 'split' ? Math.min(span(d) * colW - 4, 96) : Math.min(colW - 4, 64);
    const hidden = d.kind === 'stub' ? countSplits(n) : 0;
    const lines = d.kind === 'split' ? [question(n)] : d.kind === 'stub' ? [`+${hidden}`, counts(n, w)] : [counts(n, w)];
    return { x: cx - w / 2, y: area.y + p.depth * rowH, w, h: BOX_H, cx, level: p.depth, kind: d.kind, lines, node: n, hidden };
  });
  return { nodes, links: lay.links };
}

// The questions a point is asked on its way down, and the leaf it reaches.
export function pathOf(tree, p) {
  const steps = [];
  let n = tree;
  while (!n.leaf) {
    const yes = p[n.feature] < n.threshold;
    steps.push({ node: n, yes });
    n = yes ? n.left : n.right;
  }
  return { steps, leaf: n };
}

// Every depth-2 tree whose root is a midpoint split, each side then split as well
// as it can be: the lowest total leaf impurity, sum of (n_leaf / n) · G_leaf.
let d2 = null;
export function depth2Search() {
  if (d2) return d2;
  const imp = (rows) => (rows.length * gini(classCounts(rows.map((i) => TR.y[i])))) / N_TRAIN;
  const side = (rows) => {
    const s = bestSplit(TR.X, TR.y, rows);
    return s ? imp(s.left) + imp(s.right) : imp(rows);
  };
  let best = null;
  for (const f of [0, 1]) {
    const v = [...new Set(TR.X.map((x) => x[f]))].sort((a, b) => a - b);
    for (let k = 0; k + 1 < v.length; k++) {
      const t = (v[k] + v[k + 1]) / 2;
      const total = side(ALL.filter((i) => TR.X[i][f] < t)) + side(ALL.filter((i) => TR.X[i][f] >= t));
      if (!best || total < best.impurity - 1e-12) best = { feature: f, threshold: t, impurity: total };
    }
  }
  const greedy = leafBoxes(treeAt(2)).reduce((s, b) => s + (b.node.n * gini(b.node.counts)) / N_TRAIN, 0);
  d2 = { greedy, best };
  return d2;
}

// A share as a percentage: one decimal place, but exactly 1 is "100%".
export const pct = (v) => (v === 1 ? '100%' : `${(v * 100).toFixed(1)}%`);
const side = (c) => `${c[0]} ○, ${c[1]} △`;

export function readout(step, { t = T_GUESS, depth = 1, q = Q0 } = {}) {
  if (step === 0) {
    const s = splitScore(0, T_GUESS);
    const right = Math.max(...s.left) + Math.max(...s.right);
    return [`x₁ < ${fmtT(T_GUESS)}?`, `yes: ${side(s.left)}`, `no: ${side(s.right)}`, `accuracy: ${pct(right / N_TRAIN)}`];
  }
  if (step === 1) {
    const s = splitScore(0, t);
    return [`x₁ < ${fmtT(t)}?`, `yes: ${side(s.left)}`, `no: ${side(s.right)}`, `weighted G = ${fmtG(s.weighted)}`];
  }
  if (step === 2) {
    const rc = rootChoice();
    return [`best x₁: G = ${fmtG(rc.byFeature[0].impurity)}`, `best x₂: G = ${fmtG(rc.byFeature[1].impurity)}`, `root: ${question(rc.best)}`];
  }
  const r = depthTable()[clamp(depth, 1, 8) - 1];
  if (step === 3) return [`max depth ${r.d}`, `leaves: ${r.leaves}`, `train accuracy: ${pct(r.train)}`];
  if (step === 4) return [`max depth ${r.d}`, `train accuracy: ${pct(r.train)}`, `test accuracy: ${pct(r.test)}`];
  const p = pathOf(treeAt(bestDepth()), q);
  return [...p.steps.map((s) => `${question(s.node)}? ${s.yes ? 'yes' : 'no'}`), `leaf: ${side(p.leaf.counts)}`, `prediction: class ${p.leaf.cls}`];
}

export function pageValues() {
  const rows = depthTable();
  const b = bestDepth();
  const rc = rootChoice();
  const s5 = splitScore(0, T_GUESS);
  const m = minInterval(0);
  const g = depth2Search();
  const t2 = treeAt(2);
  const p = pathOf(treeAt(b), Q0);
  const full = treeStats(fullTree());
  const rule = ruleAccuracy();
  const [n0, n1] = classCounts(TR.y);
  return {
    'n-train': String(N_TRAIN),
    'n-test': String(N_TEST),
    n0: String(n0),
    n1: String(n1),
    flipped: String(TR.flipped.length),
    'test-flipped': String(TE.flipped.length),
    'rule-test': pct(rule.test),
    'g-left0': String(s5.left[0]),
    'g-left1': String(s5.left[1]),
    'g-right0': String(s5.right[0]),
    'g-right1': String(s5.right[1]),
    'g-acc': pct((Math.max(...s5.left) + Math.max(...s5.right)) / N_TRAIN),
    'g-gl': fmtG(s5.gLeft),
    'g-gr': fmtG(s5.gRight),
    'g-w': fmtG(s5.weighted),
    'parent-g': fmtG(parentGini()),
    'x1-t': fmtT(rc.byFeature[0].threshold),
    'x1-g': fmtG(rc.byFeature[0].impurity),
    'x2-t': fmtT(rc.byFeature[1].threshold),
    'x2-g': fmtG(rc.byFeature[1].impurity),
    'x1-a': m.a.toFixed(3),
    'x1-b': m.b.toFixed(3),
    'best-d': String(b),
    'best-test': pct(rows[b - 1].test),
    'best-train': pct(rows[b - 1].train),
    'best-leaves': String(rows[b - 1].leaves),
    'full-d': String(full.depth),
    'full-leaves': String(full.leaves),
    'full-test': pct(rows[7].test),
    'd1-test': pct(rows[0].test),
    'd3-test': pct(rows[2].test),
    'greedy-g': fmtG(g.greedy),
    'best2-g': fmtG(g.best.impurity),
    'best2-t': fmtT(g.best.threshold),
    'd2-l-t': fmtT(t2.left.threshold),
    'd2-r-t': fmtT(t2.right.threshold),
    'q-leaf0': String(p.leaf.counts[0]),
    'q-leaf1': String(p.leaf.counts[1]),
    'q-cls': String(p.leaf.cls),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
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
const G_DOM = [0.38, 0.5];
const ACC_DOM = [0.6, 1];
const BOX_Y = 8;
const ROW = 20;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const rows = depthTable();
  const best = bestDepth();
  const rc = rootChoice();
  let t = T_GUESS;
  let depth = 1;
  let q = Q0.slice();

  const X = linearScale(DOMAIN, [PX0, PX1]);
  const Y = linearScale(DOMAIN, [PY0, PY1]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Eighty training points on two features, x₁ across and x₂ up: class 0 as cyan circles, class 1 as violet triangles. Shaded boxes show the class a tree predicts, with a line where the prediction changes. On the right, depending on the step, a tree diagram, a chart of weighted Gini impurity against the threshold, or a chart of training and test accuracy against depth.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'dt-plot' }, defs);
  el('rect', { x: PX0, y: PY1, width: PX1 - PX0, height: PY0 - PY1 }, clip);

  const regionG = el('g', { class: 'region dt-region' }, svg);
  // The class boundary, styled by ml.css inside a .region group.
  const edgeG = el('path', { class: 'boundary' }, el('g', { class: 'region' }, svg));
  axes(svg, X, Y, { xLabel: 'x₁', yLabel: 'x₂' });

  // Split lines for steps 2 and 3.
  const guideG = el('g', { 'clip-path': 'url(#dt-plot)' }, svg);
  const tLine = el('line', { y1: PY0, y2: PY1, stroke: STRONG, 'stroke-width': 2 }, guideG);
  const runnerLine = el('line', { y1: PY0, y2: PY1, stroke: STRONG, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }, guideG);
  const leafOutline = el('rect', { fill: 'none', stroke: STRONG, 'stroke-width': 2, 'stroke-dasharray': '6 3' }, guideG);

  const pointsG = el('g', {}, svg);
  TR.X.forEach((x, i) => classMarker(pointsG, X(x[0]), Y(x[1]), TR.y[i], { r: 4.5 }));

  // The point followed down the tree: a hollow diamond on a focusable hit area.
  const queryG = el('g', {}, svg);
  const focusRing = el('circle', { r: 14, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, queryG);
  const diamond = el('path', { fill: BG, stroke: STRONG, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, queryG);
  const handle = el('circle', {
    class: 'handle', r: 16, fill: 'transparent', tabindex: -1, role: 'button',
    'aria-roledescription': 'movable point',
  }, queryG);
  handle.style.outline = 'none';
  let focused = false;

  // Key, above the scatter.
  const keyG = el('g', {}, svg);
  classMarker(keyG, PX0 + 6, 24, 0, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 16, y: 28, text: 'class 0' }, keyG);
  classMarker(keyG, PX0 + 112, 24, 1, { r: 5 });
  el('text', { class: 'stage-text', x: PX0 + 122, y: 28, text: 'class 1' }, keyG);

  // Read-out box.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { rx: 6, fill: SURFACE, 'fill-opacity': 0.92, stroke: LINE }, box);
  const rowsT = [0, 1, 2, 3].map(() => el('text', { class: 'value' }, box));

  // Tree diagram.
  const treeG = el('g', { class: 'dt-tree' }, svg);

  // Gini chart: x₁ solid with a dot, x₂ dashed with a square.
  const giniG = el('g', {}, svg);
  const TX = linearScale(DOMAIN, [CX0, CX1]);
  const GY = linearScale(G_DOM, [CY0, CY1]);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, giniG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, giniG);
  for (const g of [0.4, 0.45, 0.5]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: GY(g), y2: GY(g), stroke: LINE }, giniG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: GY(g) + 4, 'text-anchor': 'end', text: g.toFixed(2) }, giniG);
  }
  for (const v of [0, 2, 4, 6, 8, 10]) {
    el('line', { x1: TX(v), x2: TX(v), y1: CY0, y2: CY0 + 4, stroke: LINE }, giniG);
    el('text', { class: 'stage-text', x: TX(v), y: CY0 + 16, 'text-anchor': 'middle', text: String(v) }, giniG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'threshold' }, giniG);
  const gyx = CX0 - 56;
  const gyy = (CY0 + CY1) / 2;
  el('text', { class: 'stage-text', x: gyx, y: gyy, transform: `rotate(-90 ${gyx} ${gyy})`, 'text-anchor': 'middle', text: 'weighted G' }, giniG);
  const stepPath = (f) => giniSteps(f).map((s, k) => `${k ? 'L' : 'M'}${TX(s.t0).toFixed(2)},${GY(s.g).toFixed(2)}L${TX(s.t1).toFixed(2)},${GY(s.g).toFixed(2)}`).join('');
  el('path', { d: stepPath(0), fill: 'none', stroke: INK, 'stroke-width': 1.8 }, giniG);
  const x2Curve = el('path', { d: stepPath(1), fill: 'none', stroke: CYAN, 'stroke-width': 1.8, 'stroke-dasharray': '5 3' }, giniG);
  const minX1 = minInterval(0);
  const minX2 = minInterval(1);
  el('circle', { cx: TX(minX1.mid), cy: GY(minX1.g), r: 4, fill: INK, stroke: BG }, giniG);
  const minLab1 = el('text', { class: 'stage-text', x: TX(minX1.mid) + 8, y: GY(minX1.g) + 16, text: `lowest ${fmtG(minX1.g)}` }, giniG);
  minLab1.style.fill = STRONG;
  const x2Min = el('g', {}, giniG);
  el('rect', { x: TX(minX2.mid) - 4, y: GY(minX2.g) - 4, width: 8, height: 8, fill: BG, stroke: CYAN, 'stroke-width': 1.6 }, x2Min);
  el('text', { class: 'stage-text', x: TX(minX2.mid) - 8, y: GY(minX2.g) + 16, 'text-anchor': 'end', text: fmtG(minX2.g) }, x2Min).style.fill = CYAN;
  const tMark = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, giniG);
  const tDot = el('circle', { r: 4, fill: 'none', stroke: STRONG, 'stroke-width': 1.6 }, giniG);
  const gKey = el('g', {}, giniG);
  el('line', { x1: CX0 + 10, x2: CX0 + 26, y1: CY1 - 22, y2: CY1 - 22, stroke: INK, 'stroke-width': 2 }, gKey);
  el('text', { class: 'stage-text', x: CX0 + 30, y: CY1 - 18, text: 'x₁' }, gKey).style.fill = INK;
  const gKey2 = el('g', {}, gKey);
  el('line', { x1: CX0 + 70, x2: CX0 + 86, y1: CY1 - 22, y2: CY1 - 22, stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, gKey2);
  el('text', { class: 'stage-text', x: CX0 + 90, y: CY1 - 18, text: 'x₂' }, gKey2).style.fill = CYAN;

  // Accuracy chart: train solid with dots, test dashed with squares.
  const accG = el('g', {}, svg);
  const DX = linearScale([1, 8], [CX0 + 10, CX1 - 6]);
  const AY = linearScale(ACC_DOM, [CY0, CY1]);
  el('line', { x1: CX0, x2: CX0, y1: CY0, y2: CY1, stroke: LINE }, accG);
  el('line', { x1: CX0, x2: CX1, y1: CY0, y2: CY0, stroke: LINE }, accG);
  for (const a of [0.6, 0.8, 1]) {
    el('line', { x1: CX0 - 4, x2: CX0, y1: AY(a), y2: AY(a), stroke: LINE }, accG);
    el('text', { class: 'stage-text', x: CX0 - 7, y: AY(a) + 4, 'text-anchor': 'end', text: `${Math.round(a * 100)}%` }, accG);
  }
  for (const d of DEPTHS) {
    el('line', { x1: DX(d), x2: DX(d), y1: CY0, y2: CY0 + 4, stroke: LINE }, accG);
    el('text', { class: 'stage-text', x: DX(d), y: CY0 + 16, 'text-anchor': 'middle', text: String(d) }, accG);
  }
  el('text', { class: 'stage-text', x: (CX0 + CX1) / 2, y: CY0 + 34, 'text-anchor': 'middle', text: 'max depth' }, accG);
  const dMark = el('line', { y1: CY1, y2: CY0, stroke: STRONG, 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, accG);
  const bestLabel = el('text', { class: 'stage-text', x: DX(best) + 2, y: AY(rows[best - 1].test) + 22, text: `best depth = ${best}` }, accG);
  bestLabel.style.fill = STRONG;
  // A ring round the best depth's test square.
  el('circle', { cx: DX(best), cy: AY(rows[best - 1].test), r: 7.5, fill: 'none', stroke: STRONG, 'stroke-width': 1.6 }, accG);
  const series = (key, attrs) => el('path', {
    d: rows.map((r, i) => `${i ? 'L' : 'M'}${DX(r.d).toFixed(2)},${AY(r[key]).toFixed(2)}`).join(''),
    fill: 'none', 'stroke-width': 2, ...attrs,
  }, accG);
  series('train', { stroke: INK });
  series('test', { stroke: CYAN, 'stroke-dasharray': '5 3' });
  rows.forEach((r) => {
    el('circle', { cx: DX(r.d), cy: AY(r.train), r: 3, fill: INK }, accG);
    el('rect', { x: DX(r.d) - 3, y: AY(r.test) - 3, width: 6, height: 6, fill: BG, stroke: CYAN, 'stroke-width': 1.5 }, accG);
  });
  const aKey = el('g', {}, accG);
  el('line', { x1: CX0 + 10, x2: CX0 + 26, y1: CY1 - 26, y2: CY1 - 26, stroke: INK, 'stroke-width': 2 }, aKey);
  el('text', { class: 'stage-text', x: CX0 + 30, y: CY1 - 22, text: 'train' }, aKey).style.fill = INK;
  el('line', { x1: CX0 + 84, x2: CX0 + 100, y1: CY1 - 26, y2: CY1 - 26, stroke: CYAN, 'stroke-width': 2, 'stroke-dasharray': '5 3' }, aKey);
  el('text', { class: 'stage-text', x: CX0 + 104, y: CY1 - 22, text: 'test' }, aKey).style.fill = CYAN;

  // --- Controls.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const tGroup = make('div');
  const tLabel = make('label');
  const tInput = make('input', { type: 'range', min: 0, max: 10, step: 0.05, value: t });
  const tOut = make('output');
  tOut.setAttribute('aria-hidden', 'true');
  tLabel.append('Threshold', make('span', { className: 'vh', textContent: ' on x₁' }), ' ', tInput, ' ', tOut);
  tGroup.append(tLabel);

  const dGroup = make('div');
  const dLabel = make('label');
  const dInput = make('input', { type: 'range', min: 1, max: 8, step: 1, value: depth });
  const dOut = make('output');
  dOut.setAttribute('aria-hidden', 'true');
  dLabel.append('Max depth', ' ', dInput, ' ', dOut);
  dGroup.append(dLabel);
  controls.append(tGroup, dGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State.
  let step = 0;
  let token = 0;
  const LAYERS = { tree: 1, gini: 0, acc: 0, query: 0 };
  let s = { ...LAYERS };
  const target = (i) => ({
    tree: i === 0 || i === 3 || i === 5 ? 1 : 0,
    gini: i === 1 || i === 2 ? 1 : 0,
    acc: i === 4 ? 1 : 0,
    query: i === 5 ? 1 : 0,
  });
  const op = (node, v) => {
    node.setAttribute('opacity', v.toFixed(3));
    node.style.display = v > 0.001 ? '' : 'none';
  };
  const curTree = () => {
    if (step === 0) return stump(0, T_GUESS);
    if (step === 1) return stump(0, t);
    if (step === 2) return treeAt(1);
    if (step === 5) return treeAt(best);
    return treeAt(depth);
  };

  function drawRegions(tree) {
    const boxes = leafBoxes(tree);
    regionG.replaceChildren();
    for (const b of boxes) {
      el('rect', {
        class: `r-${b.cls}`, x: X(b.x0), y: Y(b.y1),
        width: Math.max(0, X(b.x1) - X(b.x0)), height: Math.max(0, Y(b.y0) - Y(b.y1)),
      }, regionG);
    }
    const P = (v) => v.toFixed(2);
    edgeG.setAttribute('d', classEdges(boxes).map(([a, b, c, d]) => `M${P(X(a))},${P(Y(b))}L${P(X(c))},${P(Y(d))}`).join(''));
    return boxes;
  }

  function drawTree(tree, onPath = new Set()) {
    treeG.replaceChildren();
    const lay = treeBoxes(tree);
    const linkG = el('g', {}, treeG);
    for (const [a, b] of lay.links) {
      const pa = lay.nodes[a];
      const pb = lay.nodes[b];
      const on = onPath.has(pa.node) && onPath.has(pb.node);
      el('line', { class: on ? 'tree-link dt-on' : 'tree-link', x1: pa.cx, y1: pa.y + pa.h, x2: pb.cx, y2: pb.y }, linkG);
      if (a === 0) {
        const left = pb.cx < pa.cx;
        el('text', {
          class: 'tree-yn', x: (pa.cx + pb.cx) / 2 + (left ? -6 : 6), y: (pa.y + pa.h + pb.y) / 2 + 4,
          'text-anchor': left ? 'end' : 'start', text: left ? 'yes' : 'no',
        }, linkG);
      }
    }
    for (const b of lay.nodes) {
      const ng = el('g', {}, treeG);
      const cls = ['tree-box'];
      if (b.kind === 'stub') cls.push('dt-stub');
      if (onPath.has(b.node)) cls.push('dt-on');
      const [c0, c1] = b.node.counts;
      const title = b.kind === 'split'
        ? `Split: ${b.lines[0]}? Yes goes left, no goes right. ${c0} of class 0, ${c1} of class 1 reach it.`
        : b.kind === 'stub'
          ? `${b.hidden} more splits below, not drawn. ${c0} of class 0, ${c1} of class 1 reach this point.`
          : `Leaf: ${c0} of class 0 (circle), ${c1} of class 1 (triangle); predicts class ${b.node.cls}.`;
      el('title', { text: title }, ng);
      el('rect', { class: cls.join(' '), x: b.x, y: b.y, width: b.w, height: b.h, rx: 4 }, ng);
      if (b.kind === 'split') {
        el('text', { class: 'tree-text', x: b.cx, y: b.y + b.h / 2, 'text-anchor': 'middle', text: b.lines[0] }, ng);
      } else if (b.kind === 'leaf') {
        classMarker(ng, b.cx, b.y + 11, b.node.cls, { r: 4.5 });
        el('text', { class: 'tree-text', x: b.cx, y: b.y + 26, 'text-anchor': 'middle', text: b.lines[0] }, ng);
      } else {
        el('text', { class: 'tree-text', x: b.cx, y: b.y + 11, 'text-anchor': 'middle', text: b.lines[0] }, ng);
        el('text', { class: 'tree-text', x: b.cx, y: b.y + 26, 'text-anchor': 'middle', text: b.lines[1] }, ng);
      }
    }
    if (lay.nodes.some((b) => b.kind === 'stub')) {
      el('text', { class: 'stage-text', x: TREE_AREA.x, y: TREE_AREA.y - 10, text: 'dashed: +n splits below' }, treeG);
    }
  }

  let drawnKey = '';
  function render() {
    const tree = curTree();
    const key = `${step}:${step === 1 ? t : ''}:${step === 3 || step === 4 ? depth : ''}:${step === 5 ? q.join(',') : ''}`;
    let onPath = new Set();
    let leaf = null;
    if (step === 5) {
      const p = pathOf(tree, q);
      onPath = new Set([...p.steps.map((st) => st.node), p.leaf]);
      leaf = p.leaf;
    }
    if (key !== drawnKey) {
      drawnKey = key;
      const boxes = drawRegions(tree);
      if (step === 0 || step === 3 || step === 5) drawTree(tree, onPath);
      if (leaf) {
        const b = boxes.find((bx) => bx.node === leaf);
        leafOutline.setAttribute('x', X(b.x0).toFixed(2));
        leafOutline.setAttribute('y', Y(b.y1).toFixed(2));
        leafOutline.setAttribute('width', (X(b.x1) - X(b.x0)).toFixed(2));
        leafOutline.setAttribute('height', (Y(b.y0) - Y(b.y1)).toFixed(2));
      }
    }
    op(treeG, s.tree);
    op(giniG, s.gini);
    op(accG, s.acc);
    op(queryG, s.query);
    leafOutline.style.display = step === 5 ? '' : 'none';

    // Split lines on the scatter.
    const showT = step === 1 || step === 2;
    tLine.style.display = showT ? '' : 'none';
    runnerLine.style.display = step === 2 ? '' : 'none';
    if (step === 1) {
      const px = X(clamp(t, 0, 10)).toFixed(2);
      tLine.setAttribute('x1', px); tLine.setAttribute('x2', px);
      tLine.setAttribute('y1', PY0); tLine.setAttribute('y2', PY1);
    } else if (step === 2) {
      const py = Y(rc.best.threshold).toFixed(2);
      tLine.setAttribute('x1', PX0); tLine.setAttribute('x2', PX1);
      tLine.setAttribute('y1', py); tLine.setAttribute('y2', py);
      const px = X(rc.byFeature[0].threshold).toFixed(2);
      runnerLine.setAttribute('x1', px); runnerLine.setAttribute('x2', px);
    }

    // Gini chart: the x₂ curve only in step 3.
    const both = step === 2;
    x2Curve.style.display = both ? '' : 'none';
    x2Min.style.display = both ? '' : 'none';
    gKey2.style.display = both ? '' : 'none';
    tMark.style.display = both ? 'none' : '';
    tDot.style.display = both ? 'none' : '';
    const gNow = splitScore(0, t).weighted;
    tMark.setAttribute('x1', TX(t).toFixed(2)); tMark.setAttribute('x2', TX(t).toFixed(2));
    tDot.setAttribute('cx', TX(t).toFixed(2)); tDot.setAttribute('cy', GY(gNow).toFixed(2));

    const dx = DX(depth).toFixed(2);
    dMark.setAttribute('x1', dx); dMark.setAttribute('x2', dx);

    // The followed point.
    const qx = X(q[0]);
    const qy = Y(q[1]);
    diamond.setAttribute('d', diamondD(qx, qy, 8));
    handle.setAttribute('cx', qx.toFixed(2));
    handle.setAttribute('cy', qy.toFixed(2));
    focusRing.setAttribute('cx', qx.toFixed(2));
    focusRing.setAttribute('cy', qy.toFixed(2));
    focusRing.style.display = focused ? '' : 'none';
    handle.setAttribute('aria-label', `Point at x₁ = ${q[0].toFixed(1)}, x₂ = ${q[1].toFixed(1)}. Arrow keys move it.`);

    const texts = readout(step, { t, depth, q });
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
    const bx = 598 - w;
    boxRect.setAttribute('x', bx.toFixed(2));
    boxRect.setAttribute('y', BOX_Y);
    boxRect.setAttribute('width', w.toFixed(2));
    boxRect.setAttribute('height', 16 + ROW * texts.length);
    rowsT.forEach((r, i) => {
      r.setAttribute('x', (bx + 12).toFixed(2));
      r.setAttribute('y', BOX_Y + 8 + ROW * i + ROW / 2 + 5);
    });

    tOut.textContent = fmtT(t);
    tInput.setAttribute('aria-valuetext', `x₁ < ${fmtT(t)}`);
    dOut.textContent = String(depth);
    dInput.setAttribute('aria-valuetext', `max depth ${depth}`);
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

  const sideWords = (c) => `${c[0]} circles, class 0, and ${c[1]} triangles, class 1`;
  const spoken = (i) => {
    if (i === 0) {
      const sc = splitScore(0, T_GUESS);
      return `One question: is x₁ less than ${T_GUESS}? Yes side: ${sideWords(sc.left)}, so it predicts class 0. No side: ${sideWords(sc.right)}, so it predicts class 1.`;
    }
    if (i === 1) {
      const sc = splitScore(0, t);
      return `Threshold x₁ < ${fmtT(t)}. Yes side: ${sideWords(sc.left)}. No side: ${sideWords(sc.right)}. Weighted Gini ${fmtG(sc.weighted)}.`;
    }
    if (i === 2) return `Best split on x₁: x₁ < ${fmtT(rc.byFeature[0].threshold)}, weighted Gini ${fmtG(rc.byFeature[0].impurity)}. Best on x₂: x₂ < ${fmtT(rc.byFeature[1].threshold)}, weighted Gini ${fmtG(rc.byFeature[1].impurity)}. The tree takes the x₂ split.`;
    const r = rows[depth - 1];
    if (i === 3) {
      const note = r.depth > DISPLAY_DEPTH ? ` The diagram shows the first ${DISPLAY_DEPTH} levels; the shading shows the whole tree.` : '';
      return `Max depth ${r.d}: ${r.leaves} leaves, training accuracy ${pct(r.train)}.${note}`;
    }
    if (i === 4) return `Max depth ${r.d}: training accuracy ${pct(r.train)}, test accuracy ${pct(r.test)}. Best test accuracy at depth ${best}.`;
    const p = pathOf(treeAt(best), q);
    const qs = p.steps.map((st) => `${question(st.node)}? ${st.yes ? 'yes' : 'no'}.`).join(' ');
    return `Point at x₁ = ${q[0].toFixed(1)}, x₂ = ${q[1].toFixed(1)}. ${qs} The leaf holds ${sideWords(p.leaf.counts)}: predicted class ${p.leaf.cls}, the ${p.leaf.cls ? 'triangle' : 'circle'}.`;
  };
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  // Redraw at most once a frame while a slider or the point moves.
  let pending = false;
  const queue = () => {
    if (pending) return;
    pending = true;
    const run = () => { pending = false; render(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  };

  draggable(handle, {
    bounds: { x0: PX0, x1: PX1, y0: PY0, y1: PY1 },
    onDrag: (px, py) => {
      q = [clamp(X.invert(px), 0, 10), clamp(Y.invert(py), 0, 10)];
      queue();
      say(spoken(5));
    },
  });
  handle.addEventListener('focus', () => {
    let kb = true;
    try { kb = handle.matches(':focus-visible'); } catch { /* keep the ring */ }
    focused = kb;
    render();
  });
  handle.addEventListener('blur', () => { focused = false; render(); });

  tInput.addEventListener('input', () => {
    t = clamp(Math.round(Number(tInput.value) * 20) / 20, 0, 10);
    queue();
    say(spoken(1));
  });
  dInput.addEventListener('input', () => {
    depth = clamp(Math.round(Number(dInput.value)), 1, 8);
    queue();
    say(spoken(step === 4 ? 4 : 3));
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
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(() => { drawnKey = ''; render(); }, () => {});

  return {
    count: 6,
    show(i, animate) {
      step = i;
      showGroup(tGroup, i === 1);
      showGroup(dGroup, i === 3 || i === 4);
      const live = i === 5;
      handle.setAttribute('tabindex', live ? 0 : -1);
      if (!live && doc.activeElement === handle) handle.blur();
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
