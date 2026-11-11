// Lesson L5.2, "Embeddings: words as points": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-embeddings.test.mjs);
// the DOM is only touched inside createStage. Every loss, gradient and training
// step is embed.js's skip-gram code, and every number in the page's text comes
// from pageValues or storyValues.

import { worldSentences, WORLD } from '../assets/world.js';
import { wordTokens, skipgramPairs, skipgramInit, skipgramTrainer, cosine } from '../assets/embed.js';
import { el, trianglePoints } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const SEED_DATA = 251;
export const SEED_INIT = 252;
export const SEED_SHUFFLE = 253;
export const N_SENTENCES = 1500;
export const DIM = 2;
export const WINDOW = 2;
export const ETA = 0.05;
// Every step uses every pair (one full batch), so the run is smooth and the same
// in any browser to far more digits than the page shows. 200 steps.
export const STEPS = 200;
// The sentence drawn in step 2: the first "lives" sentence of the data.
export const EXAMPLE_KIND = ' lives ';

let DATA = null;
export function data() {
  if (DATA) return DATA;
  const sentences = worldSentences({ n: N_SENTENCES, seed: SEED_DATA });
  const { words, ids } = wordTokens(sentences);
  const pairs = skipgramPairs(ids, WINDOW);
  const example = sentences.findIndex((s) => s.includes(EXAMPLE_KIND));
  DATA = { sentences, words, ids, pairs, example };
  return DATA;
}

export const startModel = () => skipgramInit({ V: data().words.length, d: DIM, seed: SEED_INIT });

// ---- groups ----------------------------------------------------------------------

export const GROUPS = ['name', 'food', 'place', 'other'];
export const GROUP_LABEL = { name: 'names', food: 'foods', place: 'places', other: 'other words' };
export function groupOf(word) {
  if (WORLD.people.includes(word)) return 'name';
  if (WORLD.foods.includes(word)) return 'food';
  if (WORLD.places.includes(word)) return 'place';
  return 'other';
}

// ---- step 1: one-hot vectors ----------------------------------------------------

export function oneHot(i, V) {
  const v = new Array(V).fill(0);
  v[i] = 1;
  return v;
}

// Every pair of different words: the smallest and largest distance and cosine.
export function oneHotStats(V) {
  let n = 0;
  let dMin = Infinity, dMax = -Infinity, cMin = Infinity, cMax = -Infinity;
  for (let i = 0; i < V; i++) {
    const a = oneHot(i, V);
    for (let j = i + 1; j < V; j++) {
      const b = oneHot(j, V);
      let s = 0;
      for (let k = 0; k < V; k++) s += (a[k] - b[k]) ** 2;
      const d = Math.sqrt(s);
      const c = cosine(a, b);
      n++;
      dMin = Math.min(dMin, d); dMax = Math.max(dMax, d);
      cMin = Math.min(cMin, c); cMax = Math.max(cMax, c);
    }
  }
  return { pairs: n, dMin, dMax, cMin, cMax };
}

// The words drawn as one-hot rows in step 1.
export const ONEHOT_SHOWN = ['mia', 'tom', 'figs', 'plums', 'sea', 'lake'];

// ---- step 2: the pairs of one sentence --------------------------------------------

// The example sentence's tokens and, for the centre at position pos, its pairs.
export function examplePairs(pos) {
  const { words, ids, example } = data();
  const s = ids[example];
  const tokens = s.map((i) => words[i]);
  const ctx = [];
  for (let j = Math.max(0, pos - WINDOW); j <= Math.min(s.length - 1, pos + WINDOW); j++) if (j !== pos) ctx.push(j);
  return { tokens, pos, ctx, pairs: ctx.map((j) => [tokens[pos], tokens[j]]) };
}

// How many sentences hold two different foods (none: each sentence names at most one).
export function sentencesWithTwoFoods() {
  return data().sentences.filter((s) => WORLD.foods.filter((f) => s.split(/[ .]/).includes(f)).length > 1).length;
}

// ---- training --------------------------------------------------------------------

export function makeTrainer(model = startModel()) {
  const { pairs } = data();
  return skipgramTrainer({ model, pairs, batch: pairs.length, kind: 'adam', hyper: { eta: ETA }, seed: SEED_SHUFFLE });
}

// The live run: advance(k) takes at most k steps and never runs past STEPS.
export function makeLive() {
  const tr = makeTrainer();
  const live = { trainer: tr };
  live.advance = (k) => {
    for (let i = 0; i < k && tr.state.iter < STEPS && !tr.state.diverged; i++) tr.step(1);
  };
  live.reset = () => { tr.reset(); };
  live.done = () => tr.state.iter >= STEPS || tr.state.diverged;
  return live;
}

// ---- similarity -------------------------------------------------------------------

export const vecOf = (model, word) => model.Win[data().words.indexOf(word)];
// The angle between two vectors in degrees, from 0 to 180.
export function angleDeg(a, b) {
  const c = Math.max(-1, Math.min(1, cosine(a, b)));
  return (Math.acos(c) * 180) / Math.PI;
}

// Every other word ranked by cosine with `word`, best first (ties by word order).
export function ranking(model, word) {
  const { words } = data();
  const a = vecOf(model, word);
  return words
    .filter((w) => w !== word)
    .map((w) => ({ word: w, group: groupOf(w), cos: cosine(a, vecOf(model, w)), angle: angleDeg(a, vecOf(model, w)) }))
    .sort((p, q) => q.cos - p.cos);
}

// For names, foods and places: how many have a nearest neighbour (by cosine, among
// all other words) from the same group.
export function groupShares(model) {
  const { words } = data();
  const out = {};
  for (const g of ['name', 'food', 'place']) {
    const ws = words.filter((w) => groupOf(w) === g);
    out[g] = { ok: ws.filter((w) => ranking(model, w)[0].group === g).length, n: ws.length };
  }
  return out;
}

// The mean cosine over every pair of different words, one from group a and one from b.
export function meanCos(model, a, b) {
  const { words } = data();
  let s = 0;
  let n = 0;
  for (const x of words) {
    if (groupOf(x) !== a) continue;
    for (const y of words) {
      if (x === y || groupOf(y) !== b) continue;
      s += cosine(vecOf(model, x), vecOf(model, y));
      n++;
    }
  }
  return s / n;
}

// ---- drawing helpers (pure) ---------------------------------------------------------

// The picture keeps each vector's direction exactly and draws its length on a
// square-root scale, so short and long vectors both fit. R_MAX is the largest
// square-rooted length the plot holds (length 6.25).
export const R_MAX = 2.5;
export function plotPoint(v) {
  const len = Math.hypot(v[0], v[1]);
  if (len === 0) return [0, 0];
  const r = Math.sqrt(len);
  return [(v[0] / len) * r, (v[1] / len) * r];
}

// Labels next to points without overlaps where possible. items: {x, y, w} in px
// (w is the label's width); h is the label height. Each label goes to the right
// of its point, or is moved up or down (then to the left) to the first free slot.
// Returns, per item, {x, y} of the text's start (baseline) and moved: true when
// the label is away from its point (the page draws a thin leader line).
export function placeLabels(items, { h = 13, gap = 7, bounds = [0, 0, 600, 350] } = {}) {
  const placed = [];
  const out = new Array(items.length);
  const order = items.map((_, i) => i).sort((a, b) => items[a].y - items[b].y || items[a].x - items[b].x);
  const hits = (b) => placed.some((p) => b.x0 < p.x1 && b.x1 > p.x0 && b.y0 < p.y1 && b.y1 > p.y0);
  const [bx0, by0, bx1, by1] = bounds;
  for (const i of order) {
    const { x, y, w } = items[i];
    let choice = null;
    for (const side of [1, -1]) {
      for (let k = 0; k <= 10 && !choice; k++) {
        for (const sgn of k === 0 ? [0] : [1, -1]) {
          const dy = sgn * k * h;
          const tx = side > 0 ? x + gap : x - gap - w;
          const ty = y + 4 + dy;
          const box = { x0: tx - 1, x1: tx + w + 1, y0: ty - h + 2, y1: ty + 2 };
          if (box.x0 < bx0 || box.x1 > bx1 || box.y0 < by0 || box.y1 > by1) continue;
          if (!hits(box)) { choice = { x: tx, y: ty, moved: k > 0, box }; break; }
        }
      }
      if (choice) break;
    }
    if (!choice) {
      const tx = x + gap;
      const ty = y + 4;
      choice = { x: tx, y: ty, moved: false, box: { x0: tx, x1: tx + w, y0: ty - h + 2, y1: ty + 2 } };
    }
    placed.push(choice.box);
    out[i] = { x: choice.x, y: choice.y, moved: choice.moved };
  }
  return out;
}

// ---- formatting ---------------------------------------------------------------------

export function num(v, d) {
  const s = v.toFixed(d);
  if (Number(s) === 0) return (0).toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}
export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
export const shareText = (s) => `${s.ok} of ${s.n}`;

// ---- read-outs --------------------------------------------------------------------

export function liveInfo(live, running) {
  const s = live.trainer.state;
  return { iter: s.iter, loss: s.epochLoss.at(-1), start: s.epochLoss[0], running, diverged: s.diverged, done: live.done() };
}

export function readout(step, s) {
  if (step === 0) return [`${s.V} words → ${s.V} numbers each, one of them 1`, `all ${count(s.stats.pairs)} pairs of different words:`, `distance √2 = ${num(s.stats.dMax, 3)}, cosine ${num(s.stats.cMax, 0)}`];
  if (step === 1) {
    const e = s.ex;
    return [`sentence ${s.index + 1} of ${count(N_SENTENCES)} · centre: ${e.tokens[e.pos]}`, `${e.pairs.length} pairs from this centre`, `${count(s.nPairs)} pairs in the whole text`];
  }
  const where = s.live.iter === 0 ? 'random start' : `${s.live.iter} of ${STEPS} steps`;
  if (step === 2) {
    return [`step ${s.live.iter} of ${STEPS} · ${trainStatus(s.live)}`, `loss ℒ: ${num(s.live.loss, 3)}`, `start ${num(s.live.start, 3)} · ln ${s.V} = ${num(Math.log(s.V), 3)}`];
  }
  if (step === 3) {
    const top = s.rank[0];
    return [`model: ${where}`, `nearest to "${s.word}": ${top.word}`, `cosine ${num(top.cos, 3)} · angle ${num(top.angle, 1)}°`];
  }
  if (step === 4) {
    const g = s.shares;
    return [`model: ${where}`, `nearest is in the same group:`, `names ${shareText(g.name)} · foods ${shareText(g.food)} · places ${shareText(g.place)}`];
  }
  return [`model: ${where}`, `${s.V} words × ${DIM} numbers = ${s.V * DIM} in the embeddings`, 'real embeddings: hundreds of numbers per token'];
}

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const { words, pairs, sentences, example } = data();
  const V = words.length;
  const st = oneHotStats(V);
  return {
    'n-words': String(V),
    'n-sentences': count(sentences.length),
    'n-pairs': count(pairs.length),
    'n-word-pairs': count(st.pairs),
    'onehot-dist': num(st.dMax, 3),
    'ln-v': num(Math.log(V), 2),
    'example': sentences[example],
    'two-foods': String(sentencesWithTwoFoods()),
    'n-params': String(2 * V * DIM),
  };
}

// The three words that sit with the foods after the run (step 5 names them).
export const CROWD_FOOD = ['are', 'and', 'friends'];

// The values quoted from the seeded run. end(model) and start(model) are pure, so
// the drift test can apply them to a run from nudged starting weights.
export function runValues(start, end, startLoss, endLoss) {
  const v = {};
  v['loss-start'] = num(startLoss, 2);
  v['loss-end'] = num(endLoss, 2);
  const s0 = groupShares(start);
  const s1 = groupShares(end);
  for (const g of ['name', 'food', 'place']) {
    v[`start-${g}`] = String(s0[g].ok);
    v[`end-${g}`] = String(s1[g].ok);
  }
  for (const g of ['name', 'food', 'place']) {
    v[`within-${g}`] = num(meanCos(end, g, g), 2);
    v[`start-within-${g}`] = num(meanCos(start, g, g), 2);
  }
  v['name-place'] = num(meanCos(end, 'name', 'place'), 2);
  v['name-food'] = num(meanCos(end, 'name', 'food'), 2);
  v['food-place'] = num(meanCos(end, 'food', 'place'), 2);
  let fmin = Infinity;
  for (const a of WORLD.foods) for (const b of WORLD.foods) if (a !== b) fmin = Math.min(fmin, cosine(vecOf(end, a), vecOf(end, b)));
  v['food-min'] = num(fmin, 3);
  // Words that crowd in with the foods and the places in two dimensions.
  let crowd = Infinity;
  for (const w of CROWD_FOOD) for (const f of WORLD.foods) crowd = Math.min(crowd, cosine(vecOf(end, w), vecOf(end, f)));
  v['crowd-food-min'] = num(crowd, 3);
  let thePl = Infinity;
  for (const pl of WORLD.places) thePl = Math.min(thePl, cosine(vecOf(end, 'the'), vecOf(end, pl)));
  v['the-place-min'] = num(thePl, 3);
  // For every food, how many other foods come before the first word that is not a food.
  v['food-first'] = String(Math.min(...WORLD.foods.map((f) => ranking(end, f).findIndex((r) => r.group !== 'food'))));
  v['apples-pears'] = num(cosine(vecOf(end, 'apples'), vecOf(end, 'pears')), 3);
  return v;
}

let STORY = null;
export function storyValues() {
  if (STORY) return STORY;
  const live = makeLive();
  const start = live.trainer.state.model;
  while (!live.done()) live.advance(STEPS);
  const s = live.trainer.state;
  STORY = runValues(start, s.model, s.epochLoss[0], s.epochLoss[STEPS]);
  return STORY;
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const INK = '#e2e8f0';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const FONT = "'JetBrains Mono', monospace";
const RY = [292, 314, 336];

// Marker shapes: names circles, foods triangles, places squares, other words small diamonds.
const MARK = {
  name: { fill: CYAN, stroke: CYAN },
  food: { fill: VIOLET, stroke: VIOLET },
  place: { fill: STRONG, stroke: STRONG },
  other: { fill: 'none', stroke: MUTED },
};
function markerPath(group, r = 5) {
  const f = (v) => Number(v.toFixed(2));
  if (group === 'food') return trianglePoints(0, 0, r).map(([x, y], k) => `${k ? 'L' : 'M'}${f(x)},${f(y)}`).join('') + 'Z';
  if (group === 'place') { const s = f((r * Math.sqrt(Math.PI)) / 2); return `M${-s},${-s}H${s}V${s}H${-s}Z`; }
  if (group === 'other') { const s = f(r * 0.9); return `M0,${-s}L${s},0L0,${s}L${-s},0Z`; }
  return `M${-r},0A${r},${r} 0 1,0 ${r},0A${r},${r} 0 1,0 ${-r},0Z`;
}

// The scatter: centre and pixel radius for R_MAX.
const CX = 180;
const CY = 158;
const RPX = 132;
const toPx = (v) => { const [x, y] = plotPoint(v); return [CX + (x / R_MAX) * RPX, CY - (y / R_MAX) * RPX]; };

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const D = data();
  const V = D.words.length;
  const STATS = oneHotStats(V);
  let step = 0;
  let pos = 1;
  let word = 'sea';
  let live = null;

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 350');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  const marker = (parent, group, r = 5) => {
    const m = MARK[group];
    return el('path', { d: markerPath(group, r), fill: m.fill, 'fill-opacity': group === 'other' ? 0 : 0.85, stroke: m.stroke, 'stroke-width': 1.5 }, parent);
  };
  const getLive = () => {
    if (!live) live = makeLive();
    return live;
  };

  // ---- step 1: one-hot rows and the distance between them.
  const ohG = el('g', {}, svg);
  const OX = 96;
  const OW = 17;
  label(ohG, 16, 22, 'word');
  label(ohG, OX, 22, `its one-hot vector: ${V} numbers, one per word in the list`);
  ONEHOT_SHOWN.forEach((w, r) => {
    const y = 34 + r * 24;
    label(ohG, 16, y + 14, w, 'start', STRONG, 13);
    const idx = D.words.indexOf(w);
    for (let k = 0; k < V; k++) {
      const on = k === idx;
      el('rect', { x: OX + k * OW, y, width: OW - 2, height: 18, rx: 2, fill: on ? CYAN : 'none', 'fill-opacity': on ? 0.45 : 0, stroke: on ? CYAN : LINE }, ohG);
      label(ohG, OX + k * OW + (OW - 2) / 2, y + 13, on ? '1' : '0', 'middle', on ? STRONG : LINE, 10);
    }
  });
  label(ohG, 16, 196, 'distance between them (every pair of different words is the same)');
  const DTX = 96;
  const DTW = 76;
  ONEHOT_SHOWN.forEach((w, j) => label(ohG, DTX + j * DTW + DTW / 2, 216, w, 'middle', STRONG, 11));
  ONEHOT_SHOWN.forEach((a, i) => {
    if (i > 1) return;
    const y = 236 + i * 20;
    label(ohG, 16, y, a, 'start', STRONG, 11);
    ONEHOT_SHOWN.forEach((b, j) => {
      const va = oneHot(D.words.indexOf(a), V);
      const vb = oneHot(D.words.indexOf(b), V);
      const d = Math.sqrt(va.reduce((s, x, k) => s + (x - vb[k]) ** 2, 0));
      label(ohG, DTX + j * DTW + DTW / 2, y, num(d, 3), 'middle', a === b ? MUTED : INK, 11);
    });
  });

  // ---- step 2: one sentence, a centre word and its window.
  const sgG = el('g', {}, svg);
  const ex0 = examplePairs(0);
  const TW = 84;
  const TX0 = 300 - (ex0.tokens.length * TW) / 2;
  label(sgG, 16, 24, `sentence ${D.example + 1} of the text, as words (the full stop is a word too)`);
  const tokBoxes = ex0.tokens.map((w, j) => {
    const x = TX0 + j * TW;
    const r = el('rect', { x: x + 4, y: 92, width: TW - 8, height: 34, rx: 5, fill: 'none', stroke: LINE, 'stroke-width': 1.5 }, sgG);
    const t = label(sgG, x + TW / 2, 114, w, 'middle', STRONG, 14);
    const pos2 = label(sgG, x + TW / 2, 144, String(j + 1), 'middle', MUTED, 11);
    return { r, t, pos2, x: x + TW / 2 };
  });
  const arcs = el('g', {}, sgG);
  const pairsT = label(sgG, 16, 190, '', 'start', STRONG, 13);
  const pairsT2 = label(sgG, 16, 212, '', 'start', STRONG, 13);
  label(sgG, 16, 244, `the model is asked: given the centre word, give each word in the list a probability`, 'start', MUTED, 12);
  label(sgG, 16, 262, `of being one of its neighbours (up to ${WINDOW} words either side, in the same sentence)`, 'start', MUTED, 12);

  // ---- the scatter (steps 3, 4 and 6) and the ranked list (step 5).
  const scG = el('g', {}, svg);
  for (const L of [1, 4]) {
    const rr = (Math.sqrt(L) / R_MAX) * RPX;
    el('circle', { cx: CX, cy: CY, r: P(rr), fill: 'none', stroke: LINE, 'stroke-dasharray': '3 4' }, scG);
    label(scG, CX + 4, P(CY - rr - 3), `length ${L}`, 'start', MUTED, 10);
  }
  el('line', { x1: CX - RPX, x2: CX + RPX, y1: CY, y2: CY, stroke: LINE }, scG);
  el('line', { x1: CX, x2: CX, y1: CY - RPX, y2: CY + RPX, stroke: LINE }, scG);
  const rays = el('g', {}, scG);
  const leaders = el('g', {}, scG);
  const pts = D.words.map((w) => {
    const g = el('g', {}, scG);
    const m = marker(g, groupOf(w));
    const t = label(scG, 0, 0, w, 'start', groupOf(w) === 'other' ? MUTED : STRONG, 11);
    return { w, group: groupOf(w), g, m, t };
  });
  const selRing = el('circle', { r: 10, fill: 'none', stroke: CYAN, 'stroke-width': 2 }, scG);
  // Legend and loss curve on the right.
  const legG = el('g', {}, scG);
  ['name', 'food', 'place', 'other'].forEach((g, k) => {
    const y = 26 + k * 20;
    const m = marker(legG, g);
    m.setAttribute('transform', `translate(384 ${y - 4})`);
    label(legG, 396, y, GROUP_LABEL[g], 'start', g === 'other' ? MUTED : STRONG, 12);
  });
  label(legG, 372, 120, 'direction: exact', 'start', MUTED, 11);
  label(legG, 372, 134, 'length: square-rooted', 'start', MUTED, 11);
  const lossG = el('g', {}, scG);
  const LX0 = 400, LX1 = 584, LY0 = 262, LY1 = 172;
  const LMIN = 2.4, LMAX = 3.4;
  const lx = (i) => LX0 + (i / STEPS) * (LX1 - LX0);
  const ly = (v) => LY0 + ((Math.min(LMAX, Math.max(LMIN, v)) - LMIN) / (LMAX - LMIN)) * (LY1 - LY0);
  label(lossG, 372, 160, 'loss ℒ by step', 'start', STRONG, 12);
  el('line', { x1: LX0, x2: LX0, y1: LY0, y2: LY1, stroke: LINE }, lossG);
  el('line', { x1: LX0, x2: LX1, y1: LY0, y2: LY0, stroke: LINE }, lossG);
  for (const v of [2.5, 3.0]) {
    el('line', { x1: LX0, x2: LX1, y1: P(ly(v)), y2: P(ly(v)), stroke: LINE, 'stroke-dasharray': '2 4' }, lossG);
    label(lossG, LX0 - 6, P(ly(v)) + 4, num(v, 1), 'end', MUTED, 10);
  }
  for (const i of [0, 100, 200]) label(lossG, P(lx(i)), LY0 + 14, String(i), 'middle', MUTED, 10);
  const lossPath = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 2, 'stroke-linejoin': 'round' }, lossG);
  const lossDot = el('circle', { r: 3.5, fill: CYAN, stroke: BG, 'stroke-width': 1 }, lossG);
  // Step 4's table of nearest words, in place of the loss curve.
  const nnG = el('g', {}, scG);
  const nnTitle = label(nnG, 372, 162, '', 'start', STRONG, 12);
  label(nnG, 372, 180, '#    word     cosine angle', 'start', MUTED, 11);
  const nnHead = nnG.lastChild;
  nnHead.style.whiteSpace = 'pre';
  const nnRows = Array.from({ length: 5 }, (_, k) => {
    const y = 198 + k * 16;
    const t = label(nnG, 372, y, '', 'start', INK, 11);
    t.style.whiteSpace = 'pre';
    const mg = el('g', { transform: `translate(392 ${y - 4})` }, nnG);
    return { t, mg, grp: null };
  });

  // Step 5: every other word ranked by cosine, in two columns.
  const rkG = el('g', {}, svg);
  const rkTitle = label(rkG, 16, 20, '', 'start', STRONG, 13);
  const RK = Math.ceil((V - 1) / 2);
  const rkRows = Array.from({ length: V - 1 }, (_, k) => {
    const col = k < RK ? 0 : 1;
    const x = 16 + col * 292;
    const y = 44 + (k % RK) * 18.5;
    const n = label(rkG, x + 18, y, String(k + 1), 'end', MUTED, 11);
    const mg = el('g', { transform: `translate(${x + 30} ${y - 4})` }, rkG);
    const w = label(rkG, x + 42, y, '', 'start', STRONG, 11);
    const zero = x + 160;
    el('line', { x1: zero, x2: zero, y1: y - 11, y2: y + 3, stroke: LINE }, rkG);
    const bar = el('rect', { x: zero, y: y - 9, height: 10, width: 0, fill: CYAN, 'fill-opacity': 0.5 }, rkG);
    const v = label(rkG, x + 282, y, '', 'end', INK, 11);
    return { n, mg, w, bar, v, zero, mk: null, grp: null };
  });
  label(rkG, 196, 44 + RK * 18.5 - 2, 'bar: cosine, from −1 (left) to 1 (right)', 'start', MUTED, 10);

  // ---- read-out rows.
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- controls: the centre-word slider (step 2), Train (steps 3 to 5), the word (steps 4, 5).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const pGroup = make('div');
  const pLab = make('label');
  const pIn = make('input', { type: 'range', min: '0', max: String(ex0.tokens.length - 1), step: '1', value: String(pos) });
  const pOut = make('output');
  pOut.setAttribute('aria-hidden', 'true');
  pLab.append('centre word', ' ', pIn, ' ', pOut);
  pGroup.append(pLab);
  const wGroup = make('div');
  const wLab = make('label');
  const wSel = make('select');
  for (const g of GROUPS) {
    const og = make('optgroup');
    og.label = GROUP_LABEL[g];
    for (const w of D.words.filter((x) => groupOf(x) === g)) og.append(make('option', { value: w, textContent: w }));
    wSel.append(og);
  }
  wSel.value = word;
  wLab.append('word', ' ', wSel);
  wGroup.append(wLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'train the embeddings');
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  controls.append(pGroup, trainGroup, wGroup);
  const setValueText = () => {
    const e = examplePairs(pos);
    pOut.textContent = e.tokens[pos];
    pIn.setAttribute('aria-valuetext', `word ${pos + 1} of ${e.tokens.length}: ${e.tokens[pos]}`);
  };

  function info() {
    const base = { V, stats: STATS, word };
    if (step === 0) return base;
    if (step === 1) return { ...base, ex: examplePairs(pos), index: D.example, nPairs: D.pairs.length };
    const lv = getLive();
    const m = lv.trainer.state.model;
    const s = { ...base, live: liveInfo(lv, loop.activity()), model: m };
    if (step === 3 || step === 4) s.rank = ranking(m, word);
    if (step === 4) s.shares = groupShares(m);
    return s;
  }

  function drawScatter(m, sel) {
    const wd = svg.clientWidth || 600;
    const size = wd < 500 ? 13 : 11;
    const pxs = D.words.map((w) => toPx(vecOf(m, w)));
    const items = pts.map((p, i) => ({ x: pxs[i][0], y: pxs[i][1], w: p.w.length * size * 0.62 }));
    const lab = placeLabels(items, { h: size + 2, bounds: [0, 0, 366, 286] });
    leaders.replaceChildren();
    pts.forEach((p, i) => {
      const [x, y] = pxs[i];
      p.g.setAttribute('transform', `translate(${P(x)} ${P(y)})`);
      p.t.setAttribute('x', P(lab[i].x));
      p.t.setAttribute('y', P(lab[i].y));
      p.t.setAttribute('font-size', size);
      if (lab[i].moved) {
        const lxx = lab[i].x < x ? lab[i].x + items[i].w : lab[i].x - 1;
        el('line', { x1: P(x), y1: P(y), x2: P(lxx), y2: P(lab[i].y - 4), stroke: MUTED, 'stroke-width': 0.6, 'stroke-opacity': 0.6 }, leaders);
      }
    });
    rays.replaceChildren();
    show(selRing, !!sel);
    if (sel) {
      const i = D.words.indexOf(sel.word);
      selRing.setAttribute('cx', P(pxs[i][0]));
      selRing.setAttribute('cy', P(pxs[i][1]));
      el('line', { x1: CX, y1: CY, x2: P(pxs[i][0]), y2: P(pxs[i][1]), stroke: CYAN, 'stroke-width': 1.5 }, rays);
      for (const r of sel.rank.slice(0, 5)) {
        const j = D.words.indexOf(r.word);
        el('line', { x1: CX, y1: CY, x2: P(pxs[j][0]), y2: P(pxs[j][1]), stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, rays);
      }
    }
  }

  function render() {
    show(ohG, step === 0);
    show(sgG, step === 1);
    show(scG, step === 2 || step === 3 || step === 5);
    show(rkG, step === 4);
    setValueText();
    const s = info();

    if (step === 1) {
      const e = s.ex;
      tokBoxes.forEach((b, j) => {
        const isC = j === e.pos;
        const isN = e.ctx.includes(j);
        b.r.setAttribute('stroke', isC ? CYAN : isN ? VIOLET : LINE);
        b.r.setAttribute('stroke-dasharray', isN ? '5 3' : '');
        b.r.setAttribute('stroke-width', isC ? 2.5 : 1.5);
        b.t.style.fill = isC || isN ? STRONG : MUTED;
      });
      arcs.replaceChildren();
      const cx = tokBoxes[e.pos].x;
      for (const j of e.ctx) {
        const nx = tokBoxes[j].x;
        const hgt = 18 + 14 * Math.abs(j - e.pos);
        el('path', { d: `M${P(cx)},88 Q${P((cx + nx) / 2)},${88 - 2 * hgt} ${P(nx)},88`, fill: 'none', stroke: VIOLET, 'stroke-width': 1.5, 'stroke-dasharray': '5 3' }, arcs);
      }
      const txt = e.pairs.map(([c, o]) => `(${c}, ${o})`);
      pairsT.textContent = `pairs (centre, neighbour): ${txt.slice(0, 2).join(' ')}`;
      pairsT2.textContent = txt.slice(2).join(' ');
    }

    if (step === 2 || step === 3 || step === 5) {
      drawScatter(s.model, step === 3 ? { word, rank: s.rank } : null);
      show(lossG, step !== 3);
      show(nnG, step === 3);
      if (step !== 3) {
        const L = getLive().trainer.state.epochLoss;
        lossPath.setAttribute('d', L.map((v, i) => `${i ? 'L' : 'M'}${P(lx(i))},${P(ly(v))}`).join(''));
        lossDot.setAttribute('cx', P(lx(L.length - 1)));
        lossDot.setAttribute('cy', P(ly(L.at(-1))));
      } else {
        nnTitle.textContent = `nearest to "${word}"`;
        nnRows.forEach((row, k) => {
          const r = s.rank[k];
          row.t.textContent = `${k + 1}    ${r.word.padEnd(8)} ${num(r.cos, 3).padStart(6)} ${num(r.angle, 1).padStart(5)}°`;
          row.t.style.fill = r.group === groupOf(word) ? STRONG : MUTED;
          if (row.grp !== r.group) {
            row.mg.replaceChildren();
            marker(row.mg, r.group, 4);
            row.grp = r.group;
          }
        });
      }
    }

    if (step === 4) {
      const rank = ranking(s.model, word);
      rkTitle.textContent = `every word ranked by cosine with "${word}"`;
      rkRows.forEach((row, k) => {
        const r = rank[k];
        row.n.textContent = String(k + 1);
        if (row.grp !== r.group) {
          row.mg.replaceChildren();
          marker(row.mg, r.group, 4);
          row.grp = r.group;
        }
        row.w.textContent = r.word;
        row.w.style.fill = r.group === 'other' ? MUTED : STRONG;
        const wpx = r.cos * 66;
        row.bar.setAttribute('x', P(wpx >= 0 ? row.zero : row.zero + wpx));
        row.bar.setAttribute('width', P(Math.abs(wpx)));
        row.bar.setAttribute('fill', r.cos >= 0 ? CYAN : VIOLET);
        row.v.textContent = num(r.cos, 3);
      });
    }

    const texts = readout(step, s);
    rowsT.forEach((t, k) => {
      t.textContent = texts[k] || '';
      t.setAttribute('x', step >= 2 && step !== 4 ? 372 : 16);
      t.setAttribute('font-size', step >= 2 && step !== 4 ? 11 : 13);
    });
    // The step 5 list fills the height, so its read-out goes on one line at the bottom.
    if (step === 4) {
      rowsT[0].textContent = '';
      rowsT[1].textContent = `${texts[0]} · ${texts[1]}`;
      rowsT[2].textContent = texts[2];
    }
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function ariaFor(s, texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Six words as one-hot vectors of ${V} numbers, each with a single 1, and the distance between them: ${num(STATS.dMax, 3)} for every pair of different words. ${rest}.`;
    if (step === 1) return `The sentence "${D.sentences[D.example]}" as words. The centre word is ${s.ex.tokens[s.ex.pos]}, and its neighbours within ${WINDOW} words are ${s.ex.ctx.map((j) => s.ex.tokens[j]).join(', ')}. ${rest}.`;
    if (step === 3) return `A scatter of the ${V} word vectors, with "${word}" ringed and lines to its five nearest words by cosine: ${s.rank.slice(0, 5).map((r) => `${r.word} ${num(r.cos, 3)}`).join(', ')}. ${rest}.`;
    if (step === 4) {
      const rank = ranking(s.model, word);
      return `Every other word ranked by cosine with "${word}": ${rank.map((r) => `${r.word} ${num(r.cos, 3)}`).join(', ')}. ${rest}.`;
    }
    return `A scatter of the ${V} word vectors: names as circles, foods as triangles, places as squares, other words as diamonds, and a curve of the loss by step. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  const loop = createTrainLoop({
    step: (k) => getLive().advance(k),
    render,
    done: () => getLive().done(),
    onStop: () => {
      buttons.sync();
      render();
      if (getLive().done()) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => getLive().reset(),
    reduced,
    work: (k) => getLive().advance(k),
  });
  trainGroup.addEventListener('click', () => { if (step >= 2 && step <= 4) { render(); announce(spoken()); } });

  pIn.addEventListener('input', () => {
    pos = Number(pIn.value) || 0;
    render();
    announce(spoken());
  });
  wSel.addEventListener('change', () => {
    word = wSel.value;
    render();
    announce(spoken());
  });

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i) {
      buttons.stop();
      step = i;
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      showGroup(pGroup, i === 1);
      showGroup(wGroup, i === 3 || i === 4);
      trainGroup.style.display = i >= 2 && i <= 4 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 2);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
