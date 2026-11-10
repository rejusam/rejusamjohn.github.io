// Lesson L5.3, "Predicting the next token": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-language-models.test.mjs);
// the DOM is only touched inside createStage. Every probability comes from nextProbs in
// text.js, every generated character from generateNgram, every held-out number from
// crossEntropyBits and unseenContextShare, and every number in the page's text from
// pageValues. Typed text is sanitised, shown with textContent only, and never stored
// or sent anywhere.

import { ALPHABET, worldText } from '../assets/world.js';
import {
  sanitise, charTokenise, ngramModel, nextProbs, crossEntropyBits, unseenContextShare,
  softmax, generateNgram,
} from '../assets/text.js';
import { el, trianglePoints } from '../assets/plot.js';

// ---- data and settings --------------------------------------------------------

export const TRAIN_SEED = 261;
export const HELD_SEED = 262;
export const GEN_SEED = 263;
export const SMALL_SEED = 264;
export const N_TRAIN = 2000;
export const N_HELD = 300;
export const N_SMALL = 50;
export const K = 0.01;
export const N_MAX = 8;
export const N_DEFAULT = 3;
export const TAUS = [0, 0.5, 1, 1.5];
export const GEN_LEN = 80;
export const MAX_INPUT = 80;
export const FIXED_PROMPT = 'mia li';
export const DEFAULT_PROMPT = 'mia lives ';
const A = ALPHABET.length;   // 28: space, full stop, a to z

const TEXTS = {};
export const corpus = (which = 'full') =>
  (TEXTS[which] ??= which === 'small' ? worldText({ n: N_SMALL, seed: SMALL_SEED }) : worldText({ n: N_TRAIN, seed: TRAIN_SEED }));
let HELD = null;
export const heldText = () => (HELD ??= worldText({ n: N_HELD, seed: HELD_SEED }));

const MODELS = new Map();
export function model(n, which = 'full') {
  const key = `${which}:${n}`;
  if (!MODELS.has(key)) MODELS.set(key, ngramModel(corpus(which), n, { k: K }));
  return MODELS.get(key);
}

// The last n characters of the (sanitised) text that are in the alphabet, or null if
// there are fewer than n of them. Characters outside the alphabet are skipped.
function lastKnown(text, n) {
  const ids = charTokenise(text).filter((i) => i >= 0);
  return ids.length >= n ? ids.slice(ids.length - n) : null;
}

// The counts after the prompt's context: how often each character followed it in training.
export function countsFor(prompt, n, which = 'full') {
  const { text } = sanitise(prompt, MAX_INPUT);
  const ids = lastKnown(text, n);
  const row = ids ? model(n, which).counts.get(ids.join(',')) : undefined;
  const counts = row ? row.slice() : new Array(A).fill(0);
  return { context: ids ? ids.map((i) => ALPHABET[i]).join('') : null, counts, total: counts.reduce((a, b) => a + b, 0) };
}

// Next-character probabilities after the prompt (nextProbs: add-k smoothing; uniform for
// an unseen context or fewer than n known characters).
export function predict(prompt, n, which = 'full') {
  const { text } = sanitise(prompt, MAX_INPUT);
  const c = countsFor(text, n, which);
  return { probs: nextProbs(model(n, which), text), context: c.context, seen: c.total > 0, total: c.total, counts: c.counts };
}

// p reshaped by temperature: softmax(log p / tau), which is p^(1/tau) normalised; tau 0 is the argmax.
export const tempProbs = (p, tau) => softmax(p.map((v) => Math.log(v)), tau);

// GEN_LEN new characters after the prompt (the continuation only).
export function generate(prompt, n, tau, seed, which = 'full') {
  const { text } = sanitise(prompt, MAX_INPUT);
  return generateNgram(model(n, which), { start: text, length: GEN_LEN, tau, seed }).slice(text.length);
}

// Typed text: lower-cased and capped by sanitise. Each character is a chip; ctx marks the
// characters the model uses as its context (the last n known ones).
export function promptChips(raw, n) {
  const { text, unknown } = sanitise(raw, MAX_INPUT);
  const cs = Array.from(text);
  const bad = new Set(unknown);
  const known = cs.map((_, i) => i).filter((i) => !bad.has(i));
  const ctxIdx = known.length >= n ? new Set(known.slice(known.length - n)) : new Set();
  const chips = cs.map((ch, i) => ({ ch, unknown: bad.has(i), ctx: ctxIdx.has(i) }));
  return { text, chips, unknown: unknown.length, context: ctxIdx.size ? [...ctxIdx].map((i) => cs[i]).join('') : null };
}

// Held-out and training bits per character, the share of held-out contexts never seen
// in training, and how many different contexts training has.
const STATS = new Map();
export function stats(n, which = 'full') {
  const key = `${which}:${n}`;
  if (!STATS.has(key)) {
    const m = model(n, which);
    STATS.set(key, {
      n,
      held: crossEntropyBits(m, heldText()),
      train: crossEntropyBits(m, corpus(which)),
      unseen: unseenContextShare(m, heldText()),
      contexts: m.counts.size,
    });
  }
  return STATS.get(key);
}
export const STAT_JOBS = ['full', 'small'].flatMap((which) => Array.from({ length: N_MAX }, (_, i) => [i + 1, which]));
export const statsTable = () => {
  for (const [n, which] of STAT_JOBS) stats(n, which);
  return tableNow();
};
const tableNow = () => ({
  full: Array.from({ length: N_MAX }, (_, i) => stats(i + 1, 'full')),
  small: Array.from({ length: N_MAX }, (_, i) => stats(i + 1, 'small')),
});

// ---- formatting ------------------------------------------------------------------

export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
export const shown = (s) => s.replace(/ /g, '␣');
const bits = (x) => x.toFixed(2);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
// Three decimals, or two significant figures for a very small probability.
const tiny = (p) => (p >= 0.001 ? p.toFixed(3) : p.toFixed(1 - Math.floor(Math.log10(p))));
const TAU_NAME = (t) => (t === 0 ? '0 (greedy)' : String(t));
const sup = (n) => String(n).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
const textName = (which) => (which === 'small' ? `small training text (${N_SMALL} sentences)` : `full training text (${count(N_TRAIN)} sentences)`);

// The three likeliest characters, as "k 0.488 · v 0.460 · m 0.051".
export function top3(p) {
  return p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, 3)
    .map(([v, i]) => `${shown(ALPHABET[i])} ${v.toFixed(3)}`).join(' · ');
}

// ---- read-outs ---------------------------------------------------------------------

export function readout(step, s) {
  if (step === 0) {
    return [`text so far: ${shown(FIXED_PROMPT)} · context (last ${N_DEFAULT} characters): ${shown(s.pred.context)}`,
      `likeliest next: ${top3(s.pred.probs)}`, `all ${A} probabilities add up to 1`];
  }
  if (step === 1) {
    const c = s.pred;
    const i = c.counts.indexOf(Math.max(...c.counts));
    const ctx = shown(c.context);
    return [`c(${ctx}) = ${count(c.total)}: the context ${ctx} occurs ${count(c.total)} times`,
      `p(${ALPHABET[i]} | ${ctx}) = (${c.counts[i]} + ${K}) / (${count(c.total)} + ${K} × ${A}) = ${c.probs[i].toFixed(3)}`,
      `a character never seen after ${ctx}: ${K} / ${count(c.total + K * A)} = ${tiny(K / (c.total + K * A))}`];
  }
  if (step === 2 || step === 3) {
    const p = step === 3 ? tempProbs(s.pred.probs, s.tau) : s.pred.probs;
    const ctx = s.pred.context === null ? null : shown(s.pred.context);
    let first;
    if (ctx === null) first = `fewer than ${s.n} known characters: no context`;
    else if (!s.pred.seen) first = `context ${ctx} never seen in training`;
    else first = `context ${ctx} (last ${s.n} known) · seen ${count(s.pred.total)} times`;
    const flat = !s.pred.seen;   // uniform, at every τ (an all-way tie)
    const second = (what) => (flat ? `all ${A} equal: 1/${A} = ${(1 / A).toFixed(3)} each` : `${what}: ${top3(p)}`);
    if (step === 2) return [first, second('likeliest next'), `n = ${s.n} · characters outside the ${A} skipped: ${s.unknown}`];
    return [`τ = ${TAU_NAME(s.tau)} · n = ${s.n} · context ${ctx === null ? 'none' : ctx}${ctx !== null && flat ? ' (never seen)' : ''}`,
      second('likeliest at this τ'), `seed ${s.seed} · each Generate press uses the next seed`];
  }
  if (step === 4) {
    if (!s.table) return ['computing… (a moment)', '', ''];
    const r = s.table[s.which][s.n - 1];
    return [`n = ${s.n} · ${textName(s.which)}`, `held-out: ${bits(r.held)} bits per character · training: ${bits(r.train)}`,
      `held-out contexts never seen in training: ${pct(r.unseen)}`];
  }
  if (step === 5) {
    if (!s.table) return ['computing… (a moment)', '', ''];
    const sm = s.table.small[N_MAX - 1];
    return [`n = ${N_MAX}, full text: ${count(s.table.full[N_MAX - 1].contexts)} contexts seen of ${A}${sup(N_MAX)} possible`,
      `small text, n = ${N_MAX}: ${pct(sm.unseen)} unseen, costing ${(sm.unseen * Math.log2(A)).toFixed(2)} bits per character`,
      `full text: ${pct(Math.max(...s.table.full.map((r) => r.unseen)))} unseen at every n`];
  }
  return [];
}

// ---- values written into the page ---------------------------------------------------

export function pageValues(table) {
  const c = countsFor(FIXED_PROMPT, N_DEFAULT);
  const ix = (ch) => ALPHABET.indexOf(ch);
  const sm8 = table.small[N_MAX - 1];
  return {
    'n-train': count(N_TRAIN),
    'n-held': count(N_HELD),
    'n-small': count(N_SMALL),
    'train-chars': count(corpus('full').length),
    'ctx-total': count(c.total),
    'top-count': String(c.counts[ix('k')]),
    'top-p': ((c.counts[ix('k')] + K) / (c.total + K * A)).toFixed(3),
    'v-count': String(c.counts[ix('v')]),
    'm-count': String(c.counts[ix('m')]),
    'zero-p': tiny(K / (c.total + K * A)),
    'uniform-bits': Math.log2(A).toFixed(2),
    'full-bits-1': bits(table.full[0].held),
    'full-bits-3': bits(table.full[2].held),
    'full-bits-8': bits(table.full[N_MAX - 1].held),
    'small-bits-3': bits(table.small[2].held),
    'small-bits-8': bits(sm8.held),
    'small-train-8': bits(sm8.train),
    'small-unseen-8': pct(sm8.unseen),
    'small-unseen-cost-8': (sm8.unseen * Math.log2(A)).toFixed(2),
    'full-contexts-8': String(table.full[N_MAX - 1].contexts),
    'possible-8': count(A ** N_MAX),
    'gen-len': String(GEN_LEN),
    'tie-count': String(Math.max(...countsFor('the ', N_DEFAULT).counts)),
    'n-words': String(new Set(corpus('full').split(/[ .]+/).filter(Boolean)).size),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const VIOLET = '#a78bfa';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [280, 302, 324];
const FONT = "'JetBrains Mono', monospace";

export function createStage(svg, { controls, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let n = N_DEFAULT;
  let tauIx = 2;           // tau 1
  let seed = GEN_SEED;
  let which = 'full';
  let typed = DEFAULT_PROMPT;
  let table = null;        // the held-out numbers, computed on first show of step 5 or 6
  let computing = false;   // nothing on this page animates, so reduced motion needs nothing extra

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 340');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };

  // ---- steps 1 to 4: one column per character, probabilities (or counts) as bars.
  const barG = el('g', {}, svg);
  const CX0 = 40;
  const CW = 19.5;
  const cx = (i) => CX0 + i * CW;
  const BY0 = 222;
  const BH = 170;
  const barTitle = label(barG, 16, 24, '', 'start', STRONG);
  const barLegend = label(barG, 584, 24, '', 'end', MUTED, 11);
  el('line', { x1: CX0 - 4, x2: cx(A), y1: BY0, y2: BY0, stroke: LINE }, barG);
  const ticks = [0, 0.5, 1].map((v) => {
    const y = BY0 - v * BH;
    const ln = el('line', { x1: CX0 - 4, x2: cx(A), y1: y, y2: y, stroke: LINE, 'stroke-dasharray': '2 4' }, barG);
    const t = label(barG, CX0 - 8, y + 4, String(v), 'end', MUTED, 10);
    return { ln, t, v };
  });
  const bars = ALPHABET.map((ch, i) => {
    const ref = el('rect', { x: P(cx(i) + 2), width: CW - 4, fill: 'none', stroke: VIOLET, 'stroke-width': 1.25, 'stroke-dasharray': '3 2' }, barG);
    const rect = el('rect', { x: P(cx(i) + 2), width: CW - 4, fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN, 'stroke-width': 1 }, barG);
    const val = label(barG, P(cx(i) + CW / 2), 0, '', 'middle', STRONG, 9);
    label(barG, P(cx(i) + CW / 2), BY0 + 18, shown(ch), 'middle', STRONG, 13);
    return { ref, rect, val };
  });
  label(barG, CX0 - 6, BY0 + 18, 'next', 'end', MUTED, 10);

  // ---- step 5: held-out bits per n, bars; training bits as dashed marks.
  const nG = el('g', {}, svg);
  const NX0 = 92;
  const NX1 = 584;
  const NW = (NX1 - NX0) / N_MAX;
  const NY0 = 196;
  const NH = 136;
  const YMAX = 2.5;
  const ny = (b) => NY0 - (b / YMAX) * NH;
  const nTitle = label(nG, 16, 24, '', 'start', STRONG);
  label(nG, 584, 44, 'bars: held-out · dashes: training', 'end', MUTED, 11);
  for (const b of [0, 0.5, 1, 1.5, 2, 2.5]) {
    el('line', { x1: NX0, x2: NX1, y1: P(ny(b)), y2: P(ny(b)), stroke: LINE, 'stroke-dasharray': b ? '2 4' : null }, nG);
    label(nG, NX0 - 8, P(ny(b)) + 4, b.toFixed(1), 'end', MUTED, 10);
  }
  label(nG, 16, 44, 'bits per character', 'start', MUTED, 11);
  const nBars = Array.from({ length: N_MAX }, (_, i) => {
    const x = NX0 + i * NW + 10;
    const rect = el('rect', { x: P(x), width: P(NW - 20), fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN }, nG);
    const tr = el('line', { x1: P(x - 4), x2: P(x + NW - 16), stroke: VIOLET, 'stroke-width': 2.5, 'stroke-dasharray': '5 3' }, nG);
    const val = label(nG, P(x + (NW - 20) / 2), 0, '', 'middle', STRONG, 11);
    label(nG, P(x + (NW - 20) / 2), NY0 + 17, `n=${i + 1}`, 'middle', STRONG, 12);
    const un = label(nG, P(x + (NW - 20) / 2), NY0 + 40, '', 'middle', MUTED, 11);
    return { rect, tr, val, un };
  });
  label(nG, NX0 - 8, NY0 + 40, 'unseen', 'end', MUTED, 10);

  // ---- step 6: held-out bits against n for both texts.
  const cmpG = el('g', {}, svg);
  label(cmpG, 16, 24, 'held-out bits per character, by context length n', 'start', STRONG);
  for (const b of [0, 0.5, 1, 1.5, 2, 2.5]) {
    el('line', { x1: NX0, x2: NX1, y1: P(ny(b)), y2: P(ny(b)), stroke: LINE, 'stroke-dasharray': b ? '2 4' : null }, cmpG);
    label(cmpG, NX0 - 8, P(ny(b)) + 4, b.toFixed(1), 'end', MUTED, 10);
  }
  const mx = (i) => NX0 + i * NW + NW / 2;
  for (let i = 0; i < N_MAX; i++) label(cmpG, P(mx(i)), NY0 + 17, `n=${i + 1}`, 'middle', STRONG, 12);
  const fullPath = el('path', { fill: 'none', stroke: CYAN, 'stroke-width': 2 }, cmpG);
  const smallPath = el('path', { fill: 'none', stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, cmpG);
  const fullDots = Array.from({ length: N_MAX }, () => el('circle', { r: 4, fill: CYAN }, cmpG));
  const smallDots = Array.from({ length: N_MAX }, () => el('path', { fill: VIOLET }, cmpG));
  // Legend in the empty top right of the plot.
  el('line', { x1: 380, x2: 410, y1: 76, y2: 76, stroke: CYAN, 'stroke-width': 2 }, cmpG);
  el('circle', { cx: 395, cy: 76, r: 4, fill: CYAN }, cmpG);
  label(cmpG, 418, 80, `full text (${count(N_TRAIN)})`, 'start', STRONG, 11);
  el('line', { x1: 380, x2: 410, y1: 98, y2: 98, stroke: VIOLET, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, cmpG);
  el('path', { d: tri(395, 98), fill: VIOLET }, cmpG);
  label(cmpG, 418, 102, `small text (${N_SMALL})`, 'start', STRONG, 11);
  label(cmpG, 16, 44, 'sentences in each training text in brackets', 'start', MUTED, 11);
  const cmpUn = Array.from({ length: N_MAX }, (_, i) => label(cmpG, P(mx(i)), NY0 + 40, '', 'middle', MUTED, 11));
  label(cmpG, NX0 - 8, NY0 + 40, 'unseen', 'end', MUTED, 10);
  label(cmpG, NX0 - 8, NY0 + 54, '(small)', 'end', MUTED, 10);

  const busy = label(svg, 300, 130, '', 'middle', MUTED, 13);
  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- the panel under the picture: the prompt box, its chips, generated lines. Text only.
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const panel = make('div');
  panel.className = 'token-panel';
  Object.assign(panel.style, { marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #334155' });
  const inRow = make('label');
  Object.assign(inRow.style, { display: 'grid', gap: '4px', margin: '0 0 8px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const input = make('input', { type: 'text', autocomplete: 'off', spellcheck: false, value: typed });
  input.maxLength = MAX_INPUT;
  input.setAttribute('autocapitalize', 'off');
  Object.assign(input.style, { width: '100%', boxSizing: 'border-box', padding: '6px 8px', background: BG, color: STRONG, border: '1px solid #94a3b8', borderRadius: '4px', fontFamily: FONT, fontSize: '14px' });
  inRow.append('your text so far (up to 80 characters, never stored or sent)', input);
  const pTitle = make('p');
  Object.assign(pTitle.style, { margin: '0 0 6px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const chipsBox = make('div');
  chipsBox.setAttribute('aria-hidden', 'true');
  Object.assign(chipsBox.style, { display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start', marginBottom: '6px' });
  const chipsSaid = make('p');
  chipsSaid.className = 'vh';
  const genTitle = make('p');
  Object.assign(genTitle.style, { margin: '8px 0 4px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const genBox = make('div');
  genBox.setAttribute('aria-live', 'off');
  panel.append(inRow, pTitle, chipsBox, chipsSaid, genTitle, genBox);
  svg.after(panel);

  const chip = (c) => {
    const s = make('span');
    s.className = c.unknown ? 'token-chip unknown' : 'token-chip';
    s.textContent = c.unknown ? c.ch : shown(c.ch);
    if (c.ctx) Object.assign(s.style, { borderColor: CYAN, boxShadow: `0 0 0 1px ${CYAN}`, color: STRONG });
    return s;
  };
  // One generated line: its setting, the prompt (muted) and the new characters.
  const genLine = (name, prompt, cont, current) => {
    const row = make('p');
    Object.assign(row.style, {
      margin: '0 0 4px', padding: '2px 6px', fontFamily: FONT, fontSize: '13px', lineHeight: '1.5', overflowWrap: 'anywhere',
      borderLeft: `3px solid ${current ? CYAN : 'transparent'}`, background: current ? '#1e293b' : 'transparent',
    });
    const tag = make('span', { textContent: `${current ? '▸' : ' '} ${name}  ` });
    Object.assign(tag.style, { color: current ? STRONG : MUTED, whiteSpace: 'pre' });
    const pre = make('span', { textContent: prompt });
    pre.style.color = MUTED;
    const out = make('span', { textContent: cont });
    out.style.color = current ? STRONG : '#e2e8f0';
    row.append(tag, pre, out);
    return row;
  };

  // ---- controls.
  const slider = (name, min, max, value) => {
    const group = make('div');
    const lab = make('label');
    const inp = make('input', { type: 'range', min: String(min), max: String(max), step: '1', value: String(value) });
    const out = make('output');
    out.setAttribute('aria-hidden', 'true');
    lab.append(name, ' ', inp, ' ', out);
    group.append(lab);
    return { group, inp, out };
  };
  const nCtl = slider('context n', 1, N_MAX, n);
  const tauCtl = slider('temperature τ', 0, TAUS.length - 1, tauIx);
  const genGroup = make('div');
  const genBtn = make('button', { type: 'button', textContent: 'Generate' });
  genGroup.append(genBtn);
  const smallGroup = make('div');
  const smallBtn = make('button', { type: 'button', textContent: `Small training text (${N_SMALL} sentences)` });
  smallBtn.setAttribute('aria-pressed', 'false');
  smallGroup.append(smallBtn);
  controls.append(nCtl.group, tauCtl.group, genGroup, smallGroup);

  const setValueText = () => {
    nCtl.inp.value = String(n);
    nCtl.out.textContent = String(n);
    nCtl.inp.setAttribute('aria-valuetext', `context of ${n} character${n > 1 ? 's' : ''}`);
    tauCtl.inp.value = String(tauIx);
    tauCtl.out.textContent = TAU_NAME(TAUS[tauIx]);
    tauCtl.inp.setAttribute('aria-valuetext', `temperature ${TAU_NAME(TAUS[tauIx])}`);
    smallBtn.setAttribute('aria-pressed', String(which === 'small'));
  };

  function info() {
    if (step <= 1) {
      const c = predict(FIXED_PROMPT, N_DEFAULT);
      return { pred: c, chips: promptChips(FIXED_PROMPT, N_DEFAULT) };
    }
    if (step <= 4) {
      const which4 = step === 4 ? which : 'full';
      const pc = promptChips(typed, n);
      return { pred: predict(pc.text, n, which4), chips: pc, unknown: pc.unknown, n, tau: TAUS[tauIx], seed, which: which4, table };
    }
    return { table };
  }

  function drawBars(p, { ref = null, scale = 1, valueText = (v) => v.toFixed(2).replace(/^0\./, '.') }) {
    bars.forEach((b, i) => {
      const h = (p[i] / scale) * BH;
      b.rect.setAttribute('y', P(BY0 - h));
      b.rect.setAttribute('height', P(h));
      if (ref) {
        const hr = (ref[i] / scale) * BH;
        b.ref.setAttribute('y', P(BY0 - hr));
        b.ref.setAttribute('height', P(hr));
      }
      show(b.ref, !!ref);
      const top = Math.max(h, ref ? (ref[i] / scale) * BH : 0);
      b.val.textContent = '';
      b.val.setAttribute('y', P(BY0 - top - 5));
    });
    // Values above the bars: largest first, only bars of at least a tenth of the scale,
    // and never next to a bar already labelled (the read-out gives the top three).
    const done = new Set();
    p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).forEach(([v, i]) => {
      if (v / scale < 0.1 || done.has(i - 1) || done.has(i + 1)) return;
      bars[i].val.textContent = valueText(v);
      done.add(i);
    });
  }

  function render() {
    const s = info();
    setValueText();
    show(barG, step <= 3);
    show(nG, step === 4 && !!table);
    show(cmpG, step === 5 && !!table);
    busy.textContent = step >= 4 && !table ? 'computing… (a moment)' : '';

    if (step <= 3) {
      const counting = step === 1;
      ticks.forEach((t) => { show(t.ln, !counting); show(t.t, !counting); });
      if (counting) {
        const mxC = Math.max(...s.pred.counts);
        drawBars(s.pred.counts, { scale: mxC, valueText: (v) => String(v) });
        barTitle.textContent = `how often each character followed ${shown(s.pred.context)} in the training text`;
        barLegend.textContent = 'bar height: count';
      } else if (step === 3) {
        drawBars(tempProbs(s.pred.probs, s.tau), { ref: s.pred.probs });
        barTitle.textContent = `next character at τ = ${TAU_NAME(s.tau)}`;
        barLegend.textContent = 'dashed: τ = 1, the model itself';
      } else {
        drawBars(s.pred.probs, {});
        barTitle.textContent = step === 0 ? `next character after ${shown(FIXED_PROMPT)}` : 'next character after your text';
        barLegend.textContent = 'bar height: probability';
      }
    }

    if (step === 4 && table) {
      const rows = table[which];
      nTitle.textContent = `held-out bits · ${textName(which)}`;
      nBars.forEach((b, i) => {
        const r = rows[i];
        const y = ny(Math.min(r.held, YMAX));
        b.rect.setAttribute('y', P(y));
        b.rect.setAttribute('height', P(NY0 - y));
        b.rect.setAttribute('fill-opacity', i + 1 === n ? 0.8 : 0.35);
        b.rect.setAttribute('stroke-width', i + 1 === n ? 2.5 : 1);
        const yt = P(ny(r.train));
        b.tr.setAttribute('y1', yt);
        b.tr.setAttribute('y2', yt);
        b.val.textContent = bits(r.held);
        b.val.setAttribute('y', P(Math.min(y, ny(r.train)) - 6));
        b.un.textContent = pct(r.unseen);
      });
    }

    if (step === 5 && table) {
      const pts = (rows) => rows.map((r, i) => [mx(i), ny(r.held)]);
      const fp = pts(table.full);
      const sp = pts(table.small);
      const d = (ps) => ps.map(([x, y], k) => `${k ? 'L' : 'M'}${P(x)},${P(y)}`).join('');
      fullPath.setAttribute('d', d(fp));
      smallPath.setAttribute('d', d(sp));
      fp.forEach(([x, y], i) => { fullDots[i].setAttribute('cx', P(x)); fullDots[i].setAttribute('cy', P(y)); });
      sp.forEach(([x, y], i) => smallDots[i].setAttribute('d', tri(x, y)));
      cmpUn.forEach((t, i) => { t.textContent = pct(table.small[i].unseen); });
    }

    renderPanel(s);
    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(s, texts));
  }

  function renderPanel(s) {
    show(panel, step <= 4);
    if (step > 4) return;
    show(inRow, step >= 2);
    const chips = s.chips;
    pTitle.textContent = step <= 1
      ? `the text so far; the outlined characters are the context the model looks at (n = ${N_DEFAULT})`
      : chips.context === null ? `fewer than ${n} known characters: no context yet` : `outlined: the context, the last ${n} known characters; ? marks a character outside the ${A}`;
    chipsBox.replaceChildren(...chips.chips.map(chip));
    chipsSaid.textContent = `Context: ${chips.context === null ? 'none' : chips.context.replace(/ /g, 'space ')}.`;
    const gen = step === 3 || step === 4;
    show(genTitle, gen);
    show(genBox, gen);
    if (step === 3) {
      genTitle.textContent = `${GEN_LEN} generated characters at each temperature (n = ${n}, seed ${seed})`;
      genBox.replaceChildren(...TAUS.map((t, i) => genLine(`τ = ${t.toFixed(1)}`, chips.text, generate(chips.text, n, t, seed), i === tauIx)));
    } else if (step === 4) {
      genTitle.textContent = `${GEN_LEN} generated characters at each context length (τ = 1, seed ${seed}, ${which} text)`;
      genBox.replaceChildren(...Array.from({ length: N_MAX }, (_, i) => genLine(`n = ${i + 1}`, chips.text, generate(chips.text, i + 1, 1, seed, which), i + 1 === n)));
    }
  }

  function ariaFor(s, texts) {
    const rest = texts.filter(Boolean).join('. ');
    if (step === 0) return `Bars of the probability of each of the ${A} possible next characters after the text ${FIXED_PROMPT}. ${rest}.`;
    if (step === 1) return `Bars of how often each character followed the context in the training text. ${rest}.`;
    if (step === 2) return `Bars of the probability of each next character after your text. ${rest}.`;
    if (step === 3) return `Bars of the next-character probabilities at the chosen temperature, with dashed outlines for temperature 1. ${rest}.`;
    if (step === 4) return `Bars of held-out bits per character for context lengths 1 to ${N_MAX}, with dashed marks for training bits, and the share of unseen contexts under each bar. ${rest}.`;
    return `Two lines of held-out bits per character against context length: the full training text falls and stays low; the small text is lowest at n = 3 and then rises. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).filter(Boolean).join('. ')}.`;

  // The held-out numbers: about 16 counting jobs, a few per slice, so the page stays responsive.
  function ensureTable() {
    if (table || computing) return;
    computing = true;
    const jobs = STAT_JOBS.slice();
    const tick = () => {
      const t0 = performance.now();
      while (jobs.length && performance.now() - t0 < 8) {
        const [k, w] = jobs.shift();
        stats(k, w);
      }
      if (jobs.length) { setTimeout(tick, 0); return; }
      table = tableNow();
      computing = false;
      fillValues();
      render();
      if (step >= 4) announce(spoken());
    };
    setTimeout(tick, 0);
  }

  // Values that need no held-out numbers are written now; the rest once they are computed.
  function fillValues() {
    const filled = table ? pageValues(table) : null;
    for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
      if (!filled) continue;
      const val = filled[span.dataset.value];
      if (val !== undefined) span.textContent = val;
    }
  }

  nCtl.inp.addEventListener('input', () => {
    n = Number(nCtl.inp.value) || N_DEFAULT;
    render();
    announce(spoken());
  });
  tauCtl.inp.addEventListener('input', () => {
    tauIx = Number(tauCtl.inp.value) || 0;
    render();
    announce(spoken());
  });
  genBtn.addEventListener('click', () => {
    seed += 1;
    render();
    announce(`New samples, seed ${seed}.`);
  });
  smallBtn.addEventListener('click', () => {
    which = which === 'small' ? 'full' : 'small';
    render();
    announce(spoken());
  });
  input.addEventListener('input', () => {
    // maxlength is the first cap; sanitise caps again (in characters) and lower-cases.
    typed = input.value.slice(0, MAX_INPUT * 2);
    render();
  });
  input.addEventListener('change', () => announce(spoken()));

  render();
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 6,
    show(i) {
      step = i;
      const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };
      showGroup(nCtl.group, i >= 2 && i <= 4);
      showGroup(tauCtl.group, i === 3);
      showGroup(genGroup, i === 3 || i === 4);
      showGroup(smallGroup, i === 4);
      if (i >= 4) ensureTable();
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}

function tri(x, y) {
  return trianglePoints(x, y, 4.5).map(([a, b], k) => `${k ? 'L' : 'M'}${P(a)},${P(b)}`).join('') + 'Z';
}
