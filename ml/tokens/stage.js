// Lesson L5.1, "Tokens: text as numbers": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l5-tokens.test.mjs); the
// DOM is only touched inside createStage. Every merge comes from bpeTrain in text.js,
// every tokenisation from bpeEncode, and every number in the page's text from
// pageValues. Typed text is sanitised, shown with textContent only, and never stored
// or sent anywhere.

import { ALPHABET, worldSentences } from '../assets/world.js';
import { sanitise, bpeTrain, bpeEncode } from '../assets/text.js';
import { el } from '../assets/plot.js';
import { createTrainLoop, trainButtons, trainStatus } from '../assets/train-loop.js';

// ---- data and settings --------------------------------------------------------

export const SEED = 241;
export const N_SENTENCES = 400;
export const MAX_MERGES = 60;
export const SAMPLE_SENTENCES = 2;
export const MERGE_MS = 400;
export const MAX_INPUT = 80;
export const DEFAULT_INPUT = 'zoe and tom like kiwis.';
export const BASE = ALPHABET.length;   // 28: space, full stop, a to z

let TEXT = null;
let SENTS = null;
let TK = null;
let ALL = null;
const sentences = () => (SENTS ??= worldSentences({ n: N_SENTENCES, seed: SEED }));
export const trainingText = () => (TEXT ??= sentences().join(' '));
export const sampleText = () => sentences().slice(0, SAMPLE_SENTENCES).join(' ');
// The first MAX_MERGES merges. Greedy and deterministic, so the first r of them are
// exactly what bpeTrain(text, r) gives.
export const tokeniser = () => (TK ??= bpeTrain(trainingText(), MAX_MERGES));
// Every merge this text allows (until no pair occurs twice).
export const allMerges = () => (ALL ??= bpeTrain(trainingText(), 10000));

export const mergesAt = (r) => tokeniser().merges.slice(0, r);
export const tokensAt = (s, r) => bpeEncode(s, mergesAt(r));
export const tokenCount = (r) => tokeniser().lengths[r];
export const vocabSize = (r) => BASE + r;

// Id of a token with the first r merges: its place in ALPHABET for one symbol, 28 + k
// for the string made by merge k (counting from 0); null if it has none.
export function tokenId(tok, r) {
  const i = ALPHABET.indexOf(tok);
  if (tok.length === 1 && i >= 0) return i;
  const k = mergesAt(r).findIndex(([a, b]) => a + b === tok);
  return k >= 0 ? BASE + k : null;
}

// How many times each of the 28 ids occurs in the training text.
export function idCounts() {
  const c = new Array(BASE).fill(0);
  for (const ch of trainingText()) c[ALPHABET.indexOf(ch)]++;
  return c;
}

export const oneHot = (i, size) => Array.from({ length: size }, (_, j) => (j === i ? 1 : 0));

// One merge per MERGE_MS while the loop runs: true when it is time for the next one.
export const paceMerge = (now, last, ms = MERGE_MS) => now - last >= ms;

// Typed text: lower-cased and capped by sanitise, then split with the first r merges.
// A character outside ALPHABET is its own token with no id (merges never contain one).
export function typedTokens(raw, r) {
  const { text } = sanitise(raw, MAX_INPUT);
  const chips = tokensAt(text, r).map((tok) => {
    const id = tokenId(tok, r);
    return { tok, id, unknown: id === null };
  });
  return { text, chips, unknown: chips.filter((c) => c.unknown).length, chars: Array.from(text).length };
}

// Each character with its UTF-8 bytes (1 to 4 of them, each 0 to 255).
const ENC = new TextEncoder();
export const bytesOf = (s) => Array.from(s).map((ch) => ({ ch, bytes: [...ENC.encode(ch)] }));

// ---- formatting ------------------------------------------------------------------

export const count = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
export const ratio = (r) => (Array.from(trainingText()).length / tokenCount(r)).toFixed(2);
const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
export const shown = (tok) => tok.replace(/ /g, '␣');
const mergeText = ([a, b]) => `${shown(a)} + ${shown(b)}`;

// ---- read-outs ---------------------------------------------------------------------

export function readout(step, s) {
  const nChars = count(Array.from(trainingText()).length);
  if (step === 0) return [`training text: ${N_SENTENCES} sentences, ${nChars} characters`, `one token per character: ${nChars} tokens`, `every id comes from a fixed list of ${BASE} symbols`];
  if (step === 1) {
    const ch = s.ch;
    return [`character ${s.pos + 1} of the first sentence: ${shown(ch)} → id ${s.id}`, `e${sub(s.id)}: ${BASE} numbers, 1 in entry ${s.id}, 0 in the rest`, 'entries are counted from 0, like the ids'];
  }
  if (step === 2) {
    const status = trainStatus({ running: s.running, iter: s.r, done: s.r >= MAX_MERGES });
    return [`merge ${s.r} of ${MAX_MERGES} · ${status}`, `tokens in the text: ${count(tokenCount(s.r))} (${ratio(s.r)} characters per token)`, `vocabulary size |𝒱| = ${BASE} + ${s.r} = ${vocabSize(s.r)}`];
  }
  if (step === 3) {
    const saved = s.r > 0 ? `merge ${s.r} saved ${tokeniser().merges[s.r - 1][2]} tokens` : 'no merges yet: characters only';
    return [`after ${s.r} merges: |𝒱| = ${vocabSize(s.r)}, ${count(tokenCount(s.r))} tokens`, `characters per token: ${ratio(s.r)}`, saved];
  }
  if (step === 4) return [`your text: ${s.chars} characters → ${s.chips.length} tokens`, `split with the first ${s.r} merges`, `characters outside the ${BASE} symbols: ${s.unknown} (shown as ?)`];
  if (step === 5) {
    const nb = s.bytes.reduce((a, c) => a + c.bytes.length, 0);
    return [`your text: ${s.chars} characters → ${nb} bytes`, 'each byte is one of 256 values, so none is unknown', `with our ${BASE} symbols: ${s.unknown} unknown`];
  }
  return [];
}
const sub = (n) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[d]);

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const { merges, lengths } = tokeniser();
  const n = Array.from(trainingText()).length;
  const counts = idCounts();
  const absent = ALPHABET.filter((_, i) => counts[i] === 0);
  const friendsAt = merges.findIndex(([a, b]) => a + b === 'friends') + 1;
  const words = [...new Set(trainingText().split(/[ .]+/).filter(Boolean))];
  const unfinished = words.filter((w) => tokensAt(w, MAX_MERGES).length > 1);
  return {
    'n-sentences': String(N_SENTENCES),
    'n-chars': count(n),
    'most-common': (() => { const c = ALPHABET[counts.indexOf(Math.max(...counts))]; return c === ' ' ? 'space' : c; })(),
    'absent': listText(absent),
    'first-pair': mergeText(merges[0]),
    'first-count': String(merges[0][2]),
    'second-pair': mergeText(merges[1]),
    'second-count': String(merges[1][2]),
    'friends-at': String(friendsAt),
    'max-merges': String(MAX_MERGES),
    'tokens-end': count(lengths[MAX_MERGES]),
    'vocab-end': String(vocabSize(MAX_MERGES)),
    'ratio-end': ratio(MAX_MERGES),
    'last-count': String(merges[MAX_MERGES - 1][2]),
    'unfinished': listText(unfinished),
    'possible': String(allMerges().merges.length),
    'all-words': count(allMerges().lengths.at(-1)),
  };
}

// ---------------------------------------------------------------- the stage

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const P = (v) => Number(v.toFixed(2));
const RY = [280, 302, 324];
const FONT = "'JetBrains Mono', monospace";

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let r = 0;          // merges applied, shared by steps 3 to 5
  let pos = 1;        // step 2: the character of the first sentence (0-based)
  let lastMerge = -Infinity;
  let typed = DEFAULT_INPUT;

  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 600 340');
  svg.setAttribute('role', 'img');
  const show = (node, on) => { node.style.display = on ? '' : 'none'; };
  const label = (parent, x, y, text, anchor = 'start', fill = MUTED, size = 12) => {
    const t = el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, text, ...HALO }, parent);
    t.style.fill = fill;
    return t;
  };
  const marker = el('marker', { id: 'tok-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg));
  el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, marker);
  const arrow = (parent, x1, y1, x2, y2) => el('line', { x1, y1, x2, y2, stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#tok-arrow)' }, parent);

  // Columns for the 28 symbols (steps 1 and 2).
  const CX0 = 40;
  const CW = 19.5;
  const cx = (i) => CX0 + i * CW;

  // ---- step 1: how often each id occurs in the training text.
  const freqG = el('g', {}, svg);
  label(freqG, 16, 24, 'how often each id occurs in the training text', 'start', STRONG);
  const counts = idCounts();
  const maxC = Math.max(...counts);
  const FY0 = 200;
  const FH = 150;
  el('line', { x1: CX0 - 4, x2: cx(BASE), y1: FY0, y2: FY0, stroke: LINE }, freqG);
  ALPHABET.forEach((ch, i) => {
    const h = (counts[i] / maxC) * FH;
    el('rect', { x: P(cx(i) + 2), y: P(FY0 - h), width: CW - 4, height: P(h), fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN, 'stroke-width': 1 }, freqG);
    if (counts[i] === 0) label(freqG, P(cx(i) + CW / 2), FY0 - 4, '0', 'middle', MUTED, 10);
    label(freqG, P(cx(i) + CW / 2), FY0 + 18, shown(ch), 'middle', STRONG, 13);
    label(freqG, P(cx(i) + CW / 2), FY0 + (i % 2 ? 50 : 35), String(i), 'middle', MUTED, 10);
  });
  label(freqG, CX0 - 6, FY0 + 18, 'sym', 'end', MUTED, 10);
  label(freqG, CX0 - 6, FY0 + 35, 'id', 'end', MUTED, 10);
  label(freqG, P(cx(0) + CW / 2), P(FY0 - (counts[0] / maxC) * FH - 6), count(counts[0]), 'middle', STRONG, 11);

  // ---- step 2: the fixed list and one one-hot vector.
  const hotG = el('g', {}, svg);
  label(hotG, 16, 24, `the vocabulary 𝒱: ${BASE} symbols, ids 0 to ${BASE - 1}`, 'start', STRONG);
  // Two rows of 14, so each id has room at phone width.
  const HALF = BASE / 2;
  const VW = 38;
  const vx = (i) => 40 + (i % HALF) * VW;
  const vRow = (i) => Math.floor(i / HALF);
  const vocCells = ALPHABET.map((ch, i) => {
    const y = 36 + vRow(i) * 46;
    const rect = el('rect', { x: vx(i), y, width: VW, height: 44, fill: CYAN, 'fill-opacity': 0, stroke: LINE }, hotG);
    label(hotG, vx(i) + VW / 2, y + 19, shown(ch), 'middle', STRONG, 13);
    label(hotG, vx(i) + VW / 2, y + 37, String(i), 'middle', MUTED, 11);
    return rect;
  });
  const hotPtr = el('path', { d: '', fill: 'none', stroke: CYAN, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, hotG);
  const hotTitle = label(hotG, 16, 158, '', 'start', STRONG);
  const hotCells = ALPHABET.map((_, i) => {
    const y = 170 + vRow(i) * 34;
    const rect = el('rect', { x: vx(i), y, width: VW, height: 30, fill: CYAN, 'fill-opacity': 0, stroke: LINE }, hotG);
    const t = label(hotG, vx(i) + VW / 2, y + 20, '0', 'middle', MUTED, 12);
    return { rect, t };
  });
  label(hotG, 32, 190, 'e', 'end', STRONG, 13);

  // ---- step 3: the counters, the newest merge and the latest merges.
  const mergeG = el('g', {}, svg);
  const bigLine = (y, name) => {
    label(mergeG, 16, y, name, 'start', MUTED);
    const t = label(mergeG, 16, y + 28, '', 'start', STRONG);
    t.style.fontSize = '22px';
    return t;
  };
  const mTokens = bigLine(28, 'tokens in the training text');
  const mVocab = bigLine(86, 'vocabulary size |𝒱|');
  const mNew = bigLine(144, 'newest merge (replacements)');
  const LX = 330;
  label(mergeG, LX, 28, 'merges so far, newest first', 'start', MUTED);
  const LIST = 9;
  const mList = Array.from({ length: LIST }, (_, k) => label(mergeG, LX, 52 + k * 22, '', 'start', k === 0 ? STRONG : MUTED, 13));
  const mMore = label(mergeG, LX, 52 + LIST * 22, '', 'start', MUTED, 12);

  // ---- step 4: tokens against vocabulary size over every merge.
  const plotG = el('g', {}, svg);
  const PX0 = 78;
  const PX1 = 580;
  const PY0 = 214;
  const PY1 = 30;
  const YMAX = 10000;
  const px = (v) => PX0 + ((v - BASE) / MAX_MERGES) * (PX1 - PX0);
  const py = (n) => PY0 + (n / YMAX) * (PY1 - PY0);
  for (const n of [0, 2500, 5000, 7500, 10000]) {
    el('line', { x1: PX0, x2: PX1, y1: P(py(n)), y2: P(py(n)), stroke: LINE, 'stroke-dasharray': n ? '2 4' : null }, plotG);
    label(plotG, PX0 - 8, P(py(n)) + 4, count(n), 'end');
  }
  for (const v of [28, 48, 68, 88]) label(plotG, P(px(v)), PY0 + 17, String(v), 'middle');
  label(plotG, (PX0 + PX1) / 2, PY0 + 36, 'vocabulary size |𝒱| = 28 + merges', 'middle');
  label(plotG, PX0, PY1 - 12, 'tokens in the training text', 'start');
  el('line', { x1: PX0, x2: PX0, y1: PY0, y2: PY1, stroke: LINE }, plotG);
  const L = tokeniser().lengths;
  el('path', { d: L.map((n, k) => `${k ? 'L' : 'M'}${P(px(BASE + k))},${P(py(n))}`).join(''), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, plotG);
  L.forEach((n, k) => el('circle', { cx: P(px(BASE + k)), cy: P(py(n)), r: 1.8, fill: CYAN }, plotG));
  const pDrop = el('line', { stroke: MUTED, 'stroke-dasharray': '3 3' }, plotG);
  const pDot = el('rect', { width: 11, height: 11, fill: BG, stroke: STRONG, 'stroke-width': 2 }, plotG);
  const pLab = label(plotG, 0, 0, '', 'start', STRONG, 12);

  // ---- step 5: your text against the training text, characters and tokens.
  const youG = el('g', {}, svg);
  const BX0 = 150;
  const BW = 420;
  const barRow = (y, name) => {
    label(youG, 16, y - 8, name, 'start', STRONG);
    label(youG, BX0 - 8, y + 14, 'characters', 'end');
    label(youG, BX0 - 8, y + 40, 'tokens', 'end');
    el('rect', { x: BX0, y, width: BW, height: 20, fill: 'none', stroke: MUTED, 'stroke-dasharray': '4 3' }, youG);
    const tok = el('rect', { x: BX0, y: y + 26, width: 0, height: 20, fill: CYAN, 'fill-opacity': 0.45, stroke: CYAN }, youG);
    const a = label(youG, BX0 + BW - 6, y + 15, '', 'end', STRONG);
    const b = label(youG, BX0 + 6, y + 41, '', 'start', STRONG);
    return { tok, a, b };
  };
  const youRow = barRow(44, 'your text');
  const textRow = barRow(150, 'the training text');
  label(youG, 16, 236, 'each pair of bars has its own scale', 'start', MUTED, 11);

  // ---- step 6: what a byte-level tokeniser starts from.
  const byteG = el('g', {}, svg);
  const box = (x, y, w, lines) => {
    el('rect', { x, y, width: w, height: 70, rx: 6, fill: 'none', stroke: MUTED }, byteG);
    lines.forEach((s, k) => label(byteG, x + w / 2, y + 30 + k * 18, s, 'middle', k ? MUTED : STRONG, k ? 11 : 13));
  };
  const steps = [['text', 'any script'], ['bytes', '256 values'], ['merges', 'from text'], ['token ids', 'no unknowns']];
  steps.forEach((lines, k) => {
    box(16 + k * 148, 90, 124, lines);
    if (k) arrow(byteG, 16 + k * 148 - 22, 125, 16 + k * 148 - 2, 125);
  });
  label(byteG, 16, 40, 'starting from bytes, nothing is unknown', 'start', STRONG);
  label(byteG, 16, 210, 'below: your text as bytes, 1 to 4 per character', 'start', MUTED);

  const rowsT = RY.map((y, k) => label(svg, 16, y, '', 'start', k === 0 ? STRONG : MUTED, 13));

  // ---- the token panel: HTML chips under the picture, text only.
  const panel = doc.createElement('div');
  panel.className = 'token-panel';
  Object.assign(panel.style, { marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #334155', minHeight: '7.5rem' });
  const pTitle = doc.createElement('p');
  Object.assign(pTitle.style, { margin: '0 0 6px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const inRow = doc.createElement('label');
  Object.assign(inRow.style, { display: 'grid', gap: '4px', margin: '0 0 8px', color: MUTED, fontFamily: FONT, fontSize: '12px' });
  const input = doc.createElement('input');
  input.type = 'text';
  input.maxLength = MAX_INPUT;
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.value = typed;
  input.setAttribute('autocapitalize', 'off');
  Object.assign(input.style, { width: '100%', boxSizing: 'border-box', padding: '6px 8px', background: BG, color: STRONG, border: '1px solid #94a3b8', borderRadius: '4px', fontFamily: FONT, fontSize: '14px' });
  inRow.append('your text (up to 80 characters, never stored or sent)', input);
  const chipsBox = doc.createElement('div');
  chipsBox.setAttribute('aria-hidden', 'true');
  Object.assign(chipsBox.style, { display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start' });
  const chipsSaid = doc.createElement('p');
  chipsSaid.className = 'vh';
  panel.append(inRow, pTitle, chipsBox, chipsSaid);
  svg.after(panel);

  // One chip: the token (␣ for a space) and its id (or bytes) underneath.
  const chip = (tok, idText, { unknown = false, mark = false } = {}) => {
    const w = doc.createElement('span');
    Object.assign(w.style, { display: 'inline-flex', flexDirection: 'column', alignItems: 'center', margin: '0 3px 4px 0' });
    const c = doc.createElement('span');
    c.className = unknown ? 'token-chip unknown' : 'token-chip';
    c.textContent = unknown ? tok : shown(tok);
    if (mark) Object.assign(c.style, { borderColor: CYAN, boxShadow: `0 0 0 1px ${CYAN}` });
    const i = doc.createElement('span');
    i.textContent = idText;
    Object.assign(i.style, { fontFamily: FONT, fontSize: '11px', color: MUTED });
    w.append(c, i);
    return w;
  };
  const fillChips = (items, said) => {
    chipsBox.replaceChildren(...items);
    chipsSaid.textContent = said;
  };

  // ---- controls: the character slider (step 2), Merge once + Train (step 3), the merges slider (steps 4, 5).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const sample1 = Array.from(sentences()[0]);
  const posGroup = make('div');
  const posLab = make('label');
  const posIn = make('input', { type: 'range', min: '1', max: String(sample1.length), step: '1', value: String(pos + 1) });
  const posOut = make('output');
  posOut.setAttribute('aria-hidden', 'true');
  posLab.append('character', ' ', posIn, ' ', posOut);
  posGroup.append(posLab);
  const rGroup = make('div');
  const rLab = make('label');
  const rIn = make('input', { type: 'range', min: '0', max: String(MAX_MERGES), step: '1', value: '0' });
  const rOut = make('output');
  rOut.setAttribute('aria-hidden', 'true');
  rLab.append('merges', ' ', rIn, ' ', rOut);
  rGroup.append(rLab);
  const trainGroup = make('div');
  trainGroup.setAttribute('role', 'group');
  trainGroup.setAttribute('aria-label', 'merge pairs');
  Object.assign(trainGroup.style, { alignItems: 'center', flexWrap: 'wrap', gap: '6px' });
  const once = make('button', { type: 'button', textContent: 'Merge once' });
  trainGroup.append(once);
  controls.append(posGroup, trainGroup, rGroup);

  const setValueText = () => {
    posOut.textContent = String(pos + 1);
    posIn.setAttribute('aria-valuetext', `character ${pos + 1}: ${sample1[pos] === ' ' ? 'space' : sample1[pos]}`);
    rIn.value = String(r);
    rOut.textContent = String(r);
    rIn.setAttribute('aria-valuetext', `${r} merges`);
    once.disabled = r >= MAX_MERGES;
  };

  function info() {
    if (step === 1) {
      const ch = sample1[pos];
      return { pos, ch, id: ALPHABET.indexOf(ch) };
    }
    if (step === 2 || step === 3) return { r, running: loop.activity() };
    if (step === 4) return { ...typedTokens(typed, r), r };
    if (step === 5) {
      const t = typedTokens(typed, r);
      return { ...t, bytes: bytesOf(t.text) };
    }
    return {};
  }

  function renderPanel(s) {
    const typing = step >= 4;
    show(inRow, typing);
    if (step <= 1) {
      pTitle.textContent = 'the first two sentences, one token per character, with their ids';
      const cs = Array.from(sampleText());
      fillChips(cs.map((ch, k) => chip(ch, String(ALPHABET.indexOf(ch)), { mark: step === 1 && k === pos })),
        `${cs.length} character tokens.`);
    } else if (step <= 3) {
      const toks = tokensAt(sampleText(), r);
      pTitle.textContent = `the first two sentences after ${r} merges: ${toks.length} tokens`;
      fillChips(toks.map((t) => chip(t, String(tokenId(t, r)))), `${toks.length} tokens: ${toks.map((t) => (t === ' ' ? 'space' : t)).join(', ')}.`);
    } else if (step === 4) {
      pTitle.textContent = `your text split with the first ${r} merges`;
      fillChips(s.chips.map((c) => chip(c.tok, c.unknown ? 'none' : String(c.id), { unknown: c.unknown })),
        `${s.chips.length} tokens: ${s.chips.map((c) => (c.unknown ? 'unknown' : c.tok === ' ' ? 'space' : c.tok)).join(', ')}.`);
    } else {
      pTitle.textContent = 'your text as UTF-8 bytes (each byte 0 to 255)';
      fillChips(s.bytes.map((c) => chip(c.ch, c.bytes.join(' '))), `${s.bytes.reduce((a, c) => a + c.bytes.length, 0)} bytes.`);
    }
  }

  function render() {
    show(freqG, step === 0);
    show(hotG, step === 1);
    show(mergeG, step === 2);
    show(plotG, step === 3);
    show(youG, step === 4);
    show(byteG, step === 5);
    setValueText();
    const s = info();

    if (step === 1) {
      vocCells.forEach((rect, i) => rect.setAttribute('fill-opacity', i === s.id ? 0.35 : 0));
      hotCells.forEach((c, i) => {
        const on = i === s.id;
        c.rect.setAttribute('fill-opacity', on ? 0.6 : 0);
        c.t.textContent = on ? '1' : '0';
        c.t.style.fill = on ? STRONG : MUTED;
      });
      hotTitle.textContent = `one-hot vector e${sub(s.id)} for ${shown(s.ch)} (id ${s.id})`;
      const x = vx(s.id) + VW / 2;
      hotPtr.setAttribute('d', `M${x},${80 + vRow(s.id) * 46} L${x},${170 + vRow(s.id) * 34}`);
    }

    if (step === 2) {
      const { merges } = tokeniser();
      mTokens.textContent = `${count(tokenCount(r))}  (from ${count(tokenCount(0))})`;
      mVocab.textContent = `${BASE} + ${r} = ${vocabSize(r)}`;
      mNew.textContent = r ? `${mergeText(merges[r - 1])} → ${shown(merges[r - 1][0] + merges[r - 1][1])}  (${merges[r - 1][2]})` : 'none yet: press Merge once';
      mList.forEach((t, k) => {
        const j = r - 1 - k;
        t.textContent = j >= 0 ? `${String(j + 1).padStart(2, ' ')}. ${mergeText(merges[j])} → ${shown(merges[j][0] + merges[j][1])}  id ${BASE + j}` : '';
      });
      mMore.textContent = r > LIST ? `… and ${r - LIST} earlier` : '';
    }

    if (step === 3) {
      const x = P(px(BASE + r));
      const y = P(py(tokenCount(r)));
      pDot.setAttribute('x', x - 5.5);
      pDot.setAttribute('y', y - 5.5);
      pDrop.setAttribute('x1', x);
      pDrop.setAttribute('x2', x);
      pDrop.setAttribute('y1', y + 6);
      pDrop.setAttribute('y2', PY0);
      pLab.textContent = `${r} merges: ${count(tokenCount(r))}`;
      pLab.setAttribute('x', r > 40 ? x - 10 : x + 10);
      pLab.setAttribute('text-anchor', r > 40 ? 'end' : 'start');
      pLab.setAttribute('y', y - 12);
    }

    if (step === 4) {
      const setRow = (row, chars, toks) => {
        row.tok.setAttribute('width', chars ? P((toks / chars) * BW) : 0);
        row.a.textContent = count(chars);
        row.b.textContent = count(toks);
        row.b.setAttribute('x', P(BX0 + (chars ? (toks / chars) * BW : 0) + 6));
      };
      setRow(youRow, s.chars, s.chips.length);
      setRow(textRow, Array.from(trainingText()).length, tokenCount(r));
    }

    renderPanel(s);
    const texts = readout(step, s);
    rowsT.forEach((t, k) => { t.textContent = texts[k] || ''; });
    svg.setAttribute('aria-label', ariaFor(texts));
  }

  function ariaFor(texts) {
    const rest = texts.join('. ');
    if (step === 0) return `Bars of how often each of the ${BASE} ids occurs in the training text; space is the most common. ${rest}.`;
    if (step === 1) return `The ${BASE} symbols of the vocabulary with their ids, and below them the one-hot vector of the chosen character: ${BASE} entries, one of them 1. ${rest}.`;
    if (step === 2) return `Counters for the merges: tokens in the training text, vocabulary size, the newest merge and a list of the latest merges. ${rest}.`;
    if (step === 3) return `A line of the tokens in the training text against the vocabulary size, one point per merge from 0 to ${MAX_MERGES}, falling from ${count(tokenCount(0))} to ${count(tokenCount(MAX_MERGES))}; a square marks the current merge. ${rest}.`;
    if (step === 4) return `Bars comparing characters and tokens for your text and for the training text. ${rest}.`;
    return `A sketch: any text becomes UTF-8 bytes, 256 base tokens, then merges learned from text give token ids, with no unknowns. ${rest}.`;
  }

  const spoken = () => `${readout(step, info()).join('. ')}.`;

  // ---- step 3's merges: one every MERGE_MS while running; under reduced motion straight to the end.
  const advance = (k) => { r = Math.min(MAX_MERGES, r + k); };
  const loop = createTrainLoop({
    step: () => {
      const now = performance.now();
      if (paceMerge(now, lastMerge)) {
        advance(1);
        lastMerge = now;
      }
    },
    render,
    done: () => r >= MAX_MERGES,
    onStop: () => {
      buttons.sync();
      render();
      if (r >= MAX_MERGES) announce(`Finished. ${spoken()}`);
    },
  });
  const buttons = trainButtons(trainGroup, loop, {
    onReset: () => { r = 0; lastMerge = -Infinity; },
    reduced,
    work: (k) => advance(k),
  });
  trainGroup.addEventListener('click', () => { if (step === 2) { render(); announce(spoken()); } });
  once.addEventListener('click', () => {
    buttons.stop();
    advance(1);
    render();
  });

  posIn.addEventListener('input', () => {
    pos = (Number(posIn.value) || 1) - 1;
    render();
    announce(spoken());
  });
  rIn.addEventListener('input', () => {
    r = Number(rIn.value) || 0;
    render();
    announce(spoken());
  });
  input.addEventListener('input', () => {
    // maxlength is the first cap; sanitise caps again (in characters) and lower-cases.
    typed = Array.from(input.value).slice(0, MAX_INPUT * 2).join('');
    render();
  });
  input.addEventListener('change', () => announce(spoken()));

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
      showGroup(posGroup, i === 1);
      showGroup(rGroup, i === 3 || i === 4);
      trainGroup.style.display = i === 2 ? 'flex' : 'none';
      controls.classList.toggle('has-train', i === 2);
      render();
      announce(spoken());
      return Promise.resolve();
    },
  };
}
