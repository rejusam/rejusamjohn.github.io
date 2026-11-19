// Lesson L6.3, "The loop and when to stop": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-agent-loop.test.mjs); the
// DOM is only touched inside createStage. Every trace comes from runAgent in agent.js,
// every chance from chainSuccess and stepsForTarget, and every number in the page's text
// from pageValues. All text goes in with textContent.

import { DEFAULT_CONFIG, TASKS, runAgent, chainSuccess, stepsForTarget, fmtNumber } from '../assets/agent.js';
import { createTraceView } from '../assets/trace-view.js';
import { el, linearScale, axes, pathFrom } from '../assets/plot.js';

// ---- tasks and runs --------------------------------------------------------------------

export const TASK_SHOP = 'eli-shop';
export const TASK_LOOP = 'kai-food';
export const ALT_RUN_ID = 'pair-shop'; // the memory lesson's run that alternates between two calls

export const LIMIT_MIN = 1;
export const LIMIT_MAX = 10;
export const START_LIMIT = 3;

export const limitRun = (L) => runAgent(TASK_SHOP, { stepLimit: L });
export const loopConfig = (repeatCheck) => ({ onNotFound: 'try again', repeatCheck });
export const loopRun = (repeatCheck) => runAgent(TASK_LOOP, loopConfig(repeatCheck));
export const stepsNeeded = () => runAgent(TASK_SHOP).steps;

// The run steps 1 and 2 show.
export const stepRun = () => runAgent(TASK_SHOP);

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function limitLine(r, L) {
  if (r.stop === 'done') return `step limit ${L}: done after ${plural(r.steps, 'step')}: ${r.answer}`;
  return `step limit ${L}: stopped by the ${r.stop} after ${plural(r.steps, 'step')}, no answer`;
}

// ---- every step can go wrong: p to the power k ----------------------------------------

// The slider holds a whole number n from 80 to 99; p = n / 100 has no float drift.
export const P_MIN = 80;
export const P_MAX = 99;
export const START_P = 95;
export const K_MAX = 20;
export const K_EXAMPLE = 10;
export const TARGET = 0.5;
export const pOf = (n) => n / 100;

export const chartPoints = (p) => Array.from({ length: K_MAX }, (_, i) => [i + 1, chainSuccess(p, i + 1)]);

// k <= log(target) / log(p): the largest whole k is stepsForTarget(p, target).
export const logBound = (p, target) => Math.log(target) / Math.log(p);

// A chance printed with two decimals, rounded half up (the site's rounding rule).
export const chanceText = (x) => (Math.floor(x * 100 + 0.5) / 100).toFixed(2);

export const chartReadout = (p) => `p = ${fmtNumber(p)}: all ${K_EXAMPLE} steps succeed with chance ${chanceText(chainSuccess(p, K_EXAMPLE))}`;
export function targetReadout(p) {
  const k = stepsForTarget(p, TARGET);
  return `chance at least ${fmtNumber(TARGET)} for up to ${plural(k, 'step')}${k > K_MAX ? ' (past the right edge)' : ''}`;
}

// Model calls for a run the repeat check does not stop: n steps needed (Infinity if it never answers), limit L.
export const callsFor = (n, L) => Math.min(n, L);

// ---- cost: one step is one model call ----------------------------------------------------

export const COST_RUNS = [
  { id: 'eli-8', label: `the eli task, step limit ${DEFAULT_CONFIG.stepLimit}`, task: TASK_SHOP, config: { stepLimit: DEFAULT_CONFIG.stepLimit } },
  { id: 'eli-3', label: `the eli task, step limit ${START_LIMIT}`, task: TASK_SHOP, config: { stepLimit: START_LIMIT } },
  { id: 'kai-on', label: 'the kai task, repeat check on', task: TASK_LOOP, config: loopConfig(true) },
  { id: 'kai-off', label: 'the kai task, repeat check off', task: TASK_LOOP, config: loopConfig(false) },
].map((c) => ({ ...c, run: () => runAgent(c.task, c.config) }));

// Model calls are the plan rows (one per step); tool runs are the result rows.
export function costOf(r) {
  return {
    calls: r.messages.filter((m) => m.role === 'plan').length,
    toolRuns: r.messages.filter((m) => m.role === 'result').length,
  };
}

export const costText = (c) => `${plural(c.calls, 'model call')}, ${plural(c.toolRuns, 'tool run')}`;

const endText = (r) => (r.stop === 'done' ? 'answered' : `stopped by ${r.stop === 'step limit' ? 'the step limit' : 'the repeat check'}`);
export const costLine = (c, r) => `${c.label}: ${costText(costOf(r))}, ${endText(r)}`;

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const eli = stepRun();
  const short = limitRun(eli.steps - 1);
  const on = loopRun(true);
  const off = loopRun(false);
  const kaiCall = on.messages.find((m) => m.role === 'call');
  const kaiResult = on.messages.find((m) => m.role === 'result');
  const p = pOf(START_P);
  const kT = stepsForTarget(p, TARGET);
  const qp = 0.9;
  const v = {
    'eli-task': TASKS[TASK_SHOP].text,
    'eli-steps': String(eli.steps),
    'eli-answer': eli.answer,
    'eli-tool-runs': String(costOf(eli).toolRuns),
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
    'short-limit': String(eli.steps - 1),
    'short-result': short.messages.filter((m) => m.role === 'result').at(-1).text,
    'kai-task': TASKS[TASK_LOOP].text,
    'kai-call': kaiCall.text,
    'kai-result': kaiResult.text,
    'kai-on-steps': String(on.steps),
    'kai-on-runs': String(costOf(on).toolRuns),
    'kai-off-steps': String(off.steps),
    'kai-off-runs': String(costOf(off).toolRuns),
    'extra-calls': String(off.steps - on.steps),
    'p-example': fmtNumber(p),
    'k-example': String(K_EXAMPLE),
    'chance-example': chanceText(chainSuccess(p, K_EXAMPLE)),
    'eli-chance': chanceText(chainSuccess(p, eli.steps)),
    'target': fmtNumber(TARGET),
    'log-bound': fmtNumber(logBound(p, TARGET)),
    'target-steps': String(kT),
    'target-next': String(kT + 1),
    'chance-at-target': chanceText(chainSuccess(p, kT)),
    'chance-past-target': chanceText(chainSuccess(p, kT + 1)),
    'q-p': fmtNumber(qp),
    'q-k': String(K_EXAMPLE),
    'q-chance': chanceText(chainSuccess(qp, K_EXAMPLE)),
    'q-sum': fmtNumber(qp * K_EXAMPLE),
    'q-fail-one': fmtNumber(1 - qp),
  };
  for (const c of COST_RUNS) v[`cost-${c.id}`] = costLine(c, c.run());
  return v;
}

// ---- the stage ------------------------------------------------------------------------

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const W = 440;

// Which part of the loop drew a message: the model call, the tool run, or one of the exits.
function partOf(msg) {
  if (!msg) return null;
  if (msg.role === 'plan' || msg.role === 'call') return 'model';
  if (msg.role === 'result') return 'tool';
  if (msg.role === 'answer') return 'done';
  if (msg.role === 'stop') return msg.text === 'done' ? 'done' : msg.text;
  return null;
}

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let limit = START_LIMIT;
  let repeatCheck = true;
  let pN = START_P;
  let costIdx = 0;

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  const current = () => {
    if (step <= 1) return stepRun();
    if (step === 2) return limitRun(limit);
    if (step === 3) return loopRun(repeatCheck);
    if (step === 5) return COST_RUNS[costIdx].run();
    return null;
  };
  let result = current();

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  const host = doc.createElement('div');
  host.className = 'agent-trace';
  host.style.marginTop = '12px';
  svg.after(host);
  const tv = createTraceView(host, { tools: controls, label: 'Trace: every message in and out', reduced, announce, onSelect: () => draw() });
  // onSelect fires for every selection, the view's own included; draw() never selects, so there is no loop.
  const traceBits = [controls.querySelector('.trace-buttons'), controls.querySelector('.trace-status')].filter(Boolean);

  // ---- controls, one group per step that has one ----
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const groups = [];
  const group = (forStep, label) => {
    const g = make('div');
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', label);
    Object.assign(g.style, { display: 'none', flexWrap: 'wrap', alignItems: 'center', gap: '6px 14px' });
    controls.append(g);
    groups.push([forStep, g]);
    return g;
  };

  const rerun = (say) => {
    tv.stop();
    result = current();
    tv.show(result, { animate: false });
    tv.select(result.messages.length - 1);
    draw();
    announce(say);
  };

  // Step 3: the step limit.
  const limitIn = make('input', { type: 'range', min: String(LIMIT_MIN), max: String(LIMIT_MAX), step: '1', value: String(limit) });
  const limitOut = make('output');
  limitOut.setAttribute('aria-hidden', 'true');
  const limitLab = make('label');
  limitLab.append('step limit', ' ', limitIn, ' ', limitOut);
  group(2, 'step limit').append(limitLab);
  limitIn.addEventListener('input', () => {
    limit = Number(limitIn.value);
    syncControls();
    rerun(limitLine(limitRun(limit), limit));
  });

  // Step 4: the repeat check, a switch.
  const repeatIn = make('input', { type: 'checkbox', checked: repeatCheck });
  repeatIn.setAttribute('role', 'switch');
  repeatIn.style.accentColor = CYAN;
  const repeatLab = make('label');
  repeatLab.append(repeatIn, ' ', 'repeat check');
  group(3, 'repeat check').append(repeatLab);
  repeatIn.addEventListener('change', () => {
    repeatCheck = repeatIn.checked;
    const r = loopRun(repeatCheck);
    rerun(`Repeat check ${repeatCheck ? 'on' : 'off'}. Stopped: ${r.stop} after ${plural(r.steps, 'step')}.`);
  });

  // Step 5: p, the chance that one step succeeds.
  const pIn = make('input', { type: 'range', min: String(P_MIN), max: String(P_MAX), step: '1', value: String(pN) });
  const pOut = make('output');
  pOut.setAttribute('aria-hidden', 'true');
  const pLab = make('label');
  pLab.append('p, chance one step succeeds', ' ', pIn, ' ', pOut);
  group(4, 'chance per step').append(pLab);
  pIn.addEventListener('input', () => {
    pN = Number(pIn.value);
    syncControls();
    draw();
    announce(`${chartReadout(pOf(pN))}; ${targetReadout(pOf(pN))}.`);
  });

  // Step 6: which run to count.
  const costSel = make('select');
  COST_RUNS.forEach((c, k) => costSel.append(make('option', { value: String(k), textContent: c.label })));
  const costLab = make('label');
  costLab.append('run', ' ', costSel);
  group(5, 'run to count').append(costLab);
  costSel.addEventListener('change', () => {
    costIdx = Number(costSel.value);
    const c = COST_RUNS[costIdx];
    rerun(`${costLine(c, c.run())}.`);
  });

  function syncControls() {
    limitIn.value = String(limit);
    limitOut.textContent = String(limit);
    limitIn.setAttribute('aria-valuetext', `step limit ${limit}`);
    repeatIn.checked = repeatCheck;
    pIn.value = String(pN);
    pOut.textContent = fmtNumber(pOf(pN));
    pIn.setAttribute('aria-valuetext', `p = ${fmtNumber(pOf(pN))}`);
    costSel.value = String(costIdx);
  }

  // Sizes are in viewBox units. At phone width ml.css sets class stage-text to 15 units,
  // so every line here is laid out to fit at 15 units too (about 9 units per character).
  const label = (x, y, text, { anchor = 'start', fill = MUTED, size = 13, weight = 400, parent = svg } = {}) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, parent);

  const frame = (h) => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
    const m = el('marker', { id: 'al-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg));
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, m);
  };
  const arrow = (d, dashed = false) => el('path', { d, fill: 'none', stroke: MUTED, 'stroke-width': 1.25, 'stroke-dasharray': dashed ? '4 3' : undefined, 'marker-end': 'url(#al-arrow)' }, svg);

  // A box; the one in use is drawn with a thick outline and bold text, not by colour alone.
  // A part that is switched off is dashed.
  const box = (x, y, w, h, text, on, { dashed = false } = {}) => {
    el('rect', {
      x, y, width: w, height: h, rx: 6,
      fill: CYAN, 'fill-opacity': on ? 0.14 : 0,
      stroke: on ? CYAN : LINE, 'stroke-width': on ? 2.5 : 1.25,
      'stroke-dasharray': dashed ? '5 4' : undefined,
    }, svg);
    if (text) label(x + w / 2, y + h / 2 + 5, text, { anchor: 'middle', fill: on ? STRONG : MUTED, weight: on ? 700 : 400 });
  };

  const selectedMsg = () => {
    const i = tv.selected;
    return result && i >= 0 && i < result.messages.length ? result.messages[i] : null;
  };

  // Steps 1 to 4: the loop with its three ways out. The part that made the selected row is bold.
  function drawLoop(L, check) {
    frame(236);
    const part = partOf(selectedMsg());
    const LX = 26;
    const LW = 176;
    const RX = 264;
    const RW = 166;
    const H = 34;
    const ys = [10, 60, 110, 160];
    box(LX, ys[0], LW, H, 'steps below limit?', false);
    box(LX, ys[1], LW, H, 'model call: a step', part === 'model');
    box(LX, ys[2], LW, H, check ? 'same call again?' : 'repeat check off', false, { dashed: !check });
    box(LX, ys[3], LW, H, 'loop runs the tool', part === 'tool');
    box(RX, ys[0], RW, H, 'step limit', part === 'step limit');
    box(RX, ys[1], RW, H, 'done', part === 'done');
    box(RX, ys[2], RW, H, 'repeated call', part === 'repeated call', { dashed: !check });
    for (let k = 0; k < 3; k++) arrow(`M${LX + LW / 2},${ys[k] + H} L${LX + LW / 2},${ys[k + 1] - 2}`);
    arrow(`M${LX},${ys[3] + H / 2} L12,${ys[3] + H / 2} L12,${ys[0] + H / 2} L${LX - 2},${ys[0] + H / 2}`);
    const words = ['no', 'answer', 'yes'];
    for (let k = 0; k < 3; k++) {
      arrow(`M${LX + LW},${ys[k] + H / 2} L${RX - 2},${ys[k] + H / 2}`, k === 2 && !check);
      label((LX + LW + RX) / 2, ys[k] + H / 2 - 5, words[k], { anchor: 'middle', size: 12 });
    }
    label(LX + LW / 2 + 8, ys[1] - 4, 'yes', { size: 12 });
    label(LX + LW / 2 + 8, ys[2] - 4, 'a call', { size: 12 });
    label(LX + LW / 2 + 8, ys[3] - 4, 'no', { size: 12 });
    const line = `${plural(result.steps, 'step')}, limit ${L}, stopped: ${result.stop}`;
    label(10, 226, line, { fill: STRONG });
    const where = { model: 'the model call', tool: 'the tool run', done: 'the done exit', 'step limit': 'the step limit exit', 'repeated call': 'the repeat check exit' }[part];
    return `The loop: check the step limit, call the model (one step), stop if it answered, ${check ? 'stop if the call is the same as the one just before' : 'the repeat check is off'}, run the tool, go round again. This run: ${line}.${where ? ` Selected row: ${where}.` : ''}`;
  }

  // Step 5: the chance that all k steps succeed, p to the power k.
  function drawChart() {
    frame(354);
    const p = pOf(pN);
    const X = linearScale([0, K_MAX], [60, 424]);
    const Y = linearScale([0, 1], [212, 14]);
    axes(svg, X, Y, { xLabel: 'k, the number of steps', yLabel: 'chance all succeed', ticks: 4 });
    // The target line, dashed.
    el('line', { x1: X(0), x2: X(K_MAX), y1: Y(TARGET), y2: Y(TARGET), stroke: MUTED, 'stroke-width': 1, 'stroke-dasharray': '4 4' }, svg);
    const pts = chartPoints(p);
    el('path', { d: pathFrom(pts, X, Y), fill: 'none', stroke: CYAN, 'stroke-width': 2 }, svg);
    for (const [k, y] of pts) el('circle', { cx: X(k), cy: Y(y), r: k === K_EXAMPLE ? 5 : 2.5, fill: k === K_EXAMPLE ? BG : CYAN, stroke: CYAN, 'stroke-width': k === K_EXAMPLE ? 2.5 : 0 }, svg);
    const yk = chainSuccess(p, K_EXAMPLE);
    el('line', { x1: X(K_EXAMPLE), x2: X(K_EXAMPLE), y1: Y(0), y2: Y(yk) + 6, stroke: STRONG, 'stroke-width': 1, 'stroke-dasharray': '2 3' }, svg);
    label(10, 278, chartReadout(p), { fill: STRONG, weight: 700 });
    // At phone width the edge note goes on its own line.
    const [tLine, edge] = targetReadout(p).split(' (');
    label(10, 300, tLine, { fill: STRONG });
    if (edge) label(10, 322, `(${edge}`, { fill: STRONG });
    label(10, edge ? 344 : 322, 'assumes the steps are independent');
    return `Line chart: the chance that all k steps succeed, p to the power k, for k from 1 to ${K_MAX}, with a dashed line at ${fmtNumber(TARGET)}. ${chartReadout(p)}; ${targetReadout(p)}. This assumes each step succeeds or fails independently of the others.`;
  }

  // Step 6: one box per model call, and under it the tool run that call led to.
  function drawCost() {
    frame(176);
    const c = costOf(result);
    const calls = result.messages.filter((m) => m.role === 'call');
    label(10, 18, `model calls (steps): ${c.calls}`, { fill: STRONG, weight: 700 });
    label(10, 92, `tool runs: ${c.toolRuns}`, { fill: STRONG, weight: 700 });
    let notRun = 0;
    for (let s = 1; s <= result.steps; s++) {
      const x = 10 + (s - 1) * 52;
      box(x, 26, 44, 34, String(s), true);
      const call = calls.find((m) => m.step === s);
      const ran = result.messages.some((m) => m.role === 'result' && m.step === s);
      if (ran) box(x, 100, 44, 34, String(s), true);
      else if (call) { box(x, 100, 44, 34, '', false, { dashed: true }); notRun = s; }
    }
    const end = notRun ? `the call in step ${notRun} was not run; ${endText(result)}` : endText(result);
    label(10, 166, end, { fill: STRONG });
    return `${COST_RUNS[costIdx].label}: ${costText(c)}. ${end[0].toUpperCase()}${end.slice(1)}.`;
  }

  function draw() {
    let aria;
    if (step === 2) aria = drawLoop(limit, true);
    else if (step === 3) aria = drawLoop(DEFAULT_CONFIG.stepLimit, repeatCheck);
    else if (step === 4) aria = drawChart();
    else if (step === 5) aria = drawCost();
    else aria = drawLoop(DEFAULT_CONFIG.stepLimit, true);
    svg.setAttribute('aria-label', aria);
  }

  syncControls();

  return {
    count: 6,
    show(i, animate = false) {
      tv.stop();
      step = i;
      for (const [s, g] of groups) g.style.display = s === i ? 'flex' : 'none';
      const chart = i === 4;
      host.style.display = chart ? 'none' : '';
      for (const b of traceBits) b.style.display = chart ? 'none' : '';
      syncControls();
      result = current();
      if (chart) { draw(); return; }
      tv.show(result, { animate });
      // Opened without a run: step 2 selects the answer, the others the stop row.
      if (!(animate && !reduced)) {
        const k = i === 1 ? result.messages.findIndex((m) => m.role === 'answer') : result.messages.length - 1;
        tv.select(k);
      }
      draw();
    },
  };
}

