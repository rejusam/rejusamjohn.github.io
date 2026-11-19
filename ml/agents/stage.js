// Lesson L6.1, "What is an agent?": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-agents.test.mjs); the
// DOM is only touched inside createStage. Every trace comes from runAgent or oneAnswer
// in agent.js, every true value from the invented world, the price list and calc, and
// every number in the page's text from pageValues. All text goes in with textContent.

import { WORLD } from '../assets/world.js';
import { PRICES, RULES, TASKS, runAgent, oneAnswer, calc, fmtMoney } from '../assets/agent.js';
import { createTraceView, rulesFor } from '../assets/trace-view.js';
import { el } from '../assets/plot.js';

// ---- the two tasks --------------------------------------------------------------

export const TASK_ONE = 'mia-food';
export const TASK_MANY = 'eli-shop';

// Short names for the rule strip in step 5, in the order of RULES.
export const RULE_NAMES = [
  { id: 'R1', name: 'not found' },
  { id: 'R2', name: 'notepad' },
  { id: 'R3', name: 'lookup' },
  { id: 'R4', name: 'sum' },
  { id: 'R5', name: 'send' },
  { id: 'R6', name: 'answer' },
];

// The one answer as two rows: the task and the answer the stand-in writes with no tools.
export function oneAnswerTrace(taskId) {
  const o = oneAnswer(taskId);
  const messages = o.messages.filter((m) => m.role !== 'plan');
  return { messages, stop: 'done', steps: 1, answer: o.answer };
}

// The true answer, from the world's facts, the price list and the calculator.
export function trueAnswer(taskId) {
  const g = TASKS[taskId].goal;
  if (g.kind === 'food') return `${g.person} likes ${WORLD.likes[g.person]}.`;
  if (g.kind === 'shop') {
    const expr = g.people.map((p) => `${g.qty}*${PRICES[WORLD.likes[p]]}`).join('+');
    return `That costs ${fmtMoney(calc(expr).value)}.`;
  }
  return null;
}

export const isRight = (taskId, answer) => answer === trueAnswer(taskId);

// Which box of the loop picture a row belongs to.
const BOX_OF = { task: 'read', note: 'read', plan: 'choose', answer: 'choose', approval: 'choose', call: 'act', result: 'see', stop: 'stop' };
export const loopBox = (msg) => BOX_OF[msg.role] || 'read';

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function counterText(msg, result) {
  if (msg.role === 'task') return 'the task: before step 1';
  if (msg.role === 'stop') return `stopped after ${plural(result.steps, 'step')}`;
  return `step ${msg.step} of ${result.steps} (model call ${msg.step})`;
}

export const ruleHighlight = (msg) => rulesFor(msg).map((r) => r.id);

// Every rule id a run applies, in the order of RULES.
export function rulesUsed(result) {
  const seen = new Set();
  for (const m of result.messages) for (const id of ruleHighlight(m)) seen.add(id);
  return RULES.map((r) => r.id).filter((id) => seen.has(id));
}

const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
const money = (text) => Number(text.match(/\$(\d+\.\d\d)/)[1]);

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const oneMia = oneAnswer(TASK_ONE);
  const mia = runAgent(TASK_ONE);
  const oneEli = oneAnswer(TASK_MANY);
  const eli = runAgent(TASK_MANY);
  const g = TASKS[TASK_MANY].goal;
  const food = WORLD.likes[g.people[0]];
  const calls = eli.messages.filter((m) => m.role === 'call');
  const oneCost = money(oneEli.answer);
  const agentCost = money(eli.answer);
  const used = rulesUsed(eli);
  return {
    'one-mia': oneMia.answer,
    'true-mia': WORLD.likes[TASKS[TASK_ONE].goal.person],
    'guess-food': WORLD.foods[0],
    'one-mia-rules': listText(ruleHighlight(oneMia.messages[2])),
    'mia-steps': String(mia.steps),
    'mia-agent': mia.answer,
    'mia-call': mia.messages.find((m) => m.role === 'call').text,
    'eli-steps': String(eli.steps),
    'eli-calls': String(calls.length),
    'eli-steps-minus': `${eli.steps} − 1 = ${eli.steps - 1}`,
    'eli-calc': calls.find((m) => m.tool === 'calculator').text,
    'eli-qty': String(g.qty),
    'eli-food': food,
    'eli-price': fmtMoney(PRICES[food]),
    'agent-eli': fmtMoney(agentCost),
    'true-eli': fmtMoney(money(trueAnswer(TASK_MANY))),
    'one-eli': fmtMoney(oneCost),
    'eli-error': fmtMoney(agentCost - oneCost),
    'guess-price': fmtMoney(PRICES[Object.keys(PRICES)[0]]),
    'one-eli-rules': listText(ruleHighlight(oneEli.messages[2])),
    'rules-used': listText(used),
    'rules-unused': listText(RULES.map((r) => r.id).filter((id) => !used.includes(id))),
    'food-table': listText(WORLD.foods),
    'price-list': Object.keys(PRICES).map((f) => `${f} ${fmtMoney(PRICES[f])}`).join(', '),
    'likes-list': WORLD.people.map((p) => `${p} ${WORLD.likes[p]}`).join(', '),
  };
}

// ---- the stage ------------------------------------------------------------------------

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const W = 440;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;

  const results = [
    oneAnswerTrace(TASK_ONE),
    runAgent(TASK_ONE),
    runAgent(TASK_MANY),
    runAgent(TASK_MANY),
    runAgent(TASK_MANY),
  ];
  const oneEli = oneAnswer(TASK_MANY);

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  const host = doc.createElement('div');
  host.className = 'agent-trace';
  host.style.marginTop = '12px';
  svg.after(host);
  const tv = createTraceView(host, { tools: controls, label: 'Trace: every message in and out', reduced, announce, onSelect: () => draw() });

  // Sizes are in viewBox units. At phone width ml.css sets class stage-text to 15 units,
  // so every line here is laid out to fit at 15 units too.
  const label = (x, y, text, { anchor = 'start', fill = MUTED, size = 13, weight = 400 } = {}) =>
    el('text', { class: 'stage-text', x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, svg);

  const marker = () => {
    const m = el('marker', { id: 'ag-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg));
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: MUTED }, m);
  };
  const arrow = (d) => el('path', { d, fill: 'none', stroke: MUTED, 'stroke-width': 1.25, 'marker-end': 'url(#ag-arrow)' }, svg);

  // A box; the one in use is drawn with a thick outline and bold text, not by colour alone.
  const box = (x, y, w, h, text, on, { dashed = false, size = 13 } = {}) => {
    el('rect', {
      x, y, width: w, height: h, rx: 6,
      fill: CYAN, 'fill-opacity': on ? 0.14 : 0,
      stroke: on ? CYAN : LINE, 'stroke-width': on ? 2.5 : 1.25,
      'stroke-dasharray': dashed ? '5 4' : undefined,
    }, svg);
    label(x + w / 2, y + h / 2 + size * 0.36, text, { anchor: 'middle', fill: on ? STRONG : MUTED, size, weight: on ? 700 : 400 });
  };

  const frame = (h) => {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
    marker();
  };

  const selectedMsg = () => {
    const r = results[step];
    const i = tv.selected;
    return i >= 0 && i < r.messages.length ? r.messages[i] : null;
  };

  // Step 1: one model call, no loop, and the answer checked against the world.
  function drawOne(msg) {
    frame(112);
    const role = msg ? msg.role : null;
    box(10, 14, 90, 36, 'task', role === 'task');
    box(135, 14, 170, 36, 'model: one call', false);
    box(340, 14, 90, 36, 'answer', role === 'answer');
    arrow('M100,32 L133,32');
    arrow('M305,32 L338,32');
    const v = filled;
    label(10, 80, `check: in the invented world, mia likes ${v['true-mia']}`, { fill: STRONG });
    label(10, 100, `the answer "${v['one-mia']}" is wrong ✗`, { fill: STRONG });
    return `One model call: the task goes in and one answer comes out. In the invented world mia likes ${v['true-mia']}, so the answer "${v['one-mia']}" is wrong.`;
  }

  // Steps 2 and 3: the loop, with the box of the selected row in use.
  function drawLoop(msg) {
    frame(176);
    const r = results[step];
    const on = msg ? loopBox(msg) : null;
    const xs = { read: 6, choose: 115, act: 224, see: 333 };
    const names = { read: 'read', choose: 'choose', act: 'act', see: 'see result' };
    for (const k of Object.keys(xs)) box(xs[k], 36, 96, 36, names[k], on === k);
    arrow('M102,54 L113,54');
    arrow('M211,54 L222,54');
    arrow('M320,54 L331,54');
    arrow('M381,36 L381,16 L54,16 L54,34');
    label(219, 12, 'repeat', { anchor: 'middle', size: 12 });
    arrow('M163,72 L163,106');
    label(171, 94, 'an answer', { size: 12 });
    box(115, 108, 96, 32, 'stop', on === 'stop');
    const counter = msg ? counterText(msg, r) : '';
    label(8, 168, counter, { fill: STRONG });
    const where = on ? (on === 'see' ? 'see result' : on) : 'nothing selected';
    return `The loop: read, choose, act, see result, repeat; an answer leads to stop. Selected row: ${where}. ${counter}.`;
  }

  // Step 4: the one answer and the agent side by side, with the true cost.
  function drawCompare() {
    frame(150);
    const v = filled;
    const eli = results[3];
    el('rect', { x: 10, y: 8, width: 205, height: 92, rx: 6, fill: 'none', stroke: LINE, 'stroke-width': 1.25, 'stroke-dasharray': '5 4' }, svg);
    el('rect', { x: 225, y: 8, width: 205, height: 92, rx: 6, fill: CYAN, 'fill-opacity': 0.08, stroke: CYAN, 'stroke-width': 2 }, svg);
    label(22, 30, 'one answer', { fill: STRONG, weight: 700 });
    label(22, 48, '1 model call', { size: 12 });
    label(22, 72, oneEli.answer, { fill: STRONG });
    label(22, 90, `✗ wrong by ${v['eli-error']}`, { fill: STRONG });
    label(237, 30, 'agent', { fill: STRONG, weight: 700 });
    label(237, 48, plural(eli.steps, 'model call'), { size: 12 });
    label(237, 72, eli.answer, { fill: STRONG });
    label(237, 90, '✓ right', { fill: STRONG });
    const calcCall = eli.messages.find((m) => m.role === 'call' && m.tool === 'calculator');
    label(10, 124, `true cost: ${calcCall.text} = ${calc(calcCall.input.expr).value}, so ${v['true-eli']}`, { fill: STRONG });
    label(10, 142, `${v['eli-food']} cost ${v['eli-price']} each, from lookup`);
    return `Side by side for the eli task. One answer, 1 model call and no tools: ${oneEli.answer} Wrong by ${v['eli-error']}. Agent, ${eli.steps} steps: ${eli.answer} Right. True cost ${v['true-eli']}, from the calculator.`;
  }

  // Step 5: the six rules, with the ones the selected row applied in use.
  function drawRules(msg) {
    frame(128);
    const ids = msg ? ruleHighlight(msg) : [];
    RULE_NAMES.forEach((r, k) => box(10 + (k % 3) * 142, 6 + Math.floor(k / 3) * 48, 136, 40, `${r.id} ${r.name}`, ids.includes(r.id)));
    const text = ids.length ? `rules the selected row applied: ${listText(ids)}` : 'the selected row applies no rule';
    label(10, 120, text, { fill: STRONG });
    return `The six rules, R1 to R6. ${text}.`;
  }

  function draw() {
    const msg = selectedMsg();
    let aria;
    if (step === 0) aria = drawOne(msg);
    else if (step === 3) aria = drawCompare();
    else if (step === 4) aria = drawRules(msg);
    else aria = drawLoop(msg);
    svg.setAttribute('aria-label', aria);
  }

  // Follow the selection while Run reveals rows (the trace view moves it without a callback).
  if (typeof MutationObserver !== 'undefined') {
    new MutationObserver(() => draw()).observe(host, { subtree: true, childList: true, attributes: true, attributeFilter: ['aria-current'] });
  }

  return {
    count: 5,
    show(i, animate = false) {
      tv.stop();
      step = i;
      tv.show(results[i], { animate });
      if (i === 4 && !(animate && !reduced)) {
        const first = results[i].messages.findIndex((m) => m.role === 'plan');
        tv.select(first);
      }
      draw();
    },
  };
}
