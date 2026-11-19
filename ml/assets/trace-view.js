// Level 6 trace view: draws an agent trace (runAgent's messages) as selectable rows,
// with its own Run and Next message buttons in the lesson's tools slot.
// Pure scheduling helpers (scheduleRows, _makeScheduler) are tested in Node; the DOM is
// only touched inside createTraceView. All text goes in with textContent.
import {RULES} from './agent.js';

export const RUN_MS = 450;

export const ROLE_LABELS = {
  task: 'Task', plan: 'Plan', call: 'Tool call', result: 'Result', answer: 'Answer',
  stop: 'Stopped', note: 'Note', approval: 'Waiting for your yes',
};

// What to say in the "Why this step" box when a row carries no rule.
export const NO_RULE_TEXT = {
  task: 'No rule applies: this is the task the run starts from.',
  plan: 'No rule applies to this line: it is the plan line written before the action.',
  call: 'No rule applies to this line: it is the call the chosen action makes.',
  result: "A tool's result: the loop ran the tool. No rule applies.",
  answer: 'No rule applies to this line: it is the final answer.',
  stop: 'No rule applies: the loop stopped here.',
  note: 'No rule applies: this is a note from the loop.',
  approval: 'No rule applies: the loop waits for a yes or no.',
};

// Row indices to reveal, one per tick. Reduced motion: one tick that reveals everything.
export function scheduleRows(count, {reduced = false} = {}) {
  if (count <= 0) return [];
  if (reduced) return [count - 1];
  return Array.from({length: count}, (_, i) => i);
}

// A scheduler that can be cancelled. start() cancels any earlier run first, and a run
// that was cancelled never calls back again, even if its timer fires late.
export function _makeScheduler(timerFns = {}) {
  const set = timerFns.setTimeout || ((f, ms) => setTimeout(f, ms));
  const clear = timerFns.clearTimeout || ((id) => clearTimeout(id));
  let gen = 0;
  let timer = null;
  const cancel = () => {
    gen++;
    if (timer !== null) { clear(timer); timer = null; }
  };
  const start = (count, {reduced = false, interval = RUN_MS} = {}, onReveal, onDone) => {
    cancel();
    const mine = gen;
    const plan = scheduleRows(count, {reduced});
    let k = 0;
    const tick = () => {
      timer = null;
      if (mine !== gen) return;
      onReveal(plan[k]);
      k++;
      if (k < plan.length) timer = set(tick, interval);
      else if (onDone) onDone();
    };
    if (plan.length === 0) { if (onDone) onDone(); return; }
    tick();
  };
  return {start, cancel, get active() { return timer !== null; }};
}

export function statusText(result) {
  const n = result.steps;
  return `Stopped: ${result.stop} · ${n} ${n === 1 ? 'step' : 'steps'}`;
}

// The rules to show for a message: every id in `rules`, else `rule`, else none.
export function rulesFor(msg) {
  const ids = msg.rules && msg.rules.length ? msg.rules : (msg.rule ? [msg.rule] : []);
  return ids.map((id) => ({id, text: (RULES.find((r) => r.id === id) || {}).text || ''}));
}

// The paragraphs of the "Why this step" box for one message, as plain strings.
export function whyParagraphs(msg, unseen = false) {
  const rs = rulesFor(msg);
  const out = rs.length
    ? rs.map((r) => `${r.id} ${r.text}`)
    : [NO_RULE_TEXT[msg.role] || 'No rule applies.'];
  if (unseen) out.push('This message is outside the window the model is given, so it no longer sees it.');
  return out;
}

// Usage:
//   const tv = createTraceView(host, {tools, label: 'Trace of mia-food', onSelect, reduced, announce});
//   tv.show(runAgent(...), {animate: true});   // animate false (or reduced) shows it all
//   tv.stop();                                  // cancel any running timer; keeps rows
// onSelect(i) is called for every selection: a click, the arrow keys, select(i), and the
// view's own selection of the newest row on show, Run and Next message. A handler must
// not call select() itself.
// Options: reduced (prefers-reduced-motion); announce(text) (the lesson's live region).
// Announcements: if announce is given, status text goes there and the view's own status
// span is plain text; if not, the span is a polite live region (role=status).
// Rows marked "no longer seen" come from msg.unseen.
export function createTraceView(host, {tools, label = 'Agent trace', onSelect, reduced = false, announce} = {}) {
  const doc = host.ownerDocument;
  const mk = (tag, cls, text) => {
    const n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const sched = _makeScheduler();
  let result = null;
  let shown = 0;
  let selected = -1;

  const root = mk('div', 'trace');
  const list = mk('ol', 'trace-list');
  list.setAttribute('aria-label', label);
  const why = mk('div', 'trace-why');
  root.append(list, why);
  host.append(root);

  const bar = mk('div', 'trace-buttons');
  const run = mk('button', 'trace-run', 'Run');
  const next = mk('button', null, 'Next message');
  run.type = 'button';
  next.type = 'button';
  const status = mk('span', 'trace-status');
  if (!announce) status.setAttribute('role', 'status');
  bar.append(run, next);
  if (tools) tools.append(bar, status);

  const say = (t) => { status.textContent = t; if (announce) announce(t); };
  const total = () => (result ? result.messages.length : 0);
  const isUnseen = (i) => !!result.messages[i].unseen;

  function renderWhy() {
    why.replaceChildren();
    if (selected < 0) return;
    const m = result.messages[selected];
    why.append(mk('h4', 'trace-why-title', 'Why this step'));
    const rs = rulesFor(m);
    whyParagraphs(m, isUnseen(selected)).forEach((t, k) => {
      const p = mk('p', k < rs.length ? 'trace-rule' : null);
      if (k < rs.length) p.append(mk('strong', null, rs[k].id + ' '), doc.createTextNode(rs[k].text));
      else p.textContent = t;
      why.append(p);
    });
  }

  function select(i, {notify = true} = {}) {
    if (!result || i < 0 || i >= shown) return;
    selected = i;
    [...list.children].forEach((li, k) => {
      const b = li.firstChild;
      if (k === i) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
      li.classList.toggle('is-selected', k === i);
    });
    renderWhy();
    if (notify && onSelect) onSelect(i);
  }

  function addRow(i) {
    const m = result.messages[i];
    const li = mk('li', 'trace-row role-' + m.role);
    const unseen = isUnseen(i);
    if (unseen) li.classList.add('is-unseen');
    const b = mk('button');
    b.type = 'button';
    b.append(mk('span', 'trace-chip', ROLE_LABELS[m.role] || m.role));
    b.append(mk('span', 'trace-text', m.text));
    if (unseen) b.append(mk('span', 'trace-tag', 'no longer seen'));
    b.addEventListener('click', () => select(i));
    b.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
      if (!d) return;
      const k = i + d;
      if (k >= 0 && k < shown) { e.preventDefault(); select(k); list.children[k].firstChild.focus(); }
    });
    li.append(b);
    list.append(li);
  }

  function reveal(upTo, {follow = true} = {}) {
    const wasLast = selected < 0 || selected === shown - 1;
    while (shown <= upTo && shown < total()) { addRow(shown); shown++; }
    if (follow && wasLast) select(shown - 1);
    sync();
  }

  function sync() {
    next.disabled = !result || shown >= total();
    run.disabled = !result;
  }

  function finish() { say(statusText(result)); sync(); }

  function clear() {
    sched.cancel();
    list.replaceChildren();
    shown = 0;
    selected = -1;
    why.replaceChildren();
  }

  function play(animate) {
    clear();
    if (!result) return;
    if (!animate || reduced) { reveal(total() - 1); finish(); return; }
    say('Running…');
    sched.start(total(), {reduced: false}, (i) => reveal(i), finish);
  }

  run.addEventListener('click', () => play(true));
  next.addEventListener('click', () => {
    if (!result || shown >= total()) return;
    sched.cancel();
    reveal(shown);
    if (shown >= total()) finish(); else say(`Showing ${shown} of ${total()} messages`);
  });
  sync();

  return {
    show(r, {animate = false} = {}) {
      result = r;
      play(animate);
    },
    stop() {
      const running = sched.active;
      sched.cancel();
      if (running && result && shown < total()) say(`Paused at message ${shown} of ${total()}`);
      sync();
    },
    select(i) { select(i); },
    get selected() { return selected; },
  };
}
