import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG, PRICES, TASKS, TOOLS, runAgent, calc, callText, fmtMoney, fmtNumber, lookupText } from '../assets/agent.js';
import { WORLD } from '../assets/world.js';
import * as S from '../agent-mistakes/stage.js';
import { whyParagraphs } from '../assets/trace-view.js';

const html = readFileSync(new URL('../agent-mistakes/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../agent-mistakes/stage.js', import.meta.url), 'utf8');

const calls = (r) => r.messages.filter((m) => m.role === 'call');
const results = (r) => r.messages.filter((m) => m.role === 'result');
const plans = (r) => r.messages.filter((m) => m.role === 'plan');
const TOOL_NAMES = Object.keys(TOOLS);

test('the lesson runs match runAgent, and zoe is told pears cost $1.10', () => {
  const yes = runAgent('tell-zoe', {}, { approve: true });
  assert.equal(yes.stop, 'done');
  assert.equal(yes.answer, `Told zoe: pears cost ${fmtMoney(PRICES.pears)} each.`);
  assert.equal(yes.answer, 'Told zoe: pears cost $1.10 each.'); // the one literal
  assert.equal(S.TASK_NF, 'kai-food');
  assert.equal(S.TASK_SUM, 'sum');
  assert.equal(S.TASK_SEND, 'tell-zoe');
  assert.deepEqual(S.NOT_FOUND_OPTIONS, ['say so', 'try again', 'guess']);
  for (const o of S.NOT_FOUND_OPTIONS) assert.deepEqual(S.notFoundRun(o), runAgent('kai-food', { onNotFound: o }));
  assert.deepEqual(S.sendRun(undefined, true), runAgent('tell-zoe'));
  assert.deepEqual(S.sendRun(true, true), yes);
  assert.deepEqual(S.sendRun(false, true), runAgent('tell-zoe', {}, { approve: false }));
  assert.deepEqual(S.sendRun(undefined, false), runAgent('tell-zoe', { askBeforeChanges: false }));
  assert.deepEqual(S.sendRun(true, false), runAgent('tell-zoe', { askBeforeChanges: false }));
  for (const id of Object.keys(TASKS)) assert.deepEqual(S.defaultRun(id), runAgent(id));
  // Every run the page shows is deterministic.
  for (const r of S.PAGE_RUNS) assert.deepEqual(r.run(), r.run());
});

test('not-found stories', () => {
  // kai is not in the invented world, so lookup finds nothing.
  assert.ok(!WORLD.people.includes('kai'));
  assert.equal(lookupText('kai'), 'not found: kai');
  // Every not-found run starts the same way: step 1 writes lookup("kai") and gets "not found: kai".
  for (const o of S.NOT_FOUND_OPTIONS) {
    const r = S.notFoundRun(o);
    assert.equal(calls(r)[0].text, 'lookup("kai")');
    assert.equal(calls(r)[0].step, 1);
    assert.equal(results(r)[0].text, 'not found: kai');
    assert.equal(results(r).length, 1);
  }
  // say so: done in 2 steps, a plain "not found" answer, by R1.
  const say = S.notFoundRun('say so');
  assert.equal(say.stop, 'done');
  assert.equal(say.steps, 2);
  assert.equal(say.answer, 'I could not find kai.');
  assert.deepEqual(plans(say).map((m) => m.rule), ['R3', 'R1']);
  // try again: step 2 writes the same lookup; the repeat check stops it before it runs.
  const again = S.notFoundRun('try again');
  assert.equal(again.stop, 'repeated call');
  assert.equal(again.steps, 2);
  assert.equal(again.answer, null);
  assert.deepEqual(calls(again).map((m) => m.text), ['lookup("kai")', 'lookup("kai")']);
  assert.equal(again.messages.at(-2).role, 'call');
  // With the check off, only the step limit stops it.
  const noCheck = runAgent('kai-food', { onNotFound: 'try again', repeatCheck: false });
  assert.equal(noCheck.stop, 'step limit');
  assert.equal(noCheck.steps, DEFAULT_CONFIG.stepLimit);
  // guess: a made-up answer, the first food in the table; kai likes nothing in the world.
  const guess = S.notFoundRun('guess');
  assert.equal(guess.stop, 'done');
  assert.equal(guess.steps, 2);
  assert.equal(guess.answer, `kai likes ${WORLD.foods[0]}.`);
  assert.equal(WORLD.foods[0], 'apples');
  assert.ok(!('kai' in WORLD.likes));
  assert.deepEqual(plans(guess).at(-1).rules, ['R1', 'R6']);
  assert.ok(plans(guess).at(-1).text.startsWith('Guess (first food in the table): kai likes apples.'));
  // The answers read alike; only the trace shows the guess.
  assert.equal(S.outcomeText(say), 'done after 2 steps: I could not find kai.');
  assert.equal(S.outcomeText(again), 'stopped: repeated call after 2 steps, no answer');
  assert.equal(S.outcomeText(guess), 'done after 2 steps: kai likes apples.');
});

test('tool error story', () => {
  // The calculator cannot read "2+".
  assert.equal(S.BAD_EXPR, '2+');
  assert.deepEqual(calc('2+'), { ok: false, error: 'cannot read "2+"' });
  assert.deepEqual(TOOLS.calculator.run({ expr: '2+' }), { ok: false, text: 'cannot read "2+"' });
  const c = S.toolErrorCase();
  assert.equal(c.writtenIn, true);
  assert.deepEqual(c.messages.map((m) => m.role), ['task', 'call', 'result', 'stop']);
  assert.equal(c.messages[0].text, TASKS.sum.text);
  // Every row is true: the call row shows "2+" and the result row is the calculator's real output.
  assert.equal(c.messages[1].text, callText('calculator', { expr: '2+' }));
  assert.equal(c.messages[1].text, 'calculator("2+")');
  assert.deepEqual(c.messages[1].input, { expr: '2+' });
  assert.equal(c.messages[1].rule, undefined); // not written by the stand-in
  // Each written-in row explains itself in "Why this step".
  assert.equal(c.messages[1].why, S.WHY_CALL);
  assert.match(S.WHY_CALL, /No rule wrote this call/);
  assert.match(S.WHY_CALL, /stand-in never writes it/);
  assert.equal(c.messages[2].why, S.WHY_RESULT);
  assert.equal(c.messages[3].why, S.WHY_STOP);
  assert.deepEqual(whyParagraphs(c.messages[1]), [S.WHY_CALL]);
  assert.deepEqual(whyParagraphs(c.messages[3]), [S.WHY_STOP]);
  assert.equal(c.messages[2].text, TOOLS.calculator.run({ expr: '2+' }).text);
  assert.equal(c.stop, 'tool error');
  assert.equal(c.messages[3].text, 'tool error');
  assert.equal(c.steps, 1);
  assert.equal(c.answer, null);
  assert.deepEqual(c.state.sent, []);
  // The stop follows the loop's own rule: runAgent stops the same way when a tool returns this result.
  const failing = { ...TOOLS, calculator: { ...TOOLS.calculator, run: () => TOOLS.calculator.run({ expr: '2+' }) } };
  const real = runAgent('sum', {}, { tools: failing });
  assert.equal(real.stop, 'tool error');
  assert.equal(real.steps, 1);
  assert.equal(results(real).at(-1).text, c.messages[2].text);
  assert.equal(real.messages.at(-2).role, 'result');
  assert.equal(S.stopAfter('calculator', TOOLS.calculator.run({ expr: '2+' })), 'tool error');
  assert.equal(S.stopAfter('lookup', TOOLS.lookup.run({ key: 'kai' })), null);
  assert.equal(S.stopAfter('calculator', TOOLS.calculator.run({ expr: '2+3' })), null);
  // The stand-in itself writes a call the calculator can read.
  const sum = runAgent('sum');
  assert.equal(calls(sum)[0].text, 'calculator("3*1.25+2*0.8")');
  assert.equal(results(sum)[0].text, fmtNumber(calc('3*1.25+2*0.8').value));
  assert.equal(results(sum)[0].text, '5.35');
  // No run of the stand-in on any task with the default tools ends in a tool error.
  for (const id of Object.keys(TASKS)) assert.notEqual(runAgent(id).stop, 'tool error');
});

test('tools-needed table equals the tools called', () => {
  const t = S.needsTable();
  assert.deepEqual(t.map((row) => row.id), Object.keys(TASKS));
  assert.deepEqual(Object.keys(TASKS), ['mia-food', 'sum', 'eli-shop', 'pair-shop', 'kai-food', 'tell-zoe']);
  for (const row of t) {
    const r = runAgent(row.id);
    const called = TOOL_NAMES.filter((n) => calls(r).some((m) => m.tool === n));
    const asked = TOOL_NAMES.filter((n) => r.messages.some((m) => m.role === 'approval' && m.tool === n));
    assert.deepEqual(row.called, called, row.id);
    assert.deepEqual(row.asked, asked, row.id);
    assert.deepEqual(row.needs, TOOL_NAMES.filter((n) => called.includes(n) || asked.includes(n)), row.id);
    // With only those tools switched on, the run is the same.
    const only = Object.fromEntries(TOOL_NAMES.map((n) => [n, row.needs.includes(n)]));
    assert.deepEqual(runAgent(row.id, { tools: only }), r, row.id);
    assert.deepEqual(S.onlyNeededRun(row.id), r, row.id);
  }
  const by = Object.fromEntries(t.map((row) => [row.id, row]));
  assert.deepEqual(by['mia-food'].needs, ['lookup']);
  assert.deepEqual(by.sum.needs, ['calculator']);
  assert.deepEqual(by['eli-shop'].needs, ['calculator', 'lookup']);
  assert.deepEqual(by['pair-shop'].needs, ['calculator', 'lookup']);
  assert.deepEqual(by['kai-food'].needs, ['lookup']);
  assert.deepEqual(by['tell-zoe'].called, ['lookup']);
  assert.deepEqual(by['tell-zoe'].asked, ['send']);
  assert.deepEqual(S.unusedTools(), ['notepad']);
  // The union of the needed sets: calculator, lookup and send.
  assert.deepEqual(TOOL_NAMES.filter((n) => t.some((row) => row.needs.includes(n))), ['calculator', 'lookup', 'send']);
  assert.equal(TOOLS.send.kind, 'changes things');
  assert.deepEqual(TOOL_NAMES.filter((n) => TOOLS[n].kind === 'reads'), ['calculator', 'lookup', 'notepad']);
  // The picture's lines fit at phone width.
  for (const row of t) assert.ok(S.rowLine(row).length <= S.LINE_MAX, S.rowLine(row));
  assert.equal(S.rowLine(by['tell-zoe']), 'tell-zoe: lookup; send after a yes');
  assert.equal(S.rowLine(by['eli-shop']), 'eli-shop: calculator, lookup');
});

test('approve and decline stories', () => {
  // Default: the run stops before send, waiting for a yes. The call that writes the ask is step 2.
  const wait = S.sendRun(undefined, true);
  assert.equal(wait.stop, 'waiting for approval');
  assert.equal(wait.steps, 2);
  assert.equal(wait.answer, null);
  assert.deepEqual(wait.state.sent, []);
  assert.equal(calls(wait)[0].text, 'lookup("pears")');
  assert.equal(calls(wait)[0].step, 1);
  assert.equal(plans(wait)[1].text, 'The message is ready; ask before using send.');
  assert.equal(plans(wait)[1].rule, 'R5');
  const ask = wait.messages.find((m) => m.role === 'approval');
  assert.equal(ask.step, 2);
  assert.equal(ask.text, `Waiting for a yes before ${callText('send', { to: 'zoe', text: lookupText('pears') })}`);
  assert.equal(ask.tool, 'send');
  // Approve: send runs once, then step 3 answers. One message sent, holding $1.10.
  const yes = S.sendRun(true, true);
  assert.equal(yes.stop, 'done');
  assert.equal(yes.steps, 3);
  assert.equal(yes.state.sent.length, 1);
  assert.deepEqual(yes.state.sent[0], { to: 'zoe', text: lookupText('pears') });
  assert.ok(yes.state.sent[0].text.includes(fmtMoney(PRICES.pears)));
  assert.equal(fmtMoney(PRICES.pears), '$1.10');
  assert.equal(calls(yes).filter((m) => m.tool === 'send').length, 1);
  assert.equal(calls(yes).find((m) => m.tool === 'send').step, 2); // the send runs within step 2, no new model call
  assert.deepEqual(yes.messages.filter((m) => m.role === 'reply').map((m) => m.text), ['Yes.']);
  // Decline: nothing sent, stopped as declined after 2 steps.
  const no = S.sendRun(false, true);
  assert.equal(no.stop, 'declined');
  assert.equal(no.steps, 2);
  assert.equal(no.answer, null);
  assert.deepEqual(no.state.sent, []);
  assert.equal(calls(no).filter((m) => m.tool === 'send').length, 0);
  assert.deepEqual(no.messages.filter((m) => m.role === 'reply').map((m) => m.text), ['No.']);
  // Ask before changes off: it sends without asking.
  const off = S.sendRun(undefined, false);
  assert.equal(off.stop, 'done');
  assert.equal(off.steps, 3);
  assert.equal(off.state.sent.length, 1);
  assert.equal(off.messages.filter((m) => m.role === 'approval').length, 0);
  assert.equal(off.answer, yes.answer);
  // The sent list as the picture shows it.
  assert.deepEqual(S.sentLines(wait), ['sent: nothing']);
  assert.deepEqual(S.sentLines(no), ['sent: nothing']);
  assert.deepEqual(S.sentLines(yes), ['sent: 1 message', 'to zoe: "pears cost $1.10 each."']);
  for (const r of [wait, yes, no, off]) for (const l of S.sentLines(r)) assert.ok(l.length <= S.LINE_MAX, l);
});

test('every outcome line in the pictures fits at phone width', () => {
  const runs = [...S.PAGE_RUNS.map((p) => p.run()), ...Object.keys(TASKS).map((id) => runAgent(id))];
  for (const r of runs) {
    for (const prefix of ['', 'this run: ']) {
      const lines = S.outcomeLines(r, prefix);
      for (const l of lines) assert.ok(l.length <= S.LINE_MAX, l);
      // Joined back, the lines say the same as outcomeText.
      assert.equal(lines.join(' '), prefix + S.outcomeText(r));
    }
  }
  assert.deepEqual(S.outcomeLines(S.sendRun(true, true)), ['done after 3 steps:', 'Told zoe: pears cost $1.10 each.']);
});

test('the summary comes from the stop reasons the page produced', () => {
  const stops = [...new Set(S.PAGE_RUNS.map((p) => p.run().stop))].sort();
  assert.deepEqual(stops, ['declined', 'done', 'repeated call', 'tool error', 'waiting for approval']);
  assert.deepEqual(S.pageStops(), stops);
  const g = S.safeguards();
  assert.deepEqual(g.map((x) => x.id), ['say-so', 'repeat', 'tool-error', 'ask']);
  // Each safeguard's stops were all produced on the page by the runs it names.
  for (const x of g) {
    assert.ok(x.runs.length >= 1, x.id);
    for (const r of x.runs) assert.ok(stops.includes(r.stop), x.id);
  }
  const by = Object.fromEntries(g.map((x) => [x.id, x]));
  assert.deepEqual(by['say-so'].runs.map((r) => r.answer), ['I could not find kai.']);
  assert.deepEqual(by.repeat.runs.map((r) => r.stop), ['repeated call']);
  assert.deepEqual(by['tool-error'].runs.map((r) => r.stop), ['tool error']);
  assert.deepEqual(by.ask.runs.map((r) => r.stop), ['waiting for approval', 'declined']);
  assert.ok(by.ask.runs.every((r) => r.state.sent.length === 0));
  // Every stop reason except done belongs to exactly one safeguard.
  for (const s of stops.filter((x) => x !== 'done')) assert.equal(g.filter((x) => x.runs.some((r) => r.stop === s)).length, 1, s);
  for (const x of g) for (const l of x.lines) assert.ok(l.length <= S.LINE_MAX, l);
  assert.deepEqual(by.repeat.lines, ['repeat check', 'stopped: repeated call after 2 steps']);
  assert.deepEqual(by['tool-error'].lines, ['stop on a tool error', 'stopped: tool error after 1 step']);
  assert.deepEqual(by.ask.lines, ['ask before changes', 'stopped: waiting for approval, nothing sent', 'after a no: declined, nothing sent']);
  assert.deepEqual(by['say-so'].lines, ['the instruction "say so"', 'done: "I could not find kai." (no guess)']);
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const say = runAgent('kai-food');
  const again = runAgent('kai-food', { onNotFound: 'try again' });
  const guess = runAgent('kai-food', { onNotFound: 'guess' });
  const sum = runAgent('sum');
  const wait = runAgent('tell-zoe');
  const yes = runAgent('tell-zoe', {}, { approve: true });
  const no = runAgent('tell-zoe', {}, { approve: false });
  const off = runAgent('tell-zoe', { askBeforeChanges: false });
  const zoeAsk = callText('send', { to: 'zoe', text: lookupText('pears') });
  const expect = {
    'kai-task': TASKS['kai-food'].text,
    'kai-call': callText('lookup', { key: 'kai' }),
    'kai-result': lookupText('kai'),
    'sayso-answer': say.answer,
    'sayso-steps': String(say.steps),
    'again-steps': String(again.steps),
    'again-runs': String(results(again).length),
    'guess-answer': guess.answer,
    'first-food': WORLD.foods[0],
    'people-count': String(WORLD.people.length),
    'sum-task': TASKS.sum.text,
    'bad-call': callText('calculator', { expr: '2+' }),
    'bad-result': TOOLS.calculator.run({ expr: '2+' }).text,
    'good-call': calls(sum)[0].text,
    'good-result': results(sum)[0].text,
    'task-count': String(Object.keys(TASKS).length),
    'tool-count': String(TOOL_NAMES.length),
    'unused-tools': 'notepad',
    'send-kind': TOOLS.send.kind,
    'zoe-task': TASKS['tell-zoe'].text,
    'zoe-ask': zoeAsk,
    'wait-steps': String(wait.steps),
    'yes-steps': String(yes.steps),
    'no-steps': String(no.steps),
    'off-steps': String(off.steps),
    'yes-answer': yes.answer,
    'yes-sent': String(yes.state.sent.length),
    'pears-price': fmtMoney(PRICES.pears),
    'stop-count': String(S.pageStops().length),
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
  };
  assert.deepEqual(v, expect);
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 30, `${spans.length} spans`);
  const unesc = (t) => t.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&#39;/g, "'");
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown key ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(spans.some(([, k]) => k === key), `unused key ${key}`);
});

test('the page has the lesson parts and no free-text input', () => {
  assert.match(html, /<title>Mistakes and limits · ML Explorer<\/title>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /href="\/ml\/agent-memory\/"/);
  assert.match(html, /href="\/ml\/build-an-agent\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.equal((html.match(/<li class="step"/g) || []).length, S.STEP_COUNT);
  assert.equal(S.STEP_COUNT, 6);
  // The stand-in honesty sentence in step 1.
  const step1 = html.split('id="step-1"')[1].split('id="step-2"')[0];
  assert.match(step1, /stand-in/);
  assert.match(step1, /fixed rules/);
  assert.match(step1, /language model, which can write a wrong or malformed call/);
  // The tool-error case is labelled as written in.
  const step3 = html.split('id="step-3"')[1].split('id="step-4"')[0];
  assert.match(step3, /written-in example/i);
  assert.ok(!/<input[^>]*type="(text|search|number)"/.test(html + src));
  assert.ok(!/<textarea/.test(html));
  assert.ok(!/innerHTML|eval\(|new Function/.test(src));
  assert.ok(!/\son[a-z]+=/.test(html));
  // Every quiz option has an explanation.
  const forms = [...html.matchAll(/<form class="q" id="(q\d)"[^>]*>([\s\S]*?)<\/form>/g)];
  assert.equal(forms.length, 2);
  for (const [, id, body] of forms) {
    const opts = [...body.matchAll(/type="radio" name="q\d" value="([^"]+)"/g)].map((m) => m[1]);
    assert.ok(opts.length >= 3, id);
    for (const o of opts) assert.match(body, new RegExp(`class="why" data-for="${o}"`), `${id} ${o}`);
  }
  const text = html.replace(/<[^>]+>/g, ' ').replace(/Intelligent Agents/g, '');
  for (const w of ['thinks', 'knows', 'understands', 'wants', 'decides', 'autonomous', 'intelligent', 'remembers', 'reasons', 'creative', 'conscious']) {
    assert.ok(!new RegExp(`\\b${w}\\b`, 'i').test(text), w);
  }
  // "hallucination" at most once, in quotes.
  const h = text.match(/hallucination/gi) || [];
  assert.ok(h.length <= 1);
  if (h.length) assert.match(text, /"hallucination"/);
});
