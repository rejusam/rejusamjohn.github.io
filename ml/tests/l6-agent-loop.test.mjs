import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG, PRICES, TASKS, runAgent, chainSuccess, stepsForTarget, calc, fmtMoney, fmtNumber, callText } from '../assets/agent.js';
import { WORLD } from '../assets/world.js';
import * as S from '../agent-loop/stage.js';

const html = readFileSync(new URL('../agent-loop/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../agent-loop/stage.js', import.meta.url), 'utf8');

const TRY = { onNotFound: 'try again' };
const calls = (r) => r.messages.filter((m) => m.role === 'call');
const results = (r) => r.messages.filter((m) => m.role === 'result');
// Rounded to two decimals, half up, as the page prints a chance.
const two = (x) => (Math.round(x * 100) / 100).toFixed(2);

test('the lesson runs match runAgent, and the eli task costs $5.00', () => {
  const r = runAgent('eli-shop');
  assert.equal(r.stop, 'done');
  assert.equal(r.steps, 4);
  assert.equal(r.answer, `That costs ${fmtMoney(calc(`${TASKS['eli-shop'].goal.qty}*${PRICES[WORLD.likes.eli]}`).value)}.`);
  assert.equal(r.answer, 'That costs $5.00.'); // the one literal
  // The stage's traces are runAgent's.
  assert.deepEqual(S.stepRun(0), runAgent('eli-shop'));
  assert.deepEqual(S.stepRun(1), runAgent('eli-shop'));
  for (let L = 1; L <= 10; L++) assert.deepEqual(S.limitRun(L), runAgent('eli-shop', { stepLimit: L }));
  assert.deepEqual(S.loopRun(true), runAgent('kai-food', { ...TRY, repeatCheck: true }));
  assert.deepEqual(S.loopRun(false), runAgent('kai-food', { ...TRY, repeatCheck: false }));
  for (const c of S.COST_RUNS) assert.deepEqual(c.run(), runAgent(c.task, c.config));
});

test('the loop has three ways out, and the prose facts about the traces hold', () => {
  // The stop reasons of every run on the page: exactly done, step limit and repeated call.
  const runs = [S.stepRun(), ...Array.from({ length: 10 }, (_, i) => S.limitRun(i + 1)), S.loopRun(true), S.loopRun(false), ...S.COST_RUNS.map((c) => c.run())];
  assert.deepEqual([...new Set(runs.map((r) => r.stop))].sort(), ['done', 'repeated call', 'step limit']);
  // Each step writes at most one call, and the answer step writes none.
  for (const r of runs) {
    for (let s = 1; s <= r.steps; s++) assert.ok(r.messages.filter((m) => m.role === 'call' && m.step === s).length <= 1);
    const a = r.messages.find((m) => m.role === 'answer');
    if (a) assert.equal(r.messages.filter((m) => m.role === 'call' && m.step === a.step).length, 0);
  }
  // eli: the first three steps each call a tool; the last step answers.
  const eli = S.stepRun();
  assert.deepEqual(calls(eli).map((m) => m.step), [1, 2, 3]);
  assert.equal(eli.messages.find((m) => m.role === 'answer').step, eli.steps);
  // kai is not in the invented world.
  assert.ok(!WORLD.people.includes('kai'));
  // kai: step 1 writes the lookup, step 2 writes the same lookup again.
  assert.deepEqual(calls(S.loopRun(true)).map((m) => m.step), [1, 2]);
});

test('model calls = min(n, L) for the runs the repeat check does not stop', () => {
  const n = S.stepsNeeded();
  for (let L = 1; L <= 10; L++) assert.equal(S.costOf(S.limitRun(L)).calls, S.callsFor(n, L));
  assert.equal(S.costOf(S.stepRun()).calls, S.callsFor(n, DEFAULT_CONFIG.stepLimit));
  // kai with the check off never answers: n is unbounded, so it makes L calls.
  assert.equal(S.costOf(S.loopRun(false)).calls, S.callsFor(Infinity, DEFAULT_CONFIG.stepLimit));
  for (const c of S.COST_RUNS) {
    const r = c.run();
    if (r.stop === 'repeated call') { assert.equal(c.id, 'kai-on'); assert.equal(S.costOf(r).calls, 2); continue; }
    const need = r.stop === 'done' || c.task === 'eli-shop' ? n : Infinity;
    assert.equal(S.costOf(r).calls, S.callsFor(need, c.config.stepLimit ?? DEFAULT_CONFIG.stepLimit));
  }
  // Other ways out, named in step 1 and shown in a later lesson: waiting for a yes, and a tool error.
  assert.equal(runAgent('tell-zoe').stop, 'waiting for approval');
  const failing = { calculator: { run: () => ({ ok: false, text: 'error: bad input' }) } };
  assert.equal(runAgent('sum', {}, { tools: failing }).stop, 'tool error');
});

test('step limit story', () => {
  const needed = runAgent('eli-shop').steps;
  assert.equal(S.stepsNeeded(), needed);
  assert.equal(needed, 4);
  assert.equal(S.LIMIT_MIN, 1);
  assert.equal(S.LIMIT_MAX, 10);
  for (let L = 1; L <= 10; L++) {
    const r = S.limitRun(L);
    if (L < needed) {
      assert.equal(r.stop, 'step limit', `limit ${L}`);
      assert.equal(r.answer, null);
      assert.equal(r.steps, L);
      // Every step it had called a tool, and every call ran.
      assert.equal(calls(r).length, L);
      assert.equal(results(r).length, L);
      assert.ok(r.messages.every((m) => m.role !== 'answer'));
    } else {
      assert.equal(r.stop, 'done', `limit ${L}`);
      assert.equal(r.answer, 'That costs $5.00.');
      assert.equal(r.steps, needed);
    }
  }
  // 3 -> no answer, 4 -> done
  assert.equal(S.limitRun(3).stop, 'step limit');
  assert.equal(S.limitRun(3).answer, null);
  assert.equal(S.limitRun(4).stop, 'done');
  // The read-out under the picture.
  assert.equal(S.limitLine(S.limitRun(3), 3), 'step limit 3: stopped by the step limit after 3 steps, no answer');
  assert.equal(S.limitLine(S.limitRun(4), 4), 'step limit 4: done after 4 steps: That costs $5.00.');
  assert.equal(S.limitLine(S.limitRun(10), 10), 'step limit 10: done after 4 steps: That costs $5.00.');
  assert.equal(S.limitLine(S.limitRun(1), 1), 'step limit 1: stopped by the step limit after 1 step, no answer');
  // The run with the limit just short stops after its third tool result, before the answer.
  const short = S.limitRun(needed - 1);
  assert.equal(short.messages.at(-2).role, 'result');
  assert.equal(short.messages.at(-2).text, String(calc(`${TASKS['eli-shop'].goal.qty}*${PRICES[WORLD.likes.eli]}`).value));
  assert.equal(calls(short).at(-1).text, callText('calculator', { expr: '4*1.25' }));
});

test('repeat check story', () => {
  const on = S.loopRun(true);
  assert.equal(on.stop, 'repeated call');
  assert.equal(on.steps, 2);
  assert.equal(on.answer, null);
  // Two identical calls; the second is written but not run.
  assert.deepEqual(calls(on).map((m) => m.text), ['lookup("kai")', 'lookup("kai")']);
  assert.equal(results(on).length, 1);
  assert.equal(results(on)[0].text, 'not found: kai');
  assert.equal(on.messages.at(-2).role, 'call');
  assert.deepEqual(on.messages.filter((m) => m.role === 'plan').map((m) => m.rule), ['R3', 'R1']);
  // Off: it stops only at the step limit, whatever the limit.
  const off = S.loopRun(false);
  assert.equal(off.stop, 'step limit');
  assert.equal(off.steps, DEFAULT_CONFIG.stepLimit);
  assert.equal(off.steps, 8);
  assert.equal(calls(off).length, off.steps);
  assert.equal(results(off).length, off.steps);
  assert.ok(calls(off).every((m) => m.text === 'lookup("kai")'));
  assert.ok(results(off).every((m) => m.text === 'not found: kai'));
  for (let L = 1; L <= 10; L++) {
    const r = runAgent('kai-food', { ...TRY, repeatCheck: false, stepLimit: L });
    assert.equal(r.stop, 'step limit');
    assert.equal(r.steps, L);
  }
  // The check compares a call with the one just before it. A run that alternates between
  // two calls is not caught: the memory lesson's pair task at a small budget.
  const alt = runAgent('pair-shop', { budget: 12, stepLimit: 10, repeatCheck: true });
  const texts = calls(alt).map((m) => m.text);
  assert.equal(alt.stop, 'step limit');
  assert.equal(new Set(texts).size, 2);
  for (let i = 1; i < texts.length; i++) assert.notEqual(texts[i], texts[i - 1]);
  for (let i = 2; i < texts.length; i++) assert.equal(texts[i], texts[i - 2]);
  assert.equal(S.ALT_RUN_ID, 'pair-shop');
});

test('chart values equal chainSuccess', () => {
  assert.equal(S.P_MIN, 80);
  assert.equal(S.P_MAX, 99);
  assert.equal(S.K_MAX, 20);
  for (let n = S.P_MIN; n <= S.P_MAX; n++) {
    const p = S.pOf(n);
    assert.equal(p, n / 100);
    assert.equal(p, Number(`0.${n}`)); // no float drift in the slider value
    const pts = S.chartPoints(p);
    assert.equal(pts.length, 20);
    pts.forEach(([k, y], i) => {
      assert.equal(k, i + 1);
      assert.equal(y, chainSuccess(p, k));
      let prod = 1;
      for (let j = 0; j < k; j++) prod *= p;
      assert.ok(Math.abs(y - prod) < 1e-12);
    });
    // The log form agrees with the direct product.
    for (const t of [0.5, 0.9, 0.25]) {
      const k = stepsForTarget(p, t);
      assert.equal(k, Math.floor(S.logBound(p, t) + 1e-12));
      assert.ok(p ** k >= t);
      assert.ok(p ** (k + 1) < t);
    }
  }
  assert.equal(S.chanceText(0.5987369392383787), '0.60');
  assert.equal(S.chanceText(0.3486784401000001), '0.35');
  assert.equal(S.chanceText(1), '1.00');
  assert.equal(S.chanceText(0.005), '0.01');
});

test('quoted 0.60 and 0.35 match', () => {
  const v = S.pageValues();
  assert.equal(v['p-example'], '0.95');
  assert.equal(v['k-example'], '10');
  assert.equal(v['chance-example'], two(0.95 ** 10));
  assert.equal(v['chance-example'], '0.60');
  assert.equal(v['q-p'], '0.9');
  assert.equal(v['q-k'], '10');
  assert.equal(v['q-chance'], two(0.9 ** 10));
  assert.equal(v['q-chance'], '0.35');
  assert.equal(v['q-sum'], fmtNumber(0.9 * 10));
  assert.equal(v['q-fail-one'], fmtNumber(1 - 0.9));
  assert.equal(S.chartReadout(0.95), 'p = 0.95: all 10 steps succeed with chance 0.60');
  assert.equal(S.chartReadout(0.8), 'p = 0.8: all 10 steps succeed with chance 0.11');
  assert.equal(two(0.8 ** 10), '0.11');
  assert.equal(S.targetReadout(0.95), `chance at least 0.5 for up to ${stepsForTarget(0.95, 0.5)} steps`);
  for (let n = S.P_MIN; n <= S.P_MAX; n++) {
    const p = S.pOf(n);
    const k = stepsForTarget(p, 0.5);
    assert.equal(S.targetReadout(p).endsWith(' (past the right edge)'), k > S.K_MAX, `p ${p}`);
    assert.equal(k > S.K_MAX, n >= 97, `p ${p}`);
  }
  assert.equal(S.targetReadout(0.97), `chance at least 0.5 for up to ${stepsForTarget(0.97, 0.5)} steps (past the right edge)`);
  assert.equal(S.targetReadout(0.8), `chance at least 0.5 for up to ${Math.floor(Math.log(0.5) / Math.log(0.8))} steps`);
});

test('model calls and tool runs for each run on the page', () => {
  const rows = S.COST_RUNS.map((c) => {
    const r = c.run();
    const cost = S.costOf(r);
    // A step is one model call; a tool run is a result row.
    assert.equal(cost.calls, r.messages.filter((m) => m.role === 'plan').length);
    assert.equal(cost.calls, r.steps);
    assert.equal(cost.toolRuns, results(r).length);
    return [c.id, cost.calls, cost.toolRuns, r.stop].join(' ');
  });
  assert.deepEqual(rows, [
    'eli-8 4 3 done',
    'eli-3 3 3 step limit',
    'kai-on 2 1 repeated call',
    'kai-off 8 8 step limit',
  ]);
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const eli = runAgent('eli-shop');
  assert.equal(v['eli-task'], TASKS['eli-shop'].text);
  assert.equal(v['eli-steps'], String(eli.steps));
  assert.equal(v['eli-answer'], eli.answer);
  assert.equal(v['eli-tool-runs'], String(results(eli).length));
  assert.equal(v['default-limit'], String(DEFAULT_CONFIG.stepLimit));
  assert.equal(v['short-limit'], String(eli.steps - 1));
  assert.equal(v['short-result'], results(runAgent('eli-shop', { stepLimit: eli.steps - 1 })).at(-1).text);
  assert.equal(v['kai-task'], TASKS['kai-food'].text);
  assert.equal(v['kai-call'], 'lookup("kai")');
  assert.equal(v['kai-result'], 'not found: kai');
  const on = runAgent('kai-food', { ...TRY });
  const off = runAgent('kai-food', { ...TRY, repeatCheck: false });
  assert.equal(v['kai-on-steps'], String(on.steps));
  assert.equal(v['kai-on-runs'], String(results(on).length));
  assert.equal(v['kai-off-steps'], String(off.steps));
  assert.equal(v['kai-off-runs'], String(results(off).length));
  assert.equal(v['kai-on-rules'], 'R3 and R1');
  assert.equal(v['target'], '0.5');
  assert.equal(v['target-steps'], String(stepsForTarget(0.95, 0.5)));
  assert.equal(v['target-steps'], '13');
  assert.equal(v['log-bound'], fmtNumber(Math.log(0.5) / Math.log(0.95)));
  assert.equal(v['log-bound'], '13.51');
  assert.equal(v['chance-at-target'], two(0.95 ** 13));
  assert.equal(v['chance-past-target'], two(0.95 ** 14));
  assert.equal(v['target-next'], '14');
  assert.equal(v['eli-chance'], two(0.95 ** eli.steps));
  assert.equal(v['eli-chance'], '0.81');
  for (const c of S.COST_RUNS) {
    const r = c.run();
    assert.equal(v[`cost-${c.id}`], `${c.label}: ${S.costText(S.costOf(r))}, ${r.stop === 'done' ? 'answered' : `stopped by ${r.stop === 'step limit' ? 'the step limit' : 'the repeat check'}`}`);
  }
  assert.equal(v['cost-eli-8'], 'the eli task, step limit 8: 4 model calls, 3 tool runs, answered');
  assert.equal(v['cost-kai-on'], 'the kai task, repeat check on: 2 model calls, 1 tool run, stopped by the repeat check');
  assert.equal(v['cost-kai-off'], 'the kai task, repeat check off: 8 model calls, 8 tool runs, stopped by the step limit');
  assert.equal(v['extra-calls'], String(off.steps - on.steps));
  assert.equal(v['extra-calls'], '6');
  // Every span in the page holds exactly the computed value, and every value is used.
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 25);
  const unesc = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown value ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(html.includes(`data-value="${key}"`), `unused value ${key}`);
});

test('lesson page follows the contract', () => {
  assert.equal((html.match(/<li class="step"/g) || []).length, 6);
  assert.match(html, /<title>The loop and when to stop · ML Explorer<\/title>/);
  assert.match(html, /<h1>The loop and when to stop<\/h1>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /data-lesson="agent-loop"/);
  assert.match(html, /<link rel="canonical" href="https:\/\/rejusamjohn\.pages\.dev\/ml\/agent-loop\/">/);
  assert.match(html, /<meta property="og:image" content="https:\/\/rejusamjohn\.pages\.dev\/ml\/og\.png">/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/agent-memory\/">Next lesson: Memory →<\/a>/);
  assert.match(html, /href="\/ml\/agent-tools\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.match(html, /<script type="module" src="https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js" data-cf-beacon='\{"token": "e46c62f910fe4def882feaed064efced"\}'><\/script>/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)(?![^>]*application\/ld\+json)[^>]*>/);
  assert.doesNotMatch(html, /\son[a-z]+=/i);
  assert.equal((html.match(/<form class="q"/g) || []).length, 2);
  assert.match(html, /What does a step limit protect against\?/);
  const step1 = html.slice(html.indexOf('id="step-1"'), html.indexOf('id="step-2"'));
  assert.match(step1, /fixed rules/);
  assert.match(step1, /a real agent makes them with a language model, which can write a wrong or malformed call/);
  // The independence assumption is stated in the chart's step.
  const step5 = html.slice(html.indexOf('id="step-5"'), html.indexOf('id="step-6"'));
  assert.match(step5, /independent/);
  const step4 = html.slice(html.indexOf('id="step-4"'), html.indexOf('id="step-5"'));
  assert.match(step4, /just before it/);
  assert.match(step4, /alternates between two calls/);
  assert.match(html, /Yao et al\., "ReAct: Synergizing Reasoning and Acting in Language Models", arXiv 2210\.03629 \(2022\)/);
  assert.match(html, /class="katex"/);
});

test('wording rules', () => {
  const body = html.slice(html.indexOf('<main'), html.indexOf('<ol class="reading">'));
  assert.doesNotMatch(body, /\b(thinks?|knows?|understands?|wants?|decides|autonomous|reasons|creative|conscious)\b|intelligen/i);
  assert.doesNotMatch(src, /\b(thinks?|knows?|understands?|wants?|decides|autonomous|intelligent)\b/i);
  assert.doesNotMatch(html + src, /GPT|Gemini|Llama|LangChain|Mistral|Copilot|Hugging/i);
  assert.doesNotMatch(html, /delve|in today's world|unlock|harness the power|it's important to note|dive in/i);
  assert.doesNotMatch(src, /innerHTML|\beval\(|new Function/);
});
