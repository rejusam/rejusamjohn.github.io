// Level 6 take-home program: writes the page's agent as one small Python file
// (3.8 or later, standard library only). Pure: the text depends only on the config.
// The program holds the same rules as the stand-in in agent.js, written for the three
// build tasks only (two shopping sums and one message to send).
import {WORLD} from './world.js';
import {PRICES, TASKS, DEFAULT_CONFIG, runAgent} from './agent.js';

export const BUILD_TASKS = ['eli-shop', 'pair-shop', 'tell-zoe'];

// Row labels, the same words the trace view shows next to each message.
export const ROLE_LABELS = {task: 'Task', plan: 'Plan', call: 'Tool call', result: 'Result',
  answer: 'Answer', stop: 'Stopped', note: 'Note', approval: 'Waiting for your yes'};

export const ASK_PROMPT = 'Send this? y/n ';

function merge(config) {
  const c = config || {};
  return {...DEFAULT_CONFIG, ...c, tools: {...DEFAULT_CONFIG.tools, ...(c.tools || {})}};
}

// One printed line per message. With typed ('y' or 'n'), the terminal's question and the
// typed answer come just before the Yes./No. row, as they look on a terminal.
export function traceLines(result, typed) {
  const out = [];
  for (const m of result.messages) {
    if (typed && m.role === 'approval' && (m.text === 'Yes.' || m.text === 'No.')) out.push(ASK_PROMPT + typed);
    out.push(`${ROLE_LABELS[m.role]}: ${m.text}`);
  }
  return out;
}

// What the program prints for the three build tasks when the reader types `typed`.
export function expectedOutput(config, typed = 'y') {
  return BUILD_TASKS.map(id => traceLines(runAgent(id, config, {approve: typed === 'y'}), typed).join('\n'))
    .join('\n\n') + '\n';
}

// ---- Python literals ----
function py(v) {
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (v === null) return 'None';
  if (Array.isArray(v)) return '[' + v.map(py).join(', ') + ']';
  return '{' + Object.keys(v).map(k => `${py(k)}: ${py(v[k])}`).join(', ') + '}';
}

// NAME = {...} wrapped under 80 characters, continuation lines lined up after the brace.
function pyDict(name, obj) {
  const head = `${name} = {`;
  const pad = ' '.repeat(head.length);
  const items = Object.keys(obj).map(k => `${py(k)}: ${py(obj[k])}`);
  const lines = [];
  let cur = head;
  items.forEach((it, i) => {
    const piece = it + (i === items.length - 1 ? '}' : ',');
    const atStart = cur === head || cur === pad;
    if (!atStart && (cur + ' ' + piece).length > 79) { lines.push(cur); cur = pad + piece; }
    else cur += (atStart ? '' : ' ') + piece;
  });
  lines.push(cur);
  return lines;
}

const MARKED = [
  '# ---- To use a real language model --------------------------------',
  '# Replace call_model with a function that sends `messages` to a model',
  '# and turns its reply into a plan line and one of the same actions:',
  '#   plan, ("call", tool_name, tool_input)  or  plan, ("answer", text)',
  '# Keep any key out of this file: read it from an environment variable.',
  '# -------------------------------------------------------------------',
];

export function toPython(config) {
  const c = merge(config);
  const t = c.tools;
  const calcSums = t.calculator && c.sums === 'calculator';
  const guesses = !t.lookup || c.onNotFound === 'guess';
  const windowed = c.budget !== Infinity;
  const L = [];
  const add = (...xs) => L.push(...xs);
  const firstFood = WORLD.foods[0];
  const firstPrice = PRICES[Object.keys(PRICES)[0]];

  add('# my_agent.py: a small agent. It calls a model in a loop; each time the model writes',
    '# a plan line and chooses an action (call a tool, or answer), and the program runs the',
    '# tool and adds the result to the messages. The model here is a stand-in, the fixed',
    '# rules from the lesson page, so it runs with no model. People, foods, prices: invented.',
    '# Run it with: python3 my_agent.py   (Python 3.8 or later, standard library only)');
  if (t.calculator) add('import ast');
  add('import math');
  if (t.calculator) add('import operator');
  if (t.lookup || t.calculator) add('import re');
  add('');
  add('# Your settings from the page.',
    `STEP_LIMIT = ${c.stepLimit}  # most model calls in one run`);
  if (windowed) add(`BUDGET = ${c.budget}  # words of messages each model call is given; the task is always given`);
  add(`REPEAT_CHECK = ${py(c.repeatCheck)}  # stop if the same call comes twice in a row`);
  if (t.send) add(`ASK_BEFORE_CHANGES = ${py(c.askBeforeChanges)}  # ask on the terminal before send runs`);
  add('');
  if (t.lookup) {
    const lives = {};
    for (const p of WORLD.people) lives[p] = `${WORLD.prep[WORLD.livesIn[p]]} the ${WORLD.livesIn[p]}`;
    add('# The invented world: the food each person likes, where they live, and prices.',
      ...pyDict('LIKES', WORLD.likes), ...pyDict('LIVES', lives), ...pyDict('PRICES', PRICES));
  }
  if (guesses) {
    add('# A guess uses the first food in the food table or the first price in the price list.',
      `FIRST_FOOD = ${py(firstFood)}`, `FIRST_PRICE = ${py(firstPrice)}`);
  }
  add(...pyDict('LABELS', {task: 'Task', plan: 'Plan', call: 'Tool call', result: 'Result',
    answer: 'Answer', stop: 'Stopped', approval: 'Waiting for your yes'}));
  if (t.notepad) add('NOTES = []');
  if (t.send) add('SENT = []');
  add('',
    'def fmt_number(x):',
    '    # Round to cents, half away from zero, then drop trailing zeros: 2.40 -> 2.4.',
    '    r = math.floor(abs(x) * 100 + 0.5) / 100',
    '    s = ("%.2f" % r).rstrip("0").rstrip(".")',
    '    return "\\u2212" + s if x < 0 and r != 0 else s',
    '',
    'def fmt_money(x):',
    '    r = math.floor(abs(x) * 100 + 0.5) / 100',
    '    return ("\\u2212" if x < 0 and r != 0 else "") + "$%.2f" % r',
    '');
  if (!calcSums) {
    add('def round_one_figure(x):',
      '    # Round to one significant figure, half away from zero: 1.25 -> 1, 0.8 -> 0.8.',
      '    e = math.floor(math.log10(abs(x)))',
      '    if e < 0:',
      '        return math.floor(abs(x) * 10 ** -e + 0.5) / 10 ** -e',
      '    return math.floor(abs(x) / 10 ** e + 0.5) * 10 ** e',
      '');
  }
  add('# ---- Tools: each returns (ok, text) ----');
  if (t.calculator) {
    add('OPS = {ast.Add: operator.add, ast.Sub: operator.sub,',
      '       ast.Mult: operator.mul, ast.Div: operator.truediv}',
      '',
      'def work_out(node):',
      '    # Only numbers, minus signs and the four operations are allowed.',
      '    if isinstance(node, ast.Expression):',
      '        return work_out(node.body)',
      '    if isinstance(node, ast.Constant) and type(node.value) in (int, float):',
      '        return float(node.value)',
      '    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):',
      '        return -work_out(node.operand)',
      '    if isinstance(node, ast.BinOp) and type(node.op) in OPS:',
      '        return OPS[type(node.op)](work_out(node.left), work_out(node.right))',
      '    raise ValueError(node)',
      '',
      'def calculator(expr):',
      '    # Reads the sum as a tree of numbers and operations; nothing in the text is run.',
      '    text = expr.replace("\\u00d7", "*").replace("\\u00f7", "/")',
      '    try:',
      '        if len(text) > 60 or not re.fullmatch(r"(\\d+(\\.\\d+)?|[-+*/() ])+", text):',
      '            raise ValueError(text)',
      '        return True, fmt_number(work_out(ast.parse(text.strip(), mode="eval")))',
      '    except ZeroDivisionError:',
      '        return False, "division by zero"',
      '    except (ValueError, SyntaxError):',
      '        return False, \'cannot read "%s"\' % expr',
      '');
  }
  if (t.lookup) {
    add('def lookup(key):',
      '    # Looks up one person or one food in the invented world.',
      '    if key in LIKES:',
      '        return True, "%s likes %s and lives %s." % (key, LIKES[key], LIVES[key])',
      '    if key in PRICES:',
      '        return True, "%s cost %s each." % (key, fmt_money(PRICES[key]))',
      '    return False, "not found: " + key',
      '');
  }
  if (t.notepad) {
    add('def notepad(action, text=""):',
      '    # Writes a line to the notes, or reads all the notes back.',
      '    if action == "write":',
      '        NOTES.append(text)',
      '        return True, "noted"',
      '    if action == "read":',
      '        return True, "\\n".join(NOTES) or "notes are empty"',
      '    return False, \'cannot read action "%s"\' % action',
      '');
  }
  if (t.send) {
    add('def send(to, text):',
      '    # Changes things, so it can be asked about first. Here it only records the message.',
      '    SENT.append((to, text))',
      '    return True, "sent to " + to',
      '');
  }
  const names = ['calculator', 'lookup', 'notepad', 'send'].filter(n => t[n]);
  add('TOOLS = {' + names.map(n => `"${n}": ${n}`).join(', ') + '}',
    '',
    'def call_text(tool, tool_input):',
    '    return "%s(%s)" % (tool, ", ".join(\'"%s"\' % v for v in tool_input.values()))',
    '');
  if (windowed) {
    add('def window(messages):',
      '    # The task, then the newest messages whose words fit in BUDGET.',
      '    kept, words = [], 0',
      '    for m in reversed(messages[1:]):',
      '        words += len(m["text"].split())',
      '        if words > BUDGET:',
      '            break',
      '        kept.insert(0, m)',
      '    return [messages[0]] + kept',
      '');
  }
  if (t.lookup) {
    add('def read_facts(messages):',
      '    # Facts from the lookup results the model is given' + (t.notepad ? ', and from the notes.' : '.'),
      '    likes, price = {}, {}',
      '    texts = [m["text"] for m in messages if m["tool"] == "lookup" and m["role"] == "result"]',
      `    for text in texts${t.notepad ? ' + NOTES' : ''}:`,
      '        m = re.match(r"([a-z]+) likes ([a-z]+) ", text)',
      '        if m:',
      '            likes[m.group(1)] = m.group(2)',
      '        m = re.match(r"([a-z]+) cost \\$(\\d+\\.\\d\\d) each\\.$", text)',
      '        if m:',
      '            price[m.group(1)] = float(m.group(2))',
      '    return likes, price',
      '');
  }
  add('def sum_text(terms):',
    '    return "+".join(fmt_number(q) + "*" + fmt_number(p) for q, p in terms)',
    '',
    ...MARKED,
    'def call_model(messages):',
    '    # The stand-in: the rules from the page, first match wins. It returns a plan',
    '    # line and an action. It is given the task\'s goal, not its words.',
    '    goal, plan = messages[0]["goal"], []',
    (t.lookup ? '    likes, price = read_facts(messages)' : '    likes, price = {}, {}'),
    '    results = [m for m in messages if m["role"] == "result"]',
    '    last = results[-1] if results else {"tool": None, "text": ""}');
  if (t.lookup) {
    add('    if last["tool"] == "lookup" and last["text"].startswith("not found: "):',
      '        key = last["input"]["key"]  # rule 1: the lookup found nothing');
    if (c.onNotFound === 'say so') {
      add('        plan.append("Lookup found nothing for %s; say so." % key)',
        '        return " ".join(plan), ("answer", "I could not find %s." % key)');
    } else if (c.onNotFound === 'try again') {
      add('        plan.append("Lookup found nothing for %s; try the same lookup again." % key)',
        '        return " ".join(plan), ("call", "lookup", {"key": key})');
    } else {
      add('        if key in PRICES:',
        '            price[key] = FIRST_PRICE',
        '            plan.append("Guess (first price in the list): %s cost %s."',
        '                        % (key, fmt_money(FIRST_PRICE)))',
        '        else:',
        '            likes[key] = FIRST_FOOD',
        '            plan.append("Guess (first food in the table): %s likes %s." % (key, FIRST_FOOD))');
    }
    if (t.notepad) {
      add('    elif last["tool"] == "lookup" and last["text"] not in NOTES:  # rule 2: keep the fact',
        '        plan.append("Keep the fact about %s; use notepad." % last["input"]["key"])',
        '        return " ".join(plan), ("call", "notepad", {"action": "write", "text": last["text"]})');
    }
  }
  add('    # Rule 3: each person\'s food and its price (shopping), or one price (sending).',
    '    shop = goal["kind"] == "shop"',
    '    for person in goal["people"] if shop else [None]:');
  if (t.lookup) {
    add('        if person and person not in likes:',
      '            plan.append("Need the food %s likes; use lookup." % person)',
      '            return " ".join(plan), ("call", "lookup", {"key": person})',
      '        food = likes[person] if person else goal["food"]',
      '        if food not in price:',
      '            plan.append("Need the price of %s; use lookup." % food)',
      '            return " ".join(plan), ("call", "lookup", {"key": food})');
  } else {
    add('        if person and person not in likes:',
      '            likes[person] = FIRST_FOOD',
      '            plan.append("Guess (first food in the table): %s likes %s." % (person, FIRST_FOOD))',
      '        food = likes[person] if person else goal["food"]',
      '        if food not in price:',
      '            price[food] = FIRST_PRICE',
      '            plan.append("Guess (first price in the list): %s cost %s."',
      '                        % (food, fmt_money(FIRST_PRICE)))');
  }
  add('    if shop:  # rule 4: the sum',
    '        terms = [(goal["qty"], price[likes[p]]) for p in goal["people"]]');
  if (calcSums) {
    add('        expr = sum_text(terms)',
      '        done = [m for m in results if m["tool"] == "calculator" and m["input"]["expr"] == expr]',
      '        if not done:',
      '            plan.append("Need %s; use calculator." % expr)',
      '            return " ".join(plan), ("call", "calculator", {"expr": expr})',
      '        total = float(done[-1]["text"])');
  } else {
    add('        rounded = [(round_one_figure(q), round_one_figure(p)) for q, p in terms]',
      '        total = 0',
      '        for q, p in rounded:',
      '            total += q * p',
      '        plan.append("Guess (each number rounded to one significant figure): %s = %s."',
      '                    % (sum_text(rounded), fmt_number(total)))');
  }
  add('        plan.append("All facts are in; answer.")  # rule 6',
    '        return " ".join(plan), ("answer", "That costs %s." % fmt_money(total))');
  if (!t.send) {
    add('    plan.append("Send is off; say so.")  # rule 5',
      '    answer = "I cannot send messages, so %s was not told." % goal["to"]',
      '    return " ".join(plan), ("answer", answer)');
  } else {
    add('    text = "%s cost %s each." % (goal["food"], fmt_money(price[goal["food"]]))',
      '    if not [m for m in results if m["tool"] == "send"]:  # rule 5',
      `        plan.append("The message is ready; ${c.askBeforeChanges ? 'ask before using' : 'use'} send.")`,
      '        return " ".join(plan), ("call", "send", {"to": goal["to"], "text": text})',
      '    plan.append("All done; answer.")  # rule 6',
      '    return " ".join(plan), ("answer", "Told %s: %s" % (goal["to"], text))');
  }
  add('');
  if (t.send) {
    add('def yes():',
      `    # Ask on the terminal. With nothing to read (end of input), the answer is no.`,
      '    try:',
      `        return input("${ASK_PROMPT}").strip().lower() == "y"`,
      '    except EOFError:',
      '        return False',
      '');
  }
  add('def say(messages, role, text, tool=None, tool_input=None):',
    '    messages.append({"role": role, "text": text, "tool": tool, "input": tool_input})',
    '    print("%s: %s" % (LABELS[role], text))',
    '',
    'def loop(messages):',
    '    # Calls the model until the run stops, and returns why it stopped.');
  add(`    steps, last_call${t.send ? ', asked' : ''} = 0, None${t.send ? ', False' : ''}`,
    '    while steps < STEP_LIMIT:',
    '        steps += 1  # each model call is one step',
    `        plan, action = call_model(${windowed ? 'window(messages)' : 'messages'})`,
    '        say(messages, "plan", plan)',
    '        if action[0] == "answer":',
    '            say(messages, "answer", action[1])',
    '            return "done"',
    '        tool, tool_input = action[1], action[2]',
    '        call = call_text(tool, tool_input)');
  if (t.send) {
    add('        if tool == "send" and ASK_BEFORE_CHANGES:',
      '            say(messages, "approval", "Waiting for a yes before " + call)',
      '            if asked:  # a yes covers the first ask only',
      '                return "waiting for approval"',
      '            asked = True',
      '            if not yes():',
      '                say(messages, "approval", "No.")',
      '                return "declined"',
      '            say(messages, "approval", "Yes.")');
  }
  add('        say(messages, "call", call, tool, tool_input)',
    '        if REPEAT_CHECK and call == last_call:',
    '            return "repeated call"',
    '        last_call = call',
    '        ok, result = TOOLS[tool](**tool_input)',
    '        say(messages, "result", result, tool, tool_input)',
    '        if not ok and not result.startswith("not found: "):',
    '            return "tool error"',
    '    return "step limit"',
    '',
    'def run(task, goal):',
    '    messages = []');
  if (t.notepad) add('    NOTES.clear()');
  add('    say(messages, "task", task)',
    '    messages[0]["goal"] = goal  # the stand-in is given this; a real model is given the words',
    '    say(messages, "stop", loop(messages))',
    '',
    'TASKS = [');
  for (const id of BUILD_TASKS) add(`    (${py(TASKS[id].text)},`, `     ${py(TASKS[id].goal)}),`);
  add(']',
    '',
    'if __name__ == "__main__":',
    '    for i, (task, goal) in enumerate(TASKS):',
    '        if i:',
    '            print()',
    '        run(task, goal)');
  return L.join('\n') + '\n';
}
