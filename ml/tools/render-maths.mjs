// Pre-renders maths in ML Explorer pages (author-time tool, not shipped logic).
//
// Usage:
//   NODE_PATH=<scratchpad>/katex/node_modules node ml/tools/render-maths.mjs <file.html…>
//   Fallback: KATEX_PATH=<scratchpad>/katex/node_modules (same folder).
//
// Author maths as <span class="tex" data-tex="\frac{a}{b}"></span>, add
// data-display for display maths. The span's content is replaced with KaTeX
// HTML+MathML on every run (idempotent). Exits 1 on any KaTeX error.
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
let katex;
try {
  katex = require('katex');
} catch (e) {
  if (!process.env.KATEX_PATH) {
    console.error('Cannot load katex. Set NODE_PATH (or KATEX_PATH) to the node_modules folder that contains katex.');
    process.exit(2);
  }
  katex = createRequire(process.env.KATEX_PATH.replace(/\/?$/, '/'))('katex');
}

const decode = (s) =>
  s.replace(/&(amp|lt|gt|quot|#39|#x27);/g, (_, n) =>
    ({ amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'", '#x27': "'" })[n]);

// Opening tag of a span whose class list includes "tex".
const OPEN = /<span\b((?:[^>"']|"[^"]*"|'[^']*')*)>/g;
const attr = (attrs, name) => {
  const m = new RegExp(`(?:^|\\s)${name}(?:\\s*=\\s*(?:"([^"]*)"|'([^']*)'))?(?=\\s|/|$)`).exec(attrs);
  return m ? (m[1] ?? m[2] ?? '') : null;
};

function render(html, file) {
  let out = '';
  let pos = 0;
  let count = 0;
  OPEN.lastIndex = 0;
  let m;
  while ((m = OPEN.exec(html))) {
    const cls = attr(m[1], 'class');
    if (cls === null || !cls.split(/\s+/).includes('tex')) continue;
    const tex = attr(m[1], 'data-tex');
    if (tex === null) throw new Error(`${file}: span.tex without data-tex at offset ${m.index}`);
    const display = attr(m[1], 'data-display') !== null;
    const bodyStart = m.index + m[0].length;
    // Authored maths spans never nest elements that are spans, but the
    // rendered output does; scan with a depth counter to find our close tag.
    const tagRe = /<(\/?)span\b[^>]*>/g;
    tagRe.lastIndex = bodyStart;
    let depth = 1, t, bodyEnd = -1;
    while ((t = tagRe.exec(html))) {
      depth += t[1] ? -1 : 1;
      if (depth === 0) { bodyEnd = t.index; break; }
    }
    if (bodyEnd < 0) throw new Error(`${file}: unclosed span.tex at offset ${m.index}`);
    let rendered;
    try {
      rendered = katex.renderToString(decode(tex), {
        displayMode: display,
        output: 'htmlAndMathml',
        throwOnError: true,
        strict: 'warn',
        trust: false,
      });
    } catch (e) {
      throw new Error(`${file}: ${e.message}\n  in data-tex="${tex}"`);
    }
    out += html.slice(pos, bodyStart) + rendered;
    pos = bodyEnd;
    OPEN.lastIndex = bodyEnd;
    count++;
  }
  return { html: out + html.slice(pos), count };
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error('Usage: node render-maths.mjs <file.html…>');
  process.exit(2);
}
let failed = false;
for (const f of files) {
  try {
    const { html, count } = render(readFileSync(f, 'utf8'), f);
    writeFileSync(f, html);
    console.log(`${f}: ${count} expression${count === 1 ? '' : 's'} rendered`);
  } catch (e) {
    console.error(e.message);
    failed = true;
  }
}
process.exit(failed ? 1 : 0);
