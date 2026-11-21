// Small display helpers for the early-warning page.

// The row and its position (0 = highest risk) for a student id, or null if the id is not in
// this week's cohort for the module.
export function findStudent(file, id) {
  const text = String(id).trim();
  if (!/^\d+$/.test(text)) return null;
  const position = file.rows.findIndex(row => row[0] === Number(text));
  return position === -1 ? null : {row: file.rows[position], position};
}

// Risks come back rounded to 4 decimals, so equal risks differ by less than half of 0.0001.
export const compareRisk = (exported, live) => (Math.abs(exported - live) < 0.00005 ? 'same' : 'different');

export const formatRisk = risk => risk.toFixed(2);

export function curlSnippet(url, body) {
  return `curl -s -X POST ${url} \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body)}'`;
}

// JSON with Python's None, True and False, so the body can be pasted into Python as it is.
function pythonLiteral(value) {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (Array.isArray(value)) return `[${value.map(pythonLiteral).join(', ')}]`;
  if (typeof value === 'object') {
    return `{${Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${pythonLiteral(v)}`).join(', ')}}`;
  }
  return JSON.stringify(value);
}

export function pythonSnippet(url, body) {
  return [
    'import requests',
    '',
    `url = "${url}"`,
    `body = ${pythonLiteral(body)}`,
    'response = requests.post(url, json=body, timeout=30)',
    'print(response.status_code, response.json())',
  ].join('\n');
}

// Message for a live call that did not give a score. The proxy's own errors carry an "error"
// field; a 429 or 503 without one came from AWS, passed through by the proxy.
export function failMessage(status, body) {
  const own = body && typeof body.error === 'string' ? body.error : null;
  if ((status === 429 || status === 503) && own) return own;
  if (status === 429 || status === 503) return `The scoring API on AWS is busy (status ${status}). Try again in a minute.`;
  const detail = own ?? (status ? `status ${status}` : 'no answer');
  return `The live call failed (${detail}). Showing the exported risk.`;
}
