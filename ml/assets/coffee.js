// The Ko-fi coffee panel, ported from the home page's script.js.
// Nothing loads from Ko-fi until the panel is first opened. Esc closes it and
// focus goes back to whichever button opened it.

const KOFI_URL = 'https://ko-fi.com/rejusamjohn';
const CUP_SRC = '/assets/kofi-cup.png';

function makeFloat(doc) {
  const a = doc.createElement('a');
  a.className = 'coffee-float';
  a.href = KOFI_URL;
  a.target = '_blank';
  a.rel = 'noopener';
  a.setAttribute('aria-haspopup', 'dialog');
  a.setAttribute('aria-expanded', 'false');

  const img = doc.createElement('img');
  img.src = CUP_SRC;
  img.alt = '';
  img.width = 26;
  img.height = 21;

  const label = doc.createElement('span');
  label.className = 'coffee-label';
  label.textContent = 'Buy me a coffee';

  a.append(img, label);
  doc.body.appendChild(a);
  return a;
}

export function initCoffee(doc = document) {
  const float = doc.querySelector('.coffee-float') || makeFloat(doc);
  const handle = new URL(float.href).pathname.split('/').filter(Boolean)[0] || 'rejusamjohn';
  const openers = () => [float, ...doc.querySelectorAll('[data-coffee]')];

  let panel = null;
  let opener = null;

  function setExpanded(value) {
    for (const b of openers()) b.setAttribute('aria-expanded', String(value));
  }

  function close() {
    if (!panel || panel.hidden) return;
    panel.hidden = true;
    setExpanded(false);
    if (opener && opener.isConnected) opener.focus();
  }

  function build() {
    panel = doc.createElement('div');
    panel.className = 'coffee-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Support with a coffee');
    panel.hidden = true;

    const head = doc.createElement('div');
    head.className = 'coffee-panel-head';
    const title = doc.createElement('span');
    title.textContent = 'Buy me a coffee';
    const closeBtn = doc.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'coffee-panel-close';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', close);
    head.append(title, closeBtn);

    const iframe = doc.createElement('iframe');
    iframe.title = 'Ko-fi tip panel';
    iframe.loading = 'lazy';
    iframe.referrerPolicy = 'no-referrer';
    iframe.src = `https://ko-fi.com/${encodeURIComponent(handle)}/?hidefeed=true&widget=true&embed=true&preview=true`;

    const note = doc.createElement('p');
    note.className = 'coffee-panel-note';
    note.textContent = 'Payments are handled by Ko-fi. Nothing loads from Ko-fi until you open this panel.';

    panel.append(head, iframe, note);
    doc.body.appendChild(panel);
  }

  function open(from) {
    opener = from;
    if (!panel) build();
    panel.hidden = false;
    setExpanded(true);
    panel.querySelector('.coffee-panel-close').focus();
  }

  function toggle(from, e) {
    if (e) e.preventDefault();
    if (panel && !panel.hidden) close();
    else open(from);
  }

  float.addEventListener('click', (e) => toggle(float, e));

  for (const b of doc.querySelectorAll('[data-coffee]')) {
    b.setAttribute('aria-haspopup', 'dialog');
    b.setAttribute('aria-expanded', 'false');
    b.addEventListener('click', (e) => toggle(b, e));
  }

  doc.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
  });

  return { open: () => open(float), close };
}
