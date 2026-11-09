// Loaded as a plain blocking script in <head> on every Explorer page, so the
// JS layout is in place before first paint.
// Safety net: module scripts run before DOMContentLoaded, so if ml.js has not
// made the lesson live by then (it failed to load or threw), fall back to the
// plain reading list with every step visible.
document.documentElement.classList.add('js');
document.addEventListener('DOMContentLoaded', function () {
  var main = document.querySelector('main.lesson');
  if (main && !main.classList.contains('is-live')) main.classList.add('stage-failed');
});
