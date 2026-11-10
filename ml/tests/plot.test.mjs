import test from 'node:test';
import assert from 'node:assert/strict';
import { linearScale, clamp, pathFrom, fmt, networkLayout, pixelFill, heatmapFill, ellipsePath } from '../assets/plot.js';

test('linearScale maps and inverts', () => {
  const s = linearScale([0, 10], [50, 550]);
  assert.equal(s(5), 300);
  assert.equal(s.invert(300), 5);
});

test('ticks are nice', () => {
  const t = linearScale([0, 1], [0, 1]).ticks(5);
  const want = [0, 0.2, 0.4, 0.6, 0.8, 1];
  assert.equal(t.length, want.length);
  t.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12));
});

test('clamp keeps drag inside domain', () => {
  assert.equal(clamp(12, 0, 10), 10);
  assert.equal(clamp(-1, 0, 10), 0);
});

test('pathFrom skips non-finite', () => {
  const id = (v) => v;
  const d = pathFrom([[0, 0], [1, NaN], [2, 2]], id, id);
  assert.ok(!d.includes('NaN'));
  assert.ok(d.startsWith('M'));
});

test('tick labels use a true minus sign', () => {
  assert.equal(fmt(-2), '\u22122');
  assert.equal(fmt(-0.5), '\u22120.5');
  assert.equal(fmt(3), '3');
  assert.equal(fmt(0), '0');
  const labels = linearScale([-1.8, 1.8], [0, 100]).ticks(5).map(fmt);
  assert.deepEqual(labels, ['\u22121', '0', '1']);
});

import { regionCells, boundarySegments, treeLayout, regionFill, trianglePoints } from '../assets/plot.js';

test('boundarySegments handles three classes', () => {
  const segs = boundarySegments([[0, 1, 2]]);
  assert.equal(segs.length, 2);
  assert.deepEqual(segs, [[[1, 0], [1, 1]], [[2, 0], [2, 1]]]);
});

test('boundarySegments on a 2×2 grid', () => {
  const segs = boundarySegments([[0, 1], [0, 1]]);
  assert.deepEqual(segs, [[[1, 0], [1, 1]], [[1, 1], [1, 2]]]);
});

test('boundarySegments finds horizontal edges', () => {
  const segs = boundarySegments([[0, 0], [1, 1]]);
  assert.deepEqual(segs, [[[0, 1], [1, 1]], [[1, 1], [2, 1]]]);
});

test('boundarySegments empty for one class', () => {
  assert.deepEqual(boundarySegments([[1, 1, 1], [1, 1, 1]]), []);
});

test('regionCells covers the plot once, top row first', () => {
  const xs = linearScale([0, 6], [40, 640]);
  const ys = linearScale([0, 4], [400, 0]);
  const cells = regionCells(60, 40, xs, ys);
  assert.equal(cells.length, 2400);
  const first = cells[0];
  assert.equal(first.i, 0);
  assert.equal(first.j, 0);
  assert.ok(Math.abs(first.x - 0.05) < 1e-12);
  assert.ok(Math.abs(first.y - 3.95) < 1e-12);
  assert.equal(first.px, 40);
  assert.equal(first.py, 0);
  assert.equal(first.w, 10);
  assert.equal(first.h, 10);
  const last = cells[2399];
  assert.ok(Math.abs(last.x - 5.95) < 1e-12);
  assert.ok(Math.abs(last.y - 0.05) < 1e-12);
  assert.equal(last.px + last.w, 640);
  assert.equal(last.py + last.h, 400);
});

test('regionFill: class mode keeps the class, prob mode thresholds at 0.5', () => {
  assert.deepEqual(regionFill(2, 'class'), { cls: 2, strength: 1 });
  assert.deepEqual(regionFill(0.5, 'prob'), { cls: 1, strength: 0 });
  assert.deepEqual(regionFill(0, 'prob'), { cls: 0, strength: 1 });
  assert.deepEqual(regionFill(0.75, 'prob'), { cls: 1, strength: 0.5 });
  assert.equal(regionFill(0.25, 'prob').cls, 0);
});

test('triangle marker has the area of the circle', () => {
  const r = 5;
  const [[ax, ay], [bx, by], [cx, cy]] = trianglePoints(10, 20, r);
  const area = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2;
  assert.ok(Math.abs(area - Math.PI * r * r) < 1e-9);
  assert.ok(Math.abs((ax + bx + cx) / 3 - 10) < 1e-12);
  assert.ok(Math.abs((ay + by + cy) / 3 - 20) < 1e-12);
});

test('treeLayout puts leaves left to right', () => {
  const leaf = (c) => ({ leaf: true, counts: c, cls: c[1] > c[0] ? 1 : 0, n: c[0] + c[1] });
  const tree = {
    leaf: false, feature: 0, threshold: 3, counts: [6, 6], cls: 0, n: 12,
    left: { leaf: false, feature: 1, threshold: 1, counts: [5, 1], cls: 0, n: 6, left: leaf([4, 0]), right: leaf([1, 1]) },
    right: { leaf: false, feature: 1, threshold: 2, counts: [1, 5], cls: 1, n: 6, left: leaf([1, 2]), right: leaf([0, 3]) },
  };
  const lay = treeLayout(tree);
  assert.equal(lay.leaves, 4);
  assert.equal(lay.depth, 2);
  assert.equal(lay.nodes.length, 7);
  const leaves = lay.nodes.filter((p) => p.node.leaf);
  assert.deepEqual(leaves.map((p) => p.x), [0, 1, 2, 3]);
  assert.ok(leaves.every((p) => p.depth === 2));
  for (const [a, b] of lay.links) assert.equal(lay.nodes[b].depth, lay.nodes[a].depth + 1);
  for (const p of lay.nodes.filter((q) => !q.node.leaf)) {
    const kids = lay.links.filter(([a]) => lay.nodes[a] === p).map(([, b]) => lay.nodes[b]);
    assert.equal(kids.length, 2);
    assert.equal(p.x, (kids[0].x + kids[1].x) / 2);
  }
  assert.equal(lay.nodes[0].node, tree);
  assert.equal(lay.nodes[0].x, 1.5);
});

test('treeLayout of a single leaf', () => {
  const lay = treeLayout({ leaf: true, counts: [3, 1], cls: 0, n: 4 });
  assert.equal(lay.leaves, 1);
  assert.equal(lay.depth, 0);
  assert.deepEqual(lay.links, []);
  assert.equal(lay.nodes[0].x, 0);
});

test('networkLayout spaces layers and centres units', () => {
  const lay = networkLayout([2, 3, 1], { x: 0, y: 0, w: 300, h: 200 });
  const xs = [...new Set(lay.nodes.map((n) => n.cx))];
  assert.equal(xs.length, 3);
  assert.ok(xs[0] < xs[1] && xs[1] < xs[2]);
  assert.deepEqual(xs, [0, 150, 300]);
  const mid = lay.nodes.find((n) => n.l === 1 && n.j === 1);
  assert.equal(mid.cy, 100);
  const single = lay.nodes.find((n) => n.l === 2);
  assert.equal(single.cy, 100);
  assert.equal(lay.nodes.length, 6);
  assert.equal(lay.links.length, 2 * 3 + 3 * 1);
  const k = lay.links.find((q) => q.l === 0 && q.i === 1 && q.j === 2);
  const a = lay.nodes.find((n) => n.l === 0 && n.j === 1);
  const b = lay.nodes.find((n) => n.l === 1 && n.j === 2);
  assert.deepEqual([k.x1, k.y1, k.x2, k.y2], [a.cx, a.cy, b.cx, b.cy]);
});

test('pixelFill grey and signed', () => {
  assert.deepEqual(pixelFill(0, 'grey'), { cls: 'px-grey', opacity: 0 });
  assert.deepEqual(pixelFill(0.4, 'grey'), { cls: 'px-grey', opacity: 0.4 });
  assert.deepEqual(pixelFill(-2, { signed: 4 }), { cls: 'px-neg', opacity: 0.5 });
  assert.deepEqual(pixelFill(3, { signed: 4 }), { cls: 'px-pos', opacity: 0.75 });
  const z = pixelFill(-1, { signed: 0 });
  assert.equal(z.opacity, 0);
  assert.ok(!Number.isNaN(pixelFill(0, { signed: 0 }).opacity));
});

test('heatmapFill clamps and handles NaN', () => {
  assert.equal(heatmapFill(0.42).opacity, 0.42);
  assert.equal(heatmapFill(-0.3).opacity, 0);
  assert.equal(heatmapFill(1.7).opacity, 1);
  assert.equal(heatmapFill(NaN).opacity, 0);
});

const pathPoints = (d) => d.slice(1, -1).split('L').map((p) => p.split(' ').map(Number));

test('ellipsePath of the identity is a circle of radius k', () => {
  const pts = pathPoints(ellipsePath([1, -2], [[1, 0], [0, 1]], 2));
  for (const [x, y] of pts) assert.ok(Math.abs(Math.hypot(x - 1, y + 2) - 2) < 1e-6);
  assert.ok(pts.length >= 36);
});

test('ellipsePath axes follow the eigenvectors', () => {
  const pts = pathPoints(ellipsePath([0, 0], [[4, 0], [0, 1]], 1));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  assert.ok(Math.abs(Math.max(...xs) - 2) < 1e-6);
  assert.ok(Math.abs(Math.max(...ys) - 1) < 1e-2);
  // tilted: variance along (1,1) is 3, along (1,-1) is 1
  const t = pathPoints(ellipsePath([0, 0], [[2, 1], [1, 2]], 1));
  const far = Math.max(...t.map(([x, y]) => Math.hypot(x, y)));
  assert.ok(Math.abs(far - Math.sqrt(3)) < 1e-6);
});
