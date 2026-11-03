"""Convex hull and enclosing quadrilateral, in image pixel coordinates (y down)."""

from __future__ import annotations

import heapq

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns hull vertices, clockwise on screen (y down)."""
    pts = _drop_inner(np.asarray(points, dtype=np.float64).reshape(-1, 2))
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts
    lower, upper = _half_hull(pts), _half_hull(pts[::-1])
    return np.concatenate([lower[:-1], upper[:-1]])


def _half_hull(seq: np.ndarray) -> np.ndarray:
    """One half of the monotone chain over points sorted along x.

    A point that does not make a left turn with its current neighbors lies
    on or above a segment between two other points, so it is not a hull
    corner. Dropping all such points at once, again and again, ends at the
    same chain the one-by-one stack gives, in a few numpy passes.
    """
    keep = np.ones(len(seq), dtype=bool)
    for _ in range(64):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        a, b, q = seq[idx[:-2]], seq[idx[1:-1]], seq[idx[2:]]
        cross = (b[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (q[:, 0] - a[:, 0])
        bad = cross <= 0
        if not bad.any():
            return seq[idx]
        keep[idx[1:-1][bad]] = False
    # Rare long concave runs: finish the few points left with the plain stack.
    out: list[list[float]] = []
    for qx, qy in seq[keep].tolist():
        while len(out) >= 2:
            (ax, ay), (bx, by) = out[-2], out[-1]
            if (bx - ax) * (qy - ay) - (by - ay) * (qx - ax) <= 0:
                out.pop()
            else:
                break
        out.append([qx, qy])
    return np.array(out)


def _drop_inner(pts: np.ndarray) -> np.ndarray:
    """Drop points that lie strictly inside the polygon of the 8 extreme
    points (Akl-Toussaint). They cannot be on the hull, so the hull stays
    the same; the slow monotone chain then sees far fewer points."""
    if len(pts) < 64:
        return pts
    x, y = pts[:, 0], pts[:, 1]
    # Left, top-left, top, top-right, right, bottom-right, bottom, bottom-left (y down).
    picks = [x.argmin(), (x + y).argmin(), y.argmin(), (x - y).argmax(), x.argmax(), (x + y).argmax(),
             y.argmax(), (x - y).argmin()]
    ring = pts[picks]
    inside = np.ones(len(pts), dtype=bool)
    area2 = 0.0
    for k in range(8):
        a, b = ring[k], ring[(k + 1) % 8]
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        area2 += a[0] * b[1] - b[0] * a[1]
        inside &= cross > 1e-9 * max(1.0, float(np.hypot(*(b - a))))  # strictly left of every edge
    if area2 <= 0:  # the 8 points do not make a counterclockwise ring (degenerate input): keep all
        return pts
    return pts[~inside]


def enclosing_quad(hull: np.ndarray) -> np.ndarray:
    """A tight quadrilateral that contains the convex polygon `hull`.

    Two candidates, both containing the hull: greedy edge removal (below)
    and the smallest-area enclosing rectangle. The one whose corners stay
    closest to the hull wins. On a long, thin, bent box the greedy one can
    grow a far, thin spike (cheap in area, useless as a label); the
    rectangle cannot.
    """
    poly = [np.asarray(v, dtype=np.float64) for v in hull]
    if len(poly) < 3:
        return _pad_to_quad(poly)
    hull_arr = np.array(poly)
    candidates = [_greedy_quad(poly)]
    if len(poly) > 4:
        candidates.append(_min_area_rect(hull_arr))
    return min(candidates, key=lambda q: (round(_reach(q, hull_arr), 3), abs(_signed_area(q))))


def _reach(quad: np.ndarray, hull: np.ndarray) -> float:
    """How far the quad's farthest corner lies from the hull."""
    a = hull
    ab = np.roll(hull, -1, axis=0) - a  # hull edges
    rel = quad[:, None, :] - a[None, :, :]  # (corners, edges, 2)
    t = np.clip((rel * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(rel - t[..., None] * ab, axis=-1)
    return float(d.min(axis=1).max())


def _min_area_rect(hull: np.ndarray) -> np.ndarray:
    """Smallest-area rectangle around a convex polygon (rotating calipers:
    one side lies on a hull edge)."""
    e = np.roll(hull, -1, axis=0) - hull
    length = np.linalg.norm(e, axis=1)
    keep = length > 1e-9
    u = e[keep] / length[keep, None]  # (k, 2) edge directions
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)
    pu, pv = hull @ u.T, hull @ v.T  # (n, k)
    area = (pu.max(0) - pu.min(0)) * (pv.max(0) - pv.min(0))
    k = int(np.argmin(area))
    uk, vk = u[k], v[k]
    u0, u1, v0, v1 = pu[:, k].min(), pu[:, k].max(), pv[:, k].min(), pv[:, k].max()
    return np.array([u0 * uk + v0 * vk, u1 * uk + v0 * vk, u1 * uk + v1 * vk, u0 * uk + v1 * vk])


def _greedy_quad(poly: list[np.ndarray]) -> np.ndarray:
    """Greedy edge removal: drop the edge whose two neighbor edges, extended
    until they meet, cost the least. Each step only adds area outside the
    polygon, so the result always contains it. The cost is the added area
    plus a penalty for a new corner far from the removed edge.

    The polygon is a linked list of vertex ids and the costs sit in a heap,
    so each step costs O(log n). Ties go to the lowest vertex id, which is
    the first one in polygon order.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in poly]
    n0 = len(pts)
    nxt = [(k + 1) % n0 for k in range(n0)]
    prv = [(k - 1) % n0 for k in range(n0)]
    alive = [True] * n0
    version = [0] * n0

    def cost(u: int) -> tuple[float, np.ndarray | None]:
        """Cost of removing the edge from vertex u to the next one."""
        a, b, c, d = pts[prv[u]], pts[u], pts[nxt[u]], pts[nxt[nxt[u]]]
        x = _ray_meet(a, b, d, c)
        if x is None:
            return np.inf, None
        reach = max(float(np.linalg.norm(x - b)), float(np.linalg.norm(x - c)))
        return abs(_cross(c - b, x - b)) * 0.5 + 0.5 * reach * reach, x

    current = [cost(u) for u in range(n0)]
    heap = [(current[u][0], u, 0) for u in range(n0)]
    heapq.heapify(heap)
    count = n0
    while count > 4:
        _, u, ver = heapq.heappop(heap)
        if not alive[u] or ver != version[u]:
            continue  # an old entry
        _, x = current[u]
        if x is None:  # cannot happen for a convex polygon with > 4 vertices
            break
        v = nxt[u]
        pts[u] = x
        alive[v] = False
        nxt[u], prv[nxt[v]] = nxt[v], u
        count -= 1
        # Removing an edge changes only the costs of the edges next to it.
        for k in (prv[prv[u]], prv[u], u, nxt[u]):
            version[k] += 1
            current[k] = cost(k)
            heapq.heappush(heap, (current[k][0], k, version[k]))
    out = [pts[u] for u in range(n0) if alive[u]]
    if len(out) == 3:
        return _pad_to_quad(out)
    return np.array(out)


def order_quad(quad: np.ndarray, first_near: np.ndarray) -> np.ndarray:
    """Clockwise on screen, starting at the vertex closest to `first_near`."""
    q = np.asarray(quad, dtype=np.float64)
    if _signed_area(q) < 0:
        q = q[::-1]
    start = int(np.argmin(np.linalg.norm(q - np.asarray(first_near), axis=1)))
    return np.roll(q, -start, axis=0)


def points_in_convex_polygon(points: np.ndarray, poly: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask: which points lie inside or on the convex polygon."""
    pts = np.asarray(points, dtype=np.float64)
    q = np.asarray(poly, dtype=np.float64)
    if _signed_area(q) < 0:
        q = q[::-1]
    inside = np.ones(len(pts), dtype=bool)
    for k in range(len(q)):
        a, b = q[k], q[(k + 1) % len(q)]
        cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        inside &= cross >= -tol * max(1.0, float(np.linalg.norm(b - a)))
    return inside


def _ray_meet(a, b, d, c):
    """Point where ray a->b (past b) meets ray d->c (past c), or None."""
    r, s = b - a, c - d
    denom = _cross(r, s)
    if abs(denom) < 1e-12:
        return None
    t = _cross(d - a, s) / denom
    u = _cross(d - a, r) / denom
    if t < 1.0 - 1e-9 or u < 1.0 - 1e-9:
        return None
    return a + t * r


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _pad_to_quad(poly) -> np.ndarray:
    pts = [np.asarray(p, dtype=np.float64) for p in poly]
    if not pts:
        raise ValueError("empty polygon")
    while len(pts) < 4:
        # Split the longest edge at its midpoint. The shape stays the same.
        n = len(pts)
        k = max(range(n), key=lambda i: np.linalg.norm(pts[(i + 1) % n] - pts[i]))
        pts.insert(k + 1, (pts[k] + pts[(k + 1) % n]) / 2.0)
    return np.array(pts)
