import numpy as np

from recapture.labels.quad import convex_hull, enclosing_quad, order_quad, points_in_convex_polygon


def _area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def test_hull_of_square_with_inner_points():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [2, 7], [5, 0]])
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert _area(hull) > 0  # clockwise on screen (y down)


def test_enclosing_quad_contains_random_clouds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 300))
        pts = rng.normal(size=(n, 2)) * rng.uniform(1, 200, size=2) + rng.uniform(-500, 500, size=2)
        hull = convex_hull(pts)
        quad = enclosing_quad(hull)
        assert quad.shape == (4, 2)
        assert points_in_convex_polygon(pts, quad, tol=1e-6).all()
        # The quad should stay close to the hull, not blow up.
        assert abs(_area(quad)) <= 2.0 * abs(_area(hull)) + 1e-6


def test_enclosing_quad_is_exact_for_a_perspective_quad():
    quad = np.array([[10.0, 12.0], [200.0, 5.0], [230.0, 160.0], [-5.0, 140.0]])
    edges = [quad[i] + t * (quad[(i + 1) % 4] - quad[i]) for i in range(4) for t in np.linspace(0, 1, 50)]
    fit = enclosing_quad(convex_hull(np.array(edges)))
    fit = order_quad(fit, quad[0])
    assert np.abs(fit - quad).max() < 1e-6


def test_order_quad_starts_near_given_point():
    q = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)  # counter-clockwise on screen
    out = order_quad(q, np.array([9, 9]))
    assert out[0].tolist() == [10, 10]
    assert _area(out) > 0



# The visible part of a long, thin header box on a crumpled page (a real
# case). Plain greedy edge removal grew a 140 px spike out of it.
_THIN_BENT_HULL = [[499.32, 362.46], [499.5, 361.06], [499.77, 359.01], [500.76, 351.64], [727.5, 385.5], [748.81, 389.95], [750.22, 390.31], [751.63, 390.68], [752.09, 390.82], [752.06, 391.2], [751.93, 392.78], [751.61, 396.74], [751.58, 397.14], [751.54, 397.56], [751.5, 397.99], [751.46, 398.43], [751.33, 399.79], [751.28, 400.28], [750.73, 405.18], [627.19, 393.94], [624.76, 393.66], [623.79, 393.52], [607.77, 391.24], [606.8, 391.08], [602.94, 390.23], [545.61, 376.17]]


def test_thin_bent_box_gets_no_spike():
    from recapture.labels.quad import _reach

    hull = np.array(_THIN_BENT_HULL)
    quad = enclosing_quad(hull)
    assert points_in_convex_polygon(hull, quad, 1e-6).all()
    assert _reach(quad, hull) < 15.0



def test_hull_prefilter_keeps_the_hull():
    """Dropping inner points first must give exactly the plain monotone-chain hull."""
    from recapture.labels import quad

    rng = np.random.default_rng(0)
    for _ in range(200):
        pts = rng.normal(size=(int(rng.integers(64, 3000)), 2)) * rng.uniform(1, 50, 2) + rng.uniform(-100, 100, 2)
        if rng.random() < 0.3:  # sheared, thin sets like a bent text line
            pts[:, 1] = pts[:, 0] * rng.uniform(-1, 1) + pts[:, 1] * 0.02
        fast = quad.convex_hull(pts)
        reference = _plain_hull(pts)
        assert np.array_equal(fast, reference)


def _plain_hull(points):
    """The plain one-by-one monotone chain."""
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    p = pts.tolist()

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _greedy_reference(poly):
    """The plain O(n^2) version of quad._greedy_quad, kept to check the fast one."""
    from recapture.labels.quad import _cross, _pad_to_quad, _ray_meet

    poly = [np.asarray(v, dtype=np.float64) for v in poly]

    def cost(i):
        n = len(poly)
        a, b, c, d = poly[i - 1], poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        x = _ray_meet(a, b, d, c)
        if x is None:
            return np.inf, None
        reach = max(float(np.linalg.norm(x - b)), float(np.linalg.norm(x - c)))
        return abs(_cross(c - b, x - b)) * 0.5 + 0.5 * reach * reach, x

    costs = [cost(i) for i in range(len(poly))]
    while len(poly) > 4:
        n = len(poly)
        i = min(range(n), key=lambda k: costs[k][0])
        _, x = costs[i]
        if x is None:
            break
        j = (i + 1) % n
        poly[i] = x
        del poly[j]
        del costs[j]
        n -= 1
        i = i if j > i else i - 1
        for k in (i - 2, i - 1, i, i + 1):
            costs[k % n] = cost(k % n)
    return _pad_to_quad(poly) if len(poly) == 3 else np.array(poly)


def test_fast_greedy_matches_the_plain_version():
    from recapture.labels.quad import _greedy_quad

    rng = np.random.default_rng(1)
    for _ in range(300):
        pts = rng.normal(size=(int(rng.integers(5, 400)), 2)) * rng.uniform(1, 80, 2)
        hull = convex_hull(pts)
        if len(hull) < 5:
            continue
        assert np.array_equal(_greedy_quad(list(hull)), _greedy_reference(list(hull)))
