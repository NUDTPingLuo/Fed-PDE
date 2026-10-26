"""Check weighted centering, stratification, gradient and transport bounds."""
import json

import numpy as np
from scipy.optimize import linprog


def transport(q, r, cost):
    bins = len(q)
    constraints = np.vstack([
        np.kron(np.eye(bins), np.ones((1, bins))),
        np.kron(np.ones((1, bins)), np.eye(bins))[:-1],
    ])
    result = linprog(cost.ravel(), A_eq=constraints,
                     b_eq=np.r_[q, r[:-1]], bounds=(0, None), method="highs")
    assert result.success, result.message
    return result.fun


def geometry_checks():
    rng = np.random.default_rng(20261024)
    checks = []
    for _ in range(1000):
        clients, dim = 5, 7
        p = rng.dirichlet(np.ones(clients))
        vectors = rng.normal(size=(clients, dim))
        projection = np.kron(np.eye(clients)-np.ones((clients, 1))*p,
                             np.eye(dim))
        centered = (projection @ vectors.ravel()).reshape(clients, dim)
        np.testing.assert_allclose(centered, vectors-p@vectors, atol=1e-13)
        lhs = np.sum(p[:, None]*centered**2)
        rhs = np.sum(p[:, None]*vectors**2)-np.sum((p@vectors)**2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
    checks.append(dict(check="dimensioned_weighted_centering", cases=1000, passed=True))

    rounding_cases = 0
    for _ in range(1000):
        clients, bins = int(rng.integers(2, 11)), int(rng.integers(2, 11))
        totals = rng.integers(1, 101, size=bins)
        total = int(totals.sum())
        if total < clients:
            continue
        capacity = np.full(clients, total//clients)
        capacity[rng.permutation(clients)[:total % clients]] += 1
        counts = np.tile(totals//clients, (clients, 1))
        remaining = capacity-counts.sum(axis=1)
        for b in range(bins):
            # Random tie-breaking, then choose distinct clients of greatest capacity.
            order = rng.permutation(clients)
            order = order[np.argsort(-remaining[order], kind="stable")]
            chosen = order[:totals[b] % clients]
            counts[chosen, b] += 1
            remaining[chosen] -= 1
        np.testing.assert_array_equal(remaining, 0)
        np.testing.assert_array_equal(counts.sum(axis=0), totals)
        assert np.all((counts == totals//clients) | (counts == -(-totals//clients)))
        p, pi = capacity/total, totals/total
        q = counts/capacity[:, None]
        bound = (np.sqrt(bins)+np.linalg.norm(pi))/capacity.min()
        distances = np.linalg.norm(q-pi, axis=1)
        assert np.max(distances) <= bound+1e-12
        assert p@(distances**2) <= bound**2+1e-12
        rounding_cases += 1
    checks.append(dict(check="stratified_integer_rounding_bound", cases=rounding_cases, passed=True))

    for _ in range(1000):
        clients, bins, dim = 5, 4, 8
        p = rng.dirichlet(np.ones(clients))
        q = rng.dirichlet(np.ones(bins), size=clients)
        pi = p@q
        matrix = rng.normal(size=(dim, bins))
        basis = np.linalg.qr(np.eye(bins)[:, :-1]-np.eye(bins)[:, [-1]])[0]
        low, high = np.linalg.svd(matrix@basis, compute_uv=False)[[-1, 0]]
        histogram_variance = np.sum(p[:, None]*(q-pi)**2)
        gradient_variance = np.sum(p[:, None]*((q-pi)@matrix.T)**2)
        assert low**2*histogram_variance <= gradient_variance+1e-12
        assert gradient_variance <= high**2*histogram_variance+1e-12
        error = rng.normal(size=(clients, dim))
        error -= p@error
        estimate = matrix+rng.normal(scale=.1, size=matrix.shape)
        tau = np.linalg.norm((matrix-estimate)@basis, ord=2)
        corrected = error+(q-pi)@(matrix-estimate).T
        upper = np.sqrt(np.sum(p[:, None]*error**2))+tau*np.sqrt(histogram_variance)
        assert np.sqrt(np.sum(p[:, None]*corrected**2)) <= upper+1e-12
    checks.append(dict(check="restricted_bin_contrast_bounds", cases=1000, passed=True))

    for _ in range(50):
        bins = 5
        centers = rng.normal(size=(bins, 3))
        cost = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
        q, r = rng.dirichlet(np.ones(bins), size=2)
        distance = transport(q, r, cost)
        minimum, maximum = cost[cost > 0].min(), cost.max()
        tv = np.abs(q-r).sum()/2
        assert minimum*tv <= distance+1e-10
        assert distance <= maximum*tv+1e-10
        assert minimum*np.linalg.norm(q-r)/2 <= distance+1e-10
        assert distance <= maximum*np.sqrt(bins)*np.linalg.norm(q-r)/2+1e-10
    checks.append(dict(check="transport_tv_and_euclidean_bounds", cases=50, passed=True))
    return checks




if __name__ == "__main__":
    print(json.dumps(geometry_checks(), indent=2))
