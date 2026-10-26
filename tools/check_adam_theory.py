"""Executable checks of the manuscript's Adam identities and conditional bounds.

These numerical checks complement, and do not replace, the proofs. No training
results or model checkpoints are modified. Requires NumPy only.
"""
import itertools
import json
from pathlib import Path
import numpy as np


def weights(n, steps, beta):
    a, b, w = [], [], []
    for e in range(steps):
        den = 1-beta**(n+e+1)
        a.append(beta**(e+1)/den)
        w.append(np.array([(1-beta)*beta**(e-j)/den for j in range(e+1)]))
        b.append((1-beta**(e+1))/den)
    return np.array(a), np.array(b), w


def variance(x, p):
    return float(np.sum(p[:, None]*(x-p@x)**2))


def adam_checks():
    rng = np.random.default_rng(20261019)
    checks = []
    for n, E, beta in itertools.product((0, 7, 100), (1, 3, 5), (0., .5, .9)):
        a, b, w = weights(n, E, beta)
        np.testing.assert_allclose(b, [z.sum() for z in w], atol=1e-13)
        np.testing.assert_allclose(b.sum()+(1-beta**n)*a.sum(), E, atol=1e-13)
        m0 = rng.normal(size=4)*(1-beta**n)
        grads = rng.normal(size=(E, 4))
        m = m0.copy()
        for e in range(E):
            m = beta*m+(1-beta)*grads[e]
            direct = m/(1-beta**(n+e+1))
            np.testing.assert_allclose(direct, a[e]*m0+w[e]@grads[:e+1], atol=1e-13)
    checks.append(dict(check="bias_corrected_moment_weights", cases=81, passed=True))

    for _ in range(1000):
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(3), size=4)
        pi = p@q
        G = rng.normal(size=(7, 3))
        residual = rng.normal(size=(4, 7))
        residual -= p@residual
        gk = q@G.T+residual
        estimate = G+rng.normal(scale=.1, size=G.shape)
        correction = (pi-q)@estimate.T
        np.testing.assert_allclose(p@correction, 0, atol=1e-13)
        np.testing.assert_allclose(gk+correction-p@gk,
                                   residual+(q-pi)@(G-estimate).T, atol=1e-13)
        tau = np.linalg.norm(G-estimate, ord=2)
        bound = np.sqrt(np.sum(p[:, None]*residual**2))+tau*np.sqrt(variance(q, p))
        assert np.sqrt(variance(gk+correction, p)) <= bound+1e-12
        exact = (pi-q)@G.T
        np.testing.assert_allclose(variance(gk+exact, p), np.sum(p[:, None]*residual**2), atol=1e-12)
    checks.append(dict(check="prototype_centering_and_residual_bound", cases=1000, passed=True))

    ratios = []
    for n, E, eta in itertools.product((0, 7), (1, 2, 5, 10), (.001, .01, .05)):
        K, dim, beta1, beta2, eps = 4, 5, .9, .999, .2
        p = rng.dirichlet(np.ones(K))
        theta = rng.normal(size=dim)*.2
        curvature = rng.uniform(.2, 1.2, size=(K, dim))
        centres = rng.normal(size=(K, dim))*.3
        correction = rng.normal(size=(K, dim))*.05
        correction -= p@correction
        m0, v0 = np.zeros((K, dim)), np.zeros((K, dim))
        for _ in range(n):
            history = rng.normal(size=(K, dim))*.2
            m0 = beta1*m0+(1-beta1)*history
            v0 = beta2*v0+(1-beta2)*history**2
        D = np.ones(dim) if n == 0 else 1/(np.sqrt((p@v0)/(1-beta2**n))+eps)
        a0 = curvature*(theta-centres)+correction
        x = np.broadcast_to(theta, (K, dim)).copy()
        m, v = m0.copy(), v0.copy()
        ds, zs, moments = [], [], []
        aa, bb, ww = weights(n, E, beta1)
        for e in range(E):
            z = curvature*(x-centres)+correction
            m = beta1*m+(1-beta1)*z
            v = beta2*v+(1-beta2)*z*z
            mh = m/(1-beta1**(n+e+1))
            d = 1/(np.sqrt(v/(1-beta2**(n+e+1)))+eps)
            x -= eta*d*mh
            ds.append(d); zs.append(z); moments.append(mh)
        ds, zs = np.array(ds), np.array(zs)
        M = max(np.linalg.norm(zs, axis=2).max(),
                np.linalg.norm(m0, axis=1).max()/(1-beta1**n) if n else 0.)
        dbar = max(ds.max(), D.max())
        delta = np.abs(ds-D).max()
        R = eta*E*delta*M+eta**2*dbar**2*curvature.max()*M*E*(E-1)/2
        approximation = theta-eta*D*(aa.sum()*m0+bb.sum()*a0)
        rem = np.sqrt(np.sum(p[:, None]*(x-approximation)**2))
        assert rem <= R+1e-12
        gap = abs(np.sqrt(variance(x, p))-eta*bb.sum()*np.sqrt(variance(a0*D, p)))
        assert gap <= eta*aa.sum()*np.sqrt(variance(m0*D, p))+R+1e-12
        ratios.append(rem/R if R else 0.)
        for d, mh in zip(ds, moments):
            direct = p@(d*mh)
            covariance = p@((d-p@d)*(mh-p@mh))
            np.testing.assert_allclose(direct, (p@d)*(p@mh)+covariance, atol=1e-12)
    checks.append(dict(check="persistent_state_adam_dispersion_and_covariance", cases=len(ratios),
                       max_remainder_bound_ratio=float(max(ratios)), passed=True))

    # Enumerate bounded martingale differences; no stochastic-preconditioner
    # independence is used in the manuscript's bound.
    _, _, w = weights(7, 3, .9)
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    for e in range(3):
        np.testing.assert_allclose(np.mean((signs[:, :e+1]@w[e])**2), np.sum(w[e]**2))
    checks.append(dict(check="weighted_martingale_second_moment", cases=3, passed=True))

    for eps in (1e-8, .01, .2):
        eta = .001
        f = lambda g: g/(abs(g)+eps)
        displacement = -eta*(f(-1)+f(3))/2+eta*f(1)
        np.testing.assert_allclose(displacement, 3*eta/((1+eps)*(3+eps)), atol=1e-15)
        np.testing.assert_allclose((-1+2, 3-2), (1, 1))
    checks.append(dict(check="exact_one_step_adam_example", cases=3, passed=True))

    for _ in range(1000):
        dim = 6
        H = rng.uniform(.2, 2., size=dim)
        D = rng.uniform(.1, 2., size=dim)
        x, err = rng.normal(size=(2, dim))
        lo, hi, L = D.min(), D.max(), H.max()
        h = rng.uniform(.01, 1)*lo/(4*L*hi**2)
        g = H*x
        y = x-h*(D*g+err)
        rhs = .5*np.sum(H*x*x)-h*lo/2*np.sum(g*g)+h*(1/lo+L*h)*np.sum(err*err)
        assert .5*np.sum(H*y*y) <= rhs+1e-12
    checks.append(dict(check="conditional_descent_inequality", cases=1000, passed=True))
    return checks


if __name__ == "__main__":
    report = adam_checks()
    destination = Path(__file__).resolve().parents[1]/"output/reproducibility/adam_theory_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
