"""Solution-space discretization and without-replacement Dirichlet splitting."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog


def pairwise_euclidean(features):
    """Return Euclidean distances between rows of a two-dimensional array."""
    values = np.asarray(features, dtype=float)
    squared_norm = np.sum(values ** 2, axis=1)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (values @ values.T)
    distances = np.sqrt(np.maximum(squared, 0.0))
    if distances.shape[0] == distances.shape[1]:
        np.fill_diagonal(distances, 0.0)
    return distances


def _squared_distances(values, centers):
    values = np.asarray(values, dtype=float)
    centers = np.asarray(centers, dtype=float)
    squared = (
        np.sum(values ** 2, axis=1)[:, None]
        + np.sum(centers ** 2, axis=1)[None, :]
        - 2.0 * (values @ centers.T)
    )
    return np.maximum(squared, 0.0)


@dataclass(frozen=True)
class SolutionBins:
    """Discrete physical labels fitted directly from all training responses."""

    labels: np.ndarray
    centroids: np.ndarray
    response_cost: np.ndarray
    response_normalizer: float
    quantization_error: float
    iterations: int


def cluster_solution_responses(responses, n_bins, seed, max_iter=50):
    """Cluster the complete training solution set with deterministic K-means++.

    A single global response scale keeps distances comparable without
    feature-wise standardization that could overemphasize low-energy grid
    locations. K-means avoids a full sample-by-sample distance matrix, which is
    prohibitively expensive for high-resolution PDE fields.
    """
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 2 or not 1 < n_bins <= len(responses):
        raise ValueError("responses must be 2-D and 2 <= n_bins <= n_samples")
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses must be finite")

    centered = responses - np.mean(responses, axis=0)
    normalizer = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if normalizer <= 1e-12:
        raise ValueError("training PDE responses are effectively identical")
    values = centered / normalizer

    rng = np.random.RandomState(seed)
    center_indices = [int(rng.randint(len(values)))]
    nearest = _squared_distances(values, values[center_indices]).ravel()
    while len(center_indices) < n_bins:
        total = float(np.sum(nearest))
        if total <= 1e-15:
            remaining = np.setdiff1d(np.arange(len(values)), center_indices)
            candidate = int(remaining[0])
        else:
            candidate = int(rng.choice(len(values), p=nearest / total))
        if candidate in center_indices:
            continue
        center_indices.append(candidate)
        nearest = np.minimum(
            nearest,
            _squared_distances(values, values[[candidate]]).ravel(),
        )

    centroids = values[center_indices].copy()
    labels = np.full(len(values), -1, dtype=int)
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        squared = _squared_distances(values, centroids)
        updated_labels = np.argmin(squared, axis=1)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        updated_centroids = centroids.copy()
        nearest = squared[np.arange(len(values)), labels]
        for bin_id in range(n_bins):
            members = values[labels == bin_id]
            if len(members):
                updated_centroids[bin_id] = np.mean(members, axis=0)
            else:
                farthest = int(np.argmax(nearest))
                updated_centroids[bin_id] = values[farthest]
                nearest[farthest] = -1.0
        centroids = updated_centroids

    squared = _squared_distances(values, centroids)
    labels = np.argmin(squared, axis=1)
    if len(np.unique(labels)) != n_bins:
        raise RuntimeError("solution clustering produced an empty bin")
    quantization_error = float(np.mean(np.sqrt(
        squared[np.arange(len(values)), labels]
    )))
    return SolutionBins(
        labels=labels,
        centroids=centroids,
        response_cost=pairwise_euclidean(centroids),
        response_normalizer=normalizer,
        quantization_error=quantization_error,
        iterations=iterations,
    )


def bin_histograms(bin_ids, index_groups, n_bins):
    """Convert client sample indices into normalized bin histograms."""
    histograms = np.zeros((len(index_groups), n_bins), dtype=float)
    for client, indices in enumerate(index_groups):
        counts = np.bincount(np.asarray(bin_ids)[indices], minlength=n_bins)
        histograms[client] = counts / counts.sum()
    return histograms


def client_bin_count_matrix(bin_ids, index_groups, n_bins):
    """Return exact sample counts with shape (n_clients, n_bins)."""
    labels = np.asarray(bin_ids, dtype=int)
    return np.stack([
        np.bincount(labels[np.asarray(indices, dtype=int)], minlength=n_bins)
        for indices in index_groups
    ]).astype(int)


def serialize_client_bin_counts(counts):
    """Serialize a client-by-bin integer matrix into one reversible CSV cell."""
    matrix = np.asarray(counts, dtype=int)
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise ValueError("client bin counts must be a nonnegative 2-D matrix")
    return '|'.join(
        ';'.join(str(int(value)) for value in row) for row in matrix)


def deserialize_client_bin_counts(value):
    """Recover a client-by-bin integer matrix from a seed-level CSV cell."""
    rows = [row.split(';') for row in str(value).split('|') if row]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("invalid serialized client bin count matrix")
    matrix = np.asarray(rows, dtype=int)
    if np.any(matrix < 0):
        raise ValueError("client bin counts must be nonnegative")
    return matrix


def partition_clients(bin_ids, n_clients, alpha, minimum, seed):
    """Allocate every sample once by independent per-bin Dirichlet draws.

    The finite training set is exhausted without replacement. Extreme draws
    can leave a client too small, so a deterministic random repair moves (and
    never copies) samples from clients with surplus observations.
    """
    bin_ids = np.asarray(bin_ids, dtype=int)
    if bin_ids.ndim != 1 or len(bin_ids) == 0 or np.min(bin_ids) < 0:
        raise ValueError("bin_ids must be a non-empty vector of nonnegative IDs")
    if n_clients < 2 or alpha <= 0 or minimum < 1:
        raise ValueError("n_clients >= 2, alpha > 0, and minimum >= 1 are required")
    if n_clients * minimum > len(bin_ids):
        raise ValueError("training set cannot satisfy minimum samples per client")

    rng = np.random.RandomState(seed)
    n_bins = int(np.max(bin_ids)) + 1
    members = [rng.permutation(np.flatnonzero(bin_ids == b)) for b in range(n_bins)]
    if any(len(member) == 0 for member in members):
        raise ValueError("every solution bin must contain training samples")
    client_indices = [[] for _ in range(n_clients)]
    target_counts = np.zeros((n_clients, n_bins), dtype=float)
    for bin_id, bin_members in enumerate(members):
        proportions = rng.dirichlet(np.full(n_clients, float(alpha)))
        counts = rng.multinomial(len(bin_members), proportions)
        target_counts[:, bin_id] = len(bin_members) * proportions
        cursor = 0
        for client, count in enumerate(counts):
            client_indices[client].extend(
                bin_members[cursor:cursor + count].tolist()
            )
            cursor += count

    sizes = np.asarray([len(indices) for indices in client_indices], dtype=int)
    while np.any(sizes < minimum):
        recipient = int(np.argmin(sizes))
        surplus = np.maximum(sizes - minimum, 0)
        if surplus.sum() == 0:
            raise RuntimeError("unable to repair underfilled clients")
        donor = int(rng.choice(n_clients, p=surplus / surplus.sum()))
        position = int(rng.randint(len(client_indices[donor])))
        client_indices[recipient].append(client_indices[donor].pop(position))
        sizes[recipient] += 1
        sizes[donor] -= 1

    client_indices = [
        rng.permutation(indices).astype(int) for indices in client_indices
    ]
    realized = bin_histograms(bin_ids, client_indices, n_bins)
    target = target_counts / np.maximum(
        target_counts.sum(axis=1, keepdims=True), 1e-12
    )
    combined = np.concatenate(client_indices)
    if len(combined) != len(bin_ids) or not np.array_equal(
            np.sort(combined), np.arange(len(bin_ids))):
        raise RuntimeError("partition lost or duplicated training samples")
    return client_indices, target, realized


def partition_clients_iid(bin_ids, n_clients, minimum, seed):
    """Create an exact finite-sample IID reference without replacement.

    Samples from every solution bin are distributed as evenly as integer
    counts permit.  At the same time, total client sizes are balanced to
    differ by at most one.  Randomness affects only which client receives a
    remainder and which concrete samples are assigned; no sample is copied.

    The returned ``target`` is the global solution-bin distribution repeated
    for every client.  ``realized`` records the unavoidable integer-rounded
    empirical histograms.
    """
    bin_ids = np.asarray(bin_ids, dtype=int)
    if bin_ids.ndim != 1 or len(bin_ids) == 0 or np.min(bin_ids) < 0:
        raise ValueError("bin_ids must be a non-empty vector of nonnegative IDs")
    if n_clients < 2 or minimum < 1:
        raise ValueError("n_clients >= 2 and minimum >= 1 are required")
    if n_clients * minimum > len(bin_ids):
        raise ValueError("training set cannot satisfy minimum samples per client")

    rng = np.random.RandomState(seed)
    n_bins = int(np.max(bin_ids)) + 1
    bin_sizes = np.bincount(bin_ids, minlength=n_bins)
    if np.any(bin_sizes == 0):
        raise ValueError("every solution bin must contain training samples")

    # Give every client floor(n_b / K) observations from bin b.  The remaining
    # observations are assigned to clients that still have the largest total
    # capacity, with seeded random tie breaking.  This simultaneously balances
    # every bin and every client as tightly as finite integer counts allow.
    counts = np.tile((bin_sizes // n_clients), (n_clients, 1))
    client_target_sizes = np.full(n_clients, len(bin_ids) // n_clients, dtype=int)
    extra_clients = rng.permutation(n_clients)[:len(bin_ids) % n_clients]
    client_target_sizes[extra_clients] += 1
    capacities = client_target_sizes - counts.sum(axis=1)

    remainder_order = rng.permutation(n_bins)
    remainder_order = sorted(
        remainder_order, key=lambda b: int(bin_sizes[b] % n_clients),
        reverse=True)
    for bin_id in remainder_order:
        remainder = int(bin_sizes[bin_id] % n_clients)
        if remainder == 0:
            continue
        tie_break = rng.permutation(n_clients)
        ranked = tie_break[np.argsort(-capacities[tie_break], kind='stable')]
        chosen = ranked[:remainder]
        if np.any(capacities[chosen] <= 0):
            raise RuntimeError("unable to balance exact IID client sizes")
        counts[chosen, bin_id] += 1
        capacities[chosen] -= 1

    if np.any(capacities != 0):
        raise RuntimeError("exact IID allocation did not fill every client")

    client_indices = [[] for _ in range(n_clients)]
    for bin_id in range(n_bins):
        members = rng.permutation(np.flatnonzero(bin_ids == bin_id))
        cursor = 0
        for client in range(n_clients):
            count = int(counts[client, bin_id])
            client_indices[client].extend(members[cursor:cursor + count].tolist())
            cursor += count
        if cursor != len(members):
            raise RuntimeError("exact IID allocation did not exhaust a bin")
    client_indices = [
        rng.permutation(indices).astype(int) for indices in client_indices]

    combined = np.concatenate(client_indices)
    if len(combined) != len(bin_ids) or not np.array_equal(
            np.sort(combined), np.arange(len(bin_ids))):
        raise RuntimeError("exact IID partition lost or duplicated training samples")
    realized = bin_histograms(bin_ids, client_indices, n_bins)
    target = np.tile(bin_sizes / bin_sizes.sum(), (n_clients, 1))
    return client_indices, target, realized


def emd_histogram_distance(left, right, cost):
    """Balanced transport without mutating the supplied bin histograms.

    Extreme Dirichlet draws contain positive masses close to the LP solver's
    feasibility tolerance. HiGHS presolve can incorrectly declare the redundant
    full marginal system infeasible for these distributions. Remove one implied
    equality and solve without presolve; do not smooth or discard positive mass.
    """
    def normalized_histogram(values, name):
        histogram = np.array(values, dtype=float, copy=True)
        if histogram.ndim != 1 or len(histogram) == 0:
            raise ValueError(f"{name} histogram must be a non-empty vector")
        if not np.all(np.isfinite(histogram)) or np.any(histogram < 0):
            raise ValueError(f"{name} histogram must be finite and nonnegative")
        scale = float(np.max(histogram))
        if scale == 0:
            raise ValueError(f"{name} histogram must have positive total mass")
        # Scaling before summing also accepts very large finite sample counts.
        histogram /= scale
        histogram /= np.sum(histogram)
        return histogram

    left = normalized_histogram(left, 'left')
    right = normalized_histogram(right, 'right')
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (len(left), len(right)):
        raise ValueError("cost shape must match the two histogram lengths")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError("transport cost must be finite and nonnegative")

    # Exactly zero bins carry no transport; every positive bin is retained.
    left_support, right_support = left > 0, right > 0
    cost = cost[np.ix_(left_support, right_support)]
    left, right = left[left_support], right[right_support]
    n_left, n_right = len(left), len(right)
    if n_left == 1:
        return float(np.dot(cost[0], right))
    if n_right == 1:
        return float(np.dot(left, cost[:, 0]))

    # Equal total mass makes one column equality redundant. Omit the largest
    # right marginal so the small positive marginals remain explicitly enforced.
    retained_columns = np.arange(n_right) != np.argmax(right)
    row_constraints = np.repeat(np.eye(n_left), n_right, axis=1)
    column_constraints = np.tile(np.eye(n_right), (1, n_left))[retained_columns]
    constraints = np.concatenate([row_constraints, column_constraints], axis=0)
    rhs = np.concatenate([left, right[retained_columns]])
    result = linprog(
        cost.ravel(), A_eq=constraints, b_eq=rhs, bounds=(0, None),
        method='highs', options={
            'presolve': False,
            'primal_feasibility_tolerance': 1e-9,
            'dual_feasibility_tolerance': 1e-9,
        },
    )
    if not result.success:
        raise RuntimeError(
            f"transport LP failed for {n_left} x {n_right} positive bins: "
            f"{result.message}")
    # Check both complete marginals, including the omitted implied equality.
    # A few solver-scale residuals can accumulate in that omitted marginal.
    plan = result.x.reshape(n_left, n_right)
    tolerance = 1e-9 * max(n_left + n_right, 10)
    if (np.max(np.abs(plan.sum(axis=1) - left)) > tolerance
            or np.max(np.abs(plan.sum(axis=0) - right)) > tolerance
            or np.min(plan) < -tolerance
            or not np.isfinite(result.fun)):
        raise RuntimeError("transport LP returned an invalid marginal balance")
    return float(result.fun)


def mean_pairwise_transport(client_histograms, cost):
    values = [
        emd_histogram_distance(client_histograms[i], client_histograms[j], cost)
        for i in range(len(client_histograms))
        for j in range(i + 1, len(client_histograms))
    ]
    return float(np.mean(values)) if values else 0.0


def mean_pairwise_sample_transport(sample_groups, normalizer, max_points=128,
                                   seed=0):
    """Approximate input-space W1 using equal-size optimal matching."""
    if normalizer <= 0:
        raise ValueError("normalizer must be positive")
    rng = np.random.RandomState(seed)
    values = []
    for i in range(len(sample_groups)):
        for j in range(i + 1, len(sample_groups)):
            left = np.asarray(sample_groups[i], dtype=float)
            right = np.asarray(sample_groups[j], dtype=float)
            count = min(len(left), len(right), int(max_points))
            if count == 0:
                raise ValueError("sample groups must not be empty")
            left = left[rng.choice(len(left), count, replace=False)]
            right = right[rng.choice(len(right), count, replace=False)]
            delta = left[:, None, :] - right[None, :, :]
            cost = np.sqrt(np.sum(delta ** 2, axis=2))
            row, column = linear_sum_assignment(cost)
            values.append(float(np.mean(cost[row, column]) / normalizer))
    return float(np.mean(values)) if values else 0.0
