import numpy as np


def calc_euclidean_distance(p, q, scale=None):
    """Return Euclidean distance between 'p' and 'q'

    Parameters
    ----------------
    p: (N,) list, tuple or ndarray
        start position of distance measure
    q: (N,) list, tuple or ndarray
        end position of distance measure
    (optional) scale: (N,) list, tuple or ndarray
        scaling weights for each dimensions

    Returns
    ----------------
    dists: (N,) float
        Euclidean distance between 'p' and 'q'
    """
    # Vectorize inputs
    if not isinstance(p, type(np.ndarray)):
        p = np.array(p)
    if not isinstance(q, type(np.ndarray)):
        q = np.array(q)

    # Get distance
    if not scale:
        return np.sqrt(np.sum((q - p) ** 2.0))
    else:
        return np.sqrt(np.sum((scale * (q - p)) ** 2.0))


def calc_manhattan_distance(p, q, scale=None):
    """Return Manhattan distance between 'p' and 'q'

    Parameters
    ----------------
    p: (N,) list, tuple or ndarray
        start position of distance measure
    q: (N,) list, tuple or ndarray
        end position of distance measure
    (optional) scale: (N,) list, tuple or ndarray
        scaling weights for each dimensions

    Returns
    ----------------
    dists: (N,) float
        Manhattan distance between 'p' and 'q'
    """
    # Vectorize inputs
    if not isinstance(p, type(np.ndarray)):
        p = np.array(p)
    if not isinstance(q, type(np.ndarray)):
        q = np.array(q)

    # Get distance
    if not scale:
        return np.sum(np.abs(q - p))
    else:
        return np.sum(np.abs(scale * (q - p)))


class FireflyOptimizer(object):
    """

    Note
    ----------------
    Position updates of firefly follow bellow govening equation:
        xi(t+1) = xi(t) + beta*exp(-gamma*dist**2)*(xj(t) - xi(t)) + alpha(t)*epsilon(t)
    """

    def __init__(self, limits, alpha=1.0, beta=1.0, gamma=0.01, maxIter=100):
        # Model parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Hyper-parameters
        self.limits = limits
        self.maxIter = maxIter

    def optimize(
        self, fireflies, measure="Euclid", reduction=0.97, tol=1.0e-1
    ):
        """Run Firefly optimizer"""
        # Get input features
        num_firefly = len(fireflies)
        center = np.mean(fireflies, axis=0)

        # Select distance metric method
        if method == "Euclid":
            self.metric = calc_euclidean_distance
        elif method == "Manhattan":
            self.metric = calc_manhattan_distance
        else:
            print("WRONG DISTANCE METRIC IS USED")
            raise ValueError

        # Compute attractiveness
        attractiveness = np.array(
            [-self.metric(firefly, center) for firefly in fireflies]
        )

        # Optimize
        curr_fireflies = fireflies.copy()
        next_fireflies = np.zeros_like(fireflies)
        for n in range(self.maxIter):
            # Communication
            for i, j in np.ndindex((num_firefly, num_firefly)):
                if i == j:  # No self-attraction
                    continue
                elif attractiveness[i] < attractiveness[j]:
                    next_fireflies[i] = self.move(
                        curr_fireflies[i], curr_fireflies[j]
                    )

            # Check convergency
            mse = np.mean(
                np.sqrt(np.sum((curr_fireflies - next_fireflies) ** 2.0))
            )
            if mse < tol:
                break

            # Compute attractiveness
            curr_fireflies = next_fireflies
            attractiveness = np.array(
                [-self.metric(firefly, center) for firefly in curr_fireflies]
            )

            # Do relaxation by applying infinitesimal displacement
            curr_fireflies = self.relax(curr_fireflies, delta=0.1)

            # Reduce randomness
            self.alpha *= reduction

        return curr_fireflies

    def move(self, xi, xj):
        """Move firefly toward attractive one"""
        # Displacement
        dr = self.metric(xi, xj)

        # Attraction
        attraction = self.beta * np.exp(-self.gamma * dr**2.0)

        # Moving step scaling
        scaling = (
            (self.limits[1] - self.limits[0])
            * self.alpha
            * (np.random.random(len(xi)) - 0.5)
        )

        # Update position
        xi += np.clip(
            attraction * (xj - xi) + scaling, self.limits[0], self.limits[1]
        )

        return xi

    def relax(self, fireflies, delta=0.01):
        """Randomly distribute firefiles"""
        return np.array(
            [
                np.random.uniform(firefly - delta, firefly + delta)
                for firefly in fireflies
            ]
        )
