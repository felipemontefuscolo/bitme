import numpy as np
def solve(x, y, inv=np.linalg.pinv):
    """
    Nt = number of training
    Nx = number of features
    dim(x) = (Nx, Nt)
    dim(y) = (Ny, Nt)
    returns A* and b* such minimize J(A,b) = |Ax + b - y|
    """
    assert x.shape[1] == y.shape[1]
    Nx = x.shape[0]
    Ny = y.shape[0]
    Nt = x.shape[1]
    xm = x.mean(axis=1).reshape(Nx,1)
    ym = y.mean(axis=1).reshape(Ny,1)
    dx = x - xm
    dy = y - ym
    A = dy.dot(inv(dx))  # can be optimized
    b = ym - A.dot(xm)
    assert b.shape[0] == Ny
    return A, b

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def solve_logistic(x, y, learning_rate=0.1, max_iters=10000, tol=1.e-6):
    """
    Nt = number of training
    Nx = number of features
    dim(x) = (Nx, Nt)
    dim(y) = (Ny, Nt)
    returns A* and b* such minimize
    J(A,b) = -y * log(sigmoid(A*x + b)) - (1 - y)*log(1 - sigmoid(A*x + b))
    """
    assert x.shape[1] == y.shape[1]
    Nx = x.shape[0]
    Ny = y.shape[0]
    Nt = x.shape[1]
    A = np.zeros(Nx * Ny).reshape(Ny, Nx)
    b = np.zeros(Ny).reshape(Ny, 1)
    ones = np.ones(Nt).reshape(Nt, 1)
    r = []
    k = 0
    while (not r or r[-1] > tol) and k < max_iters:
        R = sigmoid(A.dot(x) + b.dot(ones.transpose())) - y
        dA = R.dot(x.transpose())
        A -= learning_rate * dA
        R = sigmoid(A.dot(x) + b.dot(ones.transpose())) - y
        db = R.dot(ones)
        b -= learning_rate * db
        r.append(np.linalg.norm(dA) + np.linalg.norm(db))
        k += 1
    return A, b, r


def get_A_init(x, n=None):
    """
    Get initial state for RELU layer
    x.shape() = (# features, # training)
    """
    if n is None:
        n = 2 * x.shape[0]
    box = np.array([x.min(axis=1), x.max(axis=1)]).transpose()
    assert box.shape == (x.shape[0], 2)
    k = 0
    A = []
    b = []
    n_boxes = n // (2 * x.shape[0])
    delta = .5 * (box[:, 1] - box[:, 0]) / n_boxes;
    while True:
        for side in range(x.shape[0]):
            for direction in (1,-1):
                if k >= n:
                    break
                k += 1
                A.append(np.zeros(x.shape[0]))
                A[-1][side] = float(direction)
                idx = -(direction - 1) // 2
                b.append(-direction * box[side][idx])
        if k >= n:
            break
        assert(all(delta >= 0.))
        box[:,0] += delta
        box[:,1] -= delta
        assert(all(box[:, 1] - box[:, 0] >= 0.))

    return np.array(A), np.array(b).reshape(n,1)

