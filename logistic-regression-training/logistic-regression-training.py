import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _model(X: np.ndarray, w: np.ndarray, b: np.float) -> np.ndarray:
    """
    Logistic Regression model
    """
    return _sigmoid(X @ w +b)
    
def _loss(X: np.ndarray, y: np.ndarray, w:np.ndarray, b: float):
    p = _model(X, w, b)
    loss = np.sum(y * np.log(p) + (1-y)*np.log(1-p))/-y.size
    return loss 

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[-1]) # What are better values? 
    b = 0

    for _ in range(steps):
        l = _loss(X, y, w, b)
        p = _model(X, w, b)
        w = w - lr * (X.T @ (p - y))/y.size
        b = b - lr * np.sum((p - y))/y.size
    return (w, b)