import numpy as np

# *************** acc *************** #

def accuracy(preds, labels):
    correct = (preds == labels).sum()
    acc = correct / len(labels)
    return acc.item() if hasattr(acc, 'item') else float(acc)

def accuracy_with_tolerance(preds, labels, tolerance: int = 10) -> float:
    return np.mean(np.abs(preds - labels) <= tolerance)

def kendall_tau(x, y):
    n = len(x)
    assert len(y) == n
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_x = np.sign(x[i] - x[j])
            sign_y = np.sign(y[i] - y[j])
            if sign_x * sign_y > 0:
                concordant += 1
            elif sign_x * sign_y < 0:
                discordant += 1
            # if sign_x * sign_y == 0: ties – ignored
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom

def smape(pred_lens, real_lens):
    relative_error = np.mean(2 * np.abs(pred_lens - real_lens) / (np.abs(pred_lens) + np.abs(real_lens) + 1e-8))
    return relative_error
