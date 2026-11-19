# -*- coding: utf-8 -*-
"""Hai đầu đọc KHÔNG có bước chọn siêu tham số theo dữ liệu chấm (PLAN_92 §11.2) — numpy / scipy, CPU, tất định.

  k-NN      cosine trên đặc trưng chuẩn hoá L2; điểm = TB nhãn của k láng giềng gần nhất (như knn_val của 78r, không trọng số)
  logistic  một-đấu-phần-còn-lại, mục tiêu ½‖w_c‖² + C·Σ_i loss(z_ic, y_ic) cho từng lớp c (quy ước sklearn; KHÔNG phạt hệ số chặn) — lồi chặt theo w,
            nghiệm duy nhất; giải tới hội tụ bằng L-BFGS trên tổng các lớp (tách rời theo lớp -> cùng nghiệm với giải từng lớp; tests/ so với sklearn).
            Lớp không có ca dương (hoặc toàn dương) trong lần rút: không có nghiệm hữu hạn -> điểm hằng = tỉ lệ dương của lần rút."""
import numpy as np
from scipy.optimize import minimize

from . import config as C


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn_scores(Xtr, Ytr, Xte, k, chunk=1024):
    '''Xtr [n, D], Ytr [n, L] (0 / 1), Xte [m, D] -> điểm [m, L] float32 = TB nhãn của k láng giềng cosine. k > n -> dừng (luật đặt trước).'''
    Xtr, Xte = l2_normalize(Xtr), l2_normalize(Xte)
    Ytr = np.asarray(Ytr, dtype=np.float32)
    n, k = len(Xtr), int(k)
    assert 1 <= k <= n, f'k-NN: k {k} khong hop le voi {n} anh hoc -> DUNG'
    out = np.empty((len(Xte), Ytr.shape[1]), dtype=np.float32)
    for i in range(0, len(Xte), chunk):
        S = Xte[i:i + chunk] @ Xtr.T
        top = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < n else np.broadcast_to(np.arange(n), (len(S), n))
        out[i:i + chunk] = Ytr[top].mean(1)
    return out


def standardizer(Xpool):
    '''TB / SD theo chiều của BỂ học (không dùng nhãn) -> (mu, sd) float64; sd kẹp >= 1e-6.'''
    X = np.asarray(Xpool, dtype=np.float64)
    return X.mean(0), np.maximum(X.std(0), 1e-6)


def _objective(theta, X, Y, c_reg, n):
    '''Mục tiêu CHIA n (cùng nghiệm): [½‖W‖² + C Σ softplus(z) − y z] / n và gradient. theta = [W (D x K) ; b (K)].'''
    D, K = X.shape[1], Y.shape[1]
    W, b = theta[:D * K].reshape(D, K), theta[D * K:]
    Z = X @ W + b
    loss = np.maximum(Z, 0) + np.log1p(np.exp(-np.abs(Z))) - Y * Z
    f = (0.5 * float((W * W).sum()) + c_reg * float(loss.sum())) / n
    G = c_reg * (0.5 * (1.0 + np.tanh(0.5 * Z)) - Y)                 # sigmoid ổn định số
    gW, gb = (X.T @ G + W) / n, G.sum(0) / n
    return f, np.concatenate([gW.ravel(), gb])


def fit_logistic(Xs, Y, c_reg=C.LOGISTIC_C, options=None):
    '''Xs [n, D] đã chuẩn hoá (float64), Y [n, L] (0 / 1). -> dict(W, b, lop_hoc, hang, so_vong, grad_max, hoi_tu, thong_bao).'''
    Xs, Y = np.asarray(Xs, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    n, D = Xs.shape
    pos = Y.sum(0)
    active = np.flatnonzero((pos > 0) & (pos < n))
    const = (pos / n).astype(np.float64)                              # điểm hằng cho lớp không học được
    out = dict(lop_hoc=active, hang=const, W=np.zeros((D, len(active))), b=np.zeros(len(active)), so_vong=0, grad_max=0.0, hoi_tu=True, thong_bao='-')
    if len(active) == 0:
        return out
    Ya = Y[:, active]
    p0 = np.clip(Ya.mean(0), 1e-6, 1 - 1e-6)
    theta0 = np.concatenate([np.zeros(D * len(active)), np.log(p0 / (1 - p0))])   # khởi đầu: W = 0, b = logit tỉ lệ dương (không đổi nghiệm)
    opt = dict(C.LBFGS, **(options or {}))
    res = minimize(_objective, theta0, args=(Xs, Ya, float(c_reg), n), jac=True, method='L-BFGS-B', options=opt)
    _, g = _objective(res.x, Xs, Ya, float(c_reg), n)
    gmax = float(np.abs(g).max())
    out.update(W=res.x[:D * len(active)].reshape(D, len(active)), b=res.x[D * len(active):], so_vong=int(res.nit), grad_max=gmax,
               hoi_tu=bool(res.success or gmax <= C.GRAD_OK), thong_bao=str(res.message))
    return out


def logistic_scores(model, Xs_te):
    '''-> điểm [m, L] float32: sigmoid ở lớp đã học, hằng ở lớp không học được.'''
    Xs_te = np.asarray(Xs_te, dtype=np.float64)
    out = np.tile(model['hang'][None, :], (len(Xs_te), 1))
    if len(model['lop_hoc']):
        Z = Xs_te @ model['W'] + model['b']
        out[:, model['lop_hoc']] = 0.5 * (1.0 + np.tanh(0.5 * Z))
    return out.astype(np.float32)
