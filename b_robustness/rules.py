"""Luật P2 (đặt trước, Tuấn duyệt 03/11; mốc B 06/11; = _luat_g của 85g). Đầu ML-Decoder, mAP 21 lớp:
B − X >= +0,01, CI95 > 0, > 0 ở cả hai nửa -> "B HON"; <= −0,01 và CI95 < 0 -> "B KEM HON"; còn lại "KHONG PHAN BIET".
Ở K3 luật chỉ dùng làm phán xử DẠNG LUẬT cho từng seed (mô tả); luật chính thức vẫn là B s42 ở 85g."""
from . import config


def p2_verdict(b, x, ci, threshold=config.RULE_THRESHOLD):
    '''b, x = tóm tắt (metrics.summary) đầu ML-Decoder 21 lớp của B và X; ci = CI95 ghép cặp mAP B − X. -> (chênh, ci, kết luận).'''
    dl = {f: b[f] - x[f] for f in ('ap', 'ap_A', 'ap_B', 'auc')}
    if dl['ap'] >= threshold and ci[0] > 0 and dl['ap_A'] > 0 and dl['ap_B'] > 0: v = 'B HON'
    elif dl['ap'] <= -threshold and ci[1] < 0: v = 'B KEM HON'
    else: v = 'KHONG PHAN BIET'
    return dl, ci, v
