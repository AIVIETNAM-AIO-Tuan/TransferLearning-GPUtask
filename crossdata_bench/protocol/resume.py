# -*- coding: utf-8 -*-
"""Lưu GIỮA giai đoạn cho vòng học `89` Ver2 (0.5.0; PLAN_93 §4.1 — CheXpert (A): một giai đoạn của B ~13–16 giờ, dài hơn một phiên Colab).

vindr_bench @ c81af62: pipeline.run lưu trạng thái SAU giai đoạn (1) dò lr và (2) thêm hạt; giai đoạn (3) không có điểm lưu; train_parallel luôn bắt đầu ở
bước 0. Ở đây — KHÔNG sửa gói của 89 (như protocol/heads bọc run_head) — trong khối `Checkpointer.active()`:
  - pipeline.stage_search / stage_seeds / stage_phase2 được bọc để biết giai đoạn đang chạy ('s1' / 's2' / 's3');
  - pipeline.train_parallel được thay bằng `_train_parallel` dưới đây = BẢN CHÉP của bản gốc; dòng thêm đánh dấu `+RESUME`, dòng đổi `~RESUME` (bộ kiểm:
    bỏ các dòng đánh dấu thì trùng bản gốc từng dòng):
      * nạp trạng thái ĐẦY ĐỦ của mọi đầu (trọng số, AdamW, CosineAnnealingLR, số bước, đường val, loss đang cộng) từ state['dang'] khi điểm lưu thuộc đúng
        giai đoạn + đúng danh sách đầu / số bước / tập ảnh / hạt dữ liệu / lô — lệch -> DỪNG (không chạy lại lặng lẽ);
      * dòng mẫu bắt đầu ở bước đã lưu (InfiniteStream nhận step_from: lô ở bước b chỉ phụ thuộc (hạt dữ liệu, b));
      * lưu (state['dang'] rồi save(state)) khi đã qua `every_s` giây kể từ lần lưu trước, ở ranh giới bước; cuối giai đoạn 3 lưu thêm một lần (học xong
        mà đứt trước khi ghi kết quả thì không phải học lại);
  - `save_stage` (đưa cho pipeline.run làm `save`) bỏ state['dang'] của giai đoạn đã xong trước khi ghi (tệp resume không mang điểm lưu cũ ~1,5 GB).
Chạy tiếp = chạy liền: lô ảnh theo (hạt dữ liệu, bước); tăng cường theo (hạt, lượt hoán vị, ảnh); dropout theo (hạt đầu, bước); AdamW + lịch lr nạp bằng
state_dict; loss đang cộng trong epoch nạp lại (tensor) -> chỉ còn sai số làm tròn GPU (có cả khi chạy liền hai lần). Thời gian chấm val của giai đoạn đang
dở không cộng dồn qua phiên (chỉ ảnh hưởng dòng toc_do / phut_val của JSON)."""
import contextlib
import hashlib
import time

import numpy as np

STAGES = ('s1', 's2', 's3')
EVERY_MIN = 30.0                         # lưu ~30 phút một lần (PLAN_93 §4.1)


class Interrupted(Exception):
    '''Ngắt giả (chỉ dùng ở ô chạy thử / bộ kiểm: stop_after) — trạng thái đã lưu ngay trước đó.'''


def _cpu(obj):
    '''Bản sao trên CPU của mọi tensor trong dict / list lồng nhau (state_dict của đầu / optimizer).'''
    import torch
    if torch.is_tensor(obj):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def export_head(run, curve_fields):
    '''Trạng thái ĐẦY ĐỦ của một đầu đang học (khác pipeline.export_run: có optimizer + lịch lr + loss đang cộng).'''
    import torch
    ls = run.loss_sum
    return dict(ten=run.name, head_type=run.head_type, lr=float(run.lr), seed=int(run.seed), max_steps=int(run.max_steps), step=int(run.step),
                curves={f: [float(v) for v in getattr(run, f)] for f in curve_fields},
                loss_sum=(ls.detach().to('cpu', copy=True) if torch.is_tensor(ls) else float(ls)),
                model=_cpu(run.model.state_dict()), opt=_cpu(run.optimizer.state_dict()), sched=dict(run.scheduler.state_dict()))


def import_head(run, d, device, curve_fields):
    '''Nạp trạng thái export_head vào một đầu vừa dựng lại (cùng cấu hình — kiểm trước).'''
    import torch
    same = (run.name, run.head_type, float(run.lr), int(run.seed), int(run.max_steps)) == (d['ten'], d['head_type'], d['lr'], d['seed'], d['max_steps'])
    assert same, f"diem luu giua giai doan: dau {d['ten']} KHAC dau dung lai {run.name} -> DUNG (ke hoach doi?)"
    run.model.load_state_dict(d['model'])
    run.optimizer.load_state_dict(d['opt'])
    run.scheduler.load_state_dict(d['sched'])
    run.step = int(d['step'])
    for f in curve_fields:
        setattr(run, f, list(d['curves'][f]))
    run.loss_sum = d['loss_sum'].to(device) if torch.is_tensor(d['loss_sum']) else float(d['loss_sum'])


def plan_print(runs, indices, protocol, total_steps, steps_per_epoch, stage):
    '''Vân tay kế hoạch của một lần gọi train_parallel — điểm lưu chỉ nạp khi TRÙNG.'''
    idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return dict(giai_doan=stage, dau=[r.name for r in runs], max_steps=[int(r.max_steps) for r in runs], tong_buoc=int(total_steps),
                buoc_moi_epoch=int(steps_per_epoch), so_anh=int(len(idx)), md5_anh=hashlib.md5(idx.tobytes()).hexdigest(), hat_du_lieu=int(protocol.data_seed),
                lo=int(protocol.batch_size))


class Checkpointer:
    '''state = dict giai đoạn của pipeline.run (CÙNG đối tượng truyền vào run(state=...)); save(state) -> True / False (ghi tệp resume — ở notebook là
       luu_resume của khối Drive chuẩn). every_s: giây giữa hai lần lưu. stop_after = (giai đoạn, bước): lưu rồi ném Interrupted (chạy thử / bộ kiểm).
       plan_steps = {giai đoạn: tổng bước} (để in giờ còn lại).'''

    def __init__(self, state, save, every_s=EVERY_MIN * 60, log=print, clock=time.monotonic, stop_after=None, plan_steps=None, token=None):
        self.state, self._save, self.every_s, self.log, self.clock = state, save, float(every_s), log, clock
        self.stop_after, self.plan_steps, self.token = stop_after, dict(plan_steps or {}), token
        self.stage, self.last, self.t0, self.start = None, clock(), None, 0
        self.saves = []                                          # [dict(giai_doan, buoc, dat, giay)] của phiên này

    # ---- pipeline.run gọi save(state) sau giai đoạn (1) / (2) ----
    def save_stage(self, state):
        d = state.get('dang')
        if d is not None and d['giai_doan'] in state:            # giai đoạn của điểm lưu đã xong -> bỏ (không mang ~1,5 GB thừa)
            state.pop('dang')
        self._record_session(state)
        return self._save(state)

    def _record_session(self, state):
        if self.token is not None:
            ph = state.setdefault('phien', [])
            if not ph or ph[-1].get('token') != self.token:
                ph.append(dict(token=self.token, bat_dau=time.strftime('%Y-%m-%d %H:%M:%S')))
            ph[-1]['ghi_cuoi'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # ---- train_parallel (bản chép) gọi ----
    def begin(self, runs, indices, protocol, total_steps, steps_per_epoch, timing, device, curve_fields):
        assert self.stage in STAGES, 'train_parallel goi ngoai giai doan da boc -> DUNG'
        self.fp = plan_print(runs, indices, protocol, total_steps, steps_per_epoch, self.stage)
        self.total, self.curve_fields, self.t0, self.last = int(total_steps), curve_fields, self.clock(), self.clock()
        d = self.state.get('dang')
        if d is None:
            self.start = 0
            return 0
        if d['giai_doan'] != self.stage:
            assert d['giai_doan'] in self.state, (f"diem luu giua giai doan {d['giai_doan']} nhung pipeline dang chay {self.stage} -> DUNG, gui Claude output C2-C6")
            self.state.pop('dang')                               # điểm lưu của giai đoạn đã xong (đã có kết quả giai đoạn) -> bỏ
            self.start = 0
            return 0
        diff = {k: (d['van_tay'].get(k), v) for k, v in self.fp.items() if d['van_tay'].get(k) != v}
        assert not diff, f'diem luu giua giai doan {self.stage} KHAC ke hoach dang chay {sorted(diff)} -> DUNG (khong chay lai lang le), gui Claude output C2-C6'
        assert len(d['dau']) == len(runs)
        for r, h in zip(runs, d['dau']):
            import_head(r, h, device, curve_fields)
        timing.update(d['toc_do'])
        self.start = int(d['buoc'])
        assert all(r.step == self.start for r in runs), 'diem luu: so buoc cua dau KHAC buoc da luu -> DUNG'
        self.log(f"RESUME giua giai doan {self.stage}: nap {len(runs)} dau tai buoc {self.start:,} / {self.total:,} (luu {d['t']}) -> hoc tiep, KHONG hoc lai phan da xong")
        return self.start

    def _snapshot(self, runs, step, timing):
        return dict(giai_doan=self.stage, buoc=int(step), van_tay=dict(self.fp), toc_do=dict(timing), t=time.strftime('%Y-%m-%d %H:%M:%S'),
                    dau=[export_head(r, self.curve_fields) for r in runs])

    def _eta(self, step):
        el = self.clock() - self.t0
        done = step - self.start
        if done <= 0 or el <= 0:
            return ''
        sps = el / done
        rest = self.total - step
        later = 0
        if self.plan_steps:
            order = list(STAGES)
            later = sum(self.plan_steps.get(g, 0) for g in order[order.index(self.stage) + 1:] if g not in self.state)
        return (f" | toc do DO {sps:.2f} s / buoc | giai doan nay con ~{rest * sps / 3600:.1f} gio"
                + (f", ca luot con ~{(rest + later) * sps / 3600:.1f} gio (UOC theo toc do giai doan nay)" if self.plan_steps else ''))

    def checkpoint(self, runs, step, timing, why):
        t = self.clock()
        self.state['dang'] = self._snapshot(runs, step, timing)
        self._record_session(self.state)
        ok = self._save(self.state)
        dt = self.clock() - t
        self.saves.append(dict(giai_doan=self.stage, buoc=int(step), dat=bool(ok), giay=round(dt, 1)))
        self.log(f"  [luu {why}] giai doan {self.stage} buoc {step:,} / {self.total:,} ({100 * step / max(1, self.total):.0f} %) | len Drive: "
                 f"{'DAT' if ok else 'CHUA (lan luu sau thu lai)'} ({dt:.0f} s)" + self._eta(step))
        self.last = self.clock()
        return ok

    def step_end(self, step, runs, timing):
        if self.stop_after is not None and (self.stage, step) == tuple(self.stop_after):
            self.checkpoint(runs, step, timing, 'ngat thu')
            raise Interrupted(f'ngat thu o {self.stage} buoc {step}')
        if step < self.total and self.clock() - self.last >= self.every_s:
            self.checkpoint(runs, step, timing, 'dinh ky')

    def end(self, runs, timing):
        if self.stage == 's3' and self.total > self.start:      # học xong giai đoạn 3: lưu một lần (dự đoán test + ghi kết quả chưa xảy ra)
            self.checkpoint(runs, self.total, timing, 'cuoi giai doan 3')

    # ---- thay hàm của pipeline trong khối ----
    @contextlib.contextmanager
    def active(self):
        from vindr_bench.protocols.ours import pipeline as PL
        assert not getattr(PL.train_parallel, 'crossdata_resume', False), 'train_parallel da duoc thay (khoi long nhau) -> DUNG'
        names = ('train_parallel', 'stage_search', 'stage_seeds', 'stage_phase2')
        orig = {n: getattr(PL, n) for n in names}
        tp = _make_train_parallel(self, PL)
        PL.train_parallel = tp
        for n, g in zip(names[1:], STAGES):
            setattr(PL, n, _wrap_stage(self, g, orig[n]))
        try:
            yield self
        finally:
            for n in names:
                setattr(PL, n, orig[n])
            self.stage = None


def _wrap_stage(ck, stage, fn):
    def wrapped(*a, **k):
        ck.stage = stage
        try:
            return fn(*a, **k)
        finally:
            ck.stage = None
    wrapped.__wrapped__ = fn
    return wrapped


def _make_train_parallel(ck, PL):
    import torch
    data, training, _sync = PL.data, PL.training, PL._sync

    def train_parallel(runs, backbone, dataset, indices, labels, protocol, total_steps, steps_per_epoch, on_epoch_end=None,
                       snapshot_at=None, snapshot_runs=None, num_workers=0):
        assert snapshot_at is None, 'snapshot_at khong dung o cac giai doan cua pipeline.run -> DUNG'  # +RESUME
        dev = backbone.device
        y_all = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=dev)
        timing, snapshot = dict(cho=0.0, dac=0.0, dau=0.0, anh=0), {}
        start = ck.begin(runs, indices, protocol, total_steps, steps_per_epoch, timing, dev, PL.CURVE_FIELDS)  # +RESUME
        loader = data.train_loader(dataset, indices, protocol.data_seed, start, total_steps, protocol.batch_size, num_workers, pin_memory=(dev == 'cuda'))  # ~RESUME
        t_wait = time.time()
        for s, (x, j) in enumerate(loader, start):  # ~RESUME
            timing['cho'] += time.time() - t_wait
            t1 = time.time()
            x = x.to(dev, non_blocking=True)
            y = y_all[j.to(dev, non_blocking=True)]
            blocks, last = backbone.features(x)
            _sync(dev)
            t2 = time.time()
            timing['dac'] += t2 - t1
            for run in runs:
                if run.active: training.train_step(run, blocks, last, y, protocol.loss_divisor)
            del blocks, last
            _sync(dev)
            timing['dau'] += time.time() - t2
            timing['anh'] += len(j)
            if snapshot_at is not None and s + 1 == snapshot_at:
                snapshot = {k: {n: v.detach().clone() for n, v in r.model.state_dict().items()} for k, r in (snapshot_runs or {}).items()}
            if (s + 1) % steps_per_epoch == 0:
                epoch = (s + 1) // steps_per_epoch
                done = [r for r in runs if r.max_steps >= epoch * steps_per_epoch]
                for r in done: training.close_epoch(r, steps_per_epoch)
                if on_epoch_end is not None: on_epoch_end(epoch, done)
            ck.step_end(s + 1, runs, timing)  # +RESUME
            t_wait = time.time()
        ck.end(runs, timing)  # +RESUME
        return timing, snapshot
    train_parallel.crossdata_resume = True
    return train_parallel
