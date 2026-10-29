"""Read-only CPU audit: actual 84/old-C reader code, 4 versus 5 heads.

No data, model downloads, backbone execution or training runs. Only synthetic
frozen features and three optimizer updates at the reader seam. Product source
is extracted with AST rather than reimplemented. The fifth head is the old-C
six-crop reader, with its actual forward and dropout behavior.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import types

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
STEPS = HERE / "nhanh_c" / "buoc"
OLD = HERE.parent / "78_chi_anh" / "78x_jca_v3b_dp_ema_v2_o_s42.ipynb"
COMMON = ("mldec", "mldec_hr", "fus_tb_hr", "fus_attn_hr")


def src(path):
    return path.read_text(encoding="utf-8")


def definition(code, name):
    nodes = [n for n in ast.parse(code).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    return ast.get_source_segment(code, nodes[0])


def load_defs(ns, code, names):
    for name in names:
        exec(compile(definition(code, name), f"<actual:{name}>", "exec"), ns)


def old_part2():
    cells = json.loads(OLD.read_text(encoding="utf-8"))["cells"]
    start = next(i for i, c in enumerate(cells)
                 if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# Phần 2"))
    end = next(i for i, c in enumerate(cells[start + 1:], start + 1)
               if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# Phần 3"))
    return ["".join(c["source"]) for c in cells[start:end] if c["cell_type"] == "code"]


def containing(cells, name):
    found = [s for s in cells if any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
                                    for n in ast.parse(s).body)]
    assert found, name
    # Executed notebook includes an explicit later C6 replay after DO_TEST=False.
    # Its relevant definitions and train loop are identical to the first copy.
    values = [definition(s, name) for s in found]
    assert all(v == values[0] for v in values), name
    return found[0]


def train_nodes(code):
    seed = next(n for n in ast.parse(code).body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "seed")
    epoch = next(n for n in seed.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "ep")
    batch = next(n for n in epoch.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Tuple) and n.target.elts[0].id == "j")
    arms = next(n for n in batch.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "t"
                and isinstance(n.iter, ast.Name) and n.iter.id == "ARMS")
    clip = next(n for n in batch.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare)
                and "ACCUM" in ast.unparse(n.test))
    init_names = {"_P_BERT", "_ID_BERT", "PARAM_OF", "rest", "ftpp", "opt", "steps", "warm", "sch"}
    init = [n for n in seed.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in init_names
                    for target in n.targets for t in ast.walk(target))]
    return init, arms, clip


def exec_nodes(ns, nodes):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                 "<actual:train-loop>", "exec"), ns)


def selected_arm_configuration(ns, c0, c4, five):
    """Evaluate only actual declarative arm-selection statements, not C0 I/O."""
    def selected_assignments(code, names):
        return [n for n in ast.parse(code).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]

    ns.update(R_CAO=0.3, DICH_TAT=())
    exec_nodes(ns, selected_assignments(c0, {"TEN_ARM_78T", "ARMS_GIU"}))
    names = {"_dich_bat", "_TAT_CA_DICH", "_ARMS_78T", "_GIU", "ARMS", "PHAN",
             "PHU_CFG", "ARMS_CHAM", "DOI_CHUNG", "CO_CAU", "CO_CLIP"}
    nodes = selected_assignments(c4, names)
    assert all(n.lineno < next(n.lineno for n in ast.parse(c4).body
                               if isinstance(n, ast.FunctionDef) and n.name == "build_arms") for n in nodes)
    exec_nodes(ns, nodes)
    expected = COMMON + (("fus_attn_hr_o",) if five else ())
    assert tuple(ns["ARMS"]) == expected, tuple(ns["ARMS"])
    assert ns["PHU_CFG"] == {}, ns["PHU_CFG"]
    assert not ns["CO_CAU"] and not ns["CO_CLIP"]
    assert all(not c.get("o") for t, c in ns["ARMS"].items() if t in COMMON)
    if five:
        assert ns["ARMS"]["fus_attn_hr_o"]["o"] is True
        assert ns["ARMS"]["fus_attn_hr_o"]["hr"] is True
        assert ns["ARMS"]["fus_attn_hr_o"]["fus"] == "attn"


def model_namespace(five):
    ns = dict(torch=torch, nn=nn, F=F, math=math, copy=copy, dev="cpu", SEED=43,
              C=40, D_CXR=32, D_MLD=32, D_ETOK=8, SO_O=5,
              LUONG=types.SimpleNamespace(R=4), CFG_FS={"n_view": 3})
    ns["_FORK_78Y"] = lambda: torch.random.fork_rng(devices=[])
    old = old_part2()
    c4 = containing(old, "build_arms") if five else src(STEPS / "z_C4.py")
    c6 = containing(old, "arm_forward_phu") if five else src(STEPS / "p2_C6.py")
    c0 = next(s for s in old if "TEN_ARM_78T =" in s) if five else src(STEPS / "p2_C0.py")
    load_defs(ns, src(STEPS / "z_c3.py"), ("GroupFC", "XAttn", "MLDecoder", "ASL"))
    load_defs(ns, src(STEPS / "z_C2B.py"), ("DocVung", "mldec_h_hr", "gop_view_o"))
    load_defs(ns, src(STEPS / "p2_C0.py"), ("GopStudy", "rut_giu_view"))
    load_defs(ns, src(STEPS / "z_C3C.py"), ("q_cua", "mldec_h"))
    load_defs(ns, c4, ("build_arms",))
    load_defs(ns, c6, ("logit_fus", "arm_forward_phu"))
    selected_arm_configuration(ns, c0, c4, five)
    torch.manual_seed(43000)
    ns["MOD"], ns["ETK"], ns["IMG"], ns["UPF"] = ns["build_arms"](43)
    ns["MOD"].train()
    ns.update(PHU=nn.ModuleDict(), seed=43, LR=0.001, WD=0.01,
              LR_FT_DIV=10, LR_THAP=1e-5, I_TF=list(range(48)), BATCH=6,
              ACCUM=1, EPOCHS=1, WARMUP=0.25, CLIP=0.05)
    init, arms, clip = train_nodes(c6)
    exec_nodes(ns, init)
    ns.update(crit=ns["ASL"](), N_STEP=0,
              N_CLIP={t: 0 for t in ns["ARMS"]}, N_SKIP={t: 0 for t in ns["ARMS"]},
              BO_CAT_PHU={}, CAT_PHU={}, CO_CAU=False, _hs=1.0,
              tot={t: 0.0 for t in ns["ARMS"]}, N_PHU={}, PHU_BO={},
              PHAN=("vung", "dich", "cau"), bad=[], n_lui=0,
              GEN_FUS=torch.Generator().manual_seed(43031),
              ST_FUS_T=torch.tensor([10, 10, 20, 20, 30, 40]),
              VIEW_FUS_T=torch.tensor([0, 1, 0, 2, 1, 2]))
    return ns, arms, clip


def parameters(ns):
    return {f"{t}.{k}": p for t in COMMON for k, p in ns["MOD"][t].named_parameters()}


def snapshot(ns):
    return {k: p.detach().clone() for k, p in parameters(ns).items()}


def eq_dict(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def run(five, leak_rng=False):
    ns, arms, clip = model_namespace(five)
    initial = snapshot(ns)
    init_rng = torch.get_rng_state().clone()
    g = torch.Generator().manual_seed(991)
    pf = torch.randn(6, 9, 32, generator=g)
    pf._r_vung = torch.randn(6, 4, 32, generator=g)
    pf._rows = torch.arange(6)
    pf._o11 = torch.randn(6, 5, 9, 32, generator=g)
    y = torch.randint(0, 2, (6, 40), generator=g).float()
    ns.update(pf=pf, hl=None, y=y)
    records = []
    captured = {}
    outputs = {}

    def step_pre_hook(opt, args, kwargs):
        captured.clear()
        captured.update({k: p.grad.detach().clone() for k, p in parameters(ns).items() if p.grad is not None})
        for tag in COMMON:
            values = [v for k, v in captured.items() if k.startswith(tag + ".")]
            assert values and all(torch.isfinite(v).all() for v in values), (tag, "missing/nonfinite gradients")
            assert any(torch.count_nonzero(v) for v in values), (tag, "all-zero gradients")

    ns["opt"].register_step_pre_hook(step_pre_hook)
    for tag in COMMON:
        def output_hook(module, args, output, tag=tag):
            slot = f"{tag}.{sum(k.startswith(tag + '.') for k in outputs)}"
            outputs[slot] = output.detach().clone()
        ns["MOD"][tag].out.register_forward_hook(output_hook)
    for j in range(3):
        outputs.clear()
        ns.update(j=j, n_lui=0, bad=[])
        ns["opt"].zero_grad(set_to_none=True)
        pf._giu = ns["rut_giu_view"](ns["ST_FUS_T"][pf._rows], ns["GEN_FUS"], 0.2)
        exec_nodes(ns, [arms])
        if leak_rng:
            torch.rand(1)  # Deliberate fifth-arm RNG leak negative control.
        exec_nodes(ns, [clip])
        # Exercise a real next-iterator shuffled DataLoader (same common RNG).
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(15)),
                                             batch_size=5, shuffle=True, num_workers=0)
        order = torch.cat([b[0] for b in loader])
        moments = {f"{k}.{m}": v.clone() for k, p in parameters(ns).items()
                   for m, v in ns["opt"].state.get(p, {}).items() if torch.is_tensor(v)}
        records.append(dict(logits=copy.deepcopy(outputs), grads=copy.deepcopy(captured), state=snapshot(ns), adam=moments,
                            rng=torch.get_rng_state().clone(), gen=ns["GEN_FUS"].get_state().clone(),
                            order=order, giu=pf._giu.clone()))
    for tag in COMMON:
        assert any(not torch.equal(v, records[-1]["state"][k])
                   for k, v in initial.items() if k.startswith(tag + ".")), (tag, "no parameter update")
        assert all(any(k.startswith(tag + ".") for k in record["logits"]) for record in records), (tag, "no outputs")
    return dict(initial=initial, init_rng=init_rng, records=records)


def compare(a, b):
    assert eq_dict(a["initial"], b["initial"]), "initial weights differ"
    assert torch.equal(a["init_rng"], b["init_rng"]), "post-build RNG differs"
    for j, (x, y) in enumerate(zip(a["records"], b["records"]), 1):
        for k in ("logits", "grads", "state", "adam"):
            assert eq_dict(x[k], y[k]), (j, k)
        for k in ("rng", "gen", "order", "giu"):
            assert torch.equal(x[k], y[k]), (j, k)


def main():
    torch.set_num_threads(2)
    a, b = run(False), run(True)
    compare(a, b)
    assert any(not torch.equal(v, a["records"][-1]["state"][k]) for k, v in a["initial"].items()), "no actual update"
    try:
        compare(a, run(True, leak_rng=True))
    except AssertionError as e:
        negative = str(e)
    else:
        raise AssertionError("RNG leak negative control was not detected")
    print(json.dumps(dict(result="PASS", torch=torch.__version__, device="cpu", dropout=0.1,
                          dimensions=dict(batch=6, classes=40, hidden=32, patches=9, regions=4, crops=5),
                          optimizer_calls=3, nonzero_lr_updates=2, common_heads=list(COMMON), tensors_checked=len(a["initial"]),
                          arm_configuration="actual C0/C4 assignments; PHU_CFG empty; fifth crop flag verified",
                          nonvacuity="each common head has finite nonzero gradients, outputs, and changed parameters",
                          compared=["initial weights", "post-build RNG", "GroupFC outputs", "postclip gradients", "AdamW state",
                                    "updated weights", "common RNG", "view generator/mask", "DataLoader shuffle"],
                          negative_control_detected=negative,
                          old_notebook_md5=hashlib.md5(OLD.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
