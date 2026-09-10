# -*- coding: utf-8 -*-
"""Kiem THU TU dinh nghia ten bang AST, theo TUNG DONG, xuyen cell.
   - Code chay NGAY (module-level, ke ca trong if/for/with/try): Load phai SAU Store, theo dong.
   - Than ham/class (chay tre): chi can ten duoc dinh nghia o DAU DO trong cell nay hoac truoc.
   Chay: python kiem_thutu_ast.py <notebook.ipynb> [...]"""
import ast, builtins, io, json, sys

BUILTIN = set(dir(builtins)) | {'__name__', '__file__', 'display', 'get_ipython'}
# ten do moi truong Colab/notebook hoac import * cung cap, khong the bat duoc tinh
NGOAI = {'drive', 'files', 'userdata', 'np', 'pd', 'torch', 'nn', 'F', 'tqdm', 'Image', 'T',
         'plt', 'timm', 'AutoModel', 'AutoImageProcessor', 'average_precision_score'}


def _stores(node):
    """Moi ten duoc GAN trong node (ke ca target cua for/with/except/def/class/import)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names: out.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: out.add(a.asname or a.name)
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
    return out


def _loads_ngay(node, trong_ham=False):
    """Yield (ten, dong, trong_ham) cho moi Name Load. Than ham/lambda/class -> trong_ham=True."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        # decorator, default, annotation chay NGAY; than thi chay TRE
        ngay = []
        if not isinstance(node, ast.Lambda):
            ngay += node.decorator_list
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            ngay += a.defaults + a.kw_defaults
        if isinstance(node, ast.ClassDef): ngay += node.bases + node.keywords
        for x in ngay:
            if x is not None: yield from _loads_ngay(x, trong_ham)
        body = node.body if not isinstance(node, ast.Lambda) else [node.body]
        for x in body: yield from _loads_ngay(x, True)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        yield node.id, getattr(node, 'lineno', -1), trong_ham
    # comprehension: bien lap la cuc bo, khong tinh
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        cucbo = set()
        for g in node.generators: cucbo |= _stores(g.target)
        for ten, d, th in _loads_ngay_children(node, trong_ham):
            if ten not in cucbo: yield ten, d, th
        return
    yield from _loads_ngay_children(node, trong_ham)


def _loads_ngay_children(node, trong_ham):
    for ch in ast.iter_child_nodes(node):
        yield from _loads_ngay(ch, trong_ham)


def kiem(path):
    nb = json.load(io.open(path, encoding='utf-8'))
    cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    da_co = set()            # ten dinh nghia o cac cell TRUOC
    loi = []
    for i, src in cells:
        try: tree = ast.parse(src)
        except SyntaxError as e:
            loi.append((i, e.lineno, '<CU PHAP>', str(e))); continue
        co_trong_cell = _stores(tree)                    # moi ten gan o dau do trong cell
        # duyet TUNG cau lenh module-level theo thu tu, cap nhat 'da gan den dong nay'
        gan_den_gio = set(da_co)
        for stmt in tree.body:
            # Load trong cau lenh nay (ke ca ben trong if/for/with/try — chung chay ngay)
            for ten, dong, trong_ham in _loads_ngay(stmt):
                if ten in BUILTIN or ten in NGOAI: continue
                if trong_ham:
                    if ten not in co_trong_cell and ten not in da_co:
                        loi.append((i, dong, ten, 'than ham dung ten KHONG dinh nghia o dau'))
                else:
                    if ten not in gan_den_gio:
                        # co the gan o CHINH cau lenh nay truoc khi dung (vd for x in ...: x)
                        if ten in _stores(stmt) and not isinstance(stmt, (ast.Assign, ast.AugAssign,
                                                                          ast.AnnAssign)):
                            continue
                        loi.append((i, dong, ten, 'dung TRUOC khi gan' +
                                    ('' if ten not in co_trong_cell else ' (gan o dong sau trong CUNG cell)')))
            gan_den_gio |= _stores(stmt)
        da_co |= co_trong_cell
    return loi


if __name__ == '__main__':
    tong = 0
    for p in sys.argv[1:]:
        loi = kiem(p)
        # khu trung theo (cell, ten)
        seen, uniq = set(), []
        for l in loi:
            k = (l[0], l[2])
            if k not in seen: seen.add(k); uniq.append(l)
        print(f'=== {p.split(chr(92))[-1]} — {len(uniq)} van de ===')
        for i, dong, ten, ly in uniq: print(f'  cell {i:>2} dong {dong:>4}  {ten:<22} {ly}')
        tong += len(uniq)
    sys.exit(1 if tong else 0)
