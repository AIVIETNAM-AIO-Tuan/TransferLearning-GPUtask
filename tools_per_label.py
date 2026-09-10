# -*- coding: utf-8 -*-
"""Bang prevalence + baseline TUNG NHAN.
   prevalence: doc thang tu labels.csv trong zip (nguon chuan, ca 3 split)
   AP:         tu 33 C6 (cls/patch, arm nosq518) va 33 C7 (ehr, fusion) tren VAL"""
import zipfile, io, os, sys
import numpy as np, pandas as pd

ROOT = r'c:\Users\Admin\Working_space\TransferLearning&GPUtask\TransferLearning&GPUtask'
Z = zipfile.ZipFile(os.path.join(ROOT, 'cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0.zip'))
B = 'cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0/cxr-lt-2024'
ID = ['dicom_id','subject_id','study_id','ViewPosition','ViewCodeSequence_CodeMeaning','fpath','split']

LAB40 = [c for c in pd.read_csv(Z.open(f'{B}/development_labeled_task1.csv'), nrows=0).columns
         if c not in ID]
LAB26 = set(c for c in pd.read_csv(Z.open(f'{B}/development_labeled_task2.csv'), nrows=0).columns
            if c not in ID)

F = pd.read_csv(Z.open(f'{B}/labels.csv'), usecols=['split'] + LAB40)
N = F.groupby('split')[LAB40].agg(['sum', 'size'])

# --- AP tren VAL, arm nosq518 (33 C6 + C7) ---
# cot: cls, patch(=cxr), ehr, fusion
AP = {
 'Subcutaneous Emphysema':(.3101,.5407,.0569,.5372), 'Hernia':(.4065,.5341,.0388,.5320),
 'Pneumoperitoneum':(.0024,.1110,.0022,.1000),       'Kyphosis':(.0354,.1215,.0116,.1227),
 'Hydropneumothorax':(.0092,.0935,.0122,.0964),      'Emphysema':(.1606,.2355,.0713,.2345),
 'Pneumomediastinum':(.0111,.0854,.0094,.0855),      'Pneumothorax':(.3782,.4507,.2146,.4717),
 'Mass':(.1142,.1861,.0588,.2001),                   'Calcification of the Aorta':(.0470,.0948,.0733,.0971),
 'Tuberculosis':(.0197,.0638,.0875,.0903),           'Fibrosis':(.0242,.0633,.0204,.0630),
 'Fissure':(.0244,.0577,.0162,.0579),                'Tortuous Aorta':(.0319,.0648,.0389,.0642),
 'Fracture':(.0845,.1165,.0712,.1299),               'Nodule':(.0691,.0995,.0599,.1032),
 'Pleural Thickening':(.0335,.0636,.0239,.0622),     'Normal':(.2963,.3263,.2515,.3264),
 'Pleural Effusion':(.7612,.7893,.5580,.7919),       'Edema':(.4915,.5180,.3715,.5244),
 'Rib Fracture':(.0543,.0790,.0656,.0962),           'Support Devices':(.8441,.8644,.7008,.8647),
 'Adenopathy':(.0280,.0467,.0259,.0488),             'Enlarged Cardiomediastinum':(.1415,.1599,.1377,.1645),
 'Consolidation':(.1588,.1770,.1278,.1824),          'Lung Lesion':(.0172,.0346,.0312,.0489),
 'Cardiomegaly':(.6258,.6410,.4748,.6412),           'Azygos Lobe':(.0041,.0177,.0004,.0197),
 'Infiltration':(.0434,.0562,.0563,.0622),           'Pneumonia':(.2674,.2788,.2524,.2981),
 'Pulmonary Hypertension':(.0043,.0148,.0054,.0127), 'Atelectasis':(.5435,.5526,.4084,.5523),
 'Granuloma':(.0120,.0164,.0166,.0173),              'Pleural Other':(.0033,.0074,.0034,.0076),
 'Lung Opacity':(.5304,.5341,.4171,.5439),           'Pulmonary Embolism':(.0084,.0100,.0148,.0144),
 'Round(ed) Atelectasis':(.0012,.0026,.0034,.0040),  'Clavicle Fracture':(.0006,.0010,.0012,.0009),
 'Lobar Atelectasis':(.0007,.0008,.0011,.0007),      'Infarction':(.0024,.0026,.0040,.0028),
}
_miss = set(LAB40) - set(AP)
assert not _miss, f'thieu AP cho: {sorted(_miss)}'
assert len(AP) == 40, len(AP)

rows = []
for l in LAB40:
    cls, patch, ehr, fus = AP[l]
    ntr, nva, nte = int(N.loc['train',(l,'sum')]), int(N.loc['val',(l,'sum')]), int(N.loc['test',(l,'sum')])
    Ntr, Nva = int(N.loc['train',(l,'size')]), int(N.loc['val',(l,'size')])
    prev = nva / Nva
    rows.append(dict(
        nhan=l, tier='head' if ntr >= 10000 else ('medium' if ntr >= 1000 else 'tail'),
        t2='x' if l in LAB26 else '',
        n_train=ntr, n_val=nva, n_test=nte,
        prev_train=ntr/Ntr, prev_val=prev, neg_pos=(Ntr-ntr)/max(ntr,1),
        cls=cls, patch=patch, ehr=ehr, fusion=fus,
        patch_tru_cls=patch-cls, lift_ehr=fus-patch,
        boi_so_san=patch/prev))
D = pd.DataFrame(rows)

pd.set_option('display.width', 250)
f4 = lambda v: f'{v:.4f}'

print('=' * 128)
print('BANG 1 — PREVALENCE + BASELINE TUNG NHAN   (arm nosq518, mAP muc ANH tren val 39,293 anh)')
print('=' * 128)
print(D.sort_values('prev_val', ascending=False)[
      ['nhan','tier','t2','n_train','n_val','prev_val','neg_pos','cls','patch','ehr','fusion',
       'patch_tru_cls','lift_ehr','boi_so_san']]
      .to_string(index=False, float_format=f4))

print('\n' + '=' * 128)
print('BANG 2 — XEP THEO BOI SO SAN  (patch / prevalence) — do "mo hinh hon doan bua bao nhieu lan"')
print('=' * 128)
print(D.sort_values('boi_so_san', ascending=False)[
      ['nhan','tier','n_val','prev_val','patch','boi_so_san','lift_ehr']]
      .to_string(index=False, float_format=f4))

print('\n' + '=' * 128)
print('TONG HOP THEO TIER')
print('=' * 128)
g = D.groupby('tier').agg(
        n_nhan=('nhan','size'), n_val_min=('n_val','min'), n_val_max=('n_val','max'),
        prev_val=('prev_val','mean'), cls=('cls','mean'), patch=('patch','mean'),
        ehr=('ehr','mean'), fusion=('fusion','mean'),
        patch_tru_cls=('patch_tru_cls','mean'), lift_ehr=('lift_ehr','mean'),
        boi_so_san=('boi_so_san','mean')).reindex(['head','medium','tail'])
print(g.to_string(float_format=f4))
print(f"\nTOAN BO 40 nhan: prevalence TB {D.prev_val.mean():.4f} | mAP patch {D.patch.mean():.4f} "
      f"| boi so san {D.patch.mean()/D.prev_val.mean():.2f}x")
print(f"  cls {D.cls.mean():.4f} | ehr {D.ehr.mean():.4f} | fusion {D.fusion.mean():.4f}")
print(f"  patch - cls {D.patch_tru_cls.mean():+.4f}  ({int((D.patch_tru_cls>0).sum())}/40 duong)")
print(f"  lift EHR    {D.lift_ehr.mean():+.4f}  ({int((D.lift_ehr>0).sum())}/40 duong, "
      f"{int((D.lift_ehr>0.005).sum())}/40 vuot nguong nhieu)")

print('\n' + '=' * 128)
print('MAT CAN BANG: hai loai, khong duoc gop')
print('=' * 128)
_mx, _mn = D.loc[D.n_train.idxmax()], D.loc[D.n_train.idxmin()]
print(f"  GIUA cac nhan : {_mx.nhan} {_mx.n_train:,} / {_mn.nhan} {_mn.n_train:,} = "
      f"{_mx.n_train/_mn.n_train:.0f}x  (train)")
print(f"  TRONG tung nhan: am:duong tu {D.neg_pos.min():.1f}:1 ({_mx.nhan}) "
      f"den {D.neg_pos.max():.0f}:1 ({_mn.nhan})")
print(f"  SAN AP        : tu {D.prev_val.max():.4f} ({_mx.nhan}) "
      f"den {D.prev_val.min():.5f} ({D.loc[D.prev_val.idxmin(),'nhan']}) — chenh "
      f"{D.prev_val.max()/D.prev_val.min():.0f}x")

out = os.path.join(ROOT, 'per_label_prevalence_baseline.csv')
D.sort_values('prev_val', ascending=False).to_csv(out, index=False, float_format='%.6f')
print(f'\n-> ghi {out}')
