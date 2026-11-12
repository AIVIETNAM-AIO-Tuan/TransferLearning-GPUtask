"""Tệp chạy dưới torchrun cho `linear_glori.py` của CheXFound NGUYÊN VĂN (PLAN_86 §12.2–12.3). Cần PYTHONPATH chứa repo CheXFound (commit ghim) và
thư mục gói vindr_protocol. Chạy: torchrun --nproc_per_node=1 <đường dẫn tệp này> <tham số y như linear_glori.py>.

Chỉ làm hai việc trước khi gọi main() của họ (không sửa tệp nào của họ; hai hàm được gọi theo tên trong module nên bọc từ ngoài là đủ):
  1. đăng ký tên dataset 'VinDr': chexfound.data.loaders._parse_dataset_str được bọc — tên 'VinDr' -> vindr_dataset.parse_dataset_str, tên khác ->
     hàm gốc (make_dataset của họ tra hàm này theo tên trong module lúc gọi);
  2. ở --infer: bọc linear_glori.evaluate_linear_classifiers_preds để ghi preds / target của test ra <output-dir>/du_doan_test.npz (run_infer_linear
     của họ chỉ lưu cho CheXpert / PLCO); phép tính không đổi, giá trị trả về giữ nguyên."""
import os
import sys

PRED_FILE = 'du_doan_test.npz'


def register_vindr():
    '''Bọc chexfound.data.loaders._parse_dataset_str: 'VinDr' -> lớp của ta; tên khác -> hàm gốc của họ. Trả hàm gốc (để kiểm).'''
    import chexfound.data.loaders as loaders

    from vindr_protocol.upstream import vindr_dataset
    original = loaders._parse_dataset_str
    if getattr(original, '_vindr_wrapped', False):
        return original
    def parse(dataset_str):
        if dataset_str.split(':')[0] == vindr_dataset.NAME:
            return vindr_dataset.parse_dataset_str(dataset_str)
        return original(dataset_str)
    parse._vindr_wrapped = True
    loaders._parse_dataset_str = parse
    return original


def capture_test_predictions(linear_glori, out_path):
    '''Bọc linear_glori.evaluate_linear_classifiers_preds: gọi hàm gốc, ghi preds (xác suất sigmoid của họ) + target ra out_path, trả kết quả gốc.'''
    import numpy as np
    original = linear_glori.evaluate_linear_classifiers_preds

    def wrapped(*args, **kwargs):
        res = original(*args, **kwargs)
        np.savez_compressed(out_path, preds=res['preds'].float().cpu().numpy(), target=res['target'].cpu().numpy())
        return res
    linear_glori.evaluate_linear_classifiers_preds = wrapped
    return original


def main(argv=None):
    register_vindr()
    from chexfound.eval.classification import linear_glori
    args = linear_glori.get_args_parser(description='DINOv2 linear evaluation').parse_args(argv)
    if not args.infer:
        return linear_glori.main(args)
    pred_path = os.path.join(args.output_dir, PRED_FILE)
    if os.path.exists(pred_path): os.remove(pred_path)
    capture_test_predictions(linear_glori, pred_path)
    try:
        return linear_glori.main(args)
    except NotImplementedError as e:
        # run_infer_linear của họ: tính chỉ số + trả dự đoán XONG rồi mới ghi CSV theo tên dataset; tên không có nhánh (VinDr) -> dòng cuối
        # `else: raise NotImplementedError`. Chỉ bỏ qua ĐÚNG lỗi đó (phát ra từ run_infer_linear, test là VinDr, dự đoán đã lưu); lỗi khác ném tiếp.
        import traceback
        frame = traceback.extract_tb(e.__traceback__)[-1]
        if frame.name == 'run_infer_linear' and args.test_dataset_str.startswith('VinDr') and os.path.exists(pred_path):
            print(f'[vindr_protocol] run_infer_linear: khong co nhanh ghi CSV cho VinDr (NotImplementedError o dong {frame.lineno}) -> chi so da in, '
                  f'du doan da luu {pred_path}', flush=True)
            return 0
        raise


if __name__ == '__main__':
    sys.exit(main())
