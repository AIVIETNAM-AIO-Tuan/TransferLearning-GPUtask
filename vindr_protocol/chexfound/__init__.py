"""Mã CheXFound đã chép (RPIDIAL/CheXFound @ 7e288d06532f24abedb7d00781185c36727d9d69; giấy phép MIT: LICENSE trong thư mục này).

    transforms.py  = chexfound/data/transforms.py — nguyên văn từng byte (gốc Meta / DINOv2, Apache 2.0 ở đầu tệp)
    glori.py       = chexfound/eval/classification/glori.py — bỏ ĐÚNG MỘT dòng `@torch.jit.script` (trên class GroupFC): torch mới cảnh báo
                     TorchScript lỗi thời; logit + grad trùng từng bit với bản jit (thử 10/11)
    linear.py      = create_linear_input + LinearClassifier nguyên văn từ chexfound/eval/classification/utils.py (tệp gốc import các
                     module chexfound không chép nên không import nguyên tệp được)

Bản gốc từng byte để đối chiếu (không thuộc gói): experiments/86_vindr_giao_thuc/nguon_chexfound_7e288d0/; test_86 kiểm các quan hệ trên."""
from .glori import GLoRI, create_mldecoder_input
from .linear import LinearClassifier, create_linear_input
from .transforms import make_classification_eval_transform, make_classification_train_transform
