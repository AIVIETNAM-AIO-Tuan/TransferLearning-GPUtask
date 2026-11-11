"""Mã mô hình CheXFound (DINOv2 của Meta, Apache 2.0 ở đầu từng tệp; repo CheXFound MIT) @ 7e288d06532f24abedb7d00781185c36727d9d69:

    vision_transformer.py = chexfound/models/vision_transformer.py — từng byte, CHỈ đổi dòng import tuyệt đối
                            `from chexfound.layers import ...` -> `from .layers import ...` (gói của ta không có tên `chexfound` ở mức cao nhất)
    layers/*.py           = chexfound/layers/*.py — từng byte

Không import ở `import vindr_protocol` (các tệp layers cảnh báo xFormers lúc import): backbones.load_chexfound import khi cần, chặn đúng các cảnh báo
đó. Bản gốc để đối chiếu: experiments/86_vindr_giao_thuc/nguon_chexfound_7e288d0/{models,layers}/ (test_vindr_protocol kiểm quan hệ trên)."""
