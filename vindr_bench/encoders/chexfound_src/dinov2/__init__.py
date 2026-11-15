"""Mã mô hình CheXFound (DINOv2 của Meta, Apache 2.0 ở đầu từng tệp; repo CheXFound MIT — ../LICENSE) @ 7e288d06532f24abedb7d00781185c36727d9d69,
chép từng byte từ vindr_protocol/chexfound/dinov2 của module 86 (đã chạy ở 86b / 87):

    vision_transformer.py = chexfound/models/vision_transformer.py — từng byte, CHỈ đổi dòng import tuyệt đối
                            `from chexfound.layers import ...` -> `from .layers import ...` (gói của ta không có tên `chexfound` ở mức cao nhất)
    layers/*.py           = chexfound/layers/*.py — từng byte

Không import ở `import vindr_bench` (các tệp layers cảnh báo xFormers lúc import): encoders/chexfound.py import khi cần, chặn đúng các cảnh báo đó.
tests/test_ver2.py so từng byte với bản của 86 và với bản gốc (experiments/86_vindr_giao_thuc/nguon_chexfound_7e288d0/{models,layers}/)."""
