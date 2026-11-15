"""Đầu đọc (PLAN_89 §3). mldecoder.py = đầu ML-Decoder của ta (lớp của 85f) + bộ chuyển cho giao diện đầu đọc của mã CheXFound (Ver1).
linear.py (create_linear_input + LinearClassifier) và glori.py (GLoRI) = mã công khai của CheXFound @ 7e288d0 (MIT — encoders/chexfound_src/LICENSE),
chép từng byte từ vindr_protocol/chexfound của module 86, dùng cho Ver2 (protocols/ours/heads.py). Ver1 dùng thẳng mã trong repo của họ (clone theo
commit ghim), không dùng hai tệp này."""
