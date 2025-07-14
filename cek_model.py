import torch
from ultralytics import YOLO
from torch.serialization import add_safe_globals
import ultralytics.utils.loss
import ultralytics.nn.modules.conv

try:
    # Tambahkan class yang diperlukan ke daftar aman
    add_safe_globals([
        ultralytics.utils.loss.BboxLoss,
        ultralytics.nn.modules.conv.Conv,
    ])

    # Load model dari file .pt (pastikan path-nya benar)
    model = torch.load("best.pt", weights_only=False)
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
