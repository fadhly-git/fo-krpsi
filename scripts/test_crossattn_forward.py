"""Verifikasi forward pass dan jumlah parameter modul CrossAttentionFusion (E-3).

Verifikasi ini TIDAK membutuhkan data asli atau GPU — hanya CPU + random tensor.
Dijalankan sebelum training penuh untuk memastikan:
  1. Output shape: [batch, 2] (2 kelas)
  2. Tidak ada error selama forward pass
  3. Jumlah parameter dilaporkan (untuk dibandingkan dengan E-1/E-2)

Usage:
    cd x:/
    python scripts/test_crossattn_forward.py
"""

import sys
import os

# Tambah src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import CrossAttentionFusion, build_head_cross_attention

# ------------------------------------------------------------------
# Konfigurasi sesuai rencana implementasi E-3
# ------------------------------------------------------------------
BATCH_SIZE = 2
FEATURE_DIM = 1280   # output EfficientNet-B0
DCT_DIM = 192        # 3 statistik × 64-dim (mean, variance, skewness)
ATTN_DIM = 64        # d dalam paper (single-head)
N_DCT_TOKENS = 3     # jumlah token K/V

print("=" * 60)
print("Verifikasi Forward Pass CrossAttentionFusion (E-3)")
print("=" * 60)

# ------------------------------------------------------------------
# Test 1: Instansiasi modul
# ------------------------------------------------------------------
print("\n[TEST 1] Instansiasi CrossAttentionFusion...")
try:
    fusion, head = build_head_cross_attention(
        feature_dim=FEATURE_DIM,
        dct_dim=DCT_DIM,
        attn_dim=ATTN_DIM,
        n_dct_tokens=N_DCT_TOKENS,
    )
    print("  ✓ CrossAttentionFusion dan head berhasil diinstansiasi")
except Exception as e:
    print(f"  ✗ GAGAL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Test 2: Forward pass dengan dummy data
# ------------------------------------------------------------------
print(f"\n[TEST 2] Forward pass (batch={BATCH_SIZE}, spatial={FEATURE_DIM}-dim, DCT={DCT_DIM}-dim)...")
try:
    torch.manual_seed(42)
    spatial_feat = torch.randn(BATCH_SIZE, FEATURE_DIM)
    dct_feat = torch.randn(BATCH_SIZE, DCT_DIM)

    fusion.eval()
    head.eval()
    with torch.no_grad():
        fused = fusion(spatial_feat, dct_feat)
        out = head(fused)

    assert fused.shape == (BATCH_SIZE, FEATURE_DIM), \
        f"Shape fused salah: expected ({BATCH_SIZE}, {FEATURE_DIM}), got {fused.shape}"
    assert out.shape == (BATCH_SIZE, 2), \
        f"Shape output salah: expected ({BATCH_SIZE}, 2), got {out.shape}"
    assert torch.isfinite(fused).all(), "Fused output mengandung NaN/Inf!"
    assert torch.isfinite(out).all(), "Head output mengandung NaN/Inf!"

    print(f"  ✓ fused shape: {tuple(fused.shape)}  (expected: ({BATCH_SIZE}, {FEATURE_DIM}))")
    print(f"  ✓ output shape: {tuple(out.shape)}  (expected: ({BATCH_SIZE}, 2))")
    print(f"  ✓ Semua nilai finite")
except Exception as e:
    print(f"  ✗ GAGAL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------------
# Test 3: Verifikasi residual connection
# ------------------------------------------------------------------
print("\n[TEST 3] Verifikasi residual (fused ≠ spatial_feat murni)...")
try:
    diff = (fused - spatial_feat).abs().mean().item()
    assert diff > 0.0, "Residual tidak aktif — fused == spatial_feat!"
    print(f"  ✓ Mean absolute diff (fused vs spatial_feat): {diff:.6f}  > 0")
except Exception as e:
    print(f"  ✗ GAGAL: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Test 4: Hitung parameter
# ------------------------------------------------------------------
print("\n[TEST 4] Jumlah parameter E-3...")

def count_params(module, name):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"  {name}: total={total:,}  trainable={trainable:,}")
    return total, trainable

n_fusion, _ = count_params(fusion, "CrossAttentionFusion (fusion)")
n_head,   _ = count_params(head,   "Classification head   (head) ")
n_e3_total = n_fusion + n_head

print(f"\n  E-3 (fusion + head) total:  {n_e3_total:,} parameter")

# Bandingkan dengan E-1 dan E-2
n_e1_head = (FEATURE_DIM + DCT_DIM + 1) * 2  # Linear(1472, 2): weight + bias
n_e2_head = (FEATURE_DIM + 1) * 2             # Linear(1280, 2): weight + bias
print(f"\n  Perbandingan head-only:")
print(f"    E-1 (concat Linear(1472,2)):  {n_e1_head:,} parameter")
print(f"    E-2 (Linear(1280,2)):         {n_e2_head:,} parameter")
print(f"    E-3 (CrossAttn + Linear):     {n_e3_total:,} parameter")
print(f"    Selisih E-3 vs E-1:          +{n_e3_total - n_e1_head:,} parameter")

# Breakdown detail CrossAttentionFusion
print("\n[TEST 5] Breakdown parameter CrossAttentionFusion...")
for name, param in fusion.named_parameters():
    print(f"  {name}: {tuple(param.shape)}  → {param.numel():,} param")

# ------------------------------------------------------------------
# Test 6: Gradient flow
# ------------------------------------------------------------------
print("\n[TEST 6] Gradient flow (backward pass)...")
try:
    fusion.train()
    head.train()
    spatial_feat_g = torch.randn(BATCH_SIZE, FEATURE_DIM, requires_grad=False)
    dct_feat_g = torch.randn(BATCH_SIZE, DCT_DIM, requires_grad=False)
    labels_g = torch.tensor([0, 1])

    fused_g = fusion(spatial_feat_g, dct_feat_g)
    out_g = head(fused_g)
    loss_g = torch.nn.CrossEntropyLoss()(out_g, labels_g)
    loss_g.backward()

    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in list(fusion.parameters()) + list(head.parameters())
    )
    assert grads_ok, "Ada parameter tanpa gradient atau dengan NaN/Inf gradient!"
    print(f"  ✓ loss = {loss_g.item():.6f}")
    print(f"  ✓ Semua parameter fusion + head menerima gradient finite")
except Exception as e:
    print(f"  ✗ GAGAL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------------
# Ringkasan
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("RINGKASAN VERIFIKASI")
print("=" * 60)
print(f"  Output shape:       ({BATCH_SIZE}, 2) ✓")
print(f"  Residual aktif:     Ya ✓")
print(f"  Gradient flow:      OK ✓")
print(f"  Total param E-3:    {n_e3_total:,}")
print(f"    - CrossAttentionFusion: {n_fusion:,}")
print(f"    - Head:                 {n_head:,}")
print("=" * 60)
print("\nSemua verifikasi PASSED. Siap untuk konfirmasi training.")
