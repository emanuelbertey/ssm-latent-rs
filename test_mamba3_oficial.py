"""
=== TEST DE COMPATIBILIDAD MAMBA 3 OFICIAL vs RUST ===
CORRER ESTO EN GOOGLE COLAB (necesita GPU + triton)

Paso 1: Subí este archivo a Colab
Paso 2: Instalá dependencias:
    !pip install mamba-ssm safetensors einops causal-conv1d
Paso 3: Corré este script
Paso 4: Descargá 'mamba3_test_data.safetensors'
Paso 5: Ponelo en d:/ssm-latent-rs/ y corré: cargo test --release test_mamba3
"""

import torch
from safetensors.torch import save_file
import sys, os

# Si estás corriendo desde el repo local:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba"))

from mamba_ssm.modules.mamba3 import Mamba3

def main():
    print("=" * 60)
    print("TEST MAMBA 3 OFICIAL")
    print("=" * 60)
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Crear capa Mamba3 OFICIAL con parámetros chicos para test
    d_model = 64
    d_state = 16
    expand = 2
    headdim = 16  # d_inner = 64*2 = 128, nheads = 128/16 = 8
    
    mamba3_layer = Mamba3(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
        ngroups=1,
        is_mimo=False,
        mimo_rank=1,
        is_outproj_norm=False,
        device=device,
        dtype=torch.float32,
    )
    mamba3_layer.eval()
    
    # Entrada de prueba: batch=1, seq_len=8, d_model=64
    # seq_len tiene que ser múltiplo de chunk_size o al menos razonable
    input_tensor = torch.randn(1, 8, d_model, device=device, dtype=torch.float32)
    torch.manual_seed(123)  # semilla distinta para el input
    input_tensor = torch.randn(1, 8, d_model, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = mamba3_layer(input_tensor)
    
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean:  {output.mean().item():.6f}")
    print(f"Output std:   {output.std().item():.6f}")
    
    # Guardar TODO: pesos + input + output
    tensors = {}
    
    # Input y output
    tensors["input"] = input_tensor.cpu().clone()
    tensors["expected_output"] = output.cpu().clone()
    
    # Pesos de in_proj (Linear: d_model -> d_in_proj)
    tensors["in_proj_weight"] = mamba3_layer.in_proj.weight.cpu().clone()
    
    # Pesos de out_proj (Linear: d_inner -> d_model)
    tensors["out_proj_weight"] = mamba3_layer.out_proj.weight.cpu().clone()
    
    # dt_bias
    tensors["dt_bias"] = mamba3_layer.dt_bias.cpu().clone()
    
    # B_bias, C_bias
    tensors["B_bias"] = mamba3_layer.B_bias.cpu().clone()
    tensors["C_bias"] = mamba3_layer.C_bias.cpu().clone()
    
    # B_norm, C_norm (RMSNorm weights)
    tensors["B_norm_weight"] = mamba3_layer.B_norm.weight.cpu().clone()
    tensors["C_norm_weight"] = mamba3_layer.C_norm.weight.cpu().clone()
    
    # D skip parameter
    tensors["D"] = mamba3_layer.D.cpu().clone()
    
    # Guardar config como tensor para que Rust sepa las dimensiones
    config_tensor = torch.tensor([
        d_model,      # 0
        d_state,      # 1
        expand,       # 2
        headdim,      # 3
        1,            # 4: ngroups
        1,            # 5: mimo_rank
        8,            # 6: seq_len
    ], dtype=torch.float32)
    tensors["config"] = config_tensor
    
    # Guardar
    save_file(tensors, "mamba3_test_data.safetensors")
    
    print()
    print("=" * 60)
    print("GUARDADO: mamba3_test_data.safetensors")
    print("=" * 60)
    print()
    print("Pesos guardados:")
    for name, t in tensors.items():
        print(f"  {name}: {list(t.shape)}")
    print()
    print("SIGUIENTE PASO:")
    print("  1. Descargá 'mamba3_test_data.safetensors'")
    print("  2. Ponelo en la carpeta raíz del proyecto Rust")
    print("  3. Corré: cargo test --release test_mamba3_compatibility -- --nocapture")

if __name__ == "__main__":
    main()
