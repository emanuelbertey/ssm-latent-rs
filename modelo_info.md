# Arquitectura y Parámetros del Modelo (TinyStories JEPA)

El modelo está definido en `config.toml` y estructurado en `model.rs` bajo la clase `JepaLanguageModel`. Utiliza una arquitectura basada en **Mamba-3** (SSM con números complejos y MIMO).

## Dimensiones del Modelo
Según la configuración actual (`config.toml`):
- **Dimensión del modelo (`d_model`)**: 256
- **Dimensión de estado (`d_state`)**: 16
- **Factor de expansión (`expand`)**: 2 (por lo que `d_inner` = 512)
- **Cabezales (`n_heads`)**: 4
- **Rango MIMO (`mimo_rank`)**: 1
- **Convolución 1D (`use_conv`)**: Activada, con kernel de tamaño 4.
- **Capas (SSM Blocks)**: 2 capas (está harcodeado en `model.rs` con un `for _ in 0..2`).
- **Vocabulario (`vocab_size`)**: ~50,257 tokens (tamaño estándar del tokenizer tipo GPT-Neo).

## Cálculo de Parámetros

### 1. Capas de Embeddings y Cabezales (Lo que más pesa)
- **Embedding Layer** (`256 × 50,257`): **~12.86 M**
- **Output Head** (`256 × 50,257 + bias`): **~12.91 M**
- **Input Projection** (`256 × 256 + bias`): **~65 K**

### 2. Bloque SSM (x2 capas)
Cada capa SSM (Mamba-3) cuenta con las siguientes proyecciones y parámetros:
- **`in_proj`** (256 a 1024): ~263 K
- **`out_proj`** (512 a 256): ~131 K
- **`conv1d`** (Depthwise conv 512×4): ~2.5 K
- **Proyecciones de control** (`dt_proj`, `lambda_proj`, `theta_proj`): ~20.5 K
- **Proyecciones de matrices B y C** (2 × 512 a 64): ~65.6 K
- **Otros parámetros** (A_re, A_im, D-skip, normas, bias): ~1.3 K

**Total por bloque SSM:** ~484 K parámetros.
**Para las 2 capas:** **~969 K** (casi 1 Millón de parámetros).

## Total de Parámetros
Sumando todos los componentes:
- **Parámetros "Core" (solo lógica SSM y proyecciones)**: **~1.03 Millones**
- **Embeddings + Head (Mapeo de vocabulario)**: **~25.78 Millones**

> [!IMPORTANT]  
> **Parámetros Totales: ~26.8 Millones**
> Es un modelo extremadamente ligero y optimizado ("Tiny"), donde el 96% de los parámetros se gastan exclusivamente en memorizar el vocabulario (Embeddings y Output Head) debido a que `d_model` es pequeño pero el vocabulario es enorme (50k tokens).
