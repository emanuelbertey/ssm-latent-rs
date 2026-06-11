import torch
from safetensors.torch import save_file

def main():
    print("Generando datos de prueba con PyTorch puro (sin dependencia de triton)...")
    torch.manual_seed(42)

    # Parámetros del mini-modelo de prueba
    vocab_size = 100
    d_model = 64

    # Creamos una capa Embedding idéntica a la que usaría Mamba internamente
    embedding = torch.nn.Embedding(vocab_size, d_model)

    # Tokens de prueba
    input_ids = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)

    with torch.no_grad():
        embedding_output = embedding(input_ids)

    # Guardamos pesos + entrada + salida esperada
    tensors_to_save = {
        "embedding_weight": embedding.weight.clone(),
        "input_ids": input_ids.clone(),
        "expected_embedding_output": embedding_output.clone(),
    }

    save_file(tensors_to_save, "mamba_test_data.safetensors")
    print(f"Embedding weight shape: {embedding.weight.shape}")
    print(f"Input: {input_ids}")
    print(f"Output shape: {embedding_output.shape}")
    print("Guardado en 'mamba_test_data.safetensors'")

if __name__ == "__main__":
    main()
