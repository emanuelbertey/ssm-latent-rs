use burn::module::Param;
use burn::nn::EmbeddingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use safetensors::SafeTensors;
use std::fs;

// Usamos el backend NdArray puro para el test de compatibilidad matemática
type TestBackend = burn::backend::NdArray;

#[test]
fn test_mamba_embedding_compatibility() {
    let device = Default::default();

    // 1. Cargar el archivo safetensors generado por Python
    let filepath = "mamba_test_data.safetensors";
    let bytes = match fs::read(filepath) {
        Ok(b) => b,
        Err(_) => panic!("No se pudo encontrar mamba_test_data.safetensors. Debes correr 'python export_mamba_test.py' primero para generar los pesos oficiales de prueba."),
    };
    
    let safe = SafeTensors::deserialize(&bytes).unwrap();

    // 2. Extraer los datos de safetensors
    let py_emb_weight = safe.tensor("embedding_weight").unwrap();
    let py_input_ids = safe.tensor("input_ids").unwrap();
    let py_expected_out = safe.tensor("expected_embedding_output").unwrap();

    // Parsear los bytes a slices (f32 e i64)
    let emb_weight_data: &[f32] = bytemuck::cast_slice(py_emb_weight.data());
    let input_ids_data: &[i64] = bytemuck::cast_slice(py_input_ids.data());
    let expected_out_data: &[f32] = bytemuck::cast_slice(py_expected_out.data());

    // 3. Crear tensores nativos de Burn
    // Pesos: [vocab_size=100, d_model=64]
    let weight_tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(emb_weight_data.to_vec(), py_emb_weight.shape().to_vec()),
        &device,
    );

    // Inputs: Burn usa i32 internamente para tensores Int, casteamos desde i64 (el estandar de torch.long)
    let input_ids_i32: Vec<i32> = input_ids_data.iter().map(|&x| x as i32).collect();
    let input_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(input_ids_i32, py_input_ids.shape().to_vec()),
        &device,
    );

    // Salida esperada de Python (Ground truth)
    let expected_tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(expected_out_data.to_vec(), py_expected_out.shape().to_vec()),
        &device,
    );

    // 4. Instanciar la capa Embedding de Burn (igual que en nuestro JepaLanguageModel)
    let mut burn_embedding = EmbeddingConfig::new(100, 64).init(&device);
    
    // Sobrescribimos sus pesos inicializados aleatoriamente con los pesos exactos de Python Mamba
    burn_embedding.weight = Param::from_tensor(weight_tensor);

    // 5. Ejecutar la inferencia (Forward pass) usando nuestro código en Rust
    let rust_output = burn_embedding.forward(input_tensor);

    // 6. Comparación matemática
    let diff = rust_output.sub(expected_tensor).abs().max().into_scalar();
    println!("-----------------------------------------------------");
    println!("Diferencia absoluta máxima entre Python Mamba y Rust Burn: {}", diff);
    println!("-----------------------------------------------------");

    // Aserción: la diferencia debe ser menor a 1e-5 (prácticamente cero, diferencia por precisión de float)
    assert!(
        diff < 1e-5,
        "¡Incompatibilidad detectada! Los embeddings difieren por un error de {}",
        diff
    );
}
