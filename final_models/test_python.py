import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pandas as pd
import random

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths e checkpoints
paths = {
    "Original": "data_split/test/original",
    "Segmented": "data_split/test/segmented"
}

checkpoints = {
    "ResNet50": {
        "Original": "ResNet50_SGD_original.pth",
        "Segmented": "ResNet50_SGD_segmented.pth"
    },
    "DenseNet201": {
        "Original": "DenseNet201_SGD_original.pth",
        "Segmented": "DenseNet201_SGD_segmented.pth"
    },
    "VGG19": {
        "Original": "VGG19_SGD_original.pth",
        "Segmented": "VGG19_SGD_segmented.pth"
    }
}

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_consistent_sample_indices(sample_fraction=0.3, seed=42, per_class=True):
    """
    Gera índices consistentes para ambos os datasets (Contextual e ROI-based)
    que serão reutilizados por todos os modelos
    
    Args:
        sample_fraction: Fração de amostras a selecionar (0.3 = 30%)
        seed: Seed para reprodutibilidade
        per_class: Se True, seleciona 30% de cada classe. Se False, 30% do dataset total
    """
    print("Gerando índices consistentes para amostragem...")
    print(f"Modo: {'30% por classe' if per_class else '30% do dataset total'}")
    
    # Definir seed
    random.seed(seed)
    np.random.seed(seed)
    
    sample_indices = {}
    
    for dataset_type in ["Original", "Segmented"]:
        dataset = ImageFolder(paths[dataset_type], transform=transform)
        
        if per_class:
            # Amostragem balanceada: 30% de cada classe
            selected_indices = []
            class_distribution = {}
            
            # Agrupar amostras por classe
            class_samples = {}
            for idx, (path, class_idx) in enumerate(dataset.samples):
                class_name = dataset.classes[class_idx]
                if class_name not in class_samples:
                    class_samples[class_name] = []
                class_samples[class_name].append(idx)
            
            # Selecionar 30% de cada classe
            for class_name, class_indices in class_samples.items():
                class_sample_size = int(len(class_indices) * sample_fraction)
                selected_class_indices = random.sample(class_indices, class_sample_size)
                selected_indices.extend(selected_class_indices)
                class_distribution[class_name] = {
                    'total': len(class_indices),
                    'selected': class_sample_size,
                    'percentage': (class_sample_size / len(class_indices)) * 100
                }
            
            selected_indices.sort()  # Ordenar para consistência
            
            sample_indices[dataset_type] = {
                'indices': selected_indices,
                'total_samples': len(dataset),
                'sample_size': len(selected_indices),
                'class_names': dataset.classes,
                'class_distribution': class_distribution,
                'sampling_mode': 'per_class'
            }
            
            print(f"\n{dataset_type} - Distribuição por classe:")
            for class_name, info in class_distribution.items():
                print(f"  {class_name}: {info['selected']}/{info['total']} ({info['percentage']:.1f}%)")
            print(f"  Total selecionado: {len(selected_indices)}/{len(dataset)}")
            
        else:
            # Amostragem global: 30% do dataset total
            total_samples = len(dataset)
            sample_size = int(total_samples * sample_fraction)
            
            # Gerar índices aleatórios consistentes
            indices = random.sample(range(total_samples), sample_size)
            indices.sort()  # Ordenar para consistência
            
            sample_indices[dataset_type] = {
                'indices': indices,
                'total_samples': total_samples,
                'sample_size': sample_size,
                'class_names': dataset.classes,
                'sampling_mode': 'global'
            }
            
            print(f"{dataset_type}: {sample_size}/{total_samples} amostras ({sample_fraction*100:.1f}%)")
    
    return sample_indices

def get_target_layer(model, model_name):
    """Retorna a camada target para cada modelo"""
    if model_name == "ResNet50":
        return [model.backbone.layer4[-1]]
    elif model_name == "DenseNet201":
        return [model.backbone.features.denseblock4.denselayer32.conv2]
    elif model_name == "VGG19":
        return [model.backbone.features[35]]
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

def load_model_for_evaluation(model_name, dataset_type):
    """Carrega modelo específico"""
    import sys
    sys.path.append(".")
    from ResNet50 import ResNet50
    from DenseNet201 import DenseNet201
    from VGG19 import VGG19
    
    checkpoint_path = checkpoints[model_name][dataset_type]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = len(checkpoint['classes'])
    class_names = checkpoint['classes']
    
    if model_name == "ResNet50":
        model = ResNet50(num_classes=num_classes, dropout_rate=0.3, pretrained=False).to(device)
    elif model_name == "DenseNet201":
        model = DenseNet201(num_classes=num_classes, dropout_rate=0.3, pretrained=False).to(device)
    elif model_name == "VGG19":
        model = VGG19(num_classes=num_classes, dropout_rate=0.3, pretrained=False).to(device)
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names, num_classes

def normalize_saliency_map(saliency_map):
    """Normaliza o mapa de saliência"""
    S_min = np.min(saliency_map)
    S_max = np.max(saliency_map)
    
    if S_max - S_min == 0:
        return np.zeros_like(saliency_map)
    
    normalized = (saliency_map - S_min) / (S_max - S_min)
    return normalized

def calculate_sparsity(saliency_map):
    """
    Calcula sparsidade:
    Sparsity = 1 / S'_mean onde S'_max = 1 após normalização
    """
    normalized_map = normalize_saliency_map(saliency_map)
    
    if np.mean(normalized_map) == 0:
        return 0.0
    
    sparsity = 1.0 / np.mean(normalized_map)
    return sparsity

def create_blurred_tensor(tensor, sigma=8):
    """Cria versão borrada do tensor"""
    img_np = tensor[0].cpu().numpy()
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Desnormalizar
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Aplicar blur
    blurred = np.stack([gaussian_filter(img_np[:,:,i], sigma=sigma) for i in range(3)], axis=2)
    
    # Renormalizar
    blurred = (blurred - mean) / std
    blurred = np.transpose(blurred, (2, 0, 1))
    
    return torch.from_numpy(blurred).float().unsqueeze(0).to(device)

def get_most_salient_pixels(saliency_map, fraction=0.1):
    """
    Retorna os pixels mais salientes baseado na fração especificada
    """
    flat_map = saliency_map.flatten()
    num_pixels = int(len(flat_map) * fraction)
    
    # Obter índices dos pixels mais salientes
    top_indices = np.argpartition(flat_map, -num_pixels)[-num_pixels:]
    
    # Converter para coordenadas 2D
    coords = []
    for idx in top_indices:
        y, x = np.unravel_index(idx, saliency_map.shape)
        coords.append((y, x))
    
    return coords, flat_map[top_indices]

def calculate_insertion_deletion_correlation(model, input_tensor, saliency_map, predicted_class, num_steps=4):
    """
    Calcula DC e IC
    """
    original_tensor = input_tensor.clone()
    blurred_tensor = create_blurred_tensor(input_tensor)
    
    # Normalizar mapa de saliência
    normalized_saliency = normalize_saliency_map(saliency_map)
    
    # Obter pixels ordenados por saliência (decrescente)
    H, W = saliency_map.shape
    flat_saliency = normalized_saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]  # decrescente
    
    # Calcular numero de pixels por step
    total_pixels = len(flat_saliency)
    pixels_per_step = total_pixels // num_steps
    
    deletion_scores = []
    insertion_scores = []
    saliency_scores = []
    
    # Score inicial (imagem original para deletion, borrada para insertion)
    with torch.no_grad():
        orig_output = model(original_tensor)
        orig_conf = F.softmax(orig_output, dim=1)[0, predicted_class].item()
        
        blur_output = model(blurred_tensor)
        blur_conf = F.softmax(blur_output, dim=1)[0, predicted_class].item()
    
    deletion_scores.append(orig_conf)
    insertion_scores.append(blur_conf)
    
    # Processar em steps
    current_del = original_tensor.clone()
    current_ins = blurred_tensor.clone()
    
    for step in range(num_steps):
        start_idx = step * pixels_per_step
        end_idx = min((step + 1) * pixels_per_step, total_pixels)
        
        if start_idx >= end_idx:
            break
            
        # Obter pixels para este step
        step_indices = sorted_indices[start_idx:end_idx]
        step_saliency = np.mean(flat_saliency[step_indices])
        saliency_scores.append(step_saliency)
        
        # Aplicar modificações pixel por pixel
        for idx in step_indices:
            y, x = np.unravel_index(idx, (H, W))
            
            # DELETION: substituir por pixel borrado
            current_del[:, :, y, x] = blurred_tensor[:, :, y, x]
            
            # INSERTION: substituir por pixel original  
            current_ins[:, :, y, x] = original_tensor[:, :, y, x]
        
        # Avaliar modelos
        with torch.no_grad():
            del_output = model(current_del)
            del_conf = F.softmax(del_output, dim=1)[0, predicted_class].item()
            deletion_scores.append(del_conf)
            
            ins_output = model(current_ins)
            ins_conf = F.softmax(ins_output, dim=1)[0, predicted_class].item()
            insertion_scores.append(ins_conf)
    
    # Calcular correlações
    dc = 0.0
    ic = 0.0
    
    if len(saliency_scores) > 1:
        # Para deletion: correlação entre saliência e queda de confiança
        score_drops = []
        for i in range(len(saliency_scores)):
            drop = deletion_scores[i] - deletion_scores[i+1]
            score_drops.append(drop)
        
        if len(score_drops) > 1 and np.std(score_drops) > 1e-6:
            dc_corr, dc_p = pearsonr(saliency_scores, score_drops)
            if not np.isnan(dc_corr):
                dc = dc_corr
        
        # Para insertion: correlação entre saliência e ganho de confiança
        score_gains = []
        for i in range(len(saliency_scores)):
            gain = insertion_scores[i+1] - insertion_scores[i]
            score_gains.append(gain)
        
        if len(score_gains) > 1 and np.std(score_gains) > 1e-6:
            ic_corr, ic_p = pearsonr(saliency_scores, score_gains)
            if not np.isnan(ic_corr):
                ic = ic_corr
    
    return dc, ic

def process_sample(model, cam, input_tensor, target_class, true_class):
    """Processa uma amostra e calcula todas as métricas"""
    
    # Gerar GradCAM
    targets = [ClassifierOutputTarget(target_class)]
    saliency_map = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # Calcular sparsidade
    sparsity = calculate_sparsity(saliency_map)
    
    # Calcular IC e DC
    dc, ic = calculate_insertion_deletion_correlation(model, input_tensor, saliency_map, target_class)
    
    return {
        'sparsity': sparsity,
        'ic': ic,
        'dc': dc,
        'predicted_class': target_class,
        'true_class': true_class
    }

def evaluate_model_dataset(model_name, dataset_type, sample_indices_info):
    """Avalia um modelo em um dataset específico usando índices pré-definidos"""
    
    print(f"Processando {model_name} - {dataset_type}")
    
    # Carregar modelo
    model, class_names, num_classes = load_model_for_evaluation(model_name, dataset_type)
    target_layers = get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Carregar dataset com os mesmos índices consistentes
    dataset = ImageFolder(paths[dataset_type], transform=transform)
    indices = sample_indices_info['indices']
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)
    
    results = []
    class_counts = {class_name: 0 for class_name in class_names}
    
    print(f"Processando {len(subset)} amostras (índices consistentes)...")
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Fazer predição
        with torch.no_grad():
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
        
        true_class = labels.item()
        pred_class = predicted.item()
        true_class_name = class_names[true_class]
        
        # Contar amostras por classe
        class_counts[true_class_name] += 1
        
        try:
            # Processar amostra
            metrics = process_sample(model, cam, inputs, pred_class, true_class)
            
            # Incluir índice original da amostra para rastreabilidade
            original_idx = indices[batch_idx]
            
            results.append({
                'model': model_name,
                'dataset': dataset_type,
                'sample_idx': batch_idx,
                'original_dataset_idx': original_idx, 
                'true_class': true_class_name,
                'predicted_class': class_names[pred_class],
                'sparsity': metrics['sparsity'],
                'ic': metrics['ic'],
                'dc': metrics['dc']
            })
            
        except Exception as e:
            print(f"Erro na amostra {batch_idx} (índice original {indices[batch_idx]}): {e}")
            continue
    
    print(f"Amostras por classe: {class_counts}")
    
    # Cleanup
    del cam
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    """Função principal"""
    
    # Definir seed para reprodutibilidade
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("AVALIAÇÃO DE SALIÊNCIA COM SPLIT CONSISTENTE")
    print("=" * 60)
    
    # Gerar índices consistentes uma única vez
    # per_class=True para 30% de cada classe, per_class=False para 30% do total
    sample_indices = generate_consistent_sample_indices(sample_fraction=0.3, seed=seed, per_class=True)
    
    # Validar que ambos os datasets têm o mesmo número de classes
    orig_classes = set(sample_indices['Original']['class_names'])
    seg_classes = set(sample_indices['Segmented']['class_names'])
    
    if orig_classes != seg_classes:
        print("AVISO: Classes diferentes entre Original e Segmented!")
        print(f"Original: {orig_classes}")
        print(f"Segmented: {seg_classes}")
    else:
        print(f"Classes consistentes: {len(orig_classes)} classes")
    
    print("-" * 60)
    
    all_results = []
    models = ["ResNet50", "DenseNet201", "VGG19"]
    datasets = ["Original", "Segmented"]
    
    for model_name in models:
        for dataset_type in datasets:
            try:
                # Usar os mesmos índices para cada dataset
                results = evaluate_model_dataset(
                    model_name, 
                    dataset_type, 
                    sample_indices[dataset_type]
                )
                all_results.extend(results)
                print(f"Completado: {model_name} - {dataset_type}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Erro ao processar {model_name} - {dataset_type}: {e}")
                continue
    
    # Converter para DataFrame
    df = pd.DataFrame(all_results)
    
    # Mostrar distribuição de classes
    for dataset_type in datasets:
        dataset_data = df[df['dataset'] == dataset_type]
        dataset_samples = dataset_data['original_dataset_idx'].unique()
        print(f"{dataset_type}: {len(dataset_samples)} amostras únicas")
        
        # Mostrar distribuição por classe se disponível
        if 'class_distribution' in sample_indices[dataset_type]:
            class_counts = dataset_data['true_class'].value_counts()
            print(f"  Distribuição por classe:")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count} amostras")
        
    # Verificar se as mesmas amostras foram usadas
    orig_indices = set(df[df['dataset'] == 'Original']['original_dataset_idx'].unique())
    seg_indices = set(df[df['dataset'] == 'Segmented']['original_dataset_idx'].unique())
    
    if orig_indices == seg_indices:
        print("Mesmos índices de amostra usados para ambos os datasets")
    else:
        print("ERRO: Índices diferentes entre datasets!")
        
    # Mostrar modo de amostragem usado
    sampling_mode = sample_indices['Original'].get('sampling_mode', 'unknown')
    print(f"Modo de amostragem: {sampling_mode}")
    
    # Calcular estatísticas por classe e modelo
    summary_stats = df.groupby(['model', 'dataset', 'true_class']).agg({
        'sparsity': ['mean', 'std', 'count'],
        'ic': ['mean', 'std'],
        'dc': ['mean', 'std']
    }).round(4)
    
    print("\n" + "=" * 60)
    print("RESUMO DAS MÉTRICAS POR CLASSE E MODELO:")
    print("=" * 60)
    print(summary_stats)
    
    # Salvar resultados
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'saliency_metrics_results_consistent_{timestamp}.csv'
    summary_file = f'saliency_metrics_summary_consistent_{timestamp}.csv'
    indices_file = f'sample_indices_used_{timestamp}.csv'
    
    df.to_csv(results_file, index=False)
    summary_stats.to_csv(summary_file)
    
    # Salvar informações sobre os índices usados
    indices_df = []
    for dataset_type, info in sample_indices.items():
        for idx in info['indices']:
            indices_df.append({
                'dataset': dataset_type,
                'original_index': idx,
                'total_samples': info['total_samples'],
                'sample_size': info['sample_size']
            })
    
    pd.DataFrame(indices_df).to_csv(indices_file, index=False)
    
    print(f"\nResultados salvos:")
    print(f"  • {results_file} (dados detalhados)")
    print(f"  • {summary_file} (estatísticas por classe)")  
    print(f"  • {indices_file} (índices das amostras utilizadas)")
    
    print(f"\n🔍 Total de amostras processadas: {len(df)}")
    print(f"📊 Combinações modelo-dataset: {df.groupby(['model', 'dataset']).size().to_dict()}")
    
    return df, summary_stats, sample_indices

if __name__ == "__main__":
    df_results, summary_stats, sample_indices = main()