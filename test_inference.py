from ultralytics import YOLO
import os

# Carregar o melhor modelo treinado (versão train11)
model = YOLO('runs/classify/train11/weights/best.pt')

# Caminho para algumas imagens de validação
val_dir = 'datasets/val'
classes = os.listdir(val_dir)

print(f"{'Imagem':<40} | {'Predição':<20} | {'Confiança':<10}")
print("-" * 75)

for classe in classes:
    classe_path = os.path.join(val_dir, classe)
    if not os.path.isdir(classe_path):
        continue
    
    # Pegar uma imagem de cada classe para testar
    imagens = [f for f in os.listdir(classe_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if imagens:
        img_path = os.path.join(classe_path, imagens[0])
        results = model(img_path, verbose=False)
        
        result = results[0]
        probs = result.probs
        top1_idx = probs.top1
        top1_name = result.names[top1_idx]
        top1_conf = probs.top1conf.item()
        
        print(f"{img_path:<40} | {top1_name:<20} | {top1_conf:.4f}")
