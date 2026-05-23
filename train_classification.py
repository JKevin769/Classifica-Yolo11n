from ultralytics import YOLO

# Carregar o modelo pré-treinado para classificação
model = YOLO('yolo26n-cls.pt')

# Treinar o modelo
results = model.train(
    data='datasets', 
    epochs=20, 
    imgsz=224, 
    batch=16, 
    project='runs/classify', 
    name='soja_v26'
)

# Validar o modelo
metrics = model.val()
print(f"Top 1 accuracy: {metrics.top1}")
print(f"Top 5 accuracy: {metrics.top5}")
