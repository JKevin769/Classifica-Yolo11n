from ultralytics import YOLO

# Carrega o modelo pré-treinado do YOLO26 para segmentação
model = YOLO('yolo26n-seg.pt')

# Inicia o treinamento
results = model.train(
    data='/home/vicom04/yolo11/yolo26segm/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=8, 
    project='yolo26segm/runs', 
    name='seg_soja_v1'
)

# Validação do modelo final
metrics = model.val()
print("Resultados do treinamento de segmentação:")
print(metrics)
