import json
import os
import glob

def convert_labelme_to_yolo(json_dir, output_dir, class_mapping):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        img_h = data['imageHeight']
        img_w = data['imageWidth']
        
        txt_filename = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f_out:
            for shape in data['shapes']:
                label = shape['label']
                if label in class_mapping:
                    class_id = class_mapping[label]
                else:
                    # Tenta converter para int se for um número em string
                    try:
                        class_id = int(label) - 1 # Se no LabelMe for 1-based e queremos 0-based
                    except ValueError:
                        print(f"Aviso: Label '{label}' não encontrado no mapeamento e não é numérico em {json_file}")
                        continue
                
                points = shape['points']
                # Normalizar pontos
                normalized_points = []
                for p in points:
                    normalized_points.append(p[0] / img_w)
                    normalized_points.append(p[1] / img_h)
                
                line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
                f_out.write(line + "\n")

# Mapeamento: no JSON o label é "1", queremos que seja 0 (caterpillar)
mapping = {"1": 0}

# Converter treino
convert_labelme_to_yolo(
    "yolo26segm/datasets/labels/train", 
    "yolo26segm/datasets/labels/train", 
    mapping
)

# Converter validação
convert_labelme_to_yolo(
    "yolo26segm/datasets/labels/val", 
    "yolo26segm/datasets/labels/val", 
    mapping
)

print("Conversão concluída!")
