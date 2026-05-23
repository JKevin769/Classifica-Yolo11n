import json
import os
import glob
import shutil
import numpy as np
import cv2

def convert_labelme_to_yolo(json_path, img_w, img_h, class_mapping):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Se o JSON não tiver dimensões, tenta ler do próprio arquivo se for o LabelMe antigo
    # mas geralmente os novos têm.
    h = data.get('imageHeight', img_h)
    w = data.get('imageWidth', img_w)
    
    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        # Mapeamento flexível para ferrugem
        if label in ['f1', 'f2', 'f3']:
            class_id = 1 # ferrugem
        elif label == "1" or label == "caterpillar":
            class_id = 0 # caterpillar
        else:
            # Caso existam outros labels ou o mapeamento seja diferente
            if label in class_mapping:
                class_id = class_mapping[label]
            else:
                continue # ignora se não souber o que é
        
        points = shape['points']
        normalized_points = []
        for p in points:
            normalized_points.append(p[0] / w)
            normalized_points.append(p[1] / h)
        
        line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
        yolo_lines.append(line)
    
    return yolo_lines

def organizar_dataset():
    base_path = "yolo26segm/datasets"
    mapping = {"1": 0, "caterpillar": 0, "ferrugem": 1, "f1": 1, "f2": 1}
    
    for split in ['train', 'val']:
        # 1. Criar pastas de labels se não existirem
        for cls_name in ['caterpillar', 'ferrugem', 'saudavel', 'vaquinha verde e amarela']:
            os.makedirs(os.path.join(base_path, 'labels', split, cls_name), exist_ok=True)
        
        # 2. Mover TXTs existentes da pasta 'json' para 'caterpillar'
        json_labels_dir = os.path.join(base_path, 'labels', split, 'json')
        if os.path.exists(json_labels_dir):
            txt_files = glob.glob(os.path.join(json_labels_dir, "*.txt"))
            for txt_file in txt_files:
                target = os.path.join(base_path, 'labels', split, 'caterpillar', os.path.basename(txt_file))
                shutil.copy2(txt_file, target)
                print(f"Copiado label caterpillar: {os.path.basename(txt_file)}")

        # 3. Converter JSONs de ferrugem (que estão na pasta de imagens)
        ferrugem_img_dir = os.path.join(base_path, 'images', split, 'ferrugem')
        if os.path.exists(ferrugem_img_dir):
            json_files = glob.glob(os.path.join(ferrugem_img_dir, "*.json"))
            for json_file in json_files:
                # Tenta obter dimensões da imagem correspondente se necessário, 
                # mas o convert_labelme_to_yolo já tenta ler do JSON.
                yolo_lines = convert_labelme_to_yolo(json_file, 640, 640, mapping)
                
                if yolo_lines:
                    txt_filename = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
                    txt_path = os.path.join(base_path, 'labels', split, 'ferrugem', txt_filename)
                    with open(txt_path, 'w') as f:
                        f.write("\n".join(yolo_lines))
                    print(f"Gerado label ferrugem: {txt_filename}")

        # 4. Gerar máscaras PNG para ferrugem como solicitado pelo usuário
        mask_output_dir = os.path.join("mascaras_geradas", split, "ferrugem")
        os.makedirs(mask_output_dir, exist_ok=True)
        if os.path.exists(ferrugem_img_dir):
            json_files = glob.glob(os.path.join(ferrugem_img_dir, "*.json"))
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                h, w = data['imageHeight'], data['imageWidth']
                mask = np.zeros((h, w), dtype=np.uint8)
                for shape in data['shapes']:
                    points = np.array(shape['points'], np.int32)
                    cv2.fillPoly(mask, [points], 255)
                
                mask_name = os.path.splitext(os.path.basename(json_file))[0] + ".png"
                cv2.imwrite(os.path.join(mask_output_dir, mask_name), mask)
                print(f"Máscara PNG gerada: {mask_name}")

if __name__ == "__main__":
    organizar_dataset()
    print("\n✅ Dataset reorganizado e labels convertidos!")
