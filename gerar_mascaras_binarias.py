from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import glob

def gerar_mascaras(modelo_path, input_path, output_dir="output_mascaras"):
    # Carregar o modelo
    if not os.path.exists(modelo_path):
        print(f"❌ Modelo não encontrado em: {modelo_path}")
        return
    
    model = YOLO(modelo_path)
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se o input é um arquivo ou diretório
    if os.path.isdir(input_path):
        imagens = glob.glob(os.path.join(input_path, "*.jpg")) + \
                  glob.glob(os.path.join(input_path, "*.JPG")) + \
                  glob.glob(os.path.join(input_path, "*.png"))
    else:
        imagens = [input_path]
    
    if not imagens:
        print(f"⚠️ Nenhuma imagem encontrada em: {input_path}")
        return

    print(f"🖼️ Processando {len(imagens)} imagens...")

    for img_p in imagens:
        results = model(img_p, verbose=False)
        
        for r in results:
            if r.masks is None:
                print(f"⚠️ Nenhuma máscara detectada: {os.path.basename(img_p)}")
                continue
                
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            names = r.names
            
            mascaras_por_classe = {}
            
            for i, mask in enumerate(masks):
                cls_id = int(classes[i])
                
                if mask.shape[:2] != r.orig_shape:
                    mask = cv2.resize(mask, (r.orig_shape[1], r.orig_shape[0]), interpolation=cv2.INTER_LINEAR)
                
                mask_bin = (mask > 0.5).astype(np.uint8)
                
                if cls_id not in mascaras_por_classe:
                    mascaras_por_classe[cls_id] = np.zeros(r.orig_shape, dtype=np.uint8)
                
                mascaras_por_classe[cls_id] = cv2.bitwise_or(mascaras_por_classe[cls_id], mask_bin)
                
            # Salvar máscaras e calcular áreas
            areas = {}
            for cls_id, mask_final in mascaras_por_classe.items():
                cls_name = names[cls_id]
                areas[cls_name] = np.sum(mask_final)
                
                img_final = (mask_final * 255).astype(np.uint8)
                output_path = os.path.join(output_dir, f"{base_name}_{cls_name}.png")
                cv2.imwrite(output_path, img_final)
            
            # Cálculo de Severidade (se houver folha e doenca)
            # Geralmente: 0 = folha, 1 = doenca
            folha_key = 'folha' if 'folha' in areas else (names[0] if 0 in names else None)
            doenca_key = 'doenca' if 'doenca' in areas else (names[1] if 1 in names else None)
            
            log_msg = f"✅ {base_name}:"
            if folha_key in areas and doenca_key in areas:
                area_folha = areas[folha_key]
                area_doenca = areas[doenca_key]
                if area_folha > 0:
                    severidade = (area_doenca / area_folha) * 100
                    log_msg += f" Severidade = {severidade:.2f}%"
                else:
                    log_msg += " Área da folha é zero."
            
            print(log_msg)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python gerar_mascaras_binarias.py <modelo.pt> <imagem_ou_pasta>")
        sys.exit(1)
        
    modelo = sys.argv[1]
    entrada = sys.argv[2]
    
    gerar_mascaras(modelo, entrada)
