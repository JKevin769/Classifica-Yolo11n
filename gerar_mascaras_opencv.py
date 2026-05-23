import cv2
import numpy as np
import os
import sys
import glob

def gerar_mascaras_opencv(input_path, output_dir="output_mascaras_opencv"):
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

    print(f"🖼️ Processando {len(imagens)} imagens via OpenCV Clássico...")

    for img_p in imagens:
        img = cv2.imread(img_p)
        if img is None:
            continue
            
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        
        # =====================
        # 1. Máscara da folha (B)
        # =====================
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Range para verde (ajustável dependendo da iluminação)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([90, 255, 255])

        mask_leaf = cv2.inRange(hsv, lower_green, upper_green)

        # Limpeza morfológica
        kernel = np.ones((5,5), np.uint8)
        mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel)
        mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel)

        # =====================
        # 2. Máscara das doenças (C)
        # =====================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold para manchas escuras (ajustável)
        # Usando THRESH_BINARY_INV porque as manchas costumam ser mais escuras que a folha
        _, disease = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Restringir as manchas apenas para dentro da área da folha
        mask_disease = cv2.bitwise_and(disease, mask_leaf)

        # =====================
        # Cálculo de Severidade
        # =====================
        area_leaf = np.sum(mask_leaf > 0)
        area_disease = np.sum(mask_disease > 0)
        
        severidade = 0
        if area_leaf > 0:
            severidade = (area_disease / area_leaf) * 100
        
        # =====================
        # Salvar resultados
        # =====================
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_folha_cv2.png"), mask_leaf)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_doenca_cv2.png"), mask_disease)

        print(f"✅ {base_name}: Severidade = {severidade:.2f}% (Salvo em {output_dir})")

if __name__ == "__main__":
if len(sys.argv) < 2:
     
              print("Uso: python gerar_mascaras_opencv.py <imagem_ou_pasta> [diretorio_saida]")
               sys.exit(1)
                
            entrada = sys.argv[1]
      
          saida = sys.argv[2] if len(sys.argv) > 2 else "output_mascaras_opencv"
          gerar_mascaras_opencv(entrada, saida)

