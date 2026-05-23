import json
import numpy as np
import cv2
import os
import glob

def converter_json_para_mascara(pasta_origem, pasta_destino):
    # Pega o caminho absoluto para evitar confusão de pastas
    caminho_absoluto_origem = os.path.abspath(pasta_origem)
    print(f"🔍 Procurando arquivos JSON na pasta:\n{caminho_absoluto_origem}")
    
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Busca os arquivos
    arquivos_json = glob.glob(os.path.join(pasta_origem, '*.json'))
    print(f"📦 Foram encontrados {len(arquivos_json)} arquivos JSON.")
    
    if len(arquivos_json) == 0:
        print("⚠️  Nenhum arquivo encontrado. Verifique se o caminho da pasta está correto!")
        return

    for caminho_json in arquivos_json:
        try:
            with open(caminho_json, 'r') as f:
                dados = json.load(f)
                
            altura = dados['imageHeight']
            largura = dados['imageWidth']
            
            mascara = np.zeros((altura, largura), dtype=np.uint8)
            
            for shape in dados['shapes']:
                pontos = shape['points']
                pontos_np = np.array(pontos, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mascara, [pontos_np], color=(255,))
                
            nome_arquivo = os.path.basename(caminho_json).replace('.json', '.png')
            caminho_saida = os.path.join(pasta_destino, nome_arquivo)
            
            cv2.imwrite(caminho_saida, mascara)
            print(f"✅ Máscara gerada: {caminho_saida}")
            
        except Exception as e:
            print(f"❌ Erro ao processar o arquivo {caminho_json}: {e}")

# Substitua aqui caso o caminho no seu Ubuntu seja diferente!
pasta_dos_jsons = "yolo26segm/datasets/labels/train/json" 
pasta_das_mascaras = "mascaras_geradas/train"

converter_json_para_mascara(pasta_dos_jsons, pasta_das_mascaras)
