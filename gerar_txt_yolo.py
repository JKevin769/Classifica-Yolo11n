import json
import os
import glob

def json_para_yolo_seg(pasta_origem, class_id=0):
    # Procura os arquivos JSON na pasta indicada
    arquivos_json = glob.glob(os.path.join(pasta_origem, '*.json'))
    print(f"📦 Convertendo {len(arquivos_json)} arquivos para o formato YOLO...")
    
    for caminho_json in arquivos_json:
        with open(caminho_json, 'r') as f:
            dados = json.load(f)
            
        largura = dados['imageWidth']
        altura = dados['imageHeight']
        
        # Salva o arquivo .txt na mesma pasta
        nome_arquivo = os.path.basename(caminho_json).replace('.json', '.txt')
        caminho_saida = os.path.join(pasta_origem, nome_arquivo)
        
        with open(caminho_saida, 'w') as out_file:
            for shape in dados['shapes']:
                pontos = shape['points']
                # Inicia a linha com o ID da classe (0 para lagartas, por exemplo)
                linha = f"{class_id}"
                
                for p in pontos:
                    # O YOLO exige que os pixels sejam convertidos em proporções de 0 a 1
                    x_norm = p[0] / largura
                    y_norm = p[1] / altura
                    linha += f" {x_norm:.6f} {y_norm:.6f}"
                    
                out_file.write(linha + '\n')
        
        print(f"✅ TXT gerado: {caminho_saida}")

# O caminho absoluto que estava no seu terminal
pasta_jsons = "/home/vicom04/yolo11/yolo26segm/datasets/labels/train"
json_para_yolo_seg(pasta_jsons)
