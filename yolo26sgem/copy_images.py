import os
import shutil

# Diretórios de origem (Classificação)
src_train = "/home/vicom04/yolo11/datasets/train"
src_val = "/home/vicom04/yolo11/datasets/val"

# Diretórios de destino (Segmentação)
dst_train = "/home/vicom04/yolo11/yolo26segm/datasets/images/train"
dst_val = "/home/vicom04/yolo11/yolo26segm/datasets/images/val"

def copy_images(src_root, dst_dir):
    count = 0
    if not os.path.exists(src_root):
        print(f"Erro: Diretorio {src_root} nao encontrado.")
        return
    
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                # Opcional: prefixar o nome do arquivo com a classe para evitar conflitos
                # classe = os.path.basename(root)
                # new_name = f"{classe}_{file}"
                dst_path = os.path.join(dst_dir, file)
                
                shutil.copy2(src_path, dst_path)
                count += 1
    print(f"Copiadas {count} imagens para {dst_dir}")

# Executar a cópia
copy_images(src_train, dst_train)
copy_images(src_val, dst_val)
