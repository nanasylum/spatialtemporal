import subprocess  
import os

def create_pred_ours_folder(image_path):  
    # 假设 image_path 是 keyframe 文件夹的路径  
    pred_ours_path = os.path.join(image_path, 'pred_ours_nocolor')  
    os.makedirs(pred_ours_path, exist_ok=True)  
    print(f"Created {pred_ours_path}") 
  
def execute_command(command):  
    # 执行命令并打印输出（可选）  
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)  
    print(f"Command executed: {command}")  
    if result.stdout:  
        print("Stdout:")  
        print(result.stdout)  
    if result.stderr:  
        print("Stderr:")  
        print(result.stderr)  
  
def generate_and_execute_commands(model_path, base_image_path, model_type, start_dataset=1, end_dataset=9, keyframes=5):  
    for dataset in range(start_dataset, end_dataset + 1):  
        dataset_path = f"{base_image_path}dataset{dataset}/"  
        for keyframe in range(1, keyframes + 1):  
            image_path = f"{dataset_path}keyframe{keyframe}/image_02/data/"  
            pred_path = f"{dataset_path}keyframe{keyframe}/image_02/"  
            # 创建 pred_ours 文件夹  
            create_pred_ours_folder(pred_path)
           
            # 生成并执行命令  
            command = f"python test_simple_nocolor.py --model_path {model_path} --image_path {image_path} --model_type {model_type}"  
            execute_command(command)  
            
  
# 调用函数  
model_path = './logs/endodac/models/weights_10'  
base_image_path = './endovis_data/'  
model_type = 'afsfm'  
generate_and_execute_commands(model_path, base_image_path, model_type)