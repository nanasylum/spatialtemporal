import subprocess  
import os  
  
# 数据集基础目录  
base_dir = './endovis_data/'  
  
# 数据集编号列表（1到9）  
datasets = range(1, 10)  
  
# 规则映射：根据数据集编号范围确定最后一个参数  
# 注意：这里假设的规则是基于您提供的描述，可能需要根据实际情况进行调整  
def get_last_param(dataset_num):  
    if 1 <= dataset_num <= 3:  
        return 3  
    elif 4 <= dataset_num <= 5:  
        return 4  
    elif 6 <= dataset_num <= 7:  
        return 5  
    elif dataset_num == 8:  
        return 1  
    elif dataset_num == 9:  
        return 2  
    else:  
        raise ValueError("Unsupported dataset number")  
  
# 遍历数据集并执行命令  
for dataset_num in datasets:  
    dataset_dir = os.path.join(base_dir, f'dataset{dataset_num}')  
    last_param = get_last_param(dataset_num)  
      
    # 构造并执行命令  
    # 注意：这里假设convert.py的输入和输出都是同一个目录，您可能需要根据实际情况调整  
    command = [  
        'python', 'depth_to_pointcloud_batch.py',  
        dataset_dir,  # 输入目录  
        dataset_dir,  # 假设输出目录也是这里，根据实际情况调整  
        str(last_param),  # 最后一个参数  
        dataset_dir
    ]  
      
    # 执行命令  
    subprocess.run(command, check=True)  
    print(f"Processed dataset{dataset_num} with last param {last_param}")  
  
# 注意：在实际使用中，请确保convert.py脚本具有执行权限，并且位于系统的PATH中，  
# 或者提供convert.py脚本的完整路径。