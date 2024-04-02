import os
import io 
import json
import math
from PIL import Image 
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import numpy as np
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

##将图片转换为patch
def img_to_patch(input_path, output_path, rows, cols):
    # 确保输出文件夹存在
    os.makedirs(output_path, exist_ok=True)

    # 遍历文件夹中的每个图像文件
    for filename in tqdm(os.listdir(input_path)):
        # 构建图像文件的完整路径
        image_file = os.path.join(input_path, filename)

        # 打开图像文件
        image = Image.open(image_file)

        # 获取图像的宽度和高度
        width, height = image.size

        # 计算每个块的宽度和高度
        patch_width = math.floor(width / cols)
        patch_height = math.floor(height / rows)

        # 遍历每个块的行和列
        for row in range(rows):
            for col in range(cols):
                # 计算当前块的边界框
                left = col * patch_width
                top = row * patch_height
                right = (col + 1) * patch_width
                bottom = (row + 1) * patch_height

                # 提取当前块的图像
                patch = image.crop((left, top, right, bottom))

                # 构造输出文件名
                filename_without_ext = os.path.splitext(filename)[0]
                patch_filename = f"{filename_without_ext}_patch_{row}_{col}.jpg"

                # 保存当前块的图像
                patch_path = os.path.join(output_path, patch_filename)
                patch.save(patch_path)

        # print(f"Saved patch: {patch_path}")
    return input_path, output_path

def leaf_indexing(leaf_path):
    documents = SimpleDirectoryReader(leaf_path).load_data()
    node_parser = SentenceSplitter.from_defaults()
    leaf_nodes = node_parser.get_nodes_from_documents(documents)
    ##需要补充一个Qdrant库(判断是否需要重新indexing)
    index = MultiModalVectorStoreIndex(leaf_nodes, show_progress=True)
    return index

def leaf_retriever(leaf_index, query_img, topk):
    """" Note: query must a imge_path"""
    # leaf_index = index
    leaf = leaf_index.as_retriever(similarity_top_k=topk, image_similarity_top_k=topk)
    response = leaf.image_to_image_retrieve(query_img)  ##query是image_path
    print('Leaf_retrieve Complete!')
    return response  ##topk=10

def AutoMerge(base_retriever):
    print('AutoMerge Complete!')
    retrieved_image_path = []
    retrieved_image_score = []
    for res_node in base_retriever:
        if isinstance(res_node.node, ImageNode):
            retrieved_image_path.append(res_node.node.image_path)
            retrieved_image_score.append(res_node.score)
        else:
            print('Wrong, please input ImageNode')

    retrieved_image_data = {}

    for path, score in zip(retrieved_image_path, retrieved_image_score):
        doc_name = path.split('\\')[-1].split('_patch')[0]  # 提取文档名称
        if doc_name in retrieved_image_data:
            retrieved_image_data[doc_name].append(score)
        else:
            retrieved_image_data[doc_name] = [score]
    
    merge_image = [[doc_name+'.jpg', sum(scores) / len(scores)] for doc_name, scores in retrieved_image_data.items()]
    print('AutoMerge Complete!')
    return merge_image

def rerank_embedding(image_path):
    # 加载预训练的 ResNet18 模型
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # 移除分类层，保留特征提取部分
    model.eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)

    # 使用模型获取图像的嵌入向量
    with torch.no_grad():
        embedding = model(image)

    return embedding

def Rerank(orig_img, parent_img, merge_image):
    orig_img_embedding = rerank_embedding(orig_img)

    similarity_scores = []
    img_files = os.listdir(parent_img)  # 获取父文件夹内的所有文件和文件夹

    for img_path, score in tqdm(merge_image):
        if img_path in img_files:  # 检查文件名是否存在于父文件夹中
            parent_img_path = os.path.join(parent_img, img_path)  # 构建完整的文件路径

            # 计算图像相似度
            img_similarity = torch.cosine_similarity(orig_img_embedding, rerank_embedding(parent_img_path))
            similarity = img_similarity.item()
            weighted_score = np.mean([similarity, score])
            print(weighted_score)
            similarity_scores.append((parent_img_path, similarity, weighted_score))

    similarity_scores.sort(key=lambda x: x[2], reverse=True)
    print("Rerank Complete!")
    return similarity_scores[0:3]   ##topk=3
    # return similarity_scores  ##topk=10

def parent_indexing(parent_path):
    documents = SimpleDirectoryReader(parent_path).load_data()
    node_parser = SentenceSplitter.from_defaults()
    leaf_nodes = node_parser.get_nodes_from_documents(documents)
    ##需要补充一个Qdrant库(判断是否需要重新indexing)
    index = MultiModalVectorStoreIndex(leaf_nodes, show_progress=True)
    return index

def parent_retriever(parent_index, rerank_list):
    """" Note: query must a imge_path"""
    # leaf_index = index
    parent_response = []
    average_score = []
    for path, score, score2 in rerank_list:
        # parent_retriever(parent_index, path)
        parent = parent_index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
        response = parent.image_to_image_retrieve(path)  ##query是image_path
        parent_response.append(response)
        average_score.append(score2)
    print('Parent_retrieve Complete!')
    return parent_response, average_score  ##topk=10

def plot_images(image_paths):
    # Initialize a counter to track the number of images shown
    images_shown = 0
    # Set the figure size for the entire plot
    plt.figure(figsize=(16, 9))
    # Iterate through each image path in the provided list
    for img_path in image_paths:
        # Check if the file exists
        if os.path.isfile(img_path):
            # Open the image using the Image module
            image = Image.open(img_path)
            # Create a subplot for the current image in a 2x3 grid
            plt.subplot(2, 3, images_shown + 1)
            # Display the image in the subplot
            plt.imshow(image)
            # Remove x and y ticks for clarity
            plt.xticks([])
            plt.yticks([])
            # Increment the counter for images shown
            images_shown += 1
            # Break the loop if 9 images have been shown
            if images_shown >= 9:
                break

def parent_retrieved_vis(parent_response, average_score):
    print(average_score)
    retrieved_image = []
    # retrieved_image_score = []
    retrieved_text = []
    retrieved_metadata = []
    for res_node in parent_response:
        for i in res_node:
            retrieved_image.append(i.metadata['file_path'])
            # retrieved_image_score.append(average_score)
            retrieved_text.append(i.text)
            retrieved_metadata.append(i.metadata)
    plot_images(retrieved_image)        
    
    return retrieved_image, retrieved_text, retrieved_metadata