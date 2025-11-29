import os
import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

def extract_features(path, model, processor, device):
    files = glob.glob(os.path.join(path, "*"))
    if not files: return None

    features = []
    print(f"[Info] Processing {len(files)} images from: {path}")

    for f in files:
        try:
            image = Image.open(f).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs).squeeze().cpu().numpy()
                features.append(feat)
        except Exception:
            continue

    return np.array(features) if features else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', default=["/workspace/GenAI/result_apple/original_resized/apple", "/workspace/GenAI/result_apple/blended/apple", "/workspace/GenAI/result_apple/generated/apple"])
    parser.add_argument('--labels', nargs='+', default=["Original", "Blended", "Generated"])
    parser.add_argument('--output', default="pca_result.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    data = {}
    for path, label in zip(args.dirs, args.labels):
        feats = extract_features(path, model, processor, device)
        if feats is not None:
            data[label] = feats

    if "Original" in data and len(data["Original"]) > 1:
        print("\n[Info] Calculating Eigenvectors based on 'Original' dataset...")
        pca = PCA(n_components=2)
        
        pca.fit(data["Original"]) #Eigenvectors of original
        
        plt.figure(figsize=(10, 8))
        colors = {'Original': 'blue', 'Blended': 'green', 'Generated': 'red'}
        markers = {'Original': 'o', 'Blended': '^', 'Generated': 'x'}

        for label, feats in data.items():
            proj = pca.transform(feats) #Transform 
            c = colors.get(label, 'gray')
            m = markers.get(label, '.')
            
            plt.scatter(proj[:, 0], proj[:, 1], alpha=0.6, label=label, c=c, marker=m)

        plt.title(f"CLIP Feature Space (Basis: Original) [{device.upper()}]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(args.output)
        print(f"[Success] Saved to {args.output}")
    else:
        print("[Error] 'Original' data is missing or insufficient to fit PCA.")

if __name__ == "__main__":
    main()