import numpy as np
import networkx as nx
import matplotlib
# שימוש ב-Backend שלא דורש חלון קופץ כדי למנוע קיפאון
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

file_path = './log/MUTAG/Single_comp_generatedGraphs_adj_998.npy'
output_dir = './generated_images/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    print("Loading file... please wait.")
    adj_matrices = np.load(file_path, allow_pickle=True)
    print(f"Found {len(adj_matrices)} graphs.")

    num_to_show = min(len(adj_matrices), 10)
    
    for i in range(num_to_show):
        plt.figure(figsize=(5, 5))
        adj = adj_matrices[i]
        if hasattr(adj, "toarray"):
            adj = adj.toarray()
            
        G = nx.from_numpy_array(adj)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=100, node_color='teal', with_labels=False)
        
        save_path = os.path.join(output_dir, f'molecule_{i+1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nSuccess! Check the folder: {os.path.abspath(output_dir)}")

except Exception as e:
    print(f"Error: {e}")