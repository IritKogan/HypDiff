import pickle
import networkx as nx
import matplotlib.pyplot as plt
import os

# נתיב לקובץ התוצאות
path = './log/MUTAG/_MUTAG_results'

if not os.path.exists(path):
    print(f"Error: Could not find the results file at {path}")
else:
    with open(path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Successfully loaded {len(graphs)} generated molecules!")

    # ציור 4 המולקולות הראשונות
    plt.figure(figsize=(10, 8))
    for i, G in enumerate(graphs[:4]):
        plt.subplot(2, 2, i+1)
        # שימוש ב-Spring Layout כדי שהמולקולה תיראה טוב
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=100, node_color='teal', with_labels=False)
        plt.title(f"Generated Molecule {i+1}")
    
    print("Close the plot window to finish.")
    plt.show()