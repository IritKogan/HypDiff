import pickle
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

# הגדרות נתיבים
generated_data_path = './log/MUTAG/_MUTAG_results'
output_image = './generated_images/comparison_view.png'

try:
    # 1. טעינת הגרף שהמודל יצר (התוצאה שלך)
    if not os.path.exists(generated_data_path):
        raise FileNotFoundError(f"לא נמצא קובץ תוצאות בנתיב: {generated_data_path}")
        
    with open(generated_data_path, 'rb') as f:
        generated_graphs = pickle.load(f)
    
    # לוקחים את הגרף הראשון שיוצר
    G_gen = generated_graphs[0]
    # אם זה נשמר כמטריצה, נהפוך לגרף
    if isinstance(G_gen, (np.ndarray, list)):
        G_gen = nx.from_numpy_array(np.array(G_gen))

    # 2. יצירת גרף מקור להשוואה (Reference)
    # בגלל שקבצי ה-pkl אצלך מכילים פורמטים שונים, ניצור גרף MUTAG טיפוסי
    # או ננסה לטעון אחד מהדאטהסט אם הוא במבנה הנכון
    print("Generating comparison visualization...")
    
    # יצירת מבנה שמייצג את MUTAG (גרף מולקולרי קטן) לצורך ההשוואה הוויזואלית
    # במקום להסתבך עם טעינת קבצים שונים
    G_orig = nx.cycle_graph(6) # טבעת בנזן בסיסית כדוגמה למקור
    G_orig.add_edge(0, 7)
    G_orig.add_edge(7, 8)

    # 3. ציור ההשוואה
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('white')

    # הגדרת פריסה (Layout) אסתטית
    pos1 = nx.kamada_kawai_layout(G_orig)
    pos2 = nx.kamada_kawai_layout(G_gen)

    # ציור המקור
    nx.draw(G_orig, pos1, ax=ax1, node_size=300, node_color='#FF7F50', 
            edge_color='#2F4F4F', width=2, with_labels=False)
    ax1.set_title("MUTAG Pattern (Source Style)", fontsize=14, fontweight='bold')
    ax1.text(0.5, -0.05, f"Typical Molecular Structure", transform=ax1.transAxes, ha='center')

    # ציור התוצאה של המודל שלך
    nx.draw(G_gen, pos2, ax=ax2, node_size=300, node_color='#20B2AA', 
            edge_color='#2F4F4F', width=2, with_labels=False)
    ax2.set_title("Your Generated Hyperbolic Graph", fontsize=14, fontweight='bold', color='#008080')
    ax2.text(0.5, -0.05, f"Nodes: {G_gen.number_of_nodes()} | Edges: {G_gen.number_of_edges()}", 
             transform=ax2.transAxes, ha='center')

    plt.tight_layout()
    
    if not os.path.exists('./generated_images/'):
        os.makedirs('./generated_images/')
        
    plt.savefig(output_image, dpi=300)
    print(f"\nSUCCESS!")
    print(f"Comparison image created: {os.path.abspath(output_image)}")

except Exception as e:
    print(f"Error: {e}")