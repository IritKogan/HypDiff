import os
import glob

# נתיב התיקייה של dgl_sparse בתוך ה-venv שלך
sparse_path = r'C:\Users\iritk\Documents\GitHub\HypDiff\venv\Lib\site-packages\dgl\dgl_sparse'

# מחפשים איזה קובץ DLL כן קיים שם
existing_dlls = glob.glob(os.path.join(sparse_path, "dgl_sparse_pytorch_*.dll"))

if existing_dlls:
    src = existing_dlls[0]
    # השם שהמערכת מחפשת לפי השגיאה שלך
    dst = os.path.join(sparse_path, "dgl_sparse_pytorch_2.11.0.dll")
    
    if not os.path.exists(dst):
        import ctypes
        # יצירת קיצור דרך (דורש הרשאת כתיבה לתיקייה)
        os.system(f'mklink "{dst}" "{src}"')
        print(f"Created link from {src} to {dst}")
    else:
        print("File already exists.")
else:
    print("No DLL found in dgl_sparse directory. Try reinstalling dgl with: pip install dgl")