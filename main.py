import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
# בדיקה מהירה שהצלחנו להעביר ל-CPU
print(f"Using device: {'cpu' if not torch.cuda.is_available() else 'gpu'}")

import sys

# מניעת טעינה של רכיבי DGL בעייתיים ב-Windows
os.environ['DGLBACKEND'] = 'pytorch'

# פתרון עוקף לשגיאת ה-Graphbolt ב-Windows
try:
    import dgl.graphbolt
    def mock_load(): pass
    dgl.graphbolt.load_graphbolt = mock_load
except:
    pass
from config import parser
from diff import hyperdiff,test,hyperdiff_graphset,test_graphset

import os
import time

args = parser.parse_args()
os.environ['DATAPATH'] = 'data/'
os.environ['LOG_DIR'] = 'logs/'+args.taskselect+'/'

if args.taskselect=='lptask':
    from lp_train import train
    if args.type=='train': 
        #  Hyperbolic Geometric Auto-encoding
        train(args) 
        # Hyperbolic Geometric Diffusion Process
        hyperdiff(args)

    elif args.type=='test':
        # test based on a trained model
        test(args) 

elif args.taskselect=='graphtask': 
    from graph_train import graph_train
    #  an easy encoder: encoding from adj
    args.device = 'cpu'
    args.cuda = -1
    args.UseGPU = False
    graph_train(args)
    from diff_graphset import diff_train
    diff_train(args)