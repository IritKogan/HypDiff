import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"{r['method']:8s} p{r['p_percent']}  auc={float(r['auc']):.3f}  "
          f"removed={float(r['raw_removed_mean']):.3f}  null={float(r['raw_null_mean']):.3f}")