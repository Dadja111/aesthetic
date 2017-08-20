import sys
import numpy as np
preds_path = sys.argv[1]
trues_path = sys.argv[2]
model_data = sys.argv[3]

preds = np.load(preds_path)
trues = np.load(trues_path)

print(model_data+" histo_error", np.mean(np.sum((trues - preds)**2,axis=1)))

pred_mean = np.sum(preds*np.arange(1,11), axis=1)
true_mean = np.sum(trues*np.arange(1,11), axis=1)

print(model_data+" MSE", np.mean((true_mean - pred_mean)**2))
