import numpy as np
import torch

def plot_time_series_class(data, class_name, ax):

  data_mean = np.mean(data,axis=0)
  variance = np.var(data,axis=0)

  under_line = data_mean - variance
  over_line = data_mean + variance
  ax.plot(data_mean, linewidth=2)
  ax.fill_between(np.arange(data_mean.shape[0]),under_line, over_line, alpha=0.2, color='r')
  ax.set_title(class_name)

  return 

def convert2tensor(data2tensor):

  sequences = data2tensor.tolist()
  data_tensor = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])
  n_seq, seq_len, n_features = data_tensor.shape

  return data_tensor, seq_len, n_features


def predict(model, dataset):

  predictions, losses = [], []
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.unsqueeze(0)
      seq_true = seq_true.to(next(model.parameters()).device)

      seq_pred, mu, log_var = model(seq_true)
      loss = model.loss_function(seq_true, seq_pred,  mu, log_var)
      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())

  return np.array(predictions), np.array(losses)


def plot_prediction(dataset, model, title, ax):
  
  predictions, pred_losses = predict(model, [dataset])
  ax.plot(dataset, label='true', linewidth=5)
  ax.plot(predictions[0], label='reconstructed', linewidth=3)
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
  ax.legend()

  return
