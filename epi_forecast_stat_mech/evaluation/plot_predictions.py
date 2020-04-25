# Lint as: python3
"""Plot helpers for predictions from high_level.Estimator
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt


def final_size_comparison(data,
                          predictions_dict,
                          split_day,
                          figsize=(12, 8),
                          styles=('b.', 'r+', 'go')):
  styles = itertools.cycle(styles)
  predictions_time = None
  for predictions in predictions_dict.values():
    if predictions_time is None:
      predictions_time = predictions.time
    else:
      # We require all predictions in the dictionary to have aligned time.
      np.testing.assert_array_equal(predictions_time, predictions.time)

  final_size = data.new_infections.sel(time=predictions_time).sum('time')

  plt.figure(figsize=figsize)
  for style, (name, predictions) in zip(styles, predictions_dict.items()):
    K = predictions.mean('sample').sum('time')
    plt.plot(final_size, K, style, label=name)
  plt.plot(plt.xlim(), plt.xlim(), 'k--')
  plt.legend()
  plt.title('True vs. predicted final epidemic size (post day %d)' % split_day)
  plt.xlabel('True')
  plt.ylabel('Predicted')
  plt.show()
