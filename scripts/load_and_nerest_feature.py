import numpy as np
from annoy import AnnoyIndex
from option import args


feature = np.load(args.file, allow_pickle=True)
feature = feature.item()
t = AnnoyIndex(feature.values[0].shape[0])
for i, key in enumerate(list(feature)):
  t.add_item(feature[key])
