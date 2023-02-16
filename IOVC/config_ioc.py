"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed. The device can be either
   "cpu" or "gpu", which then optimizes the model accordingly after
   training or uses the correct version for inference when testing.

   Parameter split_ratio indicates the training/validation ratio for images. If these two sets are already defined
   (and put in two different directories), the value should be zero.
"""

PARAMS = {
    "n_epochs": 20,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "shuffle": True,
    "split_ratio": 0,
    "device": "gpu",
    "use_pretrain_weights": False
}