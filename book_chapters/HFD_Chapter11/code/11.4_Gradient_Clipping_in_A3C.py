# Gradient clipping is a widely used technique to stabilize training in asynchronous reinforcement learning. 
# The following code demonstrates a practical implementation of gradient clipping in an A3C setup, 
# showcasing how gradients are capped to maintain stable updates during training.

def clip_gradients(optimizer, loss, max_grad_norm):
    gradients = optimizer.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_norm(g, max_grad_norm), v) for g, v in gradients]
    optimizer.apply_gradients(clipped_gradients)
