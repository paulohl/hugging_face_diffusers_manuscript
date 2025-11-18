# Example: Gradient clipping in A3C

def clip_gradients(optimizer, loss, max_grad_norm):
    gradients = optimizer.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_norm(g, max_grad_norm), v) for g, v in gradients]
    optimizer.apply_gradients(clipped_gradients)
