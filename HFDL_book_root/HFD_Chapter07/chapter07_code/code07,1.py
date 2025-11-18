# A common scheduling technique: 
# Reducing the learning rate by half every 10 epochs


from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch, lr):
    drop_rate = 0.5
    drop_interval = 10
    if epoch % drop_interval == 0 and epoch > 0:
        return lr * drop_rate
    return lr

lr_scheduler = LearningRateScheduler(step_decay)
model.fit(x_train, y_train, callbacks=[lr_scheduler])
