# EarlyStopping   callback (Example)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[early_stopping])
