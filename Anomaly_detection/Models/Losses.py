import tensorflow as tf

""" -----------------------------------------------------------------------------------------
Mean Squarred Error
----------------------------------------------------------------------------------------- """ 
def MSE(y_true, y_pred):
    return tf.reduce_mean((y_pred - y_true)**2, axis=-1)

""" -----------------------------------------------------------------------------------------
PSNR 
----------------------------------------------------------------------------------------- """ 
def PSNRLoss(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


