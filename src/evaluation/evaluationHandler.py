from numpy import float96
import tensorflow as tf


class EvaluationHandler:
    def __init__(self):
        pass

    """
	Loss function when we have two tracks as output - drums and accompaniments
	"""
    @staticmethod
    def drumsLossFunction(target: tf.Tensor,prediction: tf.Tensor, alpha: float = 1.0) -> float96:
        if alpha > 1 or alpha < 0:
            raise "Alpha cannot be greater than 1 or less than 0"

        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        print("Loss for iteration = "+totalLoss)
        return totalLoss
    
    @staticmethod
    def learningRateScheduler(epoch:int, lr:float)->float:
     if epoch == 20:
       lr = 1e-4 
     return lr
