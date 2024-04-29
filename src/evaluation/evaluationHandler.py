import tensorflow as tf, keras


class EvaluationHandler:
    def __init__(self):
        pass

    """
	Loss function when we have two tracks as output - drums and accompaniments
	"""
    @tf.function
    @staticmethod
    def drumsLossFunction5(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.5
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss
    
    @tf.function
    @staticmethod
    def drumsLossFunction1(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =1
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss
    
    @staticmethod
    def drumsLossFunction0(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss

    @tf.function
    @staticmethod
    def drumsLossFunction01(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.1
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss
    
    @tf.function
    @staticmethod
    def drumsLossFunction2(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.2
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss

    @tf.function
    @staticmethod
    def drumsLossFunction79(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.7919
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss
    
    @staticmethod
    def learningRateScheduler(epoch:int, lr:float)->float:
     if epoch == 20:
       lr = 1e-4 
     return lr
