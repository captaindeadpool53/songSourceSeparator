import tensorflow as tf, keras


class EvaluationHandler:
    def __init__(self):
        pass

    """
	Loss function when we have two tracks as output - drums and accompaniments
	"""
    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction5(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.5
        return EvaluationHandler.getLoss(alpha, target, prediction)

    
    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction1(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =1
        return EvaluationHandler.getLoss(alpha, target, prediction)

    
    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction0(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0
        return EvaluationHandler.getLoss(alpha, target, prediction)


    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction01(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.1
        return EvaluationHandler.getLoss(alpha, target, prediction)

    
    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction2(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.2
        return EvaluationHandler.getLoss(alpha, target, prediction)


    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction79(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.7919
        return EvaluationHandler.getLoss(alpha, target, prediction)

    
    @tf.function(reduce_retracing=True)
    @staticmethod
    def drumsLossFunction47(target: tf.Tensor,prediction: tf.Tensor) -> float:
        alpha =0.478
        return EvaluationHandler.getLoss(alpha, target, prediction)


    @tf.function(reduce_retracing=True)
    @staticmethod
    def getLoss(alpha: float, target: tf.Tensor,prediction: tf.Tensor) -> float:
        drumsTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 0] - target[..., 0]))
        accompanimentTrackLoss = tf.reduce_mean(tf.abs(prediction[..., 1] -target[...,1]))

        totalLoss = alpha * drumsTrackLoss + (1 - alpha) * accompanimentTrackLoss
        
        return totalLoss
    
    
    @staticmethod
    def learningRateScheduler(epoch:int, lr:float)->float:
     if epoch == 20:
       lr = 1e-4 
     return lr
