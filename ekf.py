import numpy as np


class ExtendedKalmanFilter():
    def __init__(self, f, jcb_f, h, jcb_h, P0, Q0):
        self.h = h          # obs function
        self.f = f          # model function
        self.jcb_h = jcb_h  # obs Jacobian
        self.jcb_f = jcb_f  # model Jacobian
        self.P = P0         # Covariance
        self.Q = Q0         # Process Noise Covariance

    def da(self, x, z, R):
        # get forcast value
        x_f = self.f(x)
        P_f = self.jcb_f(x).dot(self.P).dot(self.jcb_f(x).T) + self.Q
        
        # get Kalman matrix
        K = P_f.dot(self.jcb_h(x_f).T).dot(np.linalg.inv(self.jcb_h(x_f).dot(P_f).dot(self.jcb_h(x_f).T) + R))

        # calculate x and P
        self.P = (np.identity(int(np.sqrt(self.P.size))) - K.dot(self.jcb_h(x_f))).dot(P_f)
        return x_f + K.dot(z - self.h(x_f))
