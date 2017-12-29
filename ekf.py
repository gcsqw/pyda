import numpy as np


class ExtendedKalmanFilter():
    def __init__(self, f, jcb_f, h, jcb_h, P0, Q0, x0):
        self.h = h          # obs function
        self.f = f          # model function
        self.jcb_h = jcb_h  # obs Jacobian
        self.jcb_f = jcb_f  # model Jacobian
        self.P = P0         # Covariance
        self.Q = Q0         # Process Noise Covariance
        self.x = x0         # initial input

    def da(self, z, R):
        # get forcast value
        x_f = self.f(self.x)
        print(x_f)
        P_f = self.jcb_f(self.x).dot(self.P).dot(self.jcb_f(self.x).T) + self.Q
        
        # get Kalman matrix
        K = P_f.dot(self.jcb_h(x_f)).dot(np.linalg.inv(self.jcb_h(x_f).dot(P_f).dot(self.jcb_h(x_f).T) + R))
        print(K)

        # calculate x and P
        self.x = x_f + K.dot(z - self.h(x_f))
        # self.P = (np.identity(np.sqrt(self.P.size)) - K.dot(self.jcb_h(x_f))).dot(P_f)
