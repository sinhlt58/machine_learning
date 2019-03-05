import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  C = W.shape[1]

  for i in range(N):
    z = X[i].dot(W)
    z -= np.max(z)
    ez = np.exp(z)
    sumez = np.sum(ez)
    
    loss += -z[y[i]] + np.log(sumez)

    dz = ez / sumez
    dz[y[i]] += -1

    dw = np.dot(X[i].T.reshape(D, 1), dz.reshape(1, C))

    dW += dw

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  Z = np.dot(X, W)
  Z -= np.max(Z, axis=1).reshape(N, 1)
  EZ = np.exp(Z)
  SUMEZ = np.sum(EZ, axis=1)

  loss += np.sum(-Z[np.arange(N), y] + np.log(SUMEZ))
  loss /= N
  loss += 0.5 * reg * np.sum(W*W)

  dZ = EZ / SUMEZ.reshape(N, 1)
  dZ[np.arange(N), y] -= 1

  dW += np.dot(X.T, dZ)
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

