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
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    f_x = X[i].dot(W)
    loss += -f_x[y[i]] + np.log(np.sum(np.exp(f_x)))
    for j in range(num_class):
      if y[i] == j:
        dW[:, j] += -X[i, :].T + np.exp(f_x[y[i]]) / np.sum(np.exp(f_x)) * X[i, :].T
      else:
        dW[:, j] +=  np.exp(f_x[j]) / np.sum(np.exp(f_x)) * X[i, :].T 

  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += 2 * reg * W
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
  num_train = X.shape[0]
  dW = np.zeros_like(W)
  F_x = X.dot(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss += -np.sum(F_x[np.arange(num_train), y]) + np.sum(np.log(np.sum(np.exp(F_x),axis=-1)))
  loss /= num_train
  loss += reg * np.sum(W*W)
  F_sum = np.sum(np.exp(F_x), axis=-1)
  dW += (X.T / F_sum).dot(np.exp(F_x))
  minus = np.zeros(F_x.shape)
  minus[np.arange(num_train), y] = -1
  dW += X.T.dot(minus)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

