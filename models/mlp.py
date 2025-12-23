import numpy as np

class MLP:
    """
    Numpy MLP for regression to (xs, ys) in [0,1].
    """
    def __init__(self, input_dim: int, hidden1: int = 32, hidden2: int = 16, out_dim: int = 2, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden1) * 0.1
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * 0.1
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, out_dim) * 0.1
        self.b3 = np.zeros(out_dim)

        self.trained = False
        self.train_losses = []
        self.val_losses = []

        # normalization stats
        self.X_mean = None
        self.X_std = None
        self.Y_min = None
        self.Y_max = None

    @staticmethod
    def relu(x): 
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x):
        h1 = self.relu(x @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        return self.sigmoid(h2 @ self.W3 + self.b3)

    def _fit_normalizers(self, X_train, Y_train):
        self.X_mean = np.mean(X_train, axis=0)
        self.X_std = np.std(X_train, axis=0) + 1e-8
        self.Y_min = np.min(Y_train, axis=0)
        self.Y_max = np.max(Y_train, axis=0)

    def _norm_X(self, X):
        return (X - self.X_mean) / self.X_std

    def _norm_Y(self, Y):
        return (Y - self.Y_min) / (self.Y_max - self.Y_min + 1e-8)

    def _denorm_Y(self, Yn):
        return Yn * (self.Y_max - self.Y_min) + self.Y_min

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, lr=0.01, batch_size=32, lr_decay=0.95, decay_every=10, verbose_every=25):
        self._fit_normalizers(X_train, Y_train)

        Xtr = self._norm_X(X_train)
        Ytr = self._norm_Y(Y_train)
        Xva = self._norm_X(X_val)
        Yva = self._norm_Y(Y_val)

        n_train = len(Xtr)
        for epoch in range(epochs):
            current_lr = lr * (lr_decay ** (epoch // decay_every))
            idx = np.random.permutation(n_train)
            Xs = Xtr[idx]
            Ys = Ytr[idx]

            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]

                # forward
                h1 = self.relu(xb @ self.W1 + self.b1)
                h2 = self.relu(h1 @ self.W2 + self.b2)
                out = self.sigmoid(h2 @ self.W3 + self.b3)

                # mse + sigmoid derivative
                grad = (out - yb) * out * (1 - out)

                dW3 = (h2.T @ grad) / len(xb)
                db3 = np.sum(grad, axis=0) / len(xb)

                grad2 = (grad @ self.W3.T) * (h2 > 0)
                dW2 = (h1.T @ grad2) / len(xb)
                db2 = np.sum(grad2, axis=0) / len(xb)

                grad1 = (grad2 @ self.W2.T) * (h1 > 0)
                dW1 = (xb.T @ grad1) / len(xb)
                db1 = np.sum(grad1, axis=0) / len(xb)

                # update
                self.W3 -= current_lr * dW3; self.b3 -= current_lr * db3
                self.W2 -= current_lr * dW2; self.b2 -= current_lr * db2
                self.W1 -= current_lr * dW1; self.b1 -= current_lr * db1

                epoch_loss += np.mean((out - yb) ** 2)

            self.train_losses.append(epoch_loss / max(1, (n_train / batch_size)))

            val_out = self.forward(Xva)
            self.val_losses.append(np.mean((val_out - Yva) ** 2))

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:4d} | val_mse={self.val_losses[-1]:.6f}")

        self.trained = True

    def predict(self, X):
        if not self.trained:
            return np.full((len(X), 2), 0.5)
        Xn = self._norm_X(X)
        Yn = self.forward(Xn)
        return self._denorm_Y(Yn)
