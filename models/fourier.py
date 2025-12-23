
import numpy as np
from .mlp import MLP

class FourierMLP(MLP):
    """
    FIXED: Random Fourier features with proper normalization handling.
    
    Key improvements:
    1. Stores original input_dim for proper feature computation
    2. Uses parent's normalization methods consistently
    3. Cleaner train() implementation that inherits more from parent
    """
    def __init__(self, input_dim: int, mapping_size: int = 32, scale: float = 10.0, seed: int = 0):
        # Initialize parent with EXPANDED dimensions (2 * mapping_size)
        super().__init__(input_dim=mapping_size * 2, seed=seed, hidden1=32, hidden2=16, out_dim=2)
        
        # Store original input dimension for feature computation
        self.original_input_dim = input_dim
        self.mapping_size = mapping_size
        
        # Random Fourier feature matrix B: (original_input_dim, mapping_size)
        rng = np.random.RandomState(seed)
        self.B = rng.normal(scale=scale, size=(input_dim, mapping_size))

    def compute_features(self, x):
        """
        Transform input x (after normalization) to Fourier features.
        
        Args:
            x: normalized input of shape (n_samples, original_input_dim)
        
        Returns:
            features: [cos(2π xB), sin(2π xB)] of shape (n_samples, 2*mapping_size)
        """
        x_proj = 2 * np.pi * x @ self.B
        return np.concatenate([np.cos(x_proj), np.sin(x_proj)], axis=-1)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=100, lr=0.01, 
              batch_size=32, lr_decay=0.95, decay_every=10, verbose_every=25):
        """
        FIXED: Cleaner training that properly handles feature projection.
        
        Steps:
        1. Fit normalizers on RAW input (original_input_dim)
        2. Normalize inputs
        3. Project to Fourier space (2*mapping_size)
        4. Train on projected features
        """
        # Step 1: Fit normalizers on RAW input
        self._fit_normalizers(X_train, Y_train)

        # Step 2: Normalize RAW inputs
        Xtr_norm = self._norm_X(X_train)
        Xva_norm = self._norm_X(X_val)
        Ytr_norm = self._norm_Y(Y_train)
        Yva_norm = self._norm_Y(Y_val)

        # Step 3: Project to Fourier features
        Xtr_fourier = self.compute_features(Xtr_norm)
        Xva_fourier = self.compute_features(Xva_norm)

        # Step 4: Standard training loop on Fourier features
        n_train = len(Xtr_fourier)
        
        for epoch in range(epochs):
            current_lr = lr * (lr_decay ** (epoch // decay_every))
            idx = np.random.permutation(n_train)
            Xs = Xtr_fourier[idx]
            Ys = Ytr_norm[idx]

            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]

                # Forward pass (on Fourier features)
                h1 = self.relu(xb @ self.W1 + self.b1)
                h2 = self.relu(h1 @ self.W2 + self.b2)
                out = self.sigmoid(h2 @ self.W3 + self.b3)

                # Backward pass
                grad = (out - yb) * out * (1 - out)

                dW3 = (h2.T @ grad) / len(xb)
                db3 = np.sum(grad, axis=0) / len(xb)

                grad2 = (grad @ self.W3.T) * (h2 > 0)
                dW2 = (h1.T @ grad2) / len(xb)
                db2 = np.sum(grad2, axis=0) / len(xb)

                grad1 = (grad2 @ self.W2.T) * (h1 > 0)
                dW1 = (xb.T @ grad1) / len(xb)
                db1 = np.sum(grad1, axis=0) / len(xb)

                # Update
                self.W3 -= current_lr * dW3
                self.b3 -= current_lr * db3
                self.W2 -= current_lr * dW2
                self.b2 -= current_lr * db2
                self.W1 -= current_lr * dW1
                self.b1 -= current_lr * db1

                epoch_loss += np.mean((out - yb) ** 2)

            # Record training loss
            self.train_losses.append(epoch_loss / max(1, (n_train / batch_size)))
            
            # Validation loss (on Fourier features)
            h1_val = self.relu(Xva_fourier @ self.W1 + self.b1)
            h2_val = self.relu(h1_val @ self.W2 + self.b2)
            val_out = self.sigmoid(h2_val @ self.W3 + self.b3)
            self.val_losses.append(np.mean((val_out - Yva_norm) ** 2))

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Epoch {epoch+1:4d} | val_mse={self.val_losses[-1]:.6f}")

        self.trained = True

    def predict(self, X):
        """
        FIXED: Proper prediction pipeline.
        
        Args:
            X: Raw input of shape (n_samples, original_input_dim)
        
        Returns:
            predictions: Denormalized output (n_samples, 2)
        """
        if not self.trained:
            return np.full((len(X), 2), 0.5)
        
        # Step 1: Normalize raw input
        Xn = self._norm_X(X)
        
        # Step 2: Project to Fourier features
        Xf = self.compute_features(Xn)
        
        # Step 3: Forward pass
        h1 = self.relu(Xf @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        Yn = self.sigmoid(h2 @ self.W3 + self.b3)
        
        # Step 4: Denormalize output
        return self._denorm_Y(Yn)
