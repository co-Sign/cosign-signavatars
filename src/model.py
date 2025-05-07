import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, Flatten, Activation, 
    SpatialDropout1D, GlobalAveragePooling1D, MultiHeadAttention, Add, LayerNormalization,
    Concatenate, GRU, GlobalMaxPooling1D
)

class ASLRecognitionModel:
    def __init__(
        self, 
        seq_length=50, 
        feature_dim=4,
        num_classes=50,
        conv_filters=[64, 128, 256, 512],  # Added one more layer with higher capacity
        kernel_sizes=[7, 5, 3, 3],  # Larger kernels to capture more context
        pool_sizes=[2, 2, 2, 2],
        lstm_units=[768, 384],  # Increased capacity further
        dropout_rate=0.35,  # Higher dropout for better regularization
        l2_reg=0.0004,  # Adjusted L2 regularization
        use_attention=True,
        attention_heads=12,  # Increased attention heads
        use_residual=True,
        use_mixed_architecture=True  # New flag for mixed LSTM/GRU architecture
    ):
        """
        Enhanced CNN-LSTM hybrid model for HamNoSys recognition with attention
        
        Args:
            seq_length: Number of tokens in each sequence
            feature_dim: Dimension of features for each token
            num_classes: Number of output classes (sign types)
            conv_filters: List of filter sizes for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            pool_sizes: List of pool sizes for each MaxPooling layer
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            use_attention: Whether to use self-attention mechanism
            attention_heads: Number of attention heads if using attention
            use_residual: Whether to use residual connections
            use_mixed_architecture: Whether to use mixed LSTM/GRU architecture
        """
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_residual = use_residual
        self.use_mixed_architecture = use_mixed_architecture
        
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build enhanced CNN-LSTM hybrid model with multi-path architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.seq_length, self.feature_dim), name='input_layer')
        
        # CNN encoder layers with parallel paths
        x = inputs
        
        # Path 1: Original CNN path with residual connections
        cnn_path = x
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(self.conv_filters, self.kernel_sizes, self.pool_sizes)
        ):
            # Conv1D layer with L2 regularization
            conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name=f'conv1d_{i+1}'
            )(cnn_path)
            
            # Batch normalization
            conv = BatchNormalization(name=f'batchnorm_{i+1}')(conv)
            
            # Activation - using LeakyReLU for better gradient flow
            conv = tf.keras.layers.LeakyReLU(alpha=0.1, name=f'leakyrelu_{i+1}')(conv)
            
            # Max pooling
            pooled = MaxPooling1D(pool_size=pool_size, name=f'maxpool_{i+1}')(conv)
            
            # Residual connection if shapes match and residual is enabled
            if self.use_residual and i > 0 and pooled.shape[-1] == cnn_path.shape[-1] and pooled.shape[1] == cnn_path.shape[1]:
                pooled = Add(name=f'residual_conv_{i+1}')([pooled, cnn_path])
            
            # Use spatial dropout instead of regular dropout
            cnn_path = SpatialDropout1D(self.dropout_rate, name=f'dropout_{i+1}')(pooled)
        
        # Path 2 (optional): Direct sequence processing path for short-range features
        if self.use_mixed_architecture:
            seq_path = Conv1D(
                filters=128,
                kernel_size=1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name='seq_conv'
            )(inputs)
            seq_path = BatchNormalization(name='seq_batchnorm')(seq_path)
            seq_path = tf.keras.layers.LeakyReLU(alpha=0.1, name='seq_leakyrelu')(seq_path)
            seq_path = SpatialDropout1D(self.dropout_rate, name='seq_dropout')(seq_path)
        
        # Self-attention layers
        if self.use_attention:
            # Multi-head self-attention on CNN path
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=cnn_path.shape[-1] // self.attention_heads,
                dropout=self.dropout_rate,
                name='self_attention'
            )(cnn_path, cnn_path)
            
            # Add & norm (residual connection and layer normalization)
            attention_output = Add(name='attention_residual')([attention_output, cnn_path])
            cnn_path = LayerNormalization(name='attention_layernorm')(attention_output)
            
            # Second attention layer for more complex pattern recognition
            attention_output2 = MultiHeadAttention(
                num_heads=self.attention_heads // 2,
                key_dim=cnn_path.shape[-1] // (self.attention_heads // 2),
                dropout=self.dropout_rate,
                name='self_attention2'
            )(cnn_path, cnn_path)
            
            # Add & norm (residual connection and layer normalization)
            attention_output2 = Add(name='attention_residual2')([attention_output2, cnn_path])
            cnn_path = LayerNormalization(name='attention_layernorm2')(attention_output2)
        
        # Recurrent layers processing
        # Path 1: LSTM layers with residual connections
        lstm_out = cnn_path
        for i, units in enumerate(self.lstm_units):
            # Return sequences for all but the last LSTM layer
            return_sequences = i < len(self.lstm_units) - 1
            
            # Bidirectional LSTM
            lstm_layer = Bidirectional(
                LSTM(units, return_sequences=return_sequences, 
                     dropout=self.dropout_rate/2, recurrent_dropout=self.dropout_rate/4,
                     kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
                name=f'bilstm_{i+1}'
            )(lstm_out)
            
            # Add residual connection if possible and enabled
            if self.use_residual and return_sequences and lstm_layer.shape[-1] == lstm_out.shape[-1]:
                lstm_layer = Add(name=f'residual_{i+1}')([lstm_layer, lstm_out])
            
            # Layer normalization
            lstm_layer = LayerNormalization(name=f'layer_norm_{i+1}')(lstm_layer)
            
            lstm_out = lstm_layer
        
        # Path 2 (optional): GRU path for different feature extraction
        if self.use_mixed_architecture:
            gru_out = seq_path
            gru_layer = Bidirectional(
                GRU(256, return_sequences=True,
                    dropout=self.dropout_rate/2, recurrent_dropout=self.dropout_rate/4,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
                name='bigru_1'
            )(gru_out)
            gru_out = LayerNormalization(name='gru_norm_1')(gru_layer)
            
            # Global pooling for sequence data
            gru_max_pool = GlobalMaxPooling1D(name='gru_max_pool')(gru_out)
            gru_avg_pool = GlobalAveragePooling1D(name='gru_avg_pool')(gru_out)
            gru_pooled = Concatenate(name='gru_pooled')([gru_max_pool, gru_avg_pool])
        
        # Combine paths if using mixed architecture
        if self.use_mixed_architecture and len(self.lstm_units) > 0:
            if len(lstm_out.shape) == 3:  # If lstm_out has 3 dimensions
                lstm_pooled = GlobalAveragePooling1D(name='lstm_global_pooling')(lstm_out)
            else:
                lstm_pooled = lstm_out
            
            # Combine features from both paths
            x = Concatenate(name='combined_features')([lstm_pooled, gru_pooled])
        else:
            # Use only the LSTM path
            if len(lstm_out.shape) == 3:  # If lstm_out has 3 dimensions
                x = GlobalAveragePooling1D(name='global_pooling')(lstm_out)
            else:
                x = lstm_out
        
        # Final dense layers with improved architecture
        x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='dense_1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='dense_leakyrelu_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense')(x)
        
        # Second dense layer for better feature extraction
        x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='dense_2')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='dense_leakyrelu_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense_2')(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output',
                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='cnn_lstm_asl')
        
        return model
    
    def compile_model(self, learning_rate=0.001, clipnorm=1.0):
        """
        Compile the model with specified settings
        
        Args:
            learning_rate: Learning rate for optimizer
            clipnorm: Gradient clipping norm
        """
        # Use a fixed learning rate instead of cosine decay for easier logging
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=clipnorm,
            beta_1=0.9,   # Momentum term
            beta_2=0.999, # RMSprop term
            epsilon=1e-7  # Stability constant
        )
        
        self.model.compile(
            optimizer=optimizer,
            # Focal loss helps with class imbalance
            loss=self._categorical_focal_loss(alpha=0.3, gamma=2.5),
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
    
    def _categorical_focal_loss(self, alpha=0.25, gamma=2.0):
        """
        Focal loss for better handling of hard examples
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            
        Returns:
            Focal loss function
        """
        def categorical_focal_loss(y_true, y_pred):
            # Clip predictions to prevent NaN
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
            
            # Calculate focal loss
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = tf.pow(1.0 - y_pred, gamma) * y_true
            focal_loss = alpha * weight * cross_entropy
            
            return tf.reduce_sum(focal_loss, axis=-1)
        
        return categorical_focal_loss
    
    def get_callbacks(self, output_dir, patience=30):
        """
        Get callbacks for training with enhanced monitoring
        
        Args:
            output_dir: Directory to save model checkpoints
            patience: Patience for early stopping
            
        Returns:
            List of Keras callbacks
        """
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Early stopping callback - increased patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint for best model
        model_checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Model checkpoint for latest model
        model_checkpoint_latest = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_latest.keras'),
            monitor='val_accuracy',
            save_best_only=False,
            verbose=1
        )
        
        # Learning rate scheduler with plateau reduction
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,  # More gradual reduction
            patience=patience // 4,
            min_lr=1e-6,
            verbose=1
        )
        
        # TensorBoard logging
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # Learning rate and weight tracking callback
        class GradientTracker(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                # Simply log the current learning rate directly - much simpler now
                if hasattr(self.model.optimizer, 'lr'):
                    # Get learning rate value 
                    lr_value = self.model.optimizer.lr
                    if hasattr(lr_value, 'numpy'):
                        lr_value = float(lr_value.numpy())
                    logs['learning_rate'] = lr_value
        
        # Return all callbacks
        return [
            early_stopping, 
            model_checkpoint_best,
            model_checkpoint_latest,
            lr_schedule, 
            tensorboard, 
            GradientTracker()
        ]
    
    def summary(self):
        """
        Print model summary
        """
        return self.model.summary()
    
    def fit(self, train_dataset, validation_dataset, epochs=100, callbacks=None, class_weights=None):
        """
        Train the model
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: TensorFlow dataset for validation
            epochs: Number of epochs for training
            callbacks: List of Keras callbacks
            class_weights: Dictionary of class weights for imbalanced data
            
        Returns:
            Training history
        """
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model
        
        Args:
            test_dataset: TensorFlow dataset for testing
            
        Returns:
            Test loss and accuracy
        """
        return self.model.evaluate(test_dataset)
    
    def predict(self, x):
        """
        Make predictions
        
        Args:
            x: Input data
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(x)
    
    def save(self, filepath):
        """
        Save the model to file
        
        Args:
            filepath: Path to save the model
        """
        # Update file extension to .keras if it's still .h5
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.keras')
        self.model.save(filepath, save_format='keras')
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = tf.keras.models.load_model(filepath)
        return model 