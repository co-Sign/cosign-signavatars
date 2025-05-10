import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, Flatten, Activation, 
    SpatialDropout1D, GlobalAveragePooling1D, MultiHeadAttention, Add, LayerNormalization,
    Concatenate, GRU, GlobalMaxPooling1D, TimeDistributed, Reshape, Permute, RepeatVector
)

class ASLRecognitionModel:
    def __init__(
        self, 
        seq_length=50, 
        feature_dim=4,
        num_classes=50,
        conv_filters=[64, 128, 256, 512, 768],  # Added more layers with higher capacity
        kernel_sizes=[9, 7, 5, 3, 3],  # Larger kernels to capture more temporal context
        pool_sizes=[2, 2, 2, 2, 2],
        lstm_units=[1024, 512, 256],  # Increased capacity with more layers
        dropout_rate=0.4,  # Higher dropout for better regularization
        l2_reg=0.0005,  # Adjusted L2 regularization
        use_attention=True,
        attention_heads=16,  # Increased attention heads
        use_residual=True,
        use_mixed_architecture=True,  # Using mixed LSTM/GRU architecture
        use_self_attention=True  # New: Additional self-attention layers
    ):
        """
        Enhanced CNN-LSTM hybrid model for HamNoSys recognition with advanced attention mechanisms
        
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
            use_attention: Whether to use multi-head attention mechanism
            attention_heads: Number of attention heads if using attention
            use_residual: Whether to use residual connections
            use_mixed_architecture: Whether to use mixed LSTM/GRU architecture
            use_self_attention: Whether to use self-attention mechanisms
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
        self.use_self_attention = use_self_attention
        
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build enhanced CNN-LSTM hybrid model with multi-path architecture and advanced attention
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.seq_length, self.feature_dim), name='input_layer')
        
        # === CNN ENCODER BRANCH ===
        
        # Path 1: Deep CNN path with residual connections
        cnn_path = inputs
        cnn_tensors = []  # Store intermediate activations for skip connections
        
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
            
            # Save activation for skip connections
            cnn_tensors.append(conv)
            
            # Max pooling
            pooled = MaxPooling1D(pool_size=pool_size, name=f'maxpool_{i+1}')(conv)
            
            # Residual connection if shapes match and residual is enabled
            if self.use_residual and i > 0 and pooled.shape[-1] == cnn_path.shape[-1] and pooled.shape[1] == cnn_path.shape[1]:
                pooled = Add(name=f'residual_conv_{i+1}')([pooled, cnn_path])
            
            # Use spatial dropout for better regularization of conv features
            cnn_path = SpatialDropout1D(self.dropout_rate/2, name=f'dropout_{i+1}')(pooled)
        
        # === PARALLEL PATHWAY ===
        
        # Path 2: Parallel processing with faster temporal awareness
        if self.use_mixed_architecture:
            # Initial projection to capture different patterns
            seq_path = Conv1D(
                filters=256,
                kernel_size=3,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name='seq_conv_1'
            )(inputs)
            seq_path = BatchNormalization(name='seq_batchnorm_1')(seq_path)
            seq_path = tf.keras.layers.LeakyReLU(alpha=0.1, name='seq_leakyrelu_1')(seq_path)
            
            # Second conv layer to increase feature depth
            seq_path = Conv1D(
                filters=384,
                kernel_size=5,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name='seq_conv_2'
            )(seq_path)
            seq_path = BatchNormalization(name='seq_batchnorm_2')(seq_path)
            seq_path = tf.keras.layers.LeakyReLU(alpha=0.1, name='seq_leakyrelu_2')(seq_path)
            seq_path = SpatialDropout1D(self.dropout_rate/2, name='seq_dropout')(seq_path)
        
        # === SELF-ATTENTION MECHANISM ===
        
        # Apply self-attention if requested
        if self.use_self_attention:
            # Multi-head self-attention on CNN path
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=cnn_path.shape[-1] // min(self.attention_heads, cnn_path.shape[-1]),
                dropout=self.dropout_rate/3,
                name='self_attention_1'
            )(cnn_path, cnn_path)
            
            # Add & norm (residual connection and layer normalization)
            attention_output = Add(name='attention_residual_1')([attention_output, cnn_path])
            cnn_path = LayerNormalization(epsilon=1e-6, name='attention_layernorm_1')(attention_output)
            
            # Second attention layer with more heads for finer pattern recognition
            attention_output2 = MultiHeadAttention(
                num_heads=self.attention_heads // 2,
                key_dim=cnn_path.shape[-1] // min(self.attention_heads // 2, cnn_path.shape[-1]),
                dropout=self.dropout_rate/3,
                name='self_attention_2'
            )(cnn_path, cnn_path)
            
            # Add & norm (residual connection and layer normalization)
            attention_output2 = Add(name='attention_residual_2')([attention_output2, cnn_path])
            cnn_path = LayerNormalization(epsilon=1e-6, name='attention_layernorm_2')(attention_output2)
        
        # === SKIP CONNECTION FUSION ===
        
        # Resample and concatenate CNN tensors for enhanced feature fusion
        if len(cnn_tensors) > 2:
            # Match dimensions for concatenation
            feature_dim = cnn_path.shape[-1]
            reshaped_tensors = []
            
            for i, tensor in enumerate(cnn_tensors[-3:]):  # Use only the last 3 to avoid too much low-level info
                # Project tensor to match feature dimension
                proj = Conv1D(feature_dim, kernel_size=1, padding='same', name=f'proj_{i}')(tensor)
                
                # Match sequence length through pooling or interpolation
                if tensor.shape[1] > cnn_path.shape[1]:
                    # Downsample if longer
                    pool_size = tensor.shape[1] // cnn_path.shape[1]
                    if pool_size > 1:
                        proj = MaxPooling1D(pool_size=pool_size, name=f'skip_pool_{i}')(proj)
                    
                    # Adjust with additional 1x1 conv if needed
                    if proj.shape[1] != cnn_path.shape[1]:
                        # Use reshape and dense to match dimensions exactly
                        proj = TimeDistributed(Dense(cnn_path.shape[-1], name=f'skip_dense_{i}'))(proj)
                        proj = Reshape((cnn_path.shape[1], cnn_path.shape[-1]), name=f'skip_reshape_{i}')(proj[:, :cnn_path.shape[1], :])
                
                reshaped_tensors.append(proj)
            
            # Concatenate valid tensors with main path
            valid_tensors = [t for t in reshaped_tensors if t.shape[1] == cnn_path.shape[1]]
            if valid_tensors:
                cnn_path = Concatenate(axis=-1, name='skip_concat')([cnn_path] + valid_tensors)
                # Project back to original dimension
                cnn_path = Conv1D(feature_dim, kernel_size=1, padding='same', name='post_skip_proj')(cnn_path)
                cnn_path = BatchNormalization(name='post_skip_bn')(cnn_path)
                cnn_path = tf.keras.layers.LeakyReLU(alpha=0.1, name='post_skip_act')(cnn_path)
        
        # === RECURRENT LAYERS ===
        
        # Path 1: Stacked Bidirectional LSTM with residual connections
        lstm_out = cnn_path
        for i, units in enumerate(self.lstm_units):
            # Return sequences for all but the last LSTM layer
            return_sequences = i < len(self.lstm_units) - 1
            
            # Bidirectional LSTM with optimized dropout settings
            lstm_layer = Bidirectional(
                LSTM(units, return_sequences=return_sequences, 
                     dropout=self.dropout_rate/2, recurrent_dropout=self.dropout_rate/4,
                     kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                     recurrent_regularizer=tf.keras.regularizers.l2(self.l2_reg/2)),
                name=f'bilstm_{i+1}'
            )(lstm_out)
            
            # Add residual connection if compatible
            if self.use_residual and return_sequences and lstm_layer.shape[-1] == lstm_out.shape[-1]:
                lstm_layer = Add(name=f'residual_lstm_{i+1}')([lstm_layer, lstm_out])
            
            # Layer normalization is crucial for stable training with residuals
            lstm_layer = LayerNormalization(epsilon=1e-6, name=f'layer_norm_lstm_{i+1}')(lstm_layer)
            lstm_out = lstm_layer
        
        # Path 2: Parallel GRU path for complementary temporal features
        if self.use_mixed_architecture:
            gru_out = seq_path
            
            # First GRU layer (return sequences for stacking)
            gru_layer1 = Bidirectional(
                GRU(384, return_sequences=True,
                    dropout=self.dropout_rate/2, recurrent_dropout=self.dropout_rate/4,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                    recurrent_regularizer=tf.keras.regularizers.l2(self.l2_reg/2)),
                name='bigru_1'
            )(gru_out)
            gru_out = LayerNormalization(epsilon=1e-6, name='gru_norm_1')(gru_layer1)
            
            # Second GRU layer with different units to capture different patterns
            gru_layer2 = Bidirectional(
                GRU(256, return_sequences=True,
                    dropout=self.dropout_rate/2, recurrent_dropout=self.dropout_rate/4,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                    recurrent_regularizer=tf.keras.regularizers.l2(self.l2_reg/2)),
                name='bigru_2'
            )(gru_out)
            gru_out = LayerNormalization(epsilon=1e-6, name='gru_norm_2')(gru_layer2)
            
            # Multiple pooling strategies for better feature extraction
            gru_max_pool = GlobalMaxPooling1D(name='gru_max_pool')(gru_out)
            gru_avg_pool = GlobalAveragePooling1D(name='gru_avg_pool')(gru_out)
            gru_pooled = Concatenate(name='gru_pooled')([gru_max_pool, gru_avg_pool])
        
        # === FEATURE FUSION ===
        
        # Combine paths if using mixed architecture
        if self.use_mixed_architecture and len(self.lstm_units) > 0:
            if len(lstm_out.shape) == 3:  # If lstm_out has 3 dimensions
                # Multiple pooling for richer features
                lstm_max_pool = GlobalMaxPooling1D(name='lstm_max_pool')(lstm_out)
                lstm_avg_pool = GlobalAveragePooling1D(name='lstm_avg_pool')(lstm_out)
                lstm_pooled = Concatenate(name='lstm_pooled')([lstm_max_pool, lstm_avg_pool])
            else:
                lstm_pooled = lstm_out
            
            # Combine features from both paths
            x = Concatenate(name='combined_features')([lstm_pooled, gru_pooled])
        else:
            # Use only the LSTM path
            if len(lstm_out.shape) == 3:  # If lstm_out has 3 dimensions
                lstm_max_pool = GlobalMaxPooling1D(name='lstm_max_pool')(lstm_out)
                lstm_avg_pool = GlobalAveragePooling1D(name='lstm_avg_pool')(lstm_out)
                x = Concatenate(name='lstm_pooled')([lstm_max_pool, lstm_avg_pool])
            else:
                x = lstm_out
        
        # === CLASSIFICATION HEAD ===
        
        # Deep MLP classifier head
        x = Dense(768, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='dense_1')(x)
        x = BatchNormalization(name='dense_bn_1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='dense_leakyrelu_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense_1')(x)
        
        # Add a second dense layer
        x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='dense_2')(x)
        x = BatchNormalization(name='dense_bn_2')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='dense_leakyrelu_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_dense_2')(x)
        
        # Add a third dense layer for better feature abstraction
        x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='dense_3')(x)
        x = BatchNormalization(name='dense_bn_3')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='dense_leakyrelu_3')(x)
        x = Dropout(self.dropout_rate/2, name='dropout_dense_3')(x)
        
        # Output layer with proper regularization
        outputs = Dense(self.num_classes, activation='softmax', name='output',
                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='enhanced_cnn_bilstm_asl')
        
        return model
    
    def compile_model(self, learning_rate=0.001, clipnorm=1.0):
        """
        Compile the model with improved optimization settings
        
        Args:
            learning_rate: Learning rate for optimizer
            clipnorm: Gradient clipping norm
        """
        # Use AdamW optimizer for better weight decay handling
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=clipnorm,
            beta_1=0.9,   # Momentum term
            beta_2=0.999, # RMSprop term
            epsilon=1e-7  # Stability constant
        )
        
        self.model.compile(
            optimizer=optimizer,
            # Combined loss: focal loss for class imbalance and categorical crossentropy
            loss=self._combined_loss(),
            metrics=[
                'accuracy', 
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'), 
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.AUC(name='auc')  # Area under ROC curve
            ]
        )
    
    def _combined_loss(self, focal_weight=0.7, gamma=2.0, alpha=0.25):
        """
        Combined loss function: focal loss + categorical crossentropy
        
        Args:
            focal_weight: Weight of focal loss component
            gamma: Focusing parameter for focal loss
            alpha: Alpha parameter for focal loss
            
        Returns:
            Combined loss function
        """
        def loss_fn(y_true, y_pred):
            # Clip predictions for numerical stability
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
            
            # Categorical crossentropy component
            cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
            
            # Focal loss component
            focal_factor = tf.pow(1.0 - y_pred, gamma) * y_true
            focal = -alpha * focal_factor * tf.math.log(y_pred)
            focal_loss = tf.reduce_sum(focal, axis=-1)
            
            # Combine losses with weighting
            return focal_weight * focal_loss + (1 - focal_weight) * cce
        
        return loss_fn
    
    def get_callbacks(self, output_dir, patience=40):
        """
        Get enhanced callbacks for improved training and monitoring
        
        Args:
            output_dir: Directory to save model checkpoints
            patience: Patience for early stopping
            
        Returns:
            List of Keras callbacks
        """
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Early stopping callback with increased patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint for best model by accuracy
        model_checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Model checkpoint for lowest validation loss
        model_checkpoint_best_loss = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_best_loss.keras'),
            monitor='val_loss',
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
        
        # Learning rate scheduler with more aggressive reduction
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More aggressive reduction
            patience=patience // 5,  # Sooner LR changes
            min_lr=1e-7,
            verbose=1
        )
        
        # Cosine decay restart scheduler for better optimization
        cosine_decay = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr if epoch < 5 else  # Keep initial LR for first 5 epochs
            tf.keras.experimental.CosineDecayRestarts(
                initial_learning_rate=lr,
                first_decay_steps=10,
                t_mul=2.0,
                m_mul=0.9,
                alpha=1e-2
            )(epoch-5),
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
                # Log the current learning rate
                if hasattr(self.model.optimizer, 'lr'):
                    lr_value = self.model.optimizer.lr
                    if hasattr(lr_value, 'numpy'):
                        lr_value = float(lr_value.numpy())
                    logs['learning_rate'] = lr_value
        
        # Return all callbacks
        return [
            early_stopping, 
            model_checkpoint_best,
            model_checkpoint_best_loss,
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