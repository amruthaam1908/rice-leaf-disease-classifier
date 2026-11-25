import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class VisionTransformerClassifier:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['bacterialblight', 'blast', 'brownspot', 'tungro']
        
    def create_data_generators(self, train_path, validation_path, test_path):
        """Create data generators with augmentation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation and test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.validation_generator = test_datagen.flow_from_directory(
            validation_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Training classes: {self.train_generator.class_indices}")
        print(f"Number of training samples: {self.train_generator.samples}")
        print(f"Number of validation samples: {self.validation_generator.samples}")
        print(f"Number of test samples: {self.test_generator.samples}")
        
        # Save class indices
        with open('models/class_indices.pkl', 'wb') as f:
            pickle.dump(self.train_generator.class_indices, f)
    
    def mlp_mixer_block(self, x, mlp_dim=512, dropout_rate=0.1):
        """MLP-Mixer block implementation"""
        # Token-mixing MLP
        x1 = layers.LayerNormalization()(x)
        x1 = layers.Permute((2, 1))(x1)  # Swap channel and patch dimensions
        x1 = layers.Dense(x.shape[1], activation='gelu')(x1)  # Token mixing
        x1 = layers.Permute((2, 1))(x1)  # Swap back
        x1 = layers.Dropout(dropout_rate)(x1)
        x = layers.Add()([x, x1])
        
        # Channel-mixing MLP
        x2 = layers.LayerNormalization()(x)
        x2 = layers.Dense(mlp_dim, activation='gelu')(x2)  # Channel mixing
        x2 = layers.Dropout(dropout_rate)(x2)
        x2 = layers.Dense(x.shape[-1])(x2)
        x2 = layers.Dropout(dropout_rate)(x2)
        x = layers.Add()([x, x2])
        
        return x
    
    def build_mlp_mixer_model(self, patch_size=16, hidden_dim=512, num_blocks=8, mlp_dim=512):
        """Build MLP-Mixer model"""
        inputs = layers.Input(shape=(self.img_height, self.img_width, 3))
        
        # Patch embedding
        x = layers.Conv2D(hidden_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
        x = layers.Reshape(((self.img_height // patch_size) * (self.img_width // patch_size), hidden_dim))(x)
        
        # MLP-Mixer blocks
        for _ in range(num_blocks):
            x = self.mlp_mixer_block(x, mlp_dim)
        
        # Classification head
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(4, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        return self.model
    
    def build_vit_model(self, patch_size=16, projection_dim=64, num_heads=4, transformer_layers=4):
        """Build Vision Transformer model"""
        inputs = layers.Input(shape=(self.img_height, self.img_width, 3))
        
        # Create patches
        num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        patches = layers.Conv2D(
            projection_dim, 
            kernel_size=patch_size, 
            strides=patch_size, 
            padding='valid'
        )(inputs)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
        
        # Position embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, 
            output_dim=projection_dim
        )(positions)
        
        # Add position embedding to patches
        encoded_patches = patches + position_embedding
        
        # Transformer blocks
        for _ in range(transformer_layers):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = layers.Dense(projection_dim * 2, activation='gelu')(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x3 = layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            encoded_patches = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.3)(representation)
        
        features = layers.Dense(512, activation='gelu')(representation)
        features = layers.Dropout(0.3)(features)
        features = layers.Dense(256, activation='gelu')(features)
        features = layers.Dropout(0.3)(features)
        
        outputs = layers.Dense(4, activation='softmax')(features)
        
        self.model = Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/rice_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        
        # Load the best model
        self.model = keras.models.load_model('models/rice_disease_model.h5')
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize classifier
    classifier = VisionTransformerClassifier()
    
    # Create data generators (update paths according to your dataset structure)
    classifier.create_data_generators(
        train_path='data/train',
        validation_path='data/validation',
        test_path='data/test'
    )
    
    # Choose which model to build
    print("Choose model architecture:")
    print("1. Vision Transformer (ViT)")
    print("2. MLP-Mixer")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Build and train Vision Transformer model
        classifier.build_vit_model()
        print("Training Vision Transformer model...")
    elif choice == "2":
        # Build and train MLP-Mixer model 
        classifier.build_mlp_mixer_model()
        print("Training MLP-Mixer model...")
    else:
        print("Invalid choice. Using Vision Transformer by default.")
        classifier.build_vit_model()
    
    classifier.train_model(epochs=50) 
    
    # Evaluate model
    classifier.evaluate_model()
    
    # Plot training history
    classifier.plot_training_history()

if __name__ == "__main__":
    main()