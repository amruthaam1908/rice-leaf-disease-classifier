import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # pyright: ignore[reportMissingImports] # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard # type: ignore # type: ignore
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import datetime

class DenseNetRiceClassifier:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['bacterialblight', 'blast', 'brownspot', 'tungro']
        self.base_model = None
        
    def create_data_generators(self, train_path, validation_path, test_path):
        """Create optimized data generators with DenseNet preprocessing"""
        
        # Use DenseNet's specific preprocessing
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.2,
            fill_mode='reflect'
        )
        
        # Validation and test data generator
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
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
        os.makedirs('models', exist_ok=True)
        with open('models/class_indices.pkl', 'wb') as f:
            pickle.dump(self.train_generator.class_indices, f)
    
    def build_densenet_model(self, model_type='densenet121', freeze_backbone=True):
        """Build DenseNet model with transfer learning[citation:3][citation:10]"""
        
        # Select DenseNet variant[citation:3]
        if model_type == 'densenet121':
            self.base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif model_type == 'densenet169':
            self.base_model = tf.keras.applications.DenseNet169(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif model_type == 'densenet201':
            self.base_model = tf.keras.applications.DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        else:
            self.base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for layer in self.base_model.layers:
                layer.trainable = False
        
        # Build custom classification head
        x = self.base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Additional dense layers[citation:10]
        x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(4, activation='softmax')(x)
        
        self.model = Model(inputs=self.base_model.input, outputs=outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        print(self.model.summary())
        return self.model
    
    def unfreeze_layers(self, num_layers=50):
        """Gradually unfreeze layers for fine-tuning"""
        if self.base_model:
            # Unfreeze last N layers
            for layer in self.base_model.layers[-num_layers:]:
                layer.trainable = True
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')]
            )
            print(f"Unfroze last {num_layers} layers for fine-tuning")
    
    def train_model(self, epochs=50, fine_tune_epochs=20):
        """Train the model with comprehensive callbacks"""
        
        # Create log directory for TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/densenet_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        # Initial training with frozen backbone
        print("Initial training with frozen backbone...")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        print("\nStarting fine-tuning phase...")
        self.unfreeze_layers(num_layers=50)
        
        # Reduce learning rate for fine-tuning
        fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
            ModelCheckpoint('models/densenet_finetuned.h5', save_best_only=True)
        ]
        
        fine_tune_history = self.model.fit(
            self.train_generator,
            initial_epoch=len(self.history.history['loss']),
            epochs=len(self.history.history['loss']) + fine_tune_epochs,
            validation_data=self.validation_generator,
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])
    
    def evaluate_model(self):
        """Comprehensive model evaluation with multiple metrics"""
        
        # Load the best model
        self.model = keras.models.load_model('models/densenet_finetuned.h5')
        
        # Evaluate on test data
        print("\nEvaluating model on test data...")
        results = self.model.evaluate(self.test_generator)
        
        print(f"\nTest Metrics:")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Loss: {results[0]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        print(f"AUC: {results[4]:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Calculate and print F1 scores
        report_dict = classification_report(true_classes, predicted_classes, 
                                          target_names=self.class_names, output_dict=True)
        print("\nF1 Scores per class:")
        for class_name in self.class_names:
            print(f"{class_name}: {report_dict[class_name]['f1-score']:.4f}")
        
        # Confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes)
        
        # ROC-AUC for multi-class
        try:
            # One-hot encode true labels for AUC calculation
            true_labels = tf.keras.utils.to_categorical(true_classes, num_classes=4)
            roc_auc = roc_auc_score(true_labels, predictions, multi_class='ovr')
            print(f"\nOverall ROC-AUC (OvR): {roc_auc:.4f}")
        except:
            print("\nROC-AUC calculation requires probability scores for all classes")
        
        return results[1]  # Return accuracy
    
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   annot_kws={'size': 12})
        plt.title('Confusion Matrix - DenseNet Classifier', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2%}', 
                transform=plt.gca().transAxes, 
                ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_densenet.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        if self.history is None:
            print("No training history available")
            return
        
        history_dict = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history_dict['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history_dict['loss'], label='Training Loss')
        axes[0, 1].plot(history_dict['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history_dict:
            axes[0, 2].plot(history_dict['precision'], label='Training Precision')
            axes[0, 2].plot(history_dict['val_precision'], label='Validation Precision')
            axes[0, 2].set_title('Precision')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history_dict:
            axes[1, 0].plot(history_dict['recall'], label='Training Recall')
            axes[1, 0].plot(history_dict['val_recall'], label='Validation Recall')
            axes[1, 0].set_title('Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot AUC
        if 'auc' in history_dict:
            axes[1, 1].plot(history_dict['auc'], label='Training AUC')
            axes[1, 1].plot(history_dict['val_auc'], label='Validation AUC')
            axes[1, 1].set_title('ROC-AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history_dict:
            axes[1, 2].plot(history_dict['lr'])
            axes[1, 2].set_title('Learning Rate Schedule')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('densenet_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict disease for a single image"""
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"\nPrediction for {os.path.basename(image_path)}:")
        print(f"Disease: {self.class_names[predicted_class_idx]}")
        print(f"Confidence: {confidence:.2%}")
        
        # Display all probabilities
        print("\nAll probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {predictions[0][i]:.2%}")
        
        return predicted_class_idx, confidence

def main():
    """Main function to run the DenseNet classifier"""
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize classifier
    print("Initializing DenseNet Rice Disease Classifier...")
    classifier = DenseNetRiceClassifier(
        img_height=224,
        img_width=224,
        batch_size=32
    )
    
    # Create data generators (update paths according to your dataset structure)
    print("\nCreating data generators...")
    classifier.create_data_generators(
        train_path='data/train',
        validation_path='data/validation',
        test_path='data/test'
    )
    
    # Build DenseNet model
    print("\nBuilding DenseNet-121 model with transfer learning...")
    classifier.build_densenet_model(
        model_type='densenet121',
        freeze_backbone=True
    )
    
    # Train the model
    print("\nTraining model...")
    classifier.train_model(
        epochs=30,          # Initial training epochs
        fine_tune_epochs=20 # Fine-tuning epochs
    )
    
    # Evaluate model
    print("\nEvaluating model performance...")
    test_accuracy = classifier.evaluate_model()
    
    # Plot training history
    classifier.plot_training_history()
    
    print(f"\n{'='*50}")
    print(f"Training and evaluation complete!")
    print(f"Model saved in 'models/' directory")
    print(f"Logs available in 'logs/' directory for TensorBoard visualization")
    print(f"{'='*50}")
    
    # Optional: Test on a single image
    # classifier.predict_single_image('path/to/test/image.jpg')

if __name__ == "__main__":
    main()