import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

class FusionModel:
    def __init__(self, input_shape=(224, 224, 3), metadata_dim=2, num_classes=2):
        self.input_shape = input_shape
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        # Image branch
        image_input = tf.keras.Input(shape=self.input_shape)
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=image_input)
        for layer in base_model.layers[:-20]:  # Fine-tune last 20 layers
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Metadata branch
        metadata_input = tf.keras.Input(shape=(self.metadata_dim,))
        y = layers.Dense(32, activation='relu')(metadata_input)
        y = layers.Dropout(0.3)(y)

        # Fusion
        combined = layers.Concatenate()([x, y])
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dense(128, activation='relu')(combined)
        outputs = layers.Dense(self.num_classes, activation='softmax')(combined)

        model = Model(inputs=[image_input, metadata_input], outputs=outputs)
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='AUC')
            ]
        )

    def train(self, X_train_img, X_train_meta, y_train, X_val_img, X_val_meta, y_val):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            ModelCheckpoint(
                filepath="best_model.keras",  # ✅ Save full model
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1)
        )
        class_weight_dict = dict(enumerate(class_weights))

        history = self.model.fit(
            [X_train_img, X_train_meta], y_train,
            validation_data=([X_val_img, X_val_meta], y_val),
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        self.model.save("last_model.keras")  # ✅ Save final model
        return history

    def evaluate(self, X_test_img, X_test_meta, y_test, save_path="test_results.txt"):
        y_pred = self.model.predict([X_test_img, X_test_meta])
        y_true = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

        report = classification_report(y_true, y_pred_labels, digits=4)

        with open(save_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"AUC Score: {auc:.4f}\n")

        print("\n✅ Evaluation results saved to:", save_path)
        print(report)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
