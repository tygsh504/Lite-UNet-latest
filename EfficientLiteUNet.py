import numpy as np
import tensorflow as tf
import keras
# --- NEW: Import EfficientNetB0 instead of MobileNetV2 ---
from tensorflow.keras.applications import EfficientNetB0
from keras import optimizers
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Activation, BatchNormalization
from keras.layers import UpSampling2D, Input, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.metrics import Recall, Precision
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt 
import os 
from datetime import datetime 
import pandas as pd 

class LrHistory(Callback):
    """Callback to log the current learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'lr' not in logs and hasattr(self.model.optimizer, 'lr'):
            lr = self.model.optimizer.lr
            if callable(lr):
                lr = lr(self.model.optimizer.iterations)
            logs['lr'] = float(tf.keras.backend.get_value(lr))

class Lite_UNet():
    
    def __init__(self, args):
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.output_dir = args.output_dir
        self.result_dir = args.result_dir 
        
        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)

    def build_model(self, weights="imagenet"):
        def decoder_block(x, residual, n_filters, n_conv_layers=2):
            up = UpSampling2D((2, 2))(x)
            merge = Concatenate()([up, residual])
            
            x = Conv2D(n_filters, (3, 3), padding="same")(merge)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            for i in range(n_conv_layers-1): 
                x = Conv2D(n_filters, (3, 3), padding="same")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            return x

        def get_encoder_layers(encoder, concat_layers, output_layer):
            return [encoder.get_layer(layer).output for layer in concat_layers], encoder.get_layer(output_layer).output

        model_input = Input(shape=(self.img_height, self.img_width, 3), name="input_img")
        
        # --- NEW: EfficientNetB0 Encoder ---
        model_encoder = EfficientNetB0(
            input_tensor=model_input, 
            weights=weights, 
            include_top=False
        )
        
        # Mapped to EfficientNetB0's specific activation layers for U-Net skip connections
        concat_layers, encoder_output = get_encoder_layers(
            model_encoder,
            [
                "input_img", 
                "block2a_expand_activation", 
                "block3a_expand_activation", 
                "block4a_expand_activation", 
                "block6a_expand_activation"
            ],
            "top_activation"
        )

        filters = [3, 48, 48, 96, 192]
        x = encoder_output

        for layer_name, n_filters in zip(concat_layers[::-1], filters[::-1]):
            x = decoder_block(x, layer_name, n_filters)

        out = Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype="float32")(x)
        
        model = Model(model_input, out)
        return model

    def iou(self, y_true, y_pred):
        smooth = 1e-6
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_coef(self, y_true, y_pred):
        smooth = 1e-6
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        
        # FIX: The denominator must add y_true AND y_pred
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def define_callbacks(self):
        if self.result_dir:
            checkpoint_path = os.path.join(self.result_dir, "best_model_weights.h5")
        else:
            checkpoint_path = self.output_dir

        my_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5
            ),
            LrHistory() 
        ]
        return my_callbacks

    def compile_model(self):
        model = self.build_model()
        base_opt = tf.keras.optimizers.Adam(self.lr, clipnorm=1.0)
        opt = base_opt
        metrics = ['accuracy', self.dice_coef, self.iou, Recall(), Precision()]
        model.compile(loss=self.dice_loss, optimizer=opt, metrics=metrics)
        return model

    def train(self, train_generator, val_generator, num_train_batches, num_val_batches):
        model = self.compile_model()
        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            steps_per_epoch=num_train_batches,
            validation_steps=num_val_batches,
            callbacks=self.define_callbacks()
        )
        
        if self.result_dir:
            last_weights_path = os.path.join(self.result_dir, "last_model_weights.h5")
            model.save_weights(last_weights_path)
            print(f"Last model weights saved to: {last_weights_path}")
            
        return model, history

    def plot_history(self, history):
        if self.result_dir:
            graphs_dir = self.result_dir
        else:
            graphs_dir = r"C:\Users\User\Desktop\Lite-UNet\training_output"

        os.makedirs(graphs_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history.get('lr', []))
        plt.title('1. Learning Rate over Epochs')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history.get('loss', []), label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('2. Loss over Epochs')
        plt.ylabel('Loss (Dice Loss)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        if 'dice_coef' in history.history:
            plt.plot(history.history['dice_coef'], label='Train Dice Coef')
        if 'val_dice_coef' in history.history:
            plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
        plt.title('3. Dice Coefficient over Epochs')
        plt.ylabel('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_graphs_{timestamp}.png"
        save_path = os.path.join(graphs_dir, filename)
        
        plt.savefig(save_path) 
        plt.close() 
        
        print(f"Training graphs successfully saved to: {save_path}")

    def save_results_to_excel(self, history):
        if self.result_dir:
            save_dir = self.result_dir
        else:
            save_dir = r"C:\Users\User\Desktop\Lite-UNet\training_output"
            
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame(history.history)
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_metrics_{timestamp}.xlsx"
        save_path = os.path.join(save_dir, filename)
        
        try:
            df.to_excel(save_path, index=False)
            print(f"Training metrics successfully saved to Excel: {save_path}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")