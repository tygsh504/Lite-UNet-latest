from Dataloader import DataLoader
from EfficientLiteUNet import Lite_UNet
import argparse
import tensorflow as tf

def main():
    # Check if GPU is detected
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if len(gpus) == 0:
        print("WARNING: No GPU detected. TensorFlow is running on the CPU!")

    parser = argparse.ArgumentParser(description='Lite-UNet training script.')

    ## Arguments for Dataloader
    parser.add_argument('--train_data',default=r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\train\imgs"
                        ,type=str,help='Training images path')
    parser.add_argument('--train_annot',default=r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\train\masks"
                        ,type=str,help='Training masks path')
    parser.add_argument('--val_data',default=r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\imgs"
                        ,type=str,help='Validation images path')
    parser.add_argument('--val_annot',default=r"C:\Users\User\Desktop\EfficientUNet--\LeafDisease\val\masks"
                        ,type=str,help='Validation masks path')

    parser.add_argument('--img_width',default=480,type=int)
    parser.add_argument('--img_height',default=640,type=int)

    # Arguments for lite-UNet model
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training.')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for optimizers.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--output_dir', default='Lite-UNet_model', type=str, help='Path for saving the model after training.')

    # Directory to store Excel results and Graphs
    parser.add_argument('--result_dir', default=r"C:\Users\User\Desktop\Lite-UNet\training_output", 
                        type=str, help='Directory to store graphs and excel results.')

    args = parser.parse_args()

    dl=DataLoader(args) 
    train_gen,val_gen=dl.data_generator()
    train_steps=dl.get_train_steps_per_epoch()
    val_steps=dl.get_validation_steps_per_epoch()

    LiteUNet=Lite_UNet(args)
    # Capture the model and history object returned by train 
    model, history = LiteUNet.train(train_gen,val_gen,train_steps,val_steps)

    # Plot and save the training history
    LiteUNet.plot_history(history)

    # Save metrics to Excel
    LiteUNet.save_results_to_excel(history)

# This prevents the infinite looping on Windows!
if __name__ == '__main__':
    main()