import tensorflow as tf
from preprocess import load_datasets


def evaluate_model(model_path, data_dir):
    print(f"Evaluating model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    _, val_ds = load_datasets(data_dir)
    results = model.evaluate(val_ds)
    print(f"Results (loss, accuracy): {results}")


def main():
    evaluate_model('saved_models/species_model', 'data/species')
    evaluate_model('saved_models/disease_model', 'data/disease')


if __name__ == '__main__':
    main()