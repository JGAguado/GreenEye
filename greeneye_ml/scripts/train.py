import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from preprocess import load_datasets

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(num_classes):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # Species model
    species_train, species_val = load_datasets('data/species')
    num_species = species_train.element_spec[1].shape[-1]
    species_model = build_model(num_species)
    species_model.fit(species_train, validation_data=species_val, epochs=10)
    species_model.save(os.path.join(MODEL_DIR, 'species_model'))

    # Disease model
    disease_train, disease_val = load_datasets('data/disease')
    num_disease = disease_train.element_spec[1].shape[-1]
    disease_model = build_model(num_disease)
    disease_model.fit(disease_train, validation_data=disease_val, epochs=10)
    disease_model.save(os.path.join(MODEL_DIR, 'disease_model'))


if __name__ == '__main__':
    main()