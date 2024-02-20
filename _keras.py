# tag::train_generator_imports[]
from __future__ import absolute_import

from keras.callbacks import ModelCheckpoint  # <1>
# tag::small_network[]
from keras.layers import Activation, Conv2D, Dense, Flatten, ZeroPadding2D
from keras.models import Sequential

from alphago.encoders import OnePlaneEncoder
from alphago.test.parallel_processor import GoDataProcessor


def layers(input_shape):
    return [
        ZeroPadding2D(
            padding=3, input_shape=input_shape, data_format="channels_first"
        ),  # <1>
        Conv2D(48, (7, 7), data_format="channels_first"),
        Activation("relu"),
        ZeroPadding2D(padding=2, data_format="channels_first"),  # <2>
        Conv2D(32, (5, 5), data_format="channels_first"),
        Activation("relu"),
        ZeroPadding2D(padding=2, data_format="channels_first"),
        Conv2D(32, (5, 5), data_format="channels_first"),
        Activation("relu"),
        ZeroPadding2D(padding=2, data_format="channels_first"),
        Conv2D(32, (5, 5), data_format="channels_first"),
        Activation("relu"),
        Flatten(),
        Dense(512),
        Activation("relu"),
    ]


go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100

encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # <1>

processor = GoDataProcessor(encoder=encoder.name())  # <2>

generator = processor.load_go_data("train", num_games, use_generator=True)  # <3>
test_generator = processor.load_go_data("test", num_games, use_generator=True)

input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
print(generator.get_num_samples())
epochs = 5
batch_size = 128
model.fit_generator(
    generator=generator.generate(batch_size, num_classes),  # <1>
    epochs=epochs,
    steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
    validation_data=test_generator.generate(batch_size, num_classes),  # <3>
    validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
    callbacks=[ModelCheckpoint("../checkpoints/small_model_epoch_{epoch}.h5")],
)  # <5>

model.evaluate_generator(
    generator=test_generator.generate(batch_size, num_classes),
    steps=test_generator.get_num_samples() / batch_size,
)  # <6>
