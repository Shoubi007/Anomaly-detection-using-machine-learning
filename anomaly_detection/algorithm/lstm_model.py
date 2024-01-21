from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed

from collections import deque
import sys
#nb_classes=11+1=12
class ResearchModels():
    def __init__(self, nb_classes, model, seq_length, saved_model=None,  features_length=7):
        """
        `model` = lstm (only one for this case)
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        # self.feature = feature

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-4, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=['accuracy'])

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()

        model.add(LSTM(128, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
