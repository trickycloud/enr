import tflite_runtime.interpreter as tflite
import os

base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'model.tflite')
label_path = os.path.join(base_dir, 'labels.txt')


class Model(object):
    def __init__(self, model=model_path, labels=label_path):
        self.interpreter = tflite.Interpreter(model)
        self.interpreter.allocate_tensors()
        self.labels = self.read_labels(labels=labels)

    @staticmethod
    def read_labels(labels=label_path):
        names = {}
        with open(labels, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def load_labels(self):
        return self.labels

    def load_interpreter(self):
        return self.interpreter

    def input_details(self):
        return self.interpreter.get_input_details()

    def output_details(self):
        return self.interpreter.get_output_details()