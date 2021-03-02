# a tweaked implementation of DeepSORT

import numpy as np
import tensorflow as tf

from .algo import crop_image


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name='')

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph:
        print('Frozen Model: ')
        for i in import_graph.get_operations():
            print(i.name)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs)
    )


class ImageEncoder:
    '''
    extract the features from a patch inside an image
    with some random pretrained network (mars-small128 doesn't
    sound like a good idea for traffic flow detection btw)
    '''

    def __init__(self, model_path, input_layers, output_layers):

        self.session = tf.compat.v1.Session()

        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        #self.frozen_graph = wrap_frozen_graph(
        #    graph_def,
        #    input_layers,
        #    output_layers,
        #    False
        #)

        tf.graph_util.import_graph_def(graph_def, name='net')

        default = tf.compat.v1.get_default_graph()
        self.inputs = [
            default.get_tensor_by_name(i) for i in input_layers
        ]
        self.outputs = [
            default.get_tensor_by_name(i) for i in output_layers
        ]

        self.input_dims = [i.get_shape()[1 :] for i in self.inputs]
        self.output_dims = [i.get_shape()[1 :] for i in self.outputs]

    def forward(self, data, batch_size):
        '''
        runs the model on a given set of data

        data: list, inputs corresponding to the input layers
            (e.g. [x] if there is only one layer)

        return: list, a list of tensors, each corresponding to the output
            of an output layer over the inputs
        '''
        n_data = data[0].shape[0]

        results = [
            np.zeros((n_data, dim[0]), dtype=np.float)
            for dim in self.output_dims
        ]

        for i in range(n_data // batch_size):
            data_dict = {
                k: v[i * batch_size, i * batch_size + 1]
                for k, v in zip(self.inputs, data)
            }

            out = self.session.run(self.outputs, feed_dict=data_dict)
            print(out)

        return results

    def encode(self, image, boxes, batch_size=32):
        '''
        encodes a set of bounding boxes; this method is not universal,
        and currently only assumes a model with one input layer (unlike
        the rest of this class)
        '''
        patches = [
            crop_image(image, i, self.input_dims[0][: -1]) for i in boxes
        ]

        return self.forward([np.array(patches)], batch_size)

    def close(self):
        self.session.close()
