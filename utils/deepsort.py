# a tweaked implementation of DeepSORT


import tensorflow as tf


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

        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        #self.frozen_graph = wrap_frozen_graph(
        #    graph_def,
        #    input_layers,
        #    output_layers,
        #    False
        #)

        tf.graph_util.import_graph_def(graph_def)

ImageEncoder('../models/mars-small128.pb', ['images:0'], ['features:0'])