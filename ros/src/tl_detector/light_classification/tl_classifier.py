from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from styx_msgs.msg import TrafficLight

import argparse
import sys

import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
	
	# load labels
	self.labels = load_labels('model/labels.txt')

	# load graph, which is stored in the default session
	self.graph = load_graph('model/graph.pb')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_data = load_image('out01620.png')
        run_graph(image_data, self.labels, 'DecodeJpeg/contents:0', 'final_result:0', 1)

        # return TrafficLight.UNKNOWN


def load_image(filename):
    """Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
          num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        #   predictions will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

    return 0

def main(argv):
    """Runs inference on an image."""
    classifier = TLClassifier()
    classifier.get_classification(None)
    classifier.get_classification(None)

if __name__ == '__main__':
    tf.app.run(main=main)


