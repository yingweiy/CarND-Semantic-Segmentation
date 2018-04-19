import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    inp = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return inp, keep, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """


    def conv1x1(input):
        return tf.layers.conv2d(input, num_classes, 1, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    def upsample(input, scale=2):
        kernel_size = int(scale*4)
        return tf.layers.conv2d_transpose(input, num_classes, kernel_size, strides = (scale, scale),
                                          padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # conv 1x1, and upsample for layer 7
    layer7_fc = conv1x1(vgg_layer7_out)
    layer7_upsampled = upsample(layer7_fc)

    # conv 1x1 for layer 4
    layer4_fc = conv1x1(vgg_layer4_out)

    # skip
    layer4_merged = tf.add(layer7_upsampled, layer4_fc)
    layer4_merge_upsampled = upsample(layer4_merged)

    # layer 3 conv1x1 and  merged
    layer3_fc = conv1x1(vgg_layer3_out)
    layer3_merged = tf.add(layer4_merge_upsampled, layer3_fc)

    #l3x2 = upsample(layer3_merged)
    output = upsample(layer3_merged, scale=8)

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    #reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_term = tf.reduce_sum(tf.square(reg_vars))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=correct_label))
    #loss += 0.001*reg_term   #somehow this does not work

    optimizer = tf.train.AdamOptimizer(learning_rate)
    op = optimizer.minimize(loss)
    #tf.summary.scalar('loss', loss)

    return logits, op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    #merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter('./log',
    #                                     sess.graph)
    sess.run(tf.global_variables_initializer())


    for epoch in range(epochs):
        print('Epoch:', epoch+1)
        for image, label in get_batches_fn(batch_size):
            # training
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = {input_image: image,
                                            correct_label: label,
                                            keep_prob: 0.5,
                                            learning_rate:1e-3})
            print('Loss:', loss)

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/



    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        print('VGG Path:', vgg_path)
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        epochs = 256
        batch_size = 6

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        inp, keep, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)

        saver = tf.train.Saver()

        logits, op, loss = optimize(output, correct_label=correct_label,
                                    learning_rate=learning_rate,
                                    num_classes=num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, op, loss, inp,
                 correct_label, keep, learning_rate)

        # tf.saved_model.simple_save(sess, "./runs", )
        save_path = saver.save(sess, "./runs/model.ckpt")
        print("Model saved in path: %s" % save_path)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                       logits, keep, inp)

        # OPTIONAL: Apply the trained model to a video
        #   generate gif using the shell command below:
        #      > convert -delay 20 -loop 0 *.png segmentation.gif
        #   requires imagemagick installed on ubuntu


if __name__ == '__main__':
    run()
