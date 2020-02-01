from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer, get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

class FullyConnectedNetwork(Model):
    """Generic fully connected network."""

    @override(Model)
    def _build_layers(self, inputs, num_outputs, options, context = None):
        """Process the flattened inputs.

        Note that dict inputs will be flattened into a vector. To define a
        model that processes the components separately, use _build_layers_v2().
        """

        hiddens = options.get("fcnet_hiddens")
        activation = get_activation_fn(options.get("fcnet_activation"))
        concat_input_size = options.get("concat_input_size")
        context_input_size = options.get("context_input_size")
        concat_hidden = options.get("concat_hidden")
        if context is not None:
            context = tf.reshape(context, (-1,concat_input_size))
            if options.get("concat_context"):
                app = context[0][0:context_input_size]
                app = tf.tile(app, tf.shape(inputs)[0:1])
                app = tf.reshape(app, [-1, context_input_size])
                inputs = tf.concat([inputs, app], axis =1)

        if context is not None and not options.get("concat_context"):
            with tf.variable_scope("hyper_film"):
                x_hyp = context
                hyper_hidden_sizes = options.get("hyper_hiddens")
                for idx, hidden_size in enumerate(hyper_hidden_sizes):
                    x_hyp = tf.layers.dense(x_hyp, hidden_size, name='hidden_%d' % idx,
                        activation=activation,
                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                        bias_initializer=tf.zeros_initializer())
                # Change back to normal
                x_hyp = tf.layers.dense(x_hyp, sum(concat_hidden), name='hyper_out',#tf.layers.dense(x_hyp, 2*sum(hiddens), name='hyper_out',
                    activation=None,
                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                    bias_initializer=tf.zeros_initializer())
                # Change back to normal
                film_params = tf.split(x_hyp,  [val for val in concat_hidden], axis=1)
                #film_params = tf.split(x_hyp, [val for val in hiddens for _ in (0, 1)], axis=1)
        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for j, size in enumerate(hiddens):
                label = "fc{}".format(i)
                last_layer = tf.layers.dense(
                    last_layer,
                    size,
                    kernel_initializer= tf.keras.initializers.glorot_normal(),
                    bias_initializer=tf.zeros_initializer(),
                    activation=activation,
                    name=label)
                if context is not None and not options.get("concat_context"):
                    #last_layer = tf.einsum('ij,kj->ij', last_layer, film_params.pop(0))+ film_params.pop(0)
                    #last_layer = tf.einsum('ij,kj->ij', last_layer, film_params.pop(0)+1)+ film_params.pop(0)
                    #last_layer = tf.einsum('ij,kj->ij', last_layer, 0.5 + tf.math.sigmoid(film_params.pop(0)))+ (-0.5 + tf.math.sigmoid(film_params.pop(0)))
                    #last_layer = tf.einsum('ij,kj->ij', last_layer, 4*tf.math.sigmoid(film_params.pop(0))-2)+ (-0.5 + tf.math.sigmoid(film_params.pop(0)))
                    temp = tf.tile(film_params[j][0], tf.shape(last_layer)[0:1])
                    temp = tf.reshape(temp, [-1, concat_hidden[j]])
                    last_layer = tf.concat([last_layer, temp], axis=1)
                    print(last_layer)
                i += 1
            label = "fc_out"
            output = tf.layers.dense(
                last_layer,
                num_outputs,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                name=label)
            return output, last_layer
