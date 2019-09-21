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

        #import pdb; pdb.set_trace()
        if context is not None:
            context = tf.reshape(context, (-1,10))
            #context = tf.Print(context, ["context", context],summarize=12)
            with tf.variable_scope("hyper_film"):
                x_hyp = context
                hyper_hidden_sizes = (32, 64)
                for idx, hidden_size in enumerate(hyper_hidden_sizes):
                    x_hyp = tf.layers.dense(x_hyp, hidden_size, name='hidden_%d' % idx,
                        activation=activation,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer())
                x_hyp = tf.layers.dense(x_hyp, 2*sum(hiddens), name='hyper_out',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer())
                film_params = tf.split(x_hyp, [val for val in hiddens for _ in (0, 1)],axis=1)



        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = tf.layers.dense(
                    last_layer,
                    size,
                    kernel_initializer= tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    activation=activation,
                    name=label)
                if context is not None:
                    last_layer = tf.einsum('ij,kj->ij', last_layer, film_params.pop(0))+ film_params.pop(0)
                i += 1
            label = "fc_out"
            output = tf.layers.dense(
                last_layer,
                num_outputs,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                name=label)
            return output, last_layer
