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
    def _build_layers(self, inputs, num_outputs, options):
        """Process the flattened inputs.

        Note that dict inputs will be flattened into a vector. To define a
        model that processes the components separately, use _build_layers_v2().
        """
        #import pdb; pdb.set_trace()
        # inputs = tf.Print(inputs, ['inputs', inputs], summarize=1000)
        hiddens = options.get("fcnet_hiddens")
        activation = get_activation_fn(options.get("fcnet_activation"))
        #activation = tf.nn.relu6
        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                # skip final linear layer
                if options.get("no_final_linear") and i == len(hiddens):
                    output = tf.layers.dense(
                        last_layer,
                        num_outputs,
                        kernel_initializer=normc_initializer(1.0),
                        activation=activation,
                        name="fc_out")
                    return output, output

                label = "fc{}".format(i)
                last_layer = tf.layers.dense(
                    last_layer,
                    size,
                    kernel_initializer=normc_initializer(1.0),
                    activation=activation,
                    name=label)
                # last_layer = tf.Print(last_layer, ['last_layer', last_layer])
                i += 1

            output = tf.layers.dense(
                last_layer,
                num_outputs,
                kernel_initializer=normc_initializer(0.01),
                activation=None,
                name="fc_out")
            
            # output = tf.Print(output, ['output1:', output])
            #import pdb; pdb.set_trace()
            if not self.is_value_fn:
                mean, std = tf.split(output, 2, axis=1)
                # mean = tf.nn.tanh(mean)*1.0
                std = tf.nn.softplus(std)
                # std = tf.clip_by_value(std, -20, 2)
                output = tf.concat([mean, std], axis=1)
            # output = tf.Print(output, ['output2:', output])
            return output, last_layer
