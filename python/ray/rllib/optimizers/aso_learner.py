"""Helper class for AsyncSamplesOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from six.moves import queue
from collections import deque

from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers.aso_minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from ray.rllib.optimizers.appo_replay_buffer import APPOReplayBuffer

class LearnerThread(threading.Thread):
    """Background thread that updates the local model from sample trajectories.

    This is for use with AsyncSamplesOptimizer.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    def __init__(self, local_evaluator, minibatch_buffer_size, num_sgd_iter,
                 learner_queue_size, use_appo=False, old_policy_evaluator=None, old_policy_lag=None, use_kl_loss=False):
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_evaluator = local_evaluator
        self.old_policy_evaluator = old_policy_evaluator
        self.inqueue = queue.Queue(maxsize=16)#queue.LifoQueue(maxsize=1000)
        #self.inqueue= deque(maxlen=learner_queue_size)
        self.outqueue = queue.Queue()
        self.minibatch_buffer_size=minibatch_buffer_size
        self.minibatch_buffer = MinibatchBuffer(
            self.inqueue, minibatch_buffer_size, num_sgd_iter)
        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.load_timer = TimerStat()
        self.load_wait_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stats = {}
        self.stopped = False
        self.old_policy_lag = old_policy_lag
        self.old_policy_lag=minibatch_buffer_size*num_sgd_iter
        self.use_kl_loss=use_kl_loss
        self.counter=0
        if self.use_kl_loss:
            self.kl_list = []   
        self.use_appo = use_appo

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        with self.queue_timer:
            #batch = self.inqueue.get()
            batch, _ = self.minibatch_buffer.get()
        print(self.use_appo)
        if "old_policy_behaviour_logits" not in batch and self.use_appo:
                self.old_policy_behaviour_logits = self.old_policy_evaluator.policy_map['default_policy'].compute_actions(
                        batch["obs"],prev_action_batch=batch["prev_actions"],
                        prev_reward_batch=batch["prev_rewards"])[2]['behaviour_logits']
                batch["old_policy_behaviour_logits"] = self.old_policy_behaviour_logits

        with self.grad_timer:
            fetches = self.local_evaluator.learn_on_batch(batch)
            
            '''
            if fetches["learner_stats"]["drop_batch"]:
                self.minibatch_buffer.replace()
            '''
            #print(self.use_appo, self.old_policy_lag, self.use_kl_loss)
            self.weights_updated = True
            print(self.use_appo)
            if self.use_kl_loss and self.use_appo:
                #Collect KL divergences during last pass over minibatch buffer
                if self.counter >= self.old_policy_lag-self.minibatch_buffer_size:
                    self.kl_list.append(fetches["learner_stats"]["KL"])
                if self.counter == self.old_policy_lag-1:
                    avg_kl = sum(self.kl_list)/len(self.kl_list)
                    self.local_evaluator.for_policy(
                        lambda pi: pi.update_kl(avg_kl))
                    self.kl_list = []
                    self.counter=0
                    weights = self.local_evaluator.get_weights()
                    if self.old_policy_evaluator:
                        self.old_policy_evaluator.set_weights(weights)
                else:
                    self.counter+=1
            elif self.use_appo:
                if self.counter == self.old_policy_lag-1:
                    self.counter=0
                    weights = self.local_evaluator.get_weights()
                    if self.old_policy_evaluator:
                        self.old_policy_evaluator.set_weights(weights)
                else:
                    self.counter+=1
                    
            self.stats = get_learner_stats(fetches)

        self.outqueue.put(batch.count)
        #self.learner_queue_size.push(len(self.inqueue))
        #print("Queue Size", self.inqueue.qsize())
        self.learner_queue_size.push(self.inqueue.qsize())
