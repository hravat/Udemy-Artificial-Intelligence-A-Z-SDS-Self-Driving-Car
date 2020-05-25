# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import keras

class Network(object):

    # Define the Huber loss so that it can be used with Keras

    def __init__(self, input_size, nb_action ):
        self.input_size = input_size
        self.nb_action = nb_action
        self.learning_rate =  0.001
        self.model = self.build_model()

    def huber_loss_wrapper(**huber_loss_kwargs):
        def huber_loss_wrapped_function(y_true, y_pred):
            return keras.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

        return huber_loss_wrapped_function

    def build_model(self):
        self.temperateure=100
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(30, input_dim=self.input_size))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dense(self.nb_action))
        self.model.add(keras.layers.Lambda(lambda x: x*self.temperateure))
        self.model.add(keras.layers.Activation('softmax'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return self.model

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def predict(self, state):
        return self.model.predict(state)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, current_state_in, current_action_taken, current_reward_obtained, next_state_reached):

        self.memory.append((current_state_in, current_action_taken, current_reward_obtained, next_state_reached))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)



class Dqn():


    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.current_action_taken = 0
        self.current_reward_obtained = 0
        self.current_state_in = np.zeros([1,input_size])
        self.next_state_reached = np.zeros([1, input_size])
        self.sample_size=1000
        self.return_type ='random'
        self.nb_action = nb_action
        self.study_time = self.sample_size
        self.batch_mode = False


    def select_action(self, state):
        temperature=7 # T=100
        act_values = self.model.predict(state)

        if self.return_type=='random':
            act_idx = np.squeeze(np.argsort(-act_values), axis=0)
            idx = random.choices(population=[0, 1, 2], weights=[0.8, 0.1, 0.1])
            return int(act_idx[idx])
        elif self.return_type=='deactivate_ai':
            idx = random.choices(population=[0, 1, 2])
            return idx[0]
        else:
            return int(np.argmax(act_values))

    def learn(self, batch_size):
        print("Learning")

        if not self.batch_mode:

            minibatch = np.asarray(self.memory.sample(batch_size))

            for current_state_in, current_action_taken, current_reward_obtained, next_state_reached in minibatch:
                # the original Q-learning formula
                # http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/

                current_q = self.model.predict(current_state_in)
                next_q = self.model.predict(next_state_reached)
                target_q = float(current_reward_obtained + (self.gamma * next_q[:, np.argmax(next_q)]))
                current_q[0][current_action_taken] = target_q

                self.model.model.fit(current_state_in, current_q, epochs=1, verbose=0)
        else:
            minibatch = np.asarray(self.memory.sample(batch_size))

            current_state_in =np.asarray(minibatch[:,0])
            current_state_in=np.squeeze(np.stack(current_state_in), axis=1)

            current_action_taken =np.asarray(minibatch[:,1])
            current_action_taken = np.expand_dims(current_action_taken,axis=-1)

            current_reward_obtained =np.asarray(minibatch[:,2])
            current_reward_obtained = np.expand_dims(current_reward_obtained, axis=-1)

            next_state_reached = np.asarray(minibatch[:,3])
            next_state_reached = np.squeeze(np.stack(next_state_reached), axis=1)

            current_q = self.model.predict(current_state_in)


            next_q  = self.model.predict(next_state_reached)
            next_q=np.expand_dims(np.amax(next_q,axis=-1),axis=-1)
            target_q = current_reward_obtained+(self.gamma*next_q)

            modified_current_q = self.set_target(current_q, target_q,current_action_taken)

            self.model.model.fit(current_state_in, modified_current_q , epochs=1, verbose=0)



    def update(self, reward, new_signal):
        np_arr = np.transpose(np.expand_dims(np.asarray(new_signal),axis=-1))
        self.next_state_reached=np_arr
        self.memory.push(self.current_state_in, self.current_action_taken, self.current_reward_obtained, self.next_state_reached)

        action = self.select_action(self.next_state_reached)

        if len(self.memory.memory) > self.sample_size and self.study_time <= 0:
            self.learn(batch_size=self.sample_size)
            self.study_time = self.sample_size/2


        self.current_action_taken = action
        self.current_state_in = self.next_state_reached
        self.current_reward_obtained = reward
        self.study_time -= 1
        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]


        return action

    def score(self):
        return (sum(self.reward_window) / (len(self.reward_window) + 1.0))

    def load(self):
        self.model.load('brain.h5')
        print("Loaded model from disk")

    def save(self):
        self.model.save('brain.h5')
        print("Saved model to disk")

    def set_target(self,current_q,target_q,current_action_taken):
        # Convert numpy to one hot
        current_action_taken_list = np.squeeze(current_action_taken).tolist()
        n_values = self.nb_action
        current_action_taken_one_hot = np.eye(n_values)[current_action_taken_list]
        target_q = current_action_taken_one_hot * target_q
        current_q = current_q * (1 - current_action_taken_one_hot)

        return (target_q+current_q)