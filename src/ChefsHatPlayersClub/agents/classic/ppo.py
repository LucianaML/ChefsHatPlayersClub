# Adapted from: https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py
from ChefsHatGym.agents.chefs_hat_agent import ChefsHatAgent
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning

from keras.layers import Input, Dense, Multiply
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

import random
import numpy
import copy
import os
import sys
import urllib.request
import keras
from typing import Literal


def proximal_policy_optimization_loss():
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        ENTROPY_LOSS = 5e-3
        y_tru_valid = y_true[:, 0:200]
        old_prediction = y_true[:, 200:400]
        advantage = y_true[:, 400][0]

        prob = K.sum(y_tru_valid * y_pred, axis=-1)
        old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(
                r * advantage,
                K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
                * advantage,
            )
            + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10))
        )

    return loss


# Adapted from: https://github.com/germain-hug/Deep-RL-Keras

types = ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]


class AgentPPO(ChefsHatAgent):
    suffix = "PPO"
    actor = None
    training = False

    loadFrom = {
        "vsRandom": [
            "Trained/ppo_actor_vsRandom.hd5",
            "Trained/ppo_critic_vsRandom.hd5",
        ],
        "vsEveryone": [
            "Trained/ppo_actor_vsEveryone.hd5",
            "Trained/ppo_critic_vsEveryone.hd5",
        ],
        "vsSelf": ["Trained/ppo_actor_vsSelf.hd5", "Trained/ppo_critic_vsSelf.hd5"],
    }

    downloadFrom = {
        "vsRandom": [
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_actor_vsRandom.hd5",
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_critic_vsRandom.hd5",
        ],
        "vsEveryone": [
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_actor_vsEveryone.hd5",
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_critic_vsEveryone.hd5",
        ],
        "vsSelf": [
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_actor_vsSelf.hd5",
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/ppo_critic_vsSelf.hd5",
        ],
    }

    def __init__(
        self,
        name: str,
        continueTraining: bool = False,
        agentType: Literal["Scratch", "vsRandom", "vsEveryone", "vsSelf"] = "Scratch",
        initialEpsilon: int = 1,
        loadNetwork: str = "",
        saveFolder: str = "",
        verbose: bool = False,
        logDirectory: str = "",
    ):
        super().__init__(
            self.suffix,
            agentType + "_" + name,
            saveFolder,
        )

        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.loadNetwork = loadNetwork

        if verbose:
            self.startLogging(logDirectory)

        self.type = agentType
        self.reward = RewardOnlyWinning()

        self.startAgent()

        if not self.type == "Scratch":
            fileNameActor = os.path.join(
                os.path.abspath(sys.modules[AgentPPO.__module__].__file__)[0:-6],
                self.loadFrom[agentType][0],
            )
            fileNameCritic = os.path.join(
                os.path.abspath(sys.modules[AgentPPO.__module__].__file__)[0:-6],
                self.loadFrom[agentType][1],
            )
            if not os.path.exists(
                os.path.join(
                    os.path.abspath(sys.modules[AgentPPO.__module__].__file__)[0:-6],
                    "Trained",
                )
            ):
                os.mkdir(
                    os.path.join(
                        os.path.abspath(sys.modules[AgentPPO.__module__].__file__)[
                            0:-6
                        ],
                        "Trained",
                    )
                )

            if not os.path.exists(fileNameCritic):
                urllib.request.urlretrieve(
                    self.downloadFrom[agentType][0], fileNameActor
                )
                urllib.request.urlretrieve(
                    self.downloadFrom[agentType][1], fileNameCritic
                )

            self.loadModel([fileNameActor, fileNameCritic])

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    # PPO Functions
    def startAgent(self):
        self.hiddenLayers = 2
        self.hiddenUnits = 64
        self.gamma = 0.95  # discount rate

        # Game memory
        self.resetMemory()

        self.learning_rate = 1e-4

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0  # no exploration while testing

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990

        self.buildModel()

    def buildActorNetwork(self):
        inputSize = 28
        inp = Input((inputSize,), name="Actor_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(
                self.hiddenUnits * (i + 1),
                name="Actor_Dense" + str(i),
                activation="relu",
            )(previous)

        outputActor = Dense(200, activation="softmax", name="actor_output")(dense)

        actionsOutput = Input(shape=(200,), name="PossibleActions")

        outputPossibleActor = Multiply()([actionsOutput, outputActor])

        self.actor = Model([inp, actionsOutput], outputPossibleActor)

        self.actor.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=[proximal_policy_optimization_loss()],
        )

    def buildCriticNetwork(self):
        # Critic model
        inputSize = 28

        inp = Input((inputSize,), name="Critic_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(
                self.hiddenUnits * (i + 1),
                name="Critic_Dense" + str(i),
                activation="relu",
            )(previous)

        outputCritic = Dense(1, activation="linear", name="critic_output")(dense)

        self.critic = Model([inp], outputCritic)

        self.critic.compile(Adam(self.learning_rate), "mse")

    def buildModel(self):
        self.buildCriticNetwork()
        self.buildActorNetwork()

    def discount(self, r):
        """Compute the gamma-discounted rewards over an episode"""
        discounted_r, cumul_r = numpy.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def loadModel(self, model):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
            ENTROPY_LOSS = 5e-3
            y_tru_valid = y_true[:, 0:200]
            old_prediction = y_true[:, 200:400]
            advantage = y_true[:, 400][0]

            prob = K.sum(y_tru_valid * y_pred, axis=-1)
            old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
            r = prob / (old_prob + 1e-10)

            return -K.mean(
                K.minimum(
                    r * advantage,
                    K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
                    * advantage,
                )
                + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10))
            )

        actorModel, criticModel = model
        self.actor = load_model(actorModel, custom_objects={"loss": loss})
        self.critic = load_model(criticModel, custom_objects={"loss": loss})

        print(f"Load from: {actorModel}")
        print(f"Load from: {criticModel}")

    def updateModel(self, game, thisPlayer):
        state = numpy.array(self.states)

        action = self.actions
        reward = numpy.array(self.rewards)
        possibleActions = numpy.array(self.possibleActions)
        realEncoding = numpy.array(self.realEncoding)

        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(reward)
        state_values = self.critic(numpy.array(state))
        advantages = discounted_rewards - numpy.reshape(state_values, len(state_values))

        criticLoss = self.critic.train_on_batch([state], [reward])

        actions = []
        for i in range(len(action)):
            advantage = numpy.zeros(numpy.array(action[i]).shape)
            advantage[0] = advantages[i]
            # print ("advantages:" + str(numpy.array(advantage).shape))
            # print ("actions:" + str(numpy.array(action[i]).shape))
            # print("realEncoding:" + str(numpy.array(realEncoding[i]).shape))
            concatenated = numpy.concatenate((action[i], realEncoding[i], advantage))
            actions.append(concatenated)
        actions = numpy.array(actions)

        actorLoss = self.actor.train_on_batch([state, possibleActions], [actions])

        # Update the decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.verbose:
            self.log(
                "-- "
                + self.name
                + ": Epsilon:"
                + str(self.epsilon)
                + " - ALoss:"
                + str(actorLoss)
                + " - "
                + "CLoss: "
                + str(criticLoss)
            )

    def resetMemory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.possibleActions = []
        self.realEncoding = []

    # Agent Chefs Hat Functions

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def get_reward(self, info):
        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)

    def get_action(self, observations):
        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(numpy.array(stateVector), 0)
        possibleActions2 = copy.copy(possibleActions)

        randNum = numpy.random.rand()

        if randNum <= self.epsilon:

            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[
                0
            ].tolist()
            if len(itemindex) > 1:
                itemindex.pop()

            aIndex = itemindex[-1:]
            a = numpy.zeros(200)
            a[aIndex] = 1

            # print(f"randNum: {randNum} - epsilon: {self.epsilon}")
        else:
            sumBefore = numpy.sum(possibleActions)

            if numpy.sum(possibleActions2) > 1:
                possibleActions2[-1] = 0
            # print(
            #     f"Sum possible actions before: {sumBefore} after: {numpy.sum(possibleActions2)}"
            # )
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)
            a = self.actor([stateVector, possibleActionsVector])[0]

        return a

    def update_end_match(self, info):
        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            self.updateModel(rounds, thisPlayer)

            # save model
            if not self.saveModelIn == "":
                keras.models.save_model(
                    self.actor,
                    os.path.join(
                        self.saveModelIn,
                        "actor_Player_" + str(self.name) + ".h5",
                    ),
                )

                keras.models.save_model(
                    self.critic,
                    os.path.join(
                        self.saveModelIn,
                        "critic_Player_" + str(self.name) + ".h5",
                    ),
                )

            self.resetMemory()

    def update_my_action(self, info):
        if self.training:
            action = numpy.array(info["action"])
            observation = numpy.array(info["observation"])
            nextObservation = numpy.array(info["nextObservation"])

            reward = self.get_reward(info)

            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            realEncoding = action
            action = numpy.zeros(action.shape)
            action[numpy.argmax(realEncoding)] = 1

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.possibleActions.append(possibleActions)
            self.realEncoding.append(realEncoding)
