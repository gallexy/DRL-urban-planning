import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Create the environment
env = gym.make('CartPole-v1')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
env.reset(seed=42)

# Define the neural network model
num_actions = env.action_space.n
input_shape = env.observation_space.shape

model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=input_shape),
    layers.Dense(24, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

# Training parameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = []

# Function to choose an action
def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Function to train the model
def train_model():
    if len(memory) < batch_size:
        return
    batch = np.random.choice(memory, batch_size, replace=False)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Main training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, input_shape[0]])
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, input_shape[0]])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train_model()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()