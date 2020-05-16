import numpy as np
from NN.modeler import BB_Model
from dynamics.said import step

# define the sampling time
dt = 0.01

# define the number of save iterations
save_epochs = 500

# define number of training iterations
epochs = 100000

# define the batch size
batch_size = 1000

#define the testing size
test_size = 1000

# define the learning region of the state space as a box
x_up = np.array([10, 10])
x_lo = np.array([-10, -10])

# define the learning region of the input space as a box
u_up = np.array([10])
u_lo = np.array([-10])

# try to load the model
nb_states = 2
nb_actions = 1
model = BB_Model(nb_states, nb_actions)
loaded = model.load_model(path='said_')
if loaded: print("Said model loaded!")

# train the model
print "Training for", epochs, "epochs"
for e in range(epochs):
    # generate random samples
    x0 = np.random.rand(nb_states, batch_size)*(x_up[:,None] - x_lo[:,None]) + x_lo[:,None]
    u = np.random.rand(nb_actions, batch_size)*(u_up[:,None] - u_lo[:,None]) + u_lo[:,None]
    y = step(x0, u, dt)

    # train nn
    model.train(x0, u, y)

    # save the model
    if e % save_epochs == 0:
        model.save_model(path='Said_')
        print "Said model saved at iteration", e

        # get testing results
        x0 = np.random.rand(nb_states, test_size)*(x_up[:,None] - x_lo[:,None]) + x_lo[:,None]
        u = np.random.rand(nb_actions, test_size)*(u_up[:,None] - u_lo[:,None]) + u_lo[:,None]
        y = step(x0, u, dt)

        yhat = model.predict(x0, u)
        error = np.linalg.norm(y-yhat, axis=1)
        print "Training error over", test_size, "samples:", np.mean(error)
