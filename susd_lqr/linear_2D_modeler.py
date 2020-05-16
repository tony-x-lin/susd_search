import numpy as np
from NN.modeler import BB_Model
from dynamics.linear_2D import WB_Model

# define the sampling time
dt = 0.01

# define the number of save iterations
save_epochs = 500

# define number of training iterations
epochs = 1000000

# define the batch size
batch_size = 10000

#define the testing size
test_size = 10000

# define the learning region of the state space as a box
x_up = np.array([20, 20])
x_lo = np.array([-20, -20])

# define the learning region of the input space as a box
u_up = np.array([20, 20])
u_lo = np.array([-20, -20])

# try to load the model
nb_states = 2
nb_actions = 2
model = BB_Model(nb_states, nb_actions)
loaded = model.load_model(path='linear_2D_')
if loaded: print("2 Input Linear model loaded!")

wb_model = WB_Model(dt)

# train the model
print "Training for", epochs, "epochs"
for e in range(epochs):
    # generate random samples
    x0 = np.random.rand(nb_states, batch_size)*(x_up[:,None] - x_lo[:,None]) + x_lo[:,None]
    u = np.random.rand(nb_actions, batch_size)*(u_up[:,None] - u_lo[:,None]) + u_lo[:,None]
    y = wb_model.predict(x0, u)

    # train nn
    model.train(x0, u, y)

    # save the model
    if e % save_epochs == 0:
        model.save_model(path='linear_2D_')
        print "2 Input Linear model saved at iteration", e

        # get testing results
        x0 = np.random.rand(nb_states, test_size)*(x_up[:,None] - x_lo[:,None]) + x_lo[:,None]
        u = np.random.rand(nb_actions, test_size)*(u_up[:,None] - u_lo[:,None]) + u_lo[:,None]
        y = wb_model.predict(x0, u)

        yhat = model.predict(x0, u)
        error = np.linalg.norm(y-yhat, axis=1)
        print "Training error over", test_size, "samples:", np.mean(error)
