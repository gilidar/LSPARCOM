from FUNCS_GEN import *
from basic_model import *
import tensorflow as tf
import scipy.io
from keras import optimizers
from keras.utils import multi_gpu_model
import basic_model
import hdf5storage
# <><><><><><><><><><><><><><><><> DEFINE PARAMETERS <><><><><><><><><><><><><><><><>
numgpu = 1
size_dat = 16
ep_num = int(1e5)
ex_num_dat = 10005
ex_num_sing = 10005
ex_train_tot = 10000
ex_train_cut = 10000
batch_size = 50
batch_num = int(ex_train_cut/batch_size)
tot_batch = int(ep_num*batch_num)
upF = 4
folds_num = 10
LR = 1e-4
prox = 3
size_datN = size_dat*upF
input_dim = (size_datN, size_datN)
Lloss = weighted_mse_wL1
np.random.seed(7)

# <><><><><><><><><><><><><><><><> LOAD DATA <><><><><><><><><><><><><><><><>
# mat_contents = hdf5storage.loadmat('BTHD_VAR_REG_EDGE.mat')
# X = np.rollaxis(mat_contents['IM_ds'], 2)
# X = 5e-4*X.reshape(ex_num_sing, size_datN, size_datN, 1)
# Y = np.rollaxis(mat_contents['GT_ds'], 2)
# Y = 1e-6*Y.reshape(ex_num_sing, size_datN, size_datN, 1)

mat_contents = hdf5storage.loadmat('TUI4_VAR_REG.mat')
X = np.rollaxis(mat_contents['IM_ds'], 2)
X = 1e-3*X.reshape(ex_num_sing, size_datN, size_datN, 1)
Y = np.rollaxis(mat_contents['GT_ds'], 2)
Y = 1e-7*Y.reshape(ex_num_sing, size_datN, size_datN, 1)

val0_im = X[8, :, :, :]
val0_im = val0_im.reshape(1, size_datN, size_datN, 1)
val0_gt = Y[8, :, :, 0]
val1_im = X[17, :, :, :]
val1_im = val1_im.reshape(1, size_datN, size_datN, 1)
val1_gt = Y[17, :, :, 0]
test_ex_im = X[ex_train_tot, :, :, :]
test_ex_im = test_ex_im.reshape(1, size_datN, size_datN, 1)
test_ex_gt = Y[ex_train_tot , :, :, 0]

print("\n - Finished reading data")

# <><><><><><><><><><><><><><><><> Global parameters <><><><><><><><><><><><><><><><>
batch_losses = np.zeros(tot_batch)
ax_batch_losses = np.arange(0, tot_batch, 1)
batch_vec = np.arange(0, ex_train_cut, batch_size)
count = 0

ADAM_OPT = optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False) #tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)
with tf.device('/cpu:0'):
    model = basic_model.buildModel(input_dim, folds_num, upfactor = upF, proxtype=prox)
if numgpu > 1:
    parallel_model = multi_gpu_model(model, gpus=numgpu)
else:
    parallel_model = model
parallel_model.compile(loss=Lloss, optimizer=ADAM_OPT)
print("\n - Finished compiling model")

# Train the network
for ep in range(ep_num):
    print("\n - Training on epoch # %d" % ep)
    for ba in range(batch_num):
        batch_losses[count] = parallel_model.train_on_batch(X[batch_vec[ba]:batch_vec[ba]+batch_size, :, :, :], Y[batch_vec[ba]:batch_vec[ba]+batch_size, :, :, :])
        plt.plot(ax_batch_losses[:count+1], batch_losses[:count+1])
        plt.savefig('E:/loss_corr.png')
        plt.close('all')
        count = count + 1
    print("\n - Batch Loss in epoch # %d is: %f" % (ep, batch_losses[count - 1]))
    if ep % 5 == 0:
        #Save weights to file
        wdict = {}
        for layer in model.layers:
            info = layer.get_weights()
            for i in range(info.__len__()):
                wdict[layer.name+'_field_'+str(i)] = info[i]
        scipy.io.savemat('weights'+str(ep)+'.mat', wdict)
        # See training
        res = np.array(parallel_model.predict(val0_im))
        imagesc2test(val0_im.reshape(size_datN, size_datN), res.reshape(size_datN, size_datN), val0_gt, size_datN, ep, 0)  # Visualize
        res = np.array(parallel_model.predict(val1_im))
        imagesc2test(val1_im.reshape(size_datN, size_datN), res.reshape(size_datN, size_datN), val1_gt, size_datN, ep, 1)  # Visualize
        # Test
        res = np.array(parallel_model.predict(test_ex_im))
        imagesc2test(test_ex_im.reshape(size_datN, size_datN), res.reshape(size_datN, size_datN), test_ex_gt, size_datN, ep, 2)  # Visualize

