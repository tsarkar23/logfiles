#clean up neural-network code
import logging, time
import mxnet as mx
import numpy as np
import GPy
import GPyOpt
import sys
sys.path.append('/efs/datasets/users/tuhins/mxnet/example/image-classification')
sys.path.append('/efs/datasets/users/tuhins/mxnet/example/image-classification/symbols')
sys.path.append('/efs/datasets/users/tuhins')
from common import find_mxnet, data, fit
from common.util import download_file
from resnet import get_symbol
from args_parser import parse_args
import os

LOG_FILENAME='dist1000.log'
logging.basicConfig(filename=LOG_FILENAME,level=60)
logging.getLogger().setLevel(60)
filehandler_dbg = logging.FileHandler(LOG_FILENAME, mode='w')
file_name = 'train_params'

def _get_lr_scheduler(args, lr, lr_factor):

    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = 0#args.load_epoch if args.load_epoch else 0
    step_epochs = [200, 250]
    #lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= lr_factor

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor))
def simulated_anneal(cumul_val, cumul_period, thresh_val=0.01, temperature = 1, bump = 0.01):
    #returns a bool to randomize or not
    prob = min(1.0 , np.exp(-(cumul_val + bump) / (thresh_val * cumul_period * temperature)))
    u = np.random.uniform(0, 1)
    print("Probability is %f, Cumul Val %f, Cumul Period %f" %(prob, cumul_val, cumul_period))
    return (prob > u)

def get_iterator(args, kv):
    data_dir="/efs/datasets/users/tuhins/data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    
    args.data_train = fnames[0]
    args.data_val = fnames[1]
    (train_iter, val_iter) = data.get_rec_iter(args, kv=kv)
    return (train_iter, val_iter)

def main(net, batch_size, num_gpu, args_start, num_epochs=100, hyperparameter_list=['learning_rate'], file_name = 'train_params', bounds=[[0,1]], input_param_file='', kernel=None):
    kv = mx.kvstore.create(args_start.kv_store)
    #print('Rank %d' %(kv.rank))
    (train_iter, val_iter) = get_iterator(args_start, kv)
    args = args_start
    num_gpu = num_gpu
    ctx = [mx.gpu(i) for i in range(num_gpu)]
    bounds = bounds
    model_params=None
    num_classes = 10
    previous_accuracy = np.array([[np.float64(1.0 / num_classes)]])
    temperature = 1.0
    myFile=file_name
    num_iterations=1e6
    #Module binding
    mod = mx.mod.Module(symbol=net,
                    context=ctx)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
   
    model_data=None
    if(len(input_param_file) > 0):
        sym, arg_params, aux_params = mx.model.load_checkpoint(input_param_file, 1)
        model_data = (arg_params, aux_params)
    #mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), force_init=True)
    #mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),('momentum', 0.9)), force_init=True)
    #if args.test_io:
    tic = time.time()
    for i, batch in enumerate(train_iter):
        print('Binding done')
        for j in batch.data:
            j.wait_to_read()
        if (i+1) % 10 == 0:
            print('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, 10*batch_size/(time.time()-tic)))
            tic = time.time()
    train_iter.reset()
    lr = 0.22
    model_data = None
    optimizer_file=''
    for epoch in range(num_epochs):
        if(model_data == None):
            mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), force_init=True) #maybe you can change
        else:
            arg_params, aux_params = model_data
            mod.set_params(arg_params, aux_params)
        if(epoch>=200 and epoch < 250):
            optimizer_params=(('learning_rate', 0.01),('momentum', 0.9),('wd', 0.0001) )
        elif(epoch>=250):
            optimizer_params=(('learning_rate', 0.001),('momentum', 0.9), ('wd', 0.0001))
        else:
            optimizer_params=(('learning_rate', 0.1),('momentum', 0.9), ('wd', 0.0001)) 
            
        mod.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params, force_init=True)
        #optimizer_params,force_init=True)
        
        if(len(optimizer_file) > 0):
            mod.load_optimizer_states(optimizer_file)
            
        metric = mx.metric.create('acc')
        ctr = 0
        train_acc = 0
        for batch in train_iter:
            ctr = ctr + 1

            if(ctr < num_iterations):
                mod.forward(batch, is_train=True)       # compute predictions
                mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
                mod.backward()                          # compute gradients
                mod.update()                            # update parameters
                
                #print(type(metric.get()), metric.get()[0][1])
                train_acc = train_acc + float((metric.get())[1])
            if((ctr % 50 == 0) or ctr == (40000 / batch_size)):
                logging.log(65, 'Subepoch %d, Training %s' % (ctr, metric.get()))
                
        train_iter.reset()
        metric.reset()
        train_acc = np.float64(train_acc/ctr)
        val_accuracy = np.float64(mod.score(val_iter, ['acc'])[0][1])
        gen_diff = (train_acc - val_accuracy) #generalization error
        #print('Validation accuracy %s previous accuracy %s' %(val_accuracy, 0.1))
        logging.log(65, "Epoch %d, Validation Accuracy %f" %(epoch, val_accuracy))
        mod.save_checkpoint(myFile+('%.4f' %val_accuracy), 1)
        mod.save_optimizer_states(myFile+('%.4f' %val_accuracy)+'.states')
        optimizer_file=myFile+('%.4f' %val_accuracy)+'.states'
        sym, arg_params, aux_params = mx.model.load_checkpoint(myFile+('%.4f' %val_accuracy), 1)
        model_data = (arg_params, aux_params)
        
    # lr, lr_scheduler = _get_lr_scheduler(args, lr, 0.5)
    # mod.fit(train_iter,
    #     eval_data=val_iter,
    #     optimizer='sgd',
    #     optimizer_params={'learning_rate':lr, 'momentum':0.9,'wd' : 0.0001,
    #         'lr_scheduler': lr_scheduler,
    #         'multi_precision': True},
    #     eval_metric='acc',
    #     kvstore = args.kv_store,
    #     #batch_end_callback = mx.callback.Speedometer(batch_size, 50),            
    #     num_epoch=300,
    #     begin_epoch        = 0,
    #     #num_epoch          = args.num_epochs,
    #     #eval_data          = val,
    #     #eval_metric        = eval_metrics,
    #     #kvstore            = 'device',
    #     #optimizer          = args.optimizer,
    #     #optimizer_params   = optimizer_params,
    #     initializer        = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
    #     #arg_params         = arg_params,
    #     #aux_params         = aux_params,
    #     batch_end_callback = [mx.callback.Speedometer(batch_size, 20)],
    #     #epoch_end_callback = checkpoint,
    #     allow_missing      = True)

    directory = '/efs/datasets/users/tuhins/test'
    test = os.listdir( directory )
    #clean up
    for item in test:
        if item.endswith(".json") or item.endswith(".params") or item.endswith(".states"):
            os.remove( os.path.join( directory, item ) )

if __name__ == '__main__':
    args = parse_args()
    args.num_examples = 50000
    args.num_classes = 10
    args.batch_size = 500
    args.image_shape = '3,28,28'
    net = get_symbol(**vars(args))
    #args_start = parse_args()
    kernel=None
    num_gpu = args.num_gpu
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hyper_params =[x.strip() for x in args.hyperparameters.split(',')]
    hyper_params = hyper_params + ['Epoch']
    main(net, batch_size, num_gpu, args, num_epochs=300, hyperparameter_list=hyper_params, bounds=[[0,0.1], [0, 1.]], kernel=kernel)





