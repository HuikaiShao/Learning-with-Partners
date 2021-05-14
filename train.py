import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import nets
import time
from losses import insightface_loss

def read_image(path, num_per = 10):
    data = []
    able = []
    datas = []
    lables=[]

    roi = open(path)
    roi_path = roi.readlines()
    classnum = len(roi_path)//num_per
    trainnum = 0
    for i,image_list in enumerate(roi_path):
        if i  < len(roi_path)/2:
            datas.append(image_list[:-1])
            lables.append(int(i/num_per))
            trainnum = trainnum +1
    return datas,lables

def extractor(batch, reused = True, name = 'net'):
    feature1 = nets.con1(batch,False,reused, name = name + '_con1')
    feature2 = nets.con2(feature1,False,reused, name = name + '_con2')
    feature3 = nets.con3(feature2,False,reused, name = name + '_con3')
    #feature4 = nets.con4(feature3,False,reused, name = name + '_con4')
    feature5 = nets.fc1(feature3,False,reused, name = name + '_fc1')
    feature6 = nets.fc2(feature5,False,reused, name = name + '_fc2')
    feature7 = nets.fc3(feature6,False,reused, name = name + '_fc3')
    return feature7

def Dis_loss(feature, batch_size, omega_size):
    #feature = tf.nn.l2_normalize(feature, dim=1)
    archer_feature,sabor_feature = tf.split(feature,[omega_size,batch_size-omega_size],axis = 0)  
    archer_matrix = tf.matmul(archer_feature,tf.transpose(archer_feature))#下面这一部分内容、算法之前给你讲过
    sabor_matrix = tf.matmul(sabor_feature,tf.transpose(sabor_feature))
    archer_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[omega_size]),[omega_size,omega_size]))
    archer_sabor_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[batch_size-omega_size]),[batch_size-omega_size,omega_size]))
    sabor_diag = tf.reshape(tf.tile(tf.diag_part(sabor_matrix),[omega_size]),[omega_size,batch_size-omega_size])
    archer_distance = archer_diag + tf.transpose(archer_diag) - 2*archer_matrix
    sabor_distance = sabor_diag + archer_sabor_diag - 2*tf.matmul(archer_feature,tf.transpose(sabor_feature))

    return archer_distance, sabor_distance

bandwidths = [ 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
def MMD(fea1, fea2):
    loss = tf.cast(tf.sqrt(mix_rbf_mmd2(fea1, fea2, sigmas=bandwidths)), tf.float32)
    return loss
def dis(fea1, fea2):
    loss = tf.reduce_mean(tf.pow(tf.subtract(fea1, fea2), 2.0))
    return loss
source_size = 30
target_size = 30
omega_size = 20
capacity=1000+3*source_size
results = []
_eps=1e-8

def main():
    tf.reset_default_graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"       
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40) 
    sess_target = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    root = './saver/XJTU-UP/SFHF/MF'
    num_per = 10
    path_s1 = '/media/shaohuikai/D/shaohuikai/Database/roi_1_10 (wanzheng)/SF.txt' # image list
    path_s2 = '/media/shaohuikai/D/shaohuikai/Database/roi_1_10 (wanzheng)/HF.txt'
    path_s3 = '/media/shaohuikai/D/shaohuikai/Database/roi_1_10 (wanzheng)/LN.txt'
    path_target = '/media/shaohuikai/D/shaohuikai/Database/roi_1_10 (wanzheng)/MF.txt'

    s1_data, s1_label = read_image(path_s1, num_per)
    train_size=len(s1_data)
    s1_batch, s1_batch_label= get_batch(s1_data, s1_label, source_size, capacity, True)
    s2_data, s2_label = read_image(path_s2, num_per)
    s2_batch, s2_batch_label= get_batch(s2_data, s2_label, source_size,capacity,True)#打乱
    #s3_data, s3_label = read_image(path_s3, num_per)
    #s3_batch, s3_batch_label= get_batch(s3_data, s3_label, source_size,capacity,True)#打乱

    #s1_con = con_extractor(s1_batch, reused = False, name = 'source')
    #s2_con = con_extractor(s2_batch, reused = True, name = 'source')
    #s3_con = con_extractor(s3_batch, reused = True, name = 'source')
  
    s1_feature = extractor(s1_batch, False, name = 'sources')  
    s2_feature = extractor(s2_batch, True, name = 'sources') 
    #s3_feature = extractor(s3_batch, True, name = 'sources') 
    st1_feature = extractor(s1_batch, False, name = 'sourcet')  
    st2_feature = extractor(s2_batch, True, name = 'sourcet') 
    #st3_feature = extractor(s3_batch, True, name = 'sourcet')

    s1_loss, _ = insightface_loss(s1_feature, s1_batch_label, 100, scope='sources', reuse=False)
    s2_loss, _ = insightface_loss(s2_feature, s2_batch_label, 100, scope='sources', reuse=True)
    #s3_loss, _ = insightface_loss(s3_feature, s3_batch_label, 100, scope='sources', reuse=True)
    s_loss =  s1_loss + s2_loss #+ s3_loss 

    st1_loss, _ = insightface_loss(st1_feature, s1_batch_label, 100, scope='sources', reuse=True)
    st2_loss, _ = insightface_loss(st2_feature, s2_batch_label, 100, scope='sources', reuse=True)
    #st3_loss, _ = insightface_loss(st3_feature, s3_batch_label, 100, scope='sources', reuse=True)
    st_loss =  st1_loss + st2_loss #+ st3_loss
    
    target_data,_ = read_image(path_target)#获得target图像
    target_batch,_ = get_batch(target_data, s1_label, target_size, capacity, True)#打乱
    t_s_feature = extractor(target_batch, True, name = 'sources')
    t_feature = extractor(target_batch, True, name = 'sourcet')
    
    arc_t1, sab_t1 = Dis_loss(t_feature, source_size, omega_size)#
    arc_t_s, sab_t_s = Dis_loss(t_s_feature, source_size, omega_size)#

    arc_s1, sab_s1 = Dis_loss(s1_feature, source_size, omega_size)#
    arc_s2, sab_s2 = Dis_loss(s2_feature, source_size, omega_size)
    #arc_s3, sab_s3 = Dis_loss(s3_feature, source_size, omega_size)
    arc_st1, sab_st1 = Dis_loss(st1_feature, source_size, omega_size)
    arc_st2, sab_st2 = Dis_loss(st2_feature, source_size, omega_size)
    #arc_st3, sab_st3 = Dis_loss(st3_feature, source_size, omega_size)

    d_loss1 = dis(arc_s1, arc_st1) + dis(sab_s1, sab_st1)
    d_loss2 = dis(arc_s2, arc_st2) + dis(sab_s2, sab_st2)
    #d_loss3 = dis(arc_s3, arc_st3) + dis(sab_s3, sab_st3)
    d_loss4 = dis(arc_t1, arc_t_s) + dis(sab_t1, sab_t_s)
    dis_loss = d_loss1 + d_loss2 + d_loss4 #+ d_loss3

    dir_loss = dis(s1_feature,st1_feature) + dis(s2_feature,st2_feature) + dis(t_feature,t_s_feature) #+ dis(s3_feature,st3_feature)


    feature = tf.concat([s1_feature,s2_feature],0)
    #feature = tf.concat([feature,s3_feature],0)
    feature = tf.concat([feature,t_s_feature],0)
    #feature1 = tf.concat([st1_feature,st2_feature],0)
    #feature1 = tf.concat([feature1,st3_feature],0)

    k1 = MMD(feature, st1_feature) 
    k2 = MMD(feature, st2_feature) 
    #k3 = MMD(feature, st3_feature)
    #k4 = MMD(feature, t_s_feature)
    fea_MMD_loss = k1 + k2 #+ k3 #+ k4
    
    k_tss1 = MMD(feature, t_feature) #+ MMD(feature, t_feature) + MMD(feature, t_feature)  
    k_tss = k_tss1 + fea_MMD_loss


    loss = s_loss #+ fea_MMD_loss_s  
    loss1 = st_loss  + 0.5*k_tss#  k_tss #+ 0.1 * dis_MMD_loss + 1*fea_MMD_loss 
    a = 1
    loss3 = dis_loss + a * dir_loss#
    
    global_step = tf.Variable(0,trainable = False,name = "global_step")
    leaning_rate = tf.train.exponential_decay(0.01,global_step,100,0.96,staircase = False)
    
    opt = tf.train.RMSPropOptimizer(0.0001,0.9)              
    vars_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ('sources' ))
    vars_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ('sourcet'))
    vars_3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ('source3'))
    vars_4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = ('sources'))
    #vars = vars_1 + vars_3
    #print(vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt1 = opt.minimize(loss, global_step = global_step, var_list = vars_1) 
        opt2 = opt.minimize(loss1, global_step = global_step, var_list = vars_2)
        opt3 = opt.minimize(loss3, global_step = global_step, var_list = vars_2)
        #opt4 = opt.minimize(loss4, global_step = global_step)#, var_list = vars_3)            
  
    sess_target.run(tf.global_variables_initializer())

    t_vars = tf.trainable_variables()
    r1_vars = [var for var in t_vars if 'source' in var.name]
   
    saver = tf.train.Saver(r1_vars,max_to_keep=0)
    #saver.restore(sess_target,"./saver/XJTU-UP/CNNF/HFSFLN/model.ckpt-50000")
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess_target, coord=coord)
    print('start train_bottle')
    count=0  
    t0 = int(time.time()) 
    try:
        for e in range(50000):
            if coord.should_stop():
                break
            count=count+1    
            _, _, _,s_,l1_= sess_target.run([opt1,opt2,opt3,loss,loss3])                                                    
            if( (e+1)%10 == 0):
                print("After %d training step(s),the loss is %g,%g." % (e+1,s_,l1_))                       
            if( (e+1)%10000 == 0):
                logs_train_dir = os.path.join(root, 'model.ckpt')
                saver.save(sess_target,logs_train_dir,global_step=count)
            if( (e+1)%100 == 0):
                t = int(time.time())
                print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    

def get_batch(image, label, batch_size, Capacity,Shuffle):

    image = tf.cast(image, tf.string)
    label = tf.convert_to_tensor(label,tf.int32)
    
    input_queue = tf.train.slice_input_producer([image, label],shuffle = Shuffle,capacity = Capacity)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [124, 124])
    image = tf.image.random_crop(image, [112, 112, 3])

    image_batch,label_batch= tf.train.batch([image,label],batch_size= batch_size,num_threads= 1, capacity = Capacity)
    
    label_batch = tf.cast(label_batch, tf.int32)
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

# MMD functions
def rbf_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def get_mmd(x, y, sigma_sqr=1.0):
    x_kernel = rbf_kernel(x, x)
    y_kernel = rbf_kernel(y, y)
    xy_kernel = rbf_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


################################################################################
### Helper functions to compute variances based on kernel matrices


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est


if __name__ == '__main__':
    main()
