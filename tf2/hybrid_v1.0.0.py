""" 
 @author   Maksim Penkin
"""

import os
import re
import time
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf

import src.utils as utils
from src.utils import AccumulativeDict
from base_runner import Runner
from datasets.regression_dataset import HybridDataset
from models.generator import ResidualFusionGenerator
from models.losses import L1, L2, PSNR, SSIM, calc_generator_loss


class Engine(Runner):
    def __init__(self, args):
        super(Engine, self).__init__(args=args)
        nengine_name = os.path.splitext(os.path.basename(__file__))[0]
        self.nengine_version = [x for x in list(re.findall('v(\d+)\.(\d+)\.(\d+)', nengine_name)[0])]
        self.nengine_name = nengine_name[:nengine_name.find('_v' + '.'.join(self.nengine_version))]
    
    def train(self):
        # Assert paths and save meta-info
        self.assert_train_paths()
        utils.dump2json(obj=vars(self.args), fp=os.path.join(self.checkpoints_config, 'train_cmd.json'))
        self.assert_summary_writers()
        # Create models
        self.G = self.create_model(trainable=True)
        self.build_model(self.G, input_shape=self.input_shape)
        
        # [optional] Training restoration
        if self.checkpoints_restore is not None:
            G_from = self.args.restore_checkpoint_template.format(self.args.restore_epoch,self.args.restore_step)
            self.G.load_weights(os.path.join(self.checkpoints, G_from))
            print("[*] Training restoration from " + G_from + ": Successfully loaded weights!")
        # Create Optimizers
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Create Losses
        self.loss_regression_list = [L1(weight=1.0), PSNR(weight=0.0)]
        # Training dataset
        self.train_regression_dataset = HybridDataset(image_list = self.train_regression_list, db_basedir = self.args.db_regression_basedir,
                                                           batch_size = self.args.batch_size,
                                                           patch_size = self.args.patch_size,
                                                           ch = self.ch,
                                                           num_random_crops = self.args.num_random_crops,
                                                           num_parallel_calls = self.args.num_parallel_calls,
                                                           buffer_size = self.args.buffer_size*self.args.num_random_crops,
                                                           prefetch = self.args.prefetch)
        self.num_train_batches = self.train_regression_dataset.num_batches
        train_regression_dataset_iter = iter(self.train_regression_dataset)
        # [optional] Validation dataset
        if self.val_list != '':
            self.val_dataset = HybridDataset(image_list = self.val_list, db_basedir = self.args.db_regression_basedir,
                                                batch_size = self.args.batch_size,
                                                ch = self.ch,
                                                mode = 'val',
                                                num_parallel_calls = self.args.num_parallel_calls,
                                                buffer_size = self.args.buffer_size,
                                                prefetch = self.args.prefetch)
            self.num_val_batches = self.val_dataset.num_batches
            val_dataset_iter = iter(self.val_dataset)

        @tf.function
        def train_step(inputs):
            gt, sketch, kellner = inputs
            with tf.GradientTape() as G_tape:
                generated_output = self.G(tf.concat([sketch, kellner], axis=3), training=self.training)
                G_loss_total, G_losses = calc_generator_loss(
                                            loss_regression_objects = self.loss_regression_list,
                                            loss_regularizarion = tf.math.add_n(self.G.losses),
                                            gt=gt, generated_output=generated_output)

            G_gradients = G_tape.gradient(G_loss_total, self.G.trainable_variables)
            self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
            return G_losses

        @tf.function
        def val_step(inputs):
            gt, sketch, kellner = inputs
            generated_output = self.G(tf.concat([sketch, kellner], axis=3), training=False)
            _, G_losses = calc_generator_loss(
                                        loss_regression_objects = [L1(), L2(weight=0.0), PSNR(weight=0.0), SSIM(weight=0.0)],
                                        gt=gt, generated_output=generated_output)
            return G_losses

        # Track best metrics
        self.best_metrics = {'l1':-1, 'l2':-1, 'psnr': -1, 'ssim': -1}
        for epoch in range(self.epoch_start, self.epoch_number):
            start = time.time()
            # Train for the whole train dataset
            train_metrics = AccumulativeDict()
            train_metrics_update_cnt = 0
            for i in range(self.num_train_batches):
                G_losses = train_step(next(train_regression_dataset_iter))
                
                train_metrics = train_metrics + G_losses
                train_metrics_update_cnt += 1
                
                if self.global_step % 100 == 0:
                    train_metrics = train_metrics / train_metrics_update_cnt
                    print(utils.get_training_str(train_metrics, epoch+1, self.epoch_number, i+1, self.num_train_batches))
                    utils.save_logs(self.train_summary_writer, train_metrics, step=self.global_step)
                    
                    train_metrics = AccumulativeDict()
                    train_metrics_update_cnt = 0
                
                self.global_step += 1
            print('[*] ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': Time for epoch {0} is {1:.2f}'.format(epoch+1, time.time()-start))
            self.G.save_weights(os.path.join(self.checkpoints, 'G_e{:04d}_s{:08d}'.format(epoch+1, self.global_step)))  
            
            # [optional] Validate for the whole validation dataset, if it was set by user
            if self.val_list != '':
                print('[*] ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': Validation...')
                val_metrics = AccumulativeDict()
                for i in tqdm(range(self.num_val_batches)):
                    val_metrics = val_metrics + val_step(next(val_dataset_iter))
                val_metrics = val_metrics / self.num_val_batches
                print(val_metrics)
                utils.save_logs(self.val_summary_writer, val_metrics, step=self.global_step)
                if val_metrics['psnr'] >= self.best_metrics['psnr']:
                    print('[*] NEW BEST!')
                    self.best_metrics['epoch'] = epoch+1
                    self.best_metrics['step'] = self.global_step
                    self.best_metrics['l1'] = val_metrics['l1'].numpy()
                    self.best_metrics['l2'] = val_metrics['l2'].numpy()
                    self.best_metrics['psnr'] = val_metrics['psnr'].numpy()
                    self.best_metrics['ssim'] = val_metrics['ssim'].numpy()
                    utils.dump2json({str(key): str(value) for key, value in self.best_metrics.items()},
                        os.path.join(self.checkpoints, 'best.json'))

    def create_model(self, trainable=True):
        return ResidualFusionGenerator(num_filters = 64,
                                    num_blocks = 3,
                                    ch = self.ch,
                                    name = 'residual_fusion_generator',
                                    trainable = trainable)
    
    def build_model(self, model, input_shape, dummy_check=True):
        b,h,w,c = input_shape
        model.build(input_shape=(b, h, w, c+c))
        if dummy_check:
            x = utils.get_dummy_tensor(batch_size=b, height=h, width=w, channels=c+c)
            y = model(x, training=self.training)

if __name__ == '__main__':
    print('TF version: {}'.format(tf.__version__))
    args = utils.parse_args()
    print(args)
    engine_name = os.path.basename(__file__)

    print('Engine {} has started.'.format(engine_name))
    
    print('Engine is initializing...')
    model = Engine(args)
    if model.stage == 'train':
        print('Engine is training...')
        model.train()
    if model.stage == 'apply':
        print('Engine is being applied...')
        model.apply()
    if model.stage == 'to_proto':
        print('Engine is converting weights to PB...')
        model.to_proto()
    if model.stage == 'to_saved_model':
        print('Engine is converting weights to saved_model...')
        model.to_saved_model()
    
    print('Engine {} has stopped.'.format(engine_name))


