""" 
 @author   Maksim Penkin
"""

import os, re
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph as convert_to_constants

import src.utils as utils


class Runner:

    def __init__(self, args):
        self.args = args
        self.nengine_name = os.path.splitext(os.path.basename(__file__))[0]
        self.nengine_version = []
       
        self.stage = self.args.stage 
        self.ch = self.args.ch

        if self.stage == 'train':
            self.training = True
            self.input_shape = (self.args.batch_size, self.args.patch_size, self.args.patch_size, self.ch)

            # Regression 
            self.train_regression_list = os.path.join(self.args.train_lists, self.args.db_regression_name, 'train', self.args.db_regression_name + '_' + self.args.db_regression_postfix + '.txt')
            # [optional] Validation
            self.val_list = ''
            if self.args.validate_from_txt:
                self.val_list = os.path.join(self.args.train_lists, self.args.db_regression_name, 'val', self.args.db_regression_name + '_' + self.args.db_regression_postfix + '.txt')

            self.epoch_number = self.args.epoch_number
            self.learning_rate = self.args.learning_rate 
            self.batch_size = self.args.batch_size 
            
            # [optional] Training restoration
            if (self.args.restore_epoch is not None) or (self.args.restore_step is not None):
                self.epoch_start = self.args.restore_epoch - 1
                self.global_step = self.args.restore_step
            else:
                self.epoch_start = 0
                self.global_step = 0

        elif self.stage == 'apply':
            self.training = False
            if (self.args.restore_epoch is None) or (self.args.restore_step is None):
                raise Exception('base_runner.py: def __init__(...): error: specify both restore_epoch and restore_step; found: restore_epoch {}, restore_step {}'.format(self.args.restore_epoch, self.args.restore_step))
            self.restore_epoch, self.global_step = self.args.restore_epoch, self.args.restore_step
            self.apply_list = self.args.apply_list
            _, ext = os.path.splitext(os.path.split(self.apply_list)[-1])
            if (ext != '.txt'):
                raise Exception('base_runner.py: def __init__(...): error: apply_list extension should be either *.txt, but found {}'.format(ext))
            
            self.bitdepth_out = self.args.bitdepth_out
            if self.bitdepth_out == 8:
                self.dtype_out = np.uint8
            elif self.bitdepth_out == 16:
                self.dtype_out = np.uint16

        elif self.stage == 'to_proto':
            self.training = False
            self.input_shape = (1, self.args.patch_size, self.args.patch_size, self.ch)
            if (self.args.restore_epoch is None) or (self.args.restore_step is None):
                raise Exception('base_runner.py: def __init__(...): error: specify both restore_epoch and restore_step; found: restore_epoch {}, restore_step {}'.format(self.args.restore_epoch, self.args.restore_step))
            self.restore_epoch, self.global_step = self.args.restore_epoch, self.args.restore_step

        elif self.stage == 'to_saved_model':
            self.training = False
            self.input_shape = (1, self.args.patch_size, self.args.patch_size, self.ch)
            if (self.args.restore_epoch is None) or (self.args.restore_step is None):
                raise Exception('base_runner.py: def __init__(...): error: specify both restore_epoch and restore_step; found: restore_epoch {}, restore_step {}'.format(self.args.restore_epoch, self.args.restore_step))
            self.restore_epoch, self.global_step = self.args.restore_epoch, self.args.restore_step
    
    def train(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError

    def to_proto(self):
        self.assert_to_proto_paths()
        utils.dump2json(obj=vars(self.args), fp=os.path.join(self.checkpoints_config, 'to_proto_cmd.json'))

        model = self.create_model(trainable=False)
        self.build_model(model, input_shape=self.input_shape)
        model.load_weights(self.checkpoints_restore)
        print("[*] Successfully loaded weights!")

        # Configure model input
        input_dtype = tf.float32
        compiled_model = model.get_keras_model(tf.keras.Input(shape=self.input_shape[1:], batch_size=1, name="low"))
        compiled_model.summary()

        # Freezing model weights, obtaining graph
        infer_func = tf.function(compiled_model)
        infer_func = infer_func.get_concrete_function(tf.TensorSpec(shape=compiled_model.inputs[0].shape,
                                                                    dtype=input_dtype))

        frozen_func, graph_def = convert_to_constants(infer_func)
        input_tensors = [
            tensor for tensor in frozen_func.inputs
            if tensor.dtype != tf.resource
        ]
        output_tensors = frozen_func.outputs

        graph_def = run_graph_optimizations(
            graph_def,
            input_tensors,
            output_tensors,
            config=get_grappler_config(["constfold", "function"]),
            graph=frozen_func.graph)
        tf.io.write_graph(graph_or_graph_def=graph_def,
                          logdir=self.checkpoints_to_proto,
                          name="model.pb",
                          as_text=False)

    def to_saved_model(self):
        self.assert_to_saved_model_paths()
        utils.dump2json(obj=vars(self.args), fp=os.path.join(self.checkpoints_config, 'to_saved_model_cmd.json'))

        model = self.create_model(trainable=False)
        self.build_model(model, input_shape=self.input_shape)
        model.load_weights(self.checkpoints_restore)
        print("[*] Successfully loaded weights!")
        model.save(self.checkpoints_to_saved_model)

    def assert_train_paths(self):
        self.checkpoints = os.path.join(self.args.checkpoints, '_'.join([self.nengine_name, self.args.db_regression_name, 'v' + '.'.join(self.nengine_version), self.args.postfix]))
        self.checkpoints_config = os.path.join(self.checkpoints, 'config')
        self.checkpoints_logs = os.path.join(self.checkpoints, 'logs')
        self.checkpoints_logs_train = os.path.join(self.checkpoints, 'logs', 'train')
        if self.val_list != '':
            self.checkpoints_logs_val = os.path.join(self.checkpoints, 'logs', 'val')

        # [optional] Training restoration
        if (self.args.restore_epoch is not None) and (self.args.restore_step is not None): 
            self.checkpoints_restore = os.path.join(self.checkpoints, self.args.restore_checkpoint_template.format(self.args.restore_epoch,self.args.restore_step))
            create_folder_kwargs = dict(force=False, raise_except_if_exists=False)
            if self.args.train_force:
                print('[WARNING] restore_epoch is not None and restore_step is not None, but train_force is True! Training directories will not be cleaned, training will be restored. If you want to train from scratch, do not set restore_epoch and restore_step arguments')
        else:
            self.checkpoints_restore = None 
            create_folder_kwargs = dict(force=self.args.train_force)

        utils.create_folder(self.checkpoints, **create_folder_kwargs)
        utils.create_folder(self.checkpoints_config, **create_folder_kwargs)
        utils.create_folder(self.checkpoints_logs, **create_folder_kwargs)
        utils.create_folder(self.checkpoints_logs_train, **create_folder_kwargs)
        if self.val_list != '':
            utils.create_folder(self.checkpoints_logs_val, **create_folder_kwargs)

    def assert_apply_paths(self):
        self.checkpoints = os.path.join(self.args.checkpoints, '_'.join([self.nengine_name, self.args.db_regression_name, 'v' + '.'.join(self.nengine_version), self.args.postfix]))
        if not os.path.exists(self.checkpoints):
            raise Exception('base_runner.py: class Runner: def assert_apply_paths(...): error: checkpoints directory {} does not exist.'.format(self.checkpoints))
        
        self.checkpoints_restore = os.path.join(self.checkpoints, self.args.restore_checkpoint_template.format(self.restore_epoch,self.global_step))
        self.checkpoints_config = os.path.join(self.checkpoints, 'config')
        
        target_list = os.path.splitext(os.path.split(self.apply_list)[-1])[0]
        self.apply_savedir = os.path.join(self.args.apply_savedir, target_list + '_' + '_'.join([self.nengine_name, self.args.db_regression_name, 'v' + '.'.join(self.nengine_version), self.args.postfix]))
        
        utils.create_folder(self.checkpoints_config, force=False, raise_except_if_exists=False)
        utils.create_folder(self.apply_savedir, force=self.args.apply_force)
    
    def assert_to_proto_paths(self):
        self.checkpoints = os.path.join(self.args.checkpoints, '_'.join([self.nengine_name, self.args.db_regression_name, 'v' + '.'.join(self.nengine_version), self.args.postfix]))
        if not os.path.exists(self.checkpoints):
            raise Exception('base_runner.py: class Runner: def assert_to_proto_paths(...): error: checkpoints directory {} does not exist.'.format(self.checkpoints))
        
        self.checkpoints_restore = os.path.join(self.checkpoints, self.args.restore_checkpoint_template.format(self.restore_epoch,self.global_step))
        self.checkpoints_config = os.path.join(self.checkpoints, 'config')
        self.checkpoints_to_proto = os.path.join(self.checkpoints, 'pb_model')
        utils.create_folder(self.checkpoints_config, force=False, raise_except_if_exists=False)
        utils.create_folder(self.checkpoints_to_proto, force=False)
   
    def assert_to_saved_model_paths(self):
        self.checkpoints = os.path.join(self.args.checkpoints, '_'.join([self.nengine_name, self.args.db_regression_name, 'v' + '.'.join(self.nengine_version), self.args.postfix]))
        if not os.path.exists(self.checkpoints):
            raise Exception('base_runner.py: class Runner: def assert_to_saved_model_paths(...): error: checkpoints directory {} does not exist.'.format(self.checkpoints))
        
        self.checkpoints_restore = os.path.join(self.checkpoints, self.args.restore_checkpoint_template.format(self.restore_epoch,self.global_step))
        self.checkpoints_config = os.path.join(self.checkpoints, 'config')
        self.checkpoints_to_saved_model = os.path.join(self.checkpoints, 'saved_model')
        utils.create_folder(self.checkpoints_config, force=False, raise_except_if_exists=False)
        utils.create_folder(self.checkpoints_to_saved_model, force=False)

    def assert_summary_writers(self):
        self.train_summary_writer = tf.summary.create_file_writer(self.checkpoints_logs_train)
        if self.val_list != '':
            self.val_summary_writer = tf.summary.create_file_writer(self.checkpoints_logs_val)

    def create_model(self, trainable=True):
        raise NotImplementedError

    def build_model(self, model, input_shape, dummy_check=True):
        raise NotImplementedError

if __name__ == '__main__':
    print('TF version: {}'.format(tf.__version__))
    args = utils.parse_args()
    print(args)
    engine_name = os.path.basename(__file__)

    print('Engine {} has started.'.format(engine_name))
    
    print('Engine is initializing...')
    model = Runner(args)
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


